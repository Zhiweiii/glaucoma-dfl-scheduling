import argparse
import numpy as np
import random
import os
import logging
import json
import ast
import time

import pandas as pd
from sklearn.utils import class_weight

# import keras
import tensorflow as tf
from tensorflow import keras
logger = logging.getLogger(__name__)
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping

class EpochWeightStatsCallback(tf.keras.callbacks.Callback):
    """
    Logs per-epoch stats about the model weights:
      - global L2 norm of all weights
      - L2 norm of the weight update (difference from previous epoch)
    Saves a CSV file at the end of training.
    """
    def __init__(self, log_path, model_name):
        super().__init__()
        self.log_path = log_path
        self.model_name = model_name
        self.prev_weights = None
        self.epoch_stats = []
        self.epoch_start_time = None

    def _global_l2_norm(self, weights):
        # weights is a list of numpy arrays
        total = 0.0
        for w in weights:
            if w is None:
                continue
            w = np.asarray(w)
            total += np.sum(np.square(w))
        return float(np.sqrt(total))

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        # On the very first epoch, snapshot initial weights as baseline
        if self.prev_weights is None:
            self.prev_weights = [np.copy(w) for w in self.model.get_weights()]

    def on_epoch_end(self, epoch, logs=None):
        current_weights = self.model.get_weights()

        # L2 norm of weights at end of epoch
        weight_l2 = self._global_l2_norm(current_weights)

        # L2 norm of weight change during this epoch
        if self.prev_weights is not None:
            deltas = [
                (cw - pw) if (cw is not None and pw is not None) else 0.0
                for cw, pw in zip(current_weights, self.prev_weights)
            ]
            delta_l2 = self._global_l2_norm(deltas)
        else:
            delta_l2 = float('nan')

        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time is not None else float('nan')

        # >>> NEW: capture learning rate <<<
        current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

        # >>> NEW: capture SGD momentum if present <<<
        optimizer = self.model.optimizer
        momentum_var = getattr(optimizer, "momentum", None)
        if momentum_var is not None:
            current_momentum = float(tf.keras.backend.get_value(momentum_var))
        else:
            current_momentum = float('nan')

        # Store stats; also merge in basic logs like loss / val_loss if you want
        stat = {
            "epoch": epoch,
            "weight_l2": weight_l2,
            "weight_delta_l2": delta_l2,
            "epoch_time_sec": epoch_time,
            "learning_rate": current_lr,
            "momentum": current_momentum
        }
        if logs:
            for k, v in logs.items():
                stat[k] = float(v) if v is not None else v

        self.epoch_stats.append(stat)

        # Prepare for next epoch
        self.prev_weights = [np.copy(w) for w in current_weights]

    def on_train_end(self, logs=None):
        # Save to CSV at the end of training
        if not self.epoch_stats:
            return

        df = pd.DataFrame(self.epoch_stats)
        out_csv = os.path.join(
            self.log_path,
            f"weight_stats_{self.model_name}.csv",
        )
        df.to_csv(out_csv, index=False)
        logger.info(
            "Per-epoch weight stats for %s saved to %s",
            self.model_name,
            out_csv,
        )


def set_seeds(seed: int = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

# @keras.saving.register_keras_serializable()
@keras.utils.register_keras_serializable()
def f1_score_normal(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def preprocess_input_vgg19(x):
    return tf.keras.applications.vgg19.preprocess_input(x)

def get_data_generators(
    train_csv_path,
    valid_path,
    test_path,
    best_params,
    classes,
    x_col: str = "full_path",
    y_col: str = "class",
):
    """
    Build data generators.

    Training data:
      - Loaded from a CSV file whose rows are already in the desired order.
      - The CSV must contain at least:
            x_col: column with full image paths (default: 'full_path')
            y_col: column with class labels (default: 'class')
      - We set shuffle=False so Keras iterates rows exactly in CSV order.

    Validation/test:
      - Still loaded from directory structure using flow_from_directory.
      - These are not reordered and remain fixed for all experiments.
    """
    if not classes:
        classes = {'No_Glaucoma': 0, 'Suspected_Glaucoma': 1}

    # Training generator without augmentation: only preprocessing
    train_datagen = image.ImageDataGenerator(
        preprocessing_function=preprocess_input_vgg19
    )

    # No augmentation for val/test, just preprocessing
    val_datagen = image.ImageDataGenerator(
        preprocessing_function=preprocess_input_vgg19
    )
    test_datagen = image.ImageDataGenerator(
        preprocessing_function=preprocess_input_vgg19
    )

    # ---------- TRAIN GENERATOR: use CSV order as-is ----------
    df_train = pd.read_csv(train_csv_path)

    missing_cols = [col for col in [x_col, y_col] if col not in df_train.columns]
    if missing_cols:
        raise ValueError(
            f"train_csv_path is missing required columns: {missing_cols}"
        )

    # We use directory=None so x_col can contain absolute paths.
    # shuffle=False ensures the iteration order follows the CSV row order exactly.
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=df_train,
        directory=None,
        x_col=x_col,
        y_col=y_col,
        target_size=(224, 224),
        class_mode='binary',
        shuffle=False,
    )

    # ---------- VAL / TEST: always deterministic, no shuffle ----------
    validation_generator = val_datagen.flow_from_directory(
        valid_path,
        target_size=(224, 224),
        class_mode='binary',
        classes=classes,
        shuffle=False,
    )

    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(224, 224),
        class_mode='binary',
        classes=classes,
        shuffle=False,
    )

    logger.info("train CSV: %s", train_csv_path)
    logger.info("validation path: %s", valid_path)
    logger.info("test path: %s", test_path)

    logger.info("train_generator.class_indices: %s", train_generator.class_indices)
    logger.info("validation_generator.class_indices: %s", validation_generator.class_indices)
    logger.info("test_generator.class_indices: %s", test_generator.class_indices)

    return train_generator, validation_generator, test_generator

def train_and_evaluate(
    train_csv_path,
    valid_path,
    test_path,
    model_path,
    log_path,
    model_name,
    best_hyperparameters_json_path=None,
    classes={'No_Glaucoma': 0, 'Suspected_Glaucoma': 1},
    optimizer_name: str = "adam",
    seed: int = 42,
    x_col: str = "full_path",
    y_col: str = "class",
):

    # Ensure log directory exists
    os.makedirs(log_path, exist_ok=True)
    run_log_path = os.path.join(log_path, f"{model_name}.log")

    # Configure a file handler for this run (timestamped entries)
    file_handler = logging.FileHandler(run_log_path)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    # Avoid adding duplicate handlers if this function is ever called multiple times
    if not any(
        isinstance(h, logging.FileHandler)
        and getattr(h, "baseFilename", None) == file_handler.baseFilename
        for h in logger.handlers
    ):
        logger.addHandler(file_handler)

    logger.info("===== Starting run: %s =====", model_name)
    logger.info("Train CSV: %s", train_csv_path)
    logger.info("Valid path: %s", valid_path)
    logger.info("Test path: %s", test_path)
    logger.info("Optimizer: %s", optimizer_name)
    logger.info("Seed: %d", seed)

    # ----------------- Hyperparameters -----------------
    if best_hyperparameters_json_path is None:
        best_params = {
            "rotation_range": 5,
            "width_shift_range": 0.04972485058923855,
            "height_shift_range": 0.03008783098167697,
            "horizontal_flip": True,
            "vertical_flip": True,
            "zoom_range": -0.044852124875001065,
            "brightness_range": -0.02213535357633886,
            "use_class_weights": True,
            "pooling": "global_average",
            "dense_layers": 3,
            "units_layer_0": 64,
            "activation_func_0": "sigmoid",
            "batch_norm_0": True,
            "dropout_0": 0.09325925519992712,
            "units_layer_1": 64,
            "activation_func_1": "tanh",
            "batch_norm_1": True,
            "dropout_1": 0.17053317552512925,
            "units_layer_2": 32,
            "activation_func_2": "relu",
            "batch_norm_2": True,
            "dropout_2": 0.31655072863663397,
            "fine_tune_at": 7,
            "fine_tuning_learning_rate_adam": 0.00001115908855034341,
            "batch_size": 32,
            # You can optionally add SGD-specific defaults here:
            # "fine_tuning_learning_rate_sgd": 1e-3,
            # "sgd_momentum": 0.9,
            # "sgd_nesterov": True,
        }
    else:
        with open(best_hyperparameters_json_path, 'r') as file:
            best_params = json.load(file)

    # Seed everything and make generators with the requested ordering
    set_seeds(seed)

    train_generator, validation_generator, test_generator = get_data_generators(
        train_csv_path=train_csv_path,
        valid_path=valid_path,
        test_path=test_path,
        best_params=best_params,
        classes=classes,
        x_col=x_col,
        y_col=y_col,
    )

    K.clear_session()
    strategy = tf.distribute.OneDeviceStrategy("/GPU:0")
    with strategy.scope():
        base_model = VGG19(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3),
        )
        base_model.trainable = False

        inputs = keras.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)

        if best_params['pooling'] == 'global_average':
            x = GlobalAveragePooling2D()(x)
        else:
            x = Flatten()(x)

        for i in range(best_params['dense_layers']):
            num_units = best_params[f'units_layer_{i}']
            activation = best_params[f'activation_func_{i}']
            x = Dense(num_units, activation=activation)(x)

            if best_params[f'batch_norm_{i}']:
                # use CPU to avoid GPU JIT/XLA issues
                logger.info(
                    "Applying BatchNormalization for layer %d on CPU to avoid GPU JIT/XLA issues.",
                    i,
                )
                with tf.device("/CPU:0"):
                    x = BatchNormalization()(x)
                # workaround end
            x = Dropout(best_params[f'dropout_{i}'])(x)

        outputs = Dense(1, activation='sigmoid')(x)
        model = Model(inputs, outputs)

        base_model.trainable = True
        for layer in base_model.layers[:best_params['fine_tune_at']]:
            layer.trainable = False

        # ----------------- Optimizer selection -----------------
        opt_name = optimizer_name.lower()
        if opt_name == "adam":
            lr = best_params.get(
                "fine_tuning_learning_rate_adam",
                1e-4,
            )
            optimizer = Adam(learning_rate=lr)
        elif opt_name == "sgd":
            lr = best_params.get(
                "fine_tuning_learning_rate_sgd",
                best_params.get("fine_tuning_learning_rate_adam", 1e-3),
            )
            momentum = best_params.get("sgd_momentum", 0.9)
            nesterov = best_params.get("sgd_nesterov", True)
            optimizer = SGD(
                learning_rate=lr,
                momentum=momentum,
                nesterov=nesterov,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        model.compile(
            optimizer=optimizer,
            loss=BinaryCrossentropy(),
            metrics=[
                tf.keras.metrics.AUC(curve="ROC", name="roc_auc_score"),
                f1_score_normal,
                tf.keras.metrics.BinaryAccuracy(name="accuracy_score"),
            ],
        )

        logger.info("Model weights device location: %s", model.weights[0].device)

        # ----------------- Class weights -----------------
        class_weights = None
        if best_params['use_class_weights']:
            cw = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=np.unique(train_generator.classes),
                y=train_generator.classes,
            )
            class_weights = dict(enumerate(cw))

        # num_workers = os.cpu_count()
        num_workers = 1

        # ----------------- Training (per-epoch history saved below) -----------------
        # Callback to record per-epoch weight norms and update norms
        weight_stats_cb = EpochWeightStatsCallback(
            log_path=log_path,
            model_name=model_name,
        )

        training_log = model.fit(
            train_generator,
            epochs=100,
            validation_data=validation_generator,
            batch_size=best_params['batch_size'],
            class_weight=class_weights,
            workers=num_workers,
            use_multiprocessing=False,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=10, verbose=1),
                EarlyStopping(
                    monitor='val_roc_auc_score',
                    mode='max',
                    verbose=1,
                    patience=8,
                    restore_best_weights=True,
                ),
                weight_stats_cb,
            ],
        )

        # ----------------- Test evaluation -----------------
        # This evaluates on the test set and saves metrics as a CSV in log_path.
        test_results = model.evaluate(test_generator, verbose=0)
        metric_names = model.metrics_names  # e.g. ['loss', 'roc_auc_score', 'f1_score_normal', 'accuracy_score']
        metrics_dict = {name: value for name, value in zip(metric_names, test_results)}

        metrics_df = pd.DataFrame([metrics_dict])
        metrics_summary_csv = os.path.join(
            log_path,
            f'metrics_summary_{model_name}.csv',
        )
        metrics_df.to_csv(metrics_summary_csv, index=False)
        logger.info(
            "Test metrics for %s saved to %s",
            model_name,
            metrics_summary_csv,
        )

    # ----------------- Save model + per-epoch curves -----------------
    if model_name:
        model_save_path = os.path.join(model_path, f'{model_name}.h5')
    else:
        model_save_path = os.path.join(model_path, 'Trained_model.h5')

    model.save(model_save_path)

    hist_df = pd.DataFrame(training_log.history)
    training_history_csv = os.path.join(
        log_path,
        f'training_history_{model_name}.csv',
    )
    hist_df.to_csv(training_history_csv, index=False)
    logger.info(
        "%s: model trained; model and training history saved successfully.",
        model_name,
    )

    return model_save_path, training_history_csv, metrics_summary_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,  help='Path to load or save model')
    parser.add_argument('--model_name', type=str, help='Model name for saving or loading') # {optimizer}_{ordering}_{seed}
    parser.add_argument('--train_csv_path', type=str, help='CSV file listing training images and labels in desired order')
    parser.add_argument('--valid_path', type=str, help='Path to validation images')
    parser.add_argument('--test_path', type=str, help='Path to test images')
    parser.add_argument('--log_path', type=str, help='Path to save training logs')
    parser.add_argument('--hyperparameters_json_path', required=False, default=None, type=str, help='Path to hyperparameters JSON')
    parser.add_argument('--classes_definition', type=str, required=False, default=None, help='A dictionary of classes definition')
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam',
        choices=['adam', 'sgd'],
        help='Optimizer to use for training.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility.'
    )
    parser.add_argument(
        '--x_col',
        type=str,
        default='full_path',
        help='Column name in training CSV that stores full image paths.'
    )
    parser.add_argument(
        '--y_col',
        type=str,
        default='class',
        help='Column name in training CSV that stores class labels.'
    )

    args = parser.parse_args()

    if args.classes_definition:
        try:
            args.classes_definition = ast.literal_eval(args.classes_definition)
            if not isinstance(args.classes_definition, dict):
                print("Classes definition has to be a dictionary, e.g. {'No_Glaucoma': 0, 'Suspected_Glaucoma': 1}")
                args.classes_definition = None  
        except Exception as e:
            print("Classes definition has to be a dictionary, e.g. {'No_Glaucoma': 0, 'Suspected_Glaucoma': 1}")
            args.classes_definition = None 
    else:
        args.classes_definition = None

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    required_args = ['train_csv_path', 'valid_path', 'test_path', 'log_path', 'model_path', 'model_name']
    missing_args = [arg for arg in required_args if getattr(args, arg) is None]
    if missing_args:
        parser.error(f"Missing required arguments: {', '.join(missing_args)}")

    train_and_evaluate(
        train_csv_path=args.train_csv_path,
        valid_path=args.valid_path,
        test_path=args.test_path,
        model_path=args.model_path,
        log_path=args.log_path,
        model_name=args.model_name,
        best_hyperparameters_json_path=args.hyperparameters_json_path,
        classes=args.classes_definition,
        optimizer_name=args.optimizer,
        seed=args.seed,
        x_col=args.x_col,
        y_col=args.y_col,
    )
    
if __name__ == '__main__':
    main()




