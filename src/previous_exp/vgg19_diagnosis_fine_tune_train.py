import sys
import argparse
import numpy as np
import random
import os
import gc
from pathlib import Path, PurePath
import logging
import json

import pandas as pd
from sklearn.utils import class_weight

import keras
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.losses import BinaryCrossentropy, Hinge, SquaredHinge, LogCosh
from tensorflow.keras.metrics import AUC, Accuracy, Precision, Recall, BinaryAccuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, balanced_accuracy_score, matthews_corrcoef


def set_seeds():
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)

# Define custom F1 score metric
@keras.saving.register_keras_serializable()
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

def get_data_generators(train_path, valid_path, test_path, best_params):
    # Data generators
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input_vgg19,
        rotation_range=best_params['rotation_range'],
        width_shift_range=best_params['width_shift_range'],
        height_shift_range=best_params['height_shift_range'],
        horizontal_flip=best_params['horizontal_flip'],
        vertical_flip=best_params['vertical_flip'],
        zoom_range=[1 + best_params['zoom_range'], 1 - best_params['zoom_range']],
        brightness_range=[1 - best_params['brightness_range'], 1 + best_params['brightness_range']] if best_params['brightness_range'] != 0 else None
    )
    
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_vgg19)
    
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_vgg19)

    classes = {'2SKC_No_Glaucoma': 0, '2SKA_Suspected_Glaucoma': 1}

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(224, 224),
        class_mode='binary',
        classes = classes
    )
    
    validation_generator = val_datagen.flow_from_directory(
        valid_path,
        target_size=(224, 224),
        class_mode='binary',
        classes = classes
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(224, 224),
        class_mode='binary',
        classes = classes
    )
    
    print("train_generator.class_indices : ", train_generator.class_indices)
    print("validation_generator.class_indices : ", validation_generator.class_indices)
    print("test_generator.class_indices : ", test_generator.class_indices)
    
    return train_generator, validation_generator, test_generator

def fine_tune_model(train_path, valid_path, test_path, output_path, best_params, model_name, original_model_path):
    set_seeds()

    train_generator, validation_generator, test_generator = get_data_generators(train_path, valid_path, test_path, best_params)
    
    # Load the previously trained model
    model = load_model(original_model_path, custom_objects={'f1_score_normal': f1_score_normal})
    
    
    # Get the VGG19 base model
    base_model = model.get_layer('vgg19')  # Using 'vgg19' as the layer name
    
    # Set the VGG19 base model layers up to a certain point to non-trainable
    for layer in base_model.layers[:best_params['fine_tune_at']]:
        layer.trainable = False
    for layer in base_model.layers[best_params['fine_tune_at']:]:
        layer.trainable = True
    
    # Set trainability for layers after the base model
    for layer in model.layers:
        if layer.name not in ['vgg19']:  # This excludes the base VGG19 model
            layer.trainable = True
    
    # Print the trainable status of the layers
    for layer in model.layers:
        print(f"Layer {layer.name}: trainable = {layer.trainable}")
        if layer.name == 'vgg19':  # For the VGG19 base model
            for inner_layer in layer.layers:
                print(f"  Inner Layer {inner_layer.name}: trainable = {inner_layer.trainable}")

    
    # Compile model with a lower learning rate
    fine_tuning_lr = best_params['fine_tuning_learning_rate_adam'] * 0.1  # Reduce learning rate
    optimizer = Adam(learning_rate=fine_tuning_lr)
    model.compile(
        optimizer=optimizer,
        loss=BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.AUC(curve="ROC", name="roc_auc_score"),
            f1_score_normal,
            tf.keras.metrics.BinaryAccuracy(name="accuracy_score"),
        ]
    )
    
    # Training
    class_weights = None
    if best_params['use_class_weights']:
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
        class_weights = dict(enumerate(class_weights))

    num_workers = os.cpu_count()

    training_log = model.fit(
        train_generator,
        epochs=50,  # Reduce number of epochs for fine-tuning
        validation_data=validation_generator,
        batch_size=best_params['batch_size'],
        class_weight=class_weights,
        workers=num_workers,
        use_multiprocessing=True,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=10, verbose=1),
            EarlyStopping(monitor='val_roc_auc_score', mode='max', verbose=1, patience=8, restore_best_weights=True),
        ],
    )

    # Evaluate the model on the test set
    results = model.evaluate(test_generator)
    logging.info(f"Test results - {results}")
    print(f"Model Eval results: {results}")

    # Save the fine-tuned model
    fine_tuned_model_name = f'{model_name}'
    model.save(os.path.join(output_path, f'{fine_tuned_model_name}.h5'))
    
    # Save training history
    hist_df = pd.DataFrame(training_log.history) 
    hist_df.to_csv(os.path.join(output_path, f'training_history_{fine_tuned_model_name}.csv'), index=False)
    
    logging.info(f"{fine_tuned_model_name} Model fine-tuned, Model and training history are saved successfully.")

def main(train_path, valid_path, test_path, output_path, best_hyperparameters_json_path, model_name, original_model_path):
    logging.basicConfig(level=logging.INFO)
    
    # Load best parameters from JSON
    with open(best_hyperparameters_json_path, 'r') as file:
        best_params = json.load(file)

    fine_tune_model(train_path, valid_path, test_path, output_path, best_params, model_name, original_model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True, help='Path to the training images')
    parser.add_argument('--valid_path', type=str, required=True, help='Path to the validation images')
    parser.add_argument('--test_path', type=str, required=True, help='Path to the test images')
    parser.add_argument('--output_path', type=str, required=True, help='Path where the fine-tuned model should be saved')
    parser.add_argument('--best_hyperparameters_json_path', type=str, required=True, help='Path to the JSON file with best hyperparameters')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the fine-tuned model')
    parser.add_argument('--original_model_path', type=str, required=True, help='Path to the original trained model')
    args = parser.parse_args()

    main(args.train_path, args.valid_path, args.test_path, args.output_path, args.best_hyperparameters_json_path, args.model_name, args.original_model_path)


