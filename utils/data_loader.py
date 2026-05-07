import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_train_val_generators(data_path, batch_size=32, target_size=(224, 224)):
    # Normalización y Data Augmentation ligero
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        horizontal_flip=True,
        rotation_range=10,
        zoom_range=0.1
    )

    train_gen = datagen.flow_from_directory(
        data_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        data_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_gen, val_gen