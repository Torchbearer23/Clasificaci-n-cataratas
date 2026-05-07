import tensorflow as tf

def get_EfficientNetV2S(input_shape=(224, 224, 3), num_classes=3):
    base_model = tf.keras.applications.EfficientNetV2S(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Congelamos para Fine-tuning

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model