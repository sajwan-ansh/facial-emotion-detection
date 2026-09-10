import tensorflow as tf

from config import IMG_SIZE, NUM_CLASSES


def build_model():
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="face")
    x = inputs
    for filters, dropout in ((32, 0.20), (64, 0.25), (128, 0.30), (256, 0.35)):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.45)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="emotion")(x)
    model = tf.keras.Model(inputs, outputs, name="fer2013_cnn_v2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    return model
