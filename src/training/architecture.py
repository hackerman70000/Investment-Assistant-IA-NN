import tensorflow as tf


def build_model(input_shape: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_shape,))
    x = tf.keras.layers.Dropout(0.1)(inputs)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        loss="binary_crossentropy",
        optimizer="Adam",
        metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )

    return model
