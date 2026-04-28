import tensorflow as tf
from tensorflow.keras import layers, Model


def build_lstm_model(timesteps=12, num_features=4):

    inputs = tf.keras.Input(shape=(timesteps, num_features))

    # First LSTM layer — return_sequences=True passes the full sequence to the next layer
    x = layers.LSTM(units=64, return_sequences=True)(inputs)

    # Second LSTM layer — summarises the sequence into a single output vector
    x = layers.LSTM(units=32, return_sequences=False)(x)

    # Dense layer — interprets the LSTM output
    x = layers.Dense(units=16, activation='relu')(x)

    # Dropout — randomly deactivates 20% of neurons during training to reduce overfitting
    x = layers.Dropout(rate=0.2)(x)

    # Output — single sigmoid neuron, produces a score between 0.0 (mild) and 1.0 (severe)
    outputs = layers.Dense(units=1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs, name="PatientLSTM")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":
    model = build_lstm_model()
    model.summary()
