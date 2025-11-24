import tensorflow as tf
from tensorflow.keras import layers, models

class DeepfakeDetector:
    def __init__(self, input_shape=(128, 157, 1)): # Default shape for 5s audio @ 16kHz, n_mels=128
        self.input_shape = input_shape
        self.model = self.build_model()

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape)

        # CNN Block
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization()(x)

        # Reshape for BiLSTM (prepare spatial features for temporal processing)
        # Shape after pooling: (batch, freq, time, filters)
        # We want (batch, time, features) for BiLSTM input
        # Permute to (batch, time, freq, filters) first
        x = layers.Permute((2, 1, 3))(x)
        
        # Get current shape to reshape dynamically
        # shape is (None, time_steps, freq_bins, filters)
        # We merge freq_bins and filters
        s = x.shape
        x = layers.Reshape((-1, s[2] * s[3]))(x)

        # BiLSTM Block
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
        x = layers.Bidirectional(layers.LSTM(32))(x)

        # Dense Block
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output
        outputs = layers.Dense(1, activation='sigmoid')(x)

        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        return model

    def summary(self):
        self.model.summary()
