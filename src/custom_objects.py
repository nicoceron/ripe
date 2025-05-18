#!/usr/bin/env python3
"""
Custom objects module for Fruit Quality Classification.
This module defines custom Keras classes that need to be registered
for proper model loading.
"""

import tensorflow as tf
from tensorflow import keras

@tf.keras.utils.register_keras_serializable(package="Custom")
class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2.5, alpha=0.5, label_smoothing=0.03, **kwargs):
        super(FocalLoss, self).__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        # Apply label smoothing
        num_classes = y_true.shape[-1]
        y_true = y_true * (1.0 - self.label_smoothing) + (self.label_smoothing / num_classes)
        
        # Apply focal loss
        y_pred = tf.clip_by_value(y_pred, keras.backend.epsilon(), 1.0 - keras.backend.epsilon())
        ce = -y_true * tf.math.log(y_pred)
        weight = tf.math.pow(1.0 - y_pred, self.gamma) * y_true
        fl = self.alpha * weight * ce
        reduced_fl = tf.reduce_sum(fl, axis=-1)
        return reduced_fl
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha,
            "label_smoothing": self.label_smoothing
        })
        return config

# Dictionary of custom objects for model loading
custom_objects = {
    'FocalLoss': FocalLoss
}

def get_custom_objects():
    """Returns a dictionary of custom objects required for model loading"""
    return custom_objects 