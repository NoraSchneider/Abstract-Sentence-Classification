import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.losses import SparseCategoricalCrossentropy, KLDivergence
from tensorflow.keras.metrics import SparseCategoricalAccuracy

from sklearn.metrics import f1_score
import numpy as np


"""
    Loss function for KD-Loss
"""
def distil_loss(Y_true, Y_pred, num_labels, alpha = 0.3, T = 1):

    # Compute temperature-adjusted predictions
    teacher_probs = tf.nn.softmax(Y_pred[:, num_labels:] / T, axis=1)
    student_probs = tf.nn.softmax(Y_pred[:, :num_labels] / T, axis=1)
    
    # Compute the KL-divergence between teacher and student distribution
    distil_loss = KLDivergence()(
        teacher_probs, student_probs
    )

    # Compute standard loss
    student_loss = SparseCategoricalCrossentropy()(
        Y_true, tf.nn.softmax(Y_pred[:, :num_labels], axis=1)
    )

    # Total loss is a weighted average between the KL-divergence
    # and the normal loss
    return (1 - alpha) * student_loss + alpha * distil_loss


"""
    Custom accuracy metric for KD models
"""
class KDAccuracy(SparseCategoricalAccuracy):

    def __init__(self, num_labels, name="accuracy", dtype=None):
        super().__init__(name, dtype)
        self.num_labels = num_labels

    def update_state(self, y_true, y_pred, *args, **kwargs):
        super().update_state(y_true, y_pred[:, :self.num_labels])


"""
    Function to add KD to a model
"""
def add_KD(alpha, T, base_model, num_labels):
    
    # Create model which takes both standard inputs as well as teacher predictions
    sub = base_model()
    input1 = sub.input
    input2 = Input(shape=(num_labels, ))
    merged = Concatenate(axis=-1)([
        sub.output, 
        input2
    ])
    model = Model(inputs=[input1, input2], outputs=merged)
    
    # Compile model with adapted loss and metric
    model.compile(
        optimizer="adam", 
        loss=lambda y_true, y_pred: distil_loss(y_true, y_pred, num_labels, alpha, T), 
        metrics=[KDAccuracy()]
    )

    return model
