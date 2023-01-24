from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import os

images_path = r'E:\Non Medal Clips\images'
image_size = (128, 128)
batch_size = 64

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(images_path, validation_split=0.2, subset='both', seed=123, image_size=image_size, batch_size=batch_size)