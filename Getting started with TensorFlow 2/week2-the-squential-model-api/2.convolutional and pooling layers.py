from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
model = Sequential([
    # first dimension will always be the batch size, batch size is flexible
    Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 3)),  # (None, 30, 30, 16)
    MaxPooling2D((3, 3)),  # (None, 10, 10, 16)
    Flatten(),  # (None, 1600)
    Dense(64, activation='relu'),  # (None, 64)
    Dense(10, activation='softmax')  # (None, 10)
])
a = 1