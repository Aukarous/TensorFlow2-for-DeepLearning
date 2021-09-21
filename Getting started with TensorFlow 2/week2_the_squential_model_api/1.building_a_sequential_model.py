from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

model = Sequential([
    Flatten(input_shape=(28,28)),  # (784,)
    Dense(64, activation='relu',input_shape=(784,)),
    Dense(10, activation='softmax')
])

"""
equivalent:
model = Sequential()
model.add(Dense(64,activation='relu',input_shape=(784,)))
model.add(Dense(10,activation='softmax'))
"""
