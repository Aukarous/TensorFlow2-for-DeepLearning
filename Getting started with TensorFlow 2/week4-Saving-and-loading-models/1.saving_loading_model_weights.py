from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping


def check_point():
    """👇 this network is actually a binary classifier, the linear unit output will be squeezed
    through a sigmoid activation function that is being handled by the loss-function itself """
    model = Sequential([
        Dense(64, activation='sigmoid', input_shape=(10,)),
        Dense(1)  # 👈 output is a single neuron with a linear activation
    ])
    model.compile(optimizer='sgd', loss=BinaryCrossentropy(from_logits=True))
    """ only the model weights will be saved by this callback and not the architecture """
    checkpoint = ModelCheckpoint('my_model', save_weights_only=True)  # my_model.h5
    """ This callback will save the model weights after every epoch, 
     because we're using the same file name to save them model, the saved weights will get overwritten every epoch."""
    model.fit(X_train, y_train, epochs=10, callbacks=[checkpoint])


def early_stopping():
    model = Sequential()

check_point()

early_stopping()
