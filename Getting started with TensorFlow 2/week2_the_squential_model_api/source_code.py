import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense


def compiling_model():
    """
    Metrics will be calculated for each epoch of training along with the evaluation of the loss function
    on the training data.
    Each of these strings is a reference to another object or function and we can always use that object or function
     directly."""
    model = Sequential([
        Dense(64, activation='elu', input_shape=(32,)),  # exponential linear unit
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='sgd',  # stochastic gradient descent, others like: 'adam','rmsprop','adadelta'
        loss='binary_crossentropy',  # 'mean_squared_error', categorical_crossentropy'
        metrics=['accuracy', 'mae']  # mean_absolute_error
    )
    """ 
    Another compile method:
  
    """
    model.compile(
        # default:lr=0.01, momentum=0, nesterov momentum=False (whether or not to use)
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True),
        # if activation function of last layer is linear instead of sigmoids,
        # that is equal to 'there is no activation function and I could as well have left this argument out' as the
        # linear activation is the default.
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.7),  # default is 0.5
                 tf.keras.metrics.MeanAbsoluteError()]
    )


if __name__ == "__main__":
    sequential_model_API()
    feedforward_nn()
    convolutional_nn()
    weight_initialisation()
    compiling_model()
    optimisers_loss_metrics()
    train_model()
