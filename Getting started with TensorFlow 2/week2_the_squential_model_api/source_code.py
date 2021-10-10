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
        Dense(1, activation='sigmoid')  # when fit the model, units = 100
    ])
    model.compile(
        optimizer='sgd',  # stochastic gradient descent, others like: 'adam','rmsprop','adadelta'
        loss='binary_crossentropy',  # 'mean_squared_error', categorical_crossentropy'
        metrics=['accuracy', 'mae']  # mean_absolute_error
    )
    """ 
    Another compile model method:
    
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
    
    """

    # X_train: (num_samples, num_features), y_train: (num_samples, num_classes), both are numpy arrays
    # all of the dataset inputs have been stacked together into a single array:X_train
    # all the targets or outputs are in y_train,
    # So the first dimension of each array corresponds to the number of examples in the training set.
    # For example, if each input data point is a one-dimensional array, X_train would be a two-dimensional array with
    # the number of samples in the first dimension and the number of features in the second,Y_train would be a
    # two-dimensional array with the number of samples in the first dimension and the number of classes in the second
    # Assuming that the labels have been represented as a one-hot vector, so each row of y_train is a vector
    # of length num_classes which is all zeros, except for a one in the place corresponding to the correct class
    # Or if all the labels have a sparse representation, so just a single integer for each label, then y_train could be
    # a one-dimensional array with length equal to the number of samples.Notice that in this case, we should choose the
    # sparse_categorical_crossentropy loss function
    history = model.fit(X_train, y_train, epochs=10, batch_size=16)  # default batch_size=32
    return history


if __name__ == "__main__":
    sequential_model_API()
    feedforward_nn()
    convolutional_nn()
    weight_initialisation()
    compiling_model()
    optimisers_loss_metrics()
    train_model()
