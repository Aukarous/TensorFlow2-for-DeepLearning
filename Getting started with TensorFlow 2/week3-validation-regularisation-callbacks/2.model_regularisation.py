"""
Now, we'll see how you can include regularization techniques into the model training
that have the effect of constraining the model capacity in preventing overfitting.
In particular, we're going to look at using L2 weight regularization, which is also known
as weight decay in a context of neural networks, as well as L1 weight regularization and
you'll see how to include Dropouts in your models
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l1
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1_l2


def example1():
    model = Sequential([
        Dense(64, activation='relu',
              # 👇 kernel_regularizer=l1(0.005) or kernel_regularizer=l1_l2(l1=0.005, l2=0.001)
              kernel_regularizer=l2(0.001),
              # 👇 optional bias regularizer argument
              bias_regularizer=l2(0.001)
              ),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['acc'])
    model.fit(inputs, targets, validation_split=0.25)


def example2():
    """ 👇 Dropout also has a regularizing effect """
    model2 = Sequential([
        Dense(64, activation='relu'),
        # 👇 dropouts rate, means that each weight connection between these two dense layers is set to zero with
        # probability 0.5, this is sometimes referred to as Bernoulli Dropout, since the weights are effectively being
        # multiplied by a Bernoulli random variable.Each of the weights are randomly dropped out independently from one
        # another and Dropout has also applied independently across each element in the batch at training time"""
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model2.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['acc'])
    model2.fit(inputs, targets, validation_split=0.25)  # Training mode, with dropout
    model2.evaluate(val_inputs, val_targets)  # Testing mode, no dropout
    model2.predict(test_inputs)  # Testing mode, no dropout


if __name__ == "__main__":
    example1()
    example2()
