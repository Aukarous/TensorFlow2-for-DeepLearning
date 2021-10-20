from tensorflow.keras.callbacks import Callback

class MyCallBack(Callback):
    def on_train_begin(self, logs=None):
        # Do sth at the start of training

    def on_train_batch_begin(self, batch, logs=None):
        # Do sth at the start of every batch iteration

    def on_epoch_end(self, epoch, logs=None):
        # Do sth at the end of every epoch

history = model.fit(X_train, y_train, epochs=5, callbacks=[MyCallBack])