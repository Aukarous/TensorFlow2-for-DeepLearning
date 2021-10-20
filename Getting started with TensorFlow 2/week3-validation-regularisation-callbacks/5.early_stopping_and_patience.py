from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential([
    # 👇 输入是二维的，所以这可能是一个长度为128的单变量时间序列。
    Conv1D(16, 5, activation='relu', input_shape=(128, 1)),
    MaxPooling1D(4),
    Flatten(),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss',  # or monitor='val_accuracy
                               patience=5,
                               min_delta=0.01,
                               mode='max'
                               )
# 👇 callbacks是一个list，因为在实践中可能会传入许多callbacks函数，所有这些callbacks函数都在训练运行期间执
# 行不同的任务，所以early stoppingcallback做的，是监视神经网络在验证集上的性能，在这里用validation_split
# 关键字参数创建的验证集。它会根据表现停止训练。
model.fit(X_train, y_train, validation_split=0.2, epochs=100, callbacks=[early_stopping])
"""
一个很自然的问题是，callback是在监视那个度量的表现?early stopping callback构造函数接收一个名为monitor的关键字参数，该参数可用于设置要使用的性能指标。
这里传入了val_loss，表示用验证机的loss作为度量表现来决定什么时候终止训练。这其实是early stopping 
callback的默认设置，在这个模型中，loss为categorical_crossentropy。还可以用validation accuracy作为性能度量来决定何时停止训练。
记住，在compile模型时跟踪accuracy指标。
BTW，在这里可以看到传递给monitor的参数的真实string与model.fit返回的对象中的keys之一的string name相同。
这是一种检查应该用什么string的方法。在early stopping callbacks中设置的另外一个kwdargs是patience，默认为0。意味着一旦性能度量从一个epoch到到下一个epoch变得更糟，训练就会终止。
这可能不太理想，因为模型的性能有噪声，可能从一个epoch到下一个，性能可能上升或下降。我们真正关心的是整体的性能会提升。
这就是为什么经常把patience设置为epoch的倍数，如5，此时，只有在连续5个epoch的性能度量没有improvement时，训练才会终止。
early stopping callback也有个min delta参数，用于确定性能度量提升的数量，如这里将min delta设置为0.01，意味着validation accuracy至少要提高0.01才能算作improvement。
如果validation accuracy提高了很小如0.001，那么这将视为early stopping callback带来的性能恶化，此时patience 计数会增加1。
默认的min delta为0，意味着性能上的任何改进都足以重置patience，另外一个在early stopping callback中使用的参数是mode。如果监控的是validation 
loss，则越低越好，如果是validation accuracy，则越高越好，但early stopping callback如何知道direction呢？mode的默认为auto，会自动根据quantity的名称判断direction。
然而也可以显式指定，如将mode设置为max，意味着将我们监控的性能度量指标最大化。
"""
