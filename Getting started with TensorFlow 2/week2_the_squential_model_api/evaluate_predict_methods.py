import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([Dense(1, activation='sigmoid', input_shape=(12,))])
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy', 'mae'])
model.fit(X_train, y_train)

loss, accuracy, mae = model.evaluate(X_test, y_test)
# X_sample: (num_samples, 12)
pred = model.predict(X_sample)  # The target dimension is not used in the input to model.predict
"""
每个数组都包含许多数据示例
evaluate:迭代测试集, 并计算损失和metrics
如果在编译模型时定义了binary cross-entropy损失函数，以及accuracy，那么损失函数和metrics将在测试集上进行评估, 并通过调用model.evaluate 返回。
此时，可以保存返回的值，如果设置了很多metrics， 他们也会全部由模型返回。
input_shape: 第一维对应examples的数量，其余维度为features(对应与input_shape的第一维度)。如果example数量为1，那必须有一个等于1的虚拟维度。
如：X sample是一个只有一个sample的numpy数组，network是一个二分类器，最后一层layer只有一个neuron通过sigmoid激活函数，那么model.predict方法会返回一个数字，作为属于正类的概率。
再如：假设network是一个有三个类的多分类模型，最后一层有三个神经元（Dense的units=3）和softmax激活函数，损失函数变成了categorical_cross_entropy，假设输入的特征为12个，example
数量为2(X_sample=(2,12))
，那么model
.predict的输出为：     。输出预测的array是一个2d 的array：(2, 3)，分别对应X 
sample的数量与分类的类别数量。输出的array的每一行均为属于输入数据的网络输出，输出层是一个softmax层，each layer is a set of output probabilities emitted by the softmax function and you can see how they add up to one in each case.
"""