from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Lambda, concatenate, Input
from keras.utils import np_utils
from keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train_0 = pd.read_csv('row_dataset_new/train_tr_r2.csv').values[:, 1:]  # 源域 异分布
# train_1 = pd.read_csv('row_dataset_new/train_tr_r1_f1f7_3ds.csv').values[:, 1:]  # 目标域 同分布
test = pd.read_csv('row_dataset_new/test_tr_r1.csv').values[:, 1:]

#30f0
train_1 = pd.read_csv('row_dataset_new/train_tr_r1_30f0.csv').values[:31, 1:]
train_1_x = train_1[:, :320]
train_1_y = np.zeros((31, 10))

train_0_x = train_0[:, :320]
train_0_y = train_0[:, 320:]
# train_1_x = train_1[:, :320]
# train_1_y = train_1[:, 320:]
test_x = test[:, :320]
test_y = test[:, 320:]

train_0_y = np_utils.to_categorical(train_0_y[:, 0], num_classes=10)
# train_1_y = np_utils.to_categorical(train_1_y[:, 0], num_classes=10)
test_y = np_utils.to_categorical(test_y[:, 0], num_classes=10)

# print(train_0_y)

train_01_y = np.concatenate((train_0_y, train_1_y), axis=0)

# CNN+single
train_0_x = train_0_x.reshape((-1, 16, 20, 1))
train_1_x = train_1_x.reshape((-1, 16, 20, 1))
test_x = test_x.reshape((-1, 16, 20, 1))
train_01_x = np.concatenate((train_0_x, train_1_x), axis=0)

# tradaboost
n = train_0_x.shape[0]
m = train_1_x.shape[0]
N = 20
w = [float(1/(n+m))] * (n+m)

input_layer = Input(shape=(16, 20, 1))
con_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
con_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(con_1)
maxpool_1 = MaxPooling2D(pool_size=(4, 2))(con_2)
dropout_1 = Dropout(0.25)(maxpool_1)

con_3 = Conv2D(32, (3, 3), activation='relu', padding='same')(dropout_1)
con_4 = Conv2D(32, (3, 3), activation='relu', padding='same')(con_3)
maxpool_3 = MaxPooling2D(pool_size=(4, 2))(con_4)
dropout_3 = Dropout(0.25)(maxpool_3)

flatten_1 = Flatten()(dropout_3)
# dense_1 = Dense(256, activation='relu')(flatten_1)
# dropout_4 = Dropout(0.5)(dense_1)
dense_3 = Dense(10, activation='softmax')(flatten_1)

model = Model(inputs=input_layer, outputs=dense_3)

print(model.summary())

acc = []

train_1_y_argmax = np.argmax(train_1_y, axis=1)
train_0_y_argmax = np.argmax(train_0_y, axis=1)
print(train_1_y_argmax)

beta = 1/(1 + np.sqrt(2 * np.log(n/N)))

# tradaboost
for t in range(N):
	p_t = np.array([wi / sum(w) for wi in w])

	model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])
	# model.fit(x=train_0_x, y=train_0_y, epochs=100, shuffle=True, verbose=1)
	model.fit(x=train_01_x, y=train_01_y, sample_weight=p_t, epochs=5, shuffle=True, verbose=2)

	pred1 = model.predict(x=train_01_x[-m:, :, :, :])
	pred1 = np.argmax(pred1, axis=1)
	pred0 = model.predict(x=train_01_x[:n, :, :, :])
	pred0 = np.argmax(pred0, axis=1)
	print(pred1)

	error1 = (pred1 != train_1_y_argmax)
	error0 = (pred0 != train_0_y_argmax)
	print(~error0)
	e_t = np.dot(w[-m:], error1)/sum(w[-m:])
	print('e_t', e_t)

	# e_t 小于0.5？

	beta_t = e_t/(1-e_t)

	print(~error1)

	error00 = []
	for i in error0:
		if i:
			error00.append(np.power(beta, 1))
		else:
			error00.append(np.power(beta, 0))
	error11 = []
	for i in error1:
		if i:
			error11.append(np.power(beta, -1))
		else:
			error11.append(np.power(beta, 0))
	print(error00, error11)

	# Update
	w[:n] = np.multiply(w[:n], error00)
	w[n:] = np.multiply(w[n:], error11)

	print('w', w[:n])
	print('ww', w[n:])

	_, accuracy = model.evaluate(x=test_x, y=test_y)
	print(accuracy)
	acc.append(accuracy)

print(list(np.array(acc)))

pred = model.predict(x=test_x)

pred = np.array(pred).T
plt.imshow(-pred, 'gray')
plt.show()
# pd.DataFrame(pred).to_csv('temp_new/pred.csv')


















