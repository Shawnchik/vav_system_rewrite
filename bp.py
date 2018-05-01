from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.utils import np_utils
from keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def pre(file):
	data = pd.read_csv(file)
	data['time'] = pd.to_datetime(data['time'])
	data['hour'] = data['time'].map(lambda x: (x-data.time[0]).delta * 10e-10 / 3600 % 24)
	data = data.set_index('time')
	data = data.resample('15min').mean()
	# data_x_std = (data - data.mean()) / (data.std() + 0.000001)
	# return data_x_std.values
	return data


data_0_x = pre('FDD_data_base/0_case2.csv')
data_1_x = pre('FDD_data_base/1_case2.csv')


mean = np.array([data_0_x.mean().values, data_1_x.mean().values]).mean(axis=0)
std = np.array([data_0_x.std().values, data_1_x.std().values]).max(axis=0)

data_0_x = ((data_0_x - mean) / (std + 1e-10)).values
data_1_x = ((data_1_x - mean) / (std + 1e-10)).values

data_0_y = np.zeros(data_0_x.shape[0])
data_1_y = np.ones(data_1_x.shape[0])

# index = np.random.permutation(len(data_0_x))[:600]
# data_0_x_train = data_0_x[index, :]
# data_0_y_train = np.zeros(data_0_x_train.shape[0])

index = np.random.permutation(len(data_1_x))[:10]
data_1_x_train = data_1_x[index, :]
data_1_y_train = np.ones(data_1_x_train.shape[0])

print(data_0_x.shape, data_0_y.shape, data_1_x.shape, data_1_y.shape)

validation = 1/6

train_x = np.concatenate((data_0_x[:-int(len(data_0_x) * validation), :], data_1_x_train))
test_x = np.concatenate((data_0_x[-int(len(data_0_x) * validation):, :], data_1_x[-int(len(data_1_x) * validation):, :]))
train_y = np.concatenate((data_0_y[:-int(len(data_0_y) * validation)], data_1_y_train))
test_y = np.concatenate((data_0_y[-int(len(data_0_y) * validation):], data_1_y[-int(len(data_1_y) * validation):]))

train_y = np_utils.to_categorical(train_y, num_classes=2)
test_y = np_utils.to_categorical(test_y, num_classes=2)

print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

pd.DataFrame(train_x).to_csv('temp/train_x.csv')
pd.DataFrame(train_y).to_csv('temp/train_y.csv')
pd.DataFrame(test_x).to_csv('temp/test_x.csv')
pd.DataFrame(test_y).to_csv('temp/test_y.csv')

model = Sequential([
	Dense(16, input_dim=39, activation='relu'),
	# Dropout(0.5),
	Dense(2, activation='softmax')
])

print(model.summary())

los = []
acc = []

for i in range(20):
	model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])

	model.fit(x=train_x, y=train_y, epochs=5, shuffle=True, verbose=2)

	loss, accuracy = model.evaluate(x=test_x, y=test_y)
	print(loss, accuracy)

	los.append(loss)
	acc.append(accuracy)

w = model.get_weights()
# print(w)
pred = model.predict(x=test_x)
pd.DataFrame(pred).to_csv('temp/pred.csv')

# print(acc)
pd.DataFrame(acc).to_excel('temp/acc.xlsx')

