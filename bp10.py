from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Flatten, Lambda, Conv1D, concatenate
from keras.utils import np_utils, plot_model
from keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
# pre
train = pd.read_excel('FDD_data_base/train_08.xlsx')
test = pd.read_excel('FDD_data_base/test_09.xlsx')


def pre(data):
	data['time'] = pd.to_datetime(data['time'])
	data = data.set_index('time')
	data = data.resample('15min').mean()
	return data

train = pre(train).values
test = pre(test).values

# 分x, y
train_x = train[:, :-1]
train_y = train[:, -1].round()
test_x = test[:, :-1]
test_y = test[:, -1].round()

# 标准化
mean = np.concatenate((train_x, test_x), axis=0).mean(axis=0)
std = np.concatenate((train_x, test_x), axis=0).std(axis=0)
# print(mean, std)
train_x = (train_x - mean) / std
test_x = (test_x - mean) / std

# one_hot
train_y = np_utils.to_categorical(train_y, num_classes=11)
test_y = np_utils.to_categorical(test_y, num_classes=11)

# print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

# write
pd.DataFrame(train_x).to_csv('temp/train_x.csv')
pd.DataFrame(train_y).to_csv('temp/train_y.csv')
pd.DataFrame(test_x).to_csv('temp/test_x.csv')
pd.DataFrame(test_y).to_csv('temp/test_y.csv')
'''

# read
train_x = pd.read_csv('temp/train_x.csv').values[:, 1:]
train_y_1 = pd.read_csv('temp/train_y.csv').values[:, 1:]
train_y_2 = pd.read_csv('temp/train_y_2.csv').values[:, 1:]
test_x = pd.read_csv('temp/test_x.csv').values[:, 1:]
test_y_1 = pd.read_csv('temp/test_y.csv').values[:, 1:]
test_y_2 = pd.read_csv('temp/test_y_2.csv').values[:, 1:]


# bpnn
# model = Sequential([
# 	Dense(60, input_dim=33, activation='relu'),
# 	Dense(60, activation='relu'),
# 	Dropout(0.5),
# 	Dense(11, activation='softmax')
# ])


# 划分2个input

train_x_1 = np.array(train_x[:, :21])
# print(train_x_1)
# print(train_x_1.shape)
train_x_1 = train_x_1.reshape((-1, 7, 3))
# print(train_x_1)
# print(train_x_1.shape)
# train_x_1 = train_x_1.transpose((0, 2, 1))
# print(train_x_1)
# print(train_x_1.shape)
time = train_x[:, -1]
# print(time)
time = np.array([[[t, t, t]]for t in time])
# print(time.shape)
train_x_1 = np.concatenate((train_x_1, time), axis=1).transpose((0, 2, 1))
# print(train_x_1.shape)
train_x_2 = np.array(train_x[:, 21:])
# print(train_x_2.shape)
# print(train_y_1.shape)
# print(train_y_2)
# print(train_y_2.shape)

test_x_1 = np.array(test_x[:, :21])
test_x_1 = test_x_1.reshape((-1, 7, 3))
time = test_x[:, -1]
time = np.array([[[t, t, t]] for t in time])
test_x_1 = np.concatenate((test_x_1, time), axis=1).transpose((0, 2, 1))
test_x_2 = np.array(test_x[:, 21:])


# 2 input 2 output model

def slice(x, index):
	return x[:, index, :]

main_input = Input(shape=(3, 8))
ly_1 = Conv1D(16, 1, activation='relu')(main_input)
ly_2 = Dense(1, activation='softmax')

ro = []
for n in range(3):
	r = Lambda(slice, output_shape=(1, 16), arguments={'index': n})(ly_1)
	ro.append(ly_2(r))

ly_3 = concatenate([r for r in ro])

model = Model(inputs=main_input, outputs=ly_3)




print(model.summary())


model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

model.fit(x=train_x_1, y=train_y_2, epochs=10, shuffle=True, verbose=1)

loss, accuracy = model.evaluate(x=test_x_1, y=test_y_2)
print(loss, accuracy)
#
# w = model.get_weights()
# # print(w)
pred = model.predict(x=test_x_1)
pd.DataFrame(pred).to_csv('temp/pred.csv')

pred2 = model.predict(x=train_x_1)
pd.DataFrame(pred2).to_csv('temp/pred2.csv')












