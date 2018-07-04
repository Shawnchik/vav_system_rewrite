from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Flatten, Lambda, Conv1D, concatenate, Reshape
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
print(train_y_2)

train_y_2_1 = train_y_2[:, :2]
train_y_2_2 = train_y_2[:, 2:4]
train_y_2_3 = train_y_2[:, 4:]

test_y_2_1 = test_y_2[:, :2]
test_y_2_2 = test_y_2[:, 2:4]
test_y_2_3 = test_y_2[:, 4:]


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
ly_1 = Conv1D(16, 1, activation='relu')(ly_1)
ly_2 = Dense(2, activation='softmax')

ro = []
ri = []
for n in range(3):
	r = Lambda(slice, arguments={'index': n})(ly_1)
	ri.append(r)
	ro.append(ly_2(r))

ly_3_1 = ly_2(ri[0])
ly_3_2 = ly_2(ri[1])
ly_3_3 = ly_2(ri[2])


ly_3 = concatenate([r for r in ro])
ly_4 = concatenate([r for r in ri])

input_2 = Input(shape=(12,))
ly_5 = Dense(12, activation='relu')(ly_4)
ly_6 = concatenate([input_2, ly_5])
ly_7 = Dense(60, activation='relu')(ly_6)
ly_8 = Dense(11, activation='softmax')(ly_7)


# model = Model(inputs=[main_input, input_2], outputs=[ly_3, ly_8])
# model = Model(inputs=main_input, outputs=ly_3)
model = Model(inputs=main_input, outputs=[ly_3_1, ly_3_2, ly_3_3])


print(model.summary())


# model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001), loss_weights=[0.5, 0.5], metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])

# model.fit(x=[train_x_1, train_x_2], y=[train_y_2, train_y_1], epochs=10, shuffle=True, verbose=1)
# model.fit(x=train_x_1, y=train_y_2, epochs=10, shuffle=True, verbose=1)
model.fit(x=train_x_1, y=[train_y_2_1, train_y_2_2, train_y_2_3], epochs=5, shuffle=True, verbose=1)

# loss = model.evaluate(x=[test_x_1, test_x_2], y=[test_y_2, test_y_1])
# loss = model.evaluate(x=test_x_1, y=test_y_2)
loss = model.evaluate(x=test_x_1, y=[test_y_2_1, test_y_2_2, test_y_2_3])
print(loss)

#
# w = model.get_weights()
# # print(w)

# pred0, pred1 = model.predict(x=[test_x_1, test_x_2])
pred1, pred2, pred3 = model.predict(x=test_x_1)
pd.DataFrame(pred1).to_csv('temp/pred1.csv')
pd.DataFrame(pred2).to_csv('temp/pred2.csv')
pd.DataFrame(pred3).to_csv('temp/pred3.csv')

# print(pred0, pred1)
# pd.DataFrame(pred0).to_csv('temp/pred0.csv')
# pd.DataFrame(pred1).to_csv('temp/pred1.csv')
#
# pred2, pred3 = model.predict(x=[train_x_1, train_x_2])
# pd.DataFrame(pred2).to_csv('temp/pred2.csv')
# pd.DataFrame(pred3).to_csv('temp/pred3.csv')











