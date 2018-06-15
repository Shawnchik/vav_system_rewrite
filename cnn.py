from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Lambda, concatenate, Input
from keras.utils import np_utils
from keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# train = pd.read_csv('row_dataset_new/train.csv').values[:, 1:]
# train = pd.read_csv('row_dataset_new/train_0203.csv').values[:, 1:]
# train = pd.read_csv('row_dataset_new/train_0203+01.csv').values[:, 1:]
train = pd.read_csv('row_dataset_new/train_tr_r2+r1_f1f7.csv').values[:, 1:]
test = pd.read_csv('row_dataset_new/test_tr_r1.csv').values[:, 1:]

# three rooms
# train_x = train[:, :512]
# train_y = train[:, 512:]
# test_x = test[:, :512]
# test_y = test[:, 512:]

# single room
train_x = train[:, :320]
train_y = train[:, 320:]
test_x = test[:, :320]
test_y = test[:, 320:]

train_y_1 = np_utils.to_categorical(train_y[:, 0], num_classes=10)
train_y_2 = np_utils.to_categorical(train_y[:, 1], num_classes=4)
test_y_1 = np_utils.to_categorical(test_y[:, 0], num_classes=10)
test_y_2 = np_utils.to_categorical(test_y[:, 1], num_classes=4)

mode = 'CNN+single'

if mode == 'CNN':
	# CNN
	train_x = train_x.reshape((-1, 16, 32, 1))
	test_x = test_x.reshape((-1, 16, 32, 1))
	model = Sequential([
		Conv2D(32, (3, 3), activation='relu', input_shape=(16, 32, 1), padding='same'),
		Conv2D(32, (3, 3), activation='relu', padding='same'),
		MaxPooling2D(pool_size=(4, 2)),
		Dropout(0.25),

		Conv2D(32, (3, 3), activation='relu', padding='same'),
		Conv2D(32, (3, 3), activation='relu', padding='same'),
		MaxPooling2D(pool_size=(4, 2)),
		Dropout(0.25),

		Flatten(),
		Dense(256, activation='relu'),
		Dropout(0.5),
		Dense(10, activation='softmax')
	])

elif mode == 'BPNN':
	# BP
	model = Sequential([
		Dense(100, input_dim=512, activation='relu'),
		Dense(100, activation='relu'),
		Dense(10, activation='softmax')
	])

elif mode == 'CNN+':
	# CNN+
	train_x = train_x.reshape((-1, 16, 32, 1))
	test_x = test_x.reshape((-1, 16, 32, 1))

	def slice(x, index1, index2):
		return x[:, :, index1:index2, :]

	input_layer = Input(shape=(16, 32, 1))
	con_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
	con_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(con_1)

	slice_1 = Lambda(slice, arguments={'index1': 0, 'index2': 18})(con_2)
	maxpool_1 = MaxPooling2D(pool_size=(4, 3))(slice_1)
	dropout_1 = Dropout(0.25)(maxpool_1)
	slice_2 = Lambda(slice, arguments={'index1': 18, 'index2': 32})(con_2)
	maxpool_2 = MaxPooling2D(pool_size=(4, 1))(slice_2)
	dropout_2 = Dropout(0.25)(maxpool_2)

	concat_1 = concatenate([dropout_1, dropout_2], axis=2)
	con_3 = Conv2D(32, (3, 3), activation='relu', padding='same')(concat_1)
	maxpool_3 = MaxPooling2D(pool_size=(4, 1))(con_3)
	dropout_3 = Dropout(0.25)(maxpool_3)

	flatten_1 = Flatten()(dropout_3)
	dense_1 = Dense(256, activation='relu')(flatten_1)
	dropout_4 = Dropout(0.5)(dense_1)
	dense_2 = Dense(10, activation='softmax')(dropout_4)

	model = Model(inputs=input_layer, outputs=dense_2)

elif mode == 'CNN+room':
	# 排序
	train_x = train_x.reshape((-1, 16, 32, 1))
	test_x = test_x.reshape((-1, 16, 32, 1))

	# plt.imshow(test_x[0].reshape((16, 32)), 'gray')
	# plt.show()

	def room_group(data):
		room_data = data[:, :, :18, :].reshape((-1, 16, 6, 3, 1)).transpose((0, 1, 3, 2, 4)).reshape((-1, 16, 18, 1))
		return np.concatenate((room_data, data[:, :, 18:, :]), axis=2)

	train_x = room_group(train_x)
	test_x = room_group(test_x)

	# plt.imshow(test_x[0].reshape((16, 32)), 'gray')
	# plt.show()

	# CNN+room
	def slice(x, index1, index2):
		return x[:, :, index1:index2, :]

	input_layer = Input(shape=(16, 32, 1))
	con_1 = Conv2D(32, (5, 6), activation='relu', padding='same')(input_layer)
	con_2 = Conv2D(32, (5, 6), activation='relu', padding='same')(con_1)

	slice_1 = Lambda(slice, arguments={'index1': 0, 'index2': 18})(con_2)
	con_r = Conv2D(32, (5, 6), activation='relu', strides=(1, 6), padding='valid')(slice_1)
	maxpool_1 = MaxPooling2D(pool_size=(4, 1))(con_r)
	dropout_1 = Dropout(0.25)(maxpool_1)

	slice_2 = Lambda(slice, arguments={'index1': 18, 'index2': 32})(con_2)
	con_a = Conv2D(32, (5, 1), activation='relu', padding='valid')(slice_2)
	maxpool_2 = MaxPooling2D(pool_size=(4, 1))(con_a)
	dropout_2 = Dropout(0.25)(maxpool_2)

	concat_1 = concatenate([dropout_1, dropout_2], axis=2)
	con_3 = Conv2D(32, (3, 3), activation='relu', padding='same')(concat_1)
	maxpool_3 = MaxPooling2D(pool_size=(3, 1))(con_3)
	dropout_3 = Dropout(0.25)(maxpool_3)

	flatten_1 = Flatten()(dropout_3)
	dense_1 = Dense(256, activation='relu')(flatten_1)
	dropout_4 = Dropout(0.5)(dense_1)
	dense_2 = Dense(10, activation='softmax')(dropout_4)

	model = Model(inputs=input_layer, outputs=dense_2)

elif mode == 'CNN+single':

	# CNN+single
	train_x = train_x.reshape((-1, 16, 20, 1))
	test_x = test_x.reshape((-1, 16, 20, 1))

	model = Sequential([
		Conv2D(32, (3, 3), activation='relu', input_shape=(16, 20, 1), padding='same'),
		Conv2D(32, (3, 3), activation='relu', padding='same'),
		MaxPooling2D(pool_size=(4, 1)),
		Dropout(0.25),

		Conv2D(32, (3, 6), activation='relu', padding='same'),
		Conv2D(32, (3, 6), activation='relu', padding='same'),
		MaxPooling2D(pool_size=(4, 1)),
		Dropout(0.25),

		Flatten(),
		Dense(256, activation='relu'),
		Dropout(0.5),
		Dense(10, activation='softmax')
	])

print(model.summary())

los = []

for i in range(20):
	print(i)
	model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])
	model.fit(x=train_x, y=train_y_1, epochs=5, shuffle=True, verbose=1)
	_, loss = model.evaluate(x=test_x, y=test_y_1)
	print(loss)
	los.append(loss)

print(np.array(los))
pred = model.predict(x=test_x)
pd.DataFrame(pred).to_csv('temp_new/pred.csv')






























