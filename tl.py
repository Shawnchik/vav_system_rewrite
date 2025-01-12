from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Lambda, concatenate, Input
from keras.utils import np_utils
from keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train_0 = pd.read_csv('row_dataset_new/train_tr_r1.csv').values[:, 1:]
train_1 = pd.read_csv('row_dataset_new/train_tr_new_little+.csv').values[:, 1:]
test = pd.read_csv('row_dataset_new/test_tr_new.csv').values[:, 1:]

train_0_x = train_0[:, :320]
train_0_y = train_0[:, 320:]
train_1_x = train_1[:, :320]
train_1_y = train_1[:, 320:]
test_x = test[:, :320]
test_y = test[:, 320:]

train_0_y_1 = np_utils.to_categorical(train_0_y[:, 0], num_classes=10)
train_0_y_2 = np_utils.to_categorical(train_0_y[:, 1], num_classes=4)
train_1_y_1 = np_utils.to_categorical(train_1_y[:, 0], num_classes=5)
train_1_y_2 = np_utils.to_categorical(train_1_y[:, 1], num_classes=4)
test_y_1 = np_utils.to_categorical(test_y[:, 0], num_classes=5)
test_y_2 = np_utils.to_categorical(test_y[:, 1], num_classes=4)

# CNN+single
train_0_x = train_0_x.reshape((-1, 16, 20, 1))
train_1_x = train_1_x.reshape((-1, 16, 20, 1))
test_x = test_x.reshape((-1, 16, 20, 1))

# model = Sequential([
# 	Conv2D(32, (3, 3), activation='relu', input_shape=(16, 20, 1), padding='same'),
# 	Conv2D(32, (3, 3), activation='relu', padding='same'),
# 	MaxPooling2D(pool_size=(4, 1)),
# 	Dropout(0.25),
#
# 	Conv2D(32, (3, 6), activation='relu', padding='same'),
# 	Conv2D(32, (3, 6), activation='relu', padding='same'),
# 	MaxPooling2D(pool_size=(4, 1)),
# 	Dropout(0.25),
#
# 	Flatten(),
# 	Dense(256, activation='relu'),
# 	Dropout(0.5),
# 	Dense(10, activation='softmax')
# ])

input_layer = Input(shape=(16, 20, 1))
con_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
con_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(con_1)
maxpool_1 = MaxPooling2D(pool_size=(4, 2))(con_2)
dropout_1 = Dropout(0.25)(maxpool_1)

con_3 = Conv2D(32, (3, 3), activation='relu', padding='same')(dropout_1)
con_4 = Conv2D(32, (3, 3), activation='relu', padding='same')(con_3)
maxpool_3 = MaxPooling2D(pool_size=(4, 2))(con_4)
dropout_3 = Dropout(0.25)(maxpool_3)

# con_5 = Conv2D(32, (3, 3), activation='relu', padding='same')(dropout_3)
# con_6 = Conv2D(32, (3, 3), activation='relu', padding='same')(con_5)
# maxpool_4 = MaxPooling2D(pool_size=(1, 1))(con_6)
# dropout_4 = Dropout(0.25)(maxpool_4)
#
# con_7 = Conv2D(32, (3, 3), activation='relu', padding='same')(dropout_4)
# con_8 = Conv2D(32, (3, 3), activation='relu', padding='same')(con_7)
# maxpool_5 = MaxPooling2D(pool_size=(1, 1))(con_8)
# dropout_5 = Dropout(0.25)(maxpool_5)

flatten_1 = Flatten()(dropout_3)
# dense_1 = Dense(256, activation='relu')(flatten_1)
# # dense_2 = Dense(32, activation='relu')(dense_1)
# dropout_4 = Dropout(0.5)(dense_1)
dense_3 = Dense(10, activation='softmax')(flatten_1)

model = Model(inputs=input_layer, outputs=dense_3)

print(model.summary())

los = []

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])
model.fit(x=train_0_x, y=train_0_y_1, epochs=100, shuffle=True, verbose=1)
# _, loss = model.evaluate(x=test_x, y=test_y_1)
# print(loss)

# pred = model.predict(x=test_x)



# dense_4 = Dense(32, activation='relu')(dense_1)
# dropout_5 = Dropout(0.5)(dense_4)
dense_5 = Dense(5, activation='softmax')(flatten_1)
model2 =Model(inputs=input_layer, outputs=dense_5)

# input_layer2 = Input(shape=(16, 20, 1))
# con_5 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer2)
# con_6 = Conv2D(32, (3, 3), activation='relu', padding='same')(con_5)
# maxpool_4 = MaxPooling2D(pool_size=(4, 1))(con_6)
# dropout_1 = Dropout(0.25)(maxpool_4)
# model2 = Model(inputs=input_layer2, outputs=dense_3)


for layer in model.layers:
	layer.trainable = False

print(model2.summary())

acc = []

for i in range(20):
	model2.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.005), metrics=['accuracy'])
	model2.fit(x=train_1_x, y=train_1_y_1, epochs=5, shuffle=True, verbose=1)
	__, loss = model2.evaluate(x=test_x, y=test_y_1)
	print(loss)
	acc.append(loss)

print(list(np.array(acc)))

pred1 = model2.predict(x=test_x)

pred = np.array(pred1).T
plt.imshow(-pred, 'gray')
plt.show()
# pd.DataFrame(pred).to_csv('temp_new/pred.csv')
pd.DataFrame(pred1).to_csv('temp_new/pred1.csv')






























