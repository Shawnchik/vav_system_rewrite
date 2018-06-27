import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1小时整合
'''
# file = 'FDD_data_base_new/dataset_08_f0_r0.csv'
file = 'FDD_data_base_new/new_fault/dataset_09_f1_r1.csv'
data = pd.read_csv(file)
data['time'] = pd.to_datetime(data['time'])
data = data.set_index('time')
data = data.resample('1H').mean()
data['f'] = file[40]
data['r'] = file[43]


# for F, R in [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3), (4, 1), (4, 2), (4, 3),
#              (5, 1), (5, 2), (5, 3), (6, 1), (6, 2), (6, 3), (7, 1), (7, 2), (7, 3), (8, 0), (9, 0)]:
#   file = 'FDD_data_base_new/dataset_08_f%s_r%s.csv' % (F, R)

for F, R in [(2, 1), (3, 1), (4, 1)]:
	file = 'FDD_data_base_new/new_fault/dataset_09_f%s_r%s.csv' % (F, R)

	print(file)
	newdata = pd.read_csv(file)
	newdata['time'] = pd.to_datetime(newdata['time'])
	newdata = newdata.set_index('time')
	newdata = newdata.resample('1H').mean()

	newdata['f'] = file[40]
	newdata['r'] = file[43]

	data = pd.concat([data, newdata])

print(data.describe())
print(data)

# data.to_csv('row_dataset_new/08_row_new.csv')
data.to_csv('row_dataset_new/09_row_new.csv')
'''

# 标准化
'''
rowdata_08 = pd.read_csv('row_dataset_new/08_row_new.csv').values
rowdata_09 = pd.read_csv('row_dataset_new/09_row_new.csv').values

x = np.concatenate((rowdata_08[:, 1:33], rowdata_09[:, 1:33]))

x_mean = [25.53484261, 25.60443941, 25.51655391, 43.56143105, 44.01992471, 47.43312863,
          1015.695746, 806.5446232, 921.3538538, 686.438114, 736.4696074, 698.4346145,
          2.81847318, 43.03040572, 42.94757055, 25.27132701, 25.44905213, 25.44905213,
          32.72214425, 32.56761288, 28.93978465, 21.77211897, 15.76372563, 24.07419009,
          28.32464442, 15.79621133, 20.59561637, 0.792189678, 24.93924806, 21.1720649, 50.352769, 1.768346714]
x_std = [2.534511401, 2.528311702, 2.487958772, 20.94523974, 20.62261527, 19.33231078,
         930.782106, 824.5191916, 946.3482711, 256.3722828, 288.6350848, 259.994863, 11.28687844,
         11.46112565, 11.47296564, 2.306080259, 2.306080259, 2.306080259, 38.62324912,
         36.67110341, 13.95005415, 10.43991343, 1.525253382, 3.255178009, 25.46050501,
         1.516824258, 3.956949346, 1.025378949, 5.017182651, 5.331343727, 20.97647634, 1.299522715]

x_std = np.array(x_std)
x = (x - x_mean) / x_std

# std_train = np.concatenate((x[:17856, :], rowdata_08[:, 33:]), axis=1)
# std_test = np.concatenate((x[17856:, :], rowdata_09[:, 33:]), axis=1)
std_train = np.concatenate((x[:2976, :], rowdata_08[:, 33:]), axis=1)
std_test = np.concatenate((x[2976:, :], rowdata_09[:, 33:]), axis=1)


pd.DataFrame(std_train).to_csv('row_dataset_new/08_std_new.csv')
pd.DataFrame(std_test).to_csv('row_dataset_new/09_std_new.csv')
# pd.DataFrame([x_mean, x_std]).to_csv('row_dataset_new/std.csv')
'''

# 8-24
'''
std_train = pd.read_csv('row_dataset_new/08_std_new.csv').values[:, 1:]
std_test = pd.read_csv('row_dataset_new/09_std_new.csv').values[:, 1:]

hour = list(range(24)) * 31 * 4
index = [x > 7 for x in hour]
std_train = std_train[index]

hour = list(range(24)) * 10 * 4
index = [x > 7 for x in hour]
std_test = std_test[index]

pd.DataFrame(std_train).to_csv('row_dataset_new/08_std_new_8-24.csv')
pd.DataFrame(std_test).to_csv('row_dataset_new/09_std_new_8-24.csv')
'''

# 16合1
'''
train = pd.read_csv('row_dataset_new/08_std_new_8-24.csv').values[:, 1:]
test = pd.read_csv('row_dataset_new/09_std_new_8-24.csv').values[:, 1:]

# print(test.shape)
train_n = np.array(train[:, :32]).reshape(-1, 16*32)
test_n = np.array(test[:, :32]).reshape(-1, 16*32)

# plot test

test_show = np.array(test[:, :32]).reshape(10, 4, 16, 32).transpose((1, 0, 2, 3))
plt.figure()
for i in range(40):
		plt.subplot(10,4,i+1)
		plt.imshow(test_show[i//10, i%4, :, :], 'gray')
		plt.xticks([])
		plt.yticks([])

plt.subplots_adjust(hspace=0)
plt.show()


index = ([True] + [False] * 15) * 31 * 4
train_n = np.concatenate((train_n, train[:, 32:][index]), axis=1)

index = ([True] + [False] * 15) * 10 * 4
test_n = np.concatenate((test_n, test[:, 32:][index]), axis=1)

pd.DataFrame(train_n).to_csv('row_dataset_new/train_new.csv')
pd.DataFrame(test_n).to_csv('row_dataset_new/test_new.csv')
'''

# 学习集只有房间2和3
'''
train = pd.read_csv('row_dataset_new/train.csv').values[:, 1:]
train = train[train[:, -1] != 1]
pd.DataFrame(train).to_csv('row_dataset_new/train_0203.csv')
'''

# 学习集用03补01
'''
train0 = pd.read_csv('row_dataset_new/train_0203.csv').values[:, 1:]
train = train0[train0[:, -1] == 3]
train_x = train[:, :-2].reshape((-1, 16, 32))

train_x_room = train_x[:, :, :18].reshape((-1, 16, 6, 3)).transpose((0, 1, 3, 2)).reshape((-1, 16, 18))
train_x = np.concatenate((train_x_room, train_x[:, :, 18:]), axis=2)

train_x_new = np.concatenate((train_x[:, :, 12:18], train_x[:, :, 6:12], train_x[:, :, :6], train_x[:, :, 18:]), axis=2)

train_x_new_room = train_x_new[:, :, :18].reshape((-1, 16, 3, 6)).transpose((0, 1, 3, 2)).reshape((-1, 16, 18))
train_x_new = np.concatenate((train_x_new_room, train_x_new[:, :, 18:]), axis=2)

train_new = np.concatenate((train_x_new.reshape((-1, 512)), train[:, -2:]), axis=1)
train_new[:, -1] = 1
train = np.concatenate((train0, train_new), axis=0)
pd.DataFrame(train).to_csv('row_dataset_new/train_0203+01.csv')
'''

# 单个房间

'''
# train_tr_r1_F0F9
# 选7条(第一天)
train0 = pd.read_csv('row_dataset_new/train.csv').values[:, 1:]
train_tr_r1_f1f7 = []
i = 1
for data in train0:
	if data[-1] == 1 and data[-2] == i:
		train_tr_r1_f1f7.append(data)
		i += 1
	if i == 8:
		break
# 删r2r3
train_tr_r1_f1f7_x = np.array(train_tr_r1_f1f7)[:, :512].reshape((-1, 16, 32))
train_tr_r = train_tr_r1_f1f7_x[:, :, :18].reshape((-1, 16, 6, 3)).transpose((0, 1, 3, 2)).reshape((-1, 16, 18))
train_tr_r1_f1f7_x = np.concatenate((train_tr_r[:, :, :6], train_tr_r1_f1f7_x[:, :, 18:]), axis=2)
train_tr_r1_f1f7 = np.concatenate((train_tr_r1_f1f7_x.reshape((-1, 320)), np.array(train_tr_r1_f1f7)[:, 512:]), axis=1)
pd.DataFrame(train_tr_r1_f1f7).to_csv('row_dataset_new/train_tr_r1_f1f7.csv')
'''
'''
# train_tr_r2
# 选r2
train0 = pd.read_csv('row_dataset_new/train.csv').values[:, 1:]
train_r2 = train0[[x[-1] == 2 or x[-1] == 0 for x in train0]]
# 删r1r3
train_r2_x = np.array(train_r2)[:, :512].reshape((-1, 16, 32))
train_r = train_r2_x[:, :, :18].reshape((-1, 16, 6, 3)).transpose((0, 1, 3, 2)).reshape((-1, 16, 18))
train_r2_x = np.concatenate((train_r[:, :, 6:12], train_r2_x[:, :, 18:]), axis=2)
train_r2 = np.concatenate((train_r2_x.reshape((-1, 320)), np.array(train_r2)[:, 512:]), axis=1)
pd.DataFrame(train_r2).to_csv('row_dataset_new/train_tr_r2.csv')
'''
'''
# test_tr_r1
# 删r2r3
test0 = pd.read_csv('row_dataset_new/test_new.csv').values[:, 1:]
test_x = np.array(test0)[:, :512].reshape((-1, 16, 32))
test = test_x[:, :, :18].reshape((-1, 16, 6, 3)).transpose((0, 1, 3, 2)).reshape((-1, 16, 18))
test_x = np.concatenate((test[:, :, :6], test_x[:, :, 18:]), axis=2)
test = np.concatenate((test_x.reshape((-1, 320)), np.array(test0)[:, 512:]), axis=1)
pd.DataFrame(test).to_csv('row_dataset_new/test_tr_new.csv')
'''
'''
# train_tr_r1完整
train0 = pd.read_csv('row_dataset_new/train_new.csv').values[:, 1:]
train_r1 = train0[[x[-1] == 1 or x[-1] == 0 for x in train0]]
# 删r1r3
train_r1_x = np.array(train_r1)[:, :512].reshape((-1, 16, 32))
train_r = train_r1_x[:, :, :18].reshape((-1, 16, 6, 3)).transpose((0, 1, 3, 2)).reshape((-1, 16, 18))
train_r1_x = np.concatenate((train_r[:, :, :6], train_r1_x[:, :, 18:]), axis=2)
train_r1 = np.concatenate((train_r1_x.reshape((-1, 320)), np.array(train_r1)[:, 512:]), axis=1)
pd.DataFrame(train_r1).to_csv('row_dataset_new/train_tr_new.csv')
'''



























