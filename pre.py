import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# 1小时整合
'''
# file = 'FDD_data_base_new/dataset_08_f0_r0.csv'
file = 'FDD_data_base_new/dataset_09_f0_r0.csv'
data = pd.read_csv(file)
data['time'] = pd.to_datetime(data['time'])
data = data.set_index('time')
data = data.resample('1H').mean()
data['f'] = file[30]
data['r'] = file[33]


# for F, R in [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3), (4, 1), (4, 2), (4, 3),
#              (5, 1), (5, 2), (5, 3), (6, 1), (6, 2), (6, 3), (7, 1), (7, 2), (7, 3), (8, 0), (9, 0)]:
#   file = 'FDD_data_base_new/dataset_08_f%s_r%s.csv' % (F, R)

for F, R in [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 0), (9, 0)]:
	file = 'FDD_data_base_new/dataset_09_f%s_r%s.csv' % (F, R)

	print(file)
	newdata = pd.read_csv(file)
	newdata['time'] = pd.to_datetime(newdata['time'])
	newdata = newdata.set_index('time')
	newdata = newdata.resample('1H').mean()

	newdata['f'] = file[30]
	newdata['r'] = file[33]

	data = pd.concat([data, newdata])

print(data.describe())
print(data)

# data.to_csv('row_dataset_new/08_row.csv')
data.to_csv('row_dataset_new/09_row.csv')
'''

# 标准化
'''
rowdata_08 = pd.read_csv('row_dataset_new/08_row.csv').values
rowdata_09 = pd.read_csv('row_dataset_new/09_row.csv').values

x = np.concatenate((rowdata_08[:, 1:33], rowdata_09[:, 1:33]))

x_mean = x.mean(axis=0)
x_std = [2.534511401, 2.528311702, 2.487958772, 20.94523974, 20.62261527, 19.33231078,
         930.782106, 824.5191916, 946.3482711, 256.3722828, 288.6350848, 259.994863, 11.28687844,
         11.46112565, 11.47296564, 2.306080259, 2.306080259, 2.306080259, 38.62324912,
         36.67110341, 13.95005415, 10.43991343, 1.525253382, 3.255178009, 25.46050501,
         1.516824258, 3.956949346, 1.025378949, 5.017182651, 5.331343727, 20.97647634, 1.299522715]

x_std = np.array(x_std)
x = (x - x_mean) / x_std

std_train = np.concatenate((x[:17856, :], rowdata_08[:, 33:]), axis=1)
std_test = np.concatenate((x[17856:, :], rowdata_09[:, 33:]), axis=1)

pd.DataFrame(std_train).to_csv('row_dataset_new/08_std.csv')
pd.DataFrame(std_test).to_csv('row_dataset_new/09_std.csv')
pd.DataFrame([x_mean, x_std]).to_csv('row_dataset_new/std.csv')
'''

# 8-24
'''
std_train = pd.read_csv('row_dataset_new/08_std.csv').values[:, 1:]
std_test = pd.read_csv('row_dataset_new/09_std.csv').values[:, 1:]

hour = list(range(24)) * 31 * 24
index = [x > 7 for x in hour]
std_train = std_train[index]

hour = list(range(24)) * 10 * 10
index = [x > 7 for x in hour]
std_test = std_test[index]

pd.DataFrame(std_train).to_csv('row_dataset_new/08_std_8-24.csv')
pd.DataFrame(std_test).to_csv('row_dataset_new/09_std_8-24.csv')
'''

# 16合1
train = pd.read_csv('row_dataset_new/08_std_8-24.csv').values[:, 1:]
test = pd.read_csv('row_dataset_new/09_std_8-24.csv').values[:, 1:]

# print(test.shape)
train_n = np.array(train[:, :32]).reshape(-1, 16*32)
test_n = np.array(test[:, :32]).reshape(-1, 16*32)

# plot test
'''
test_show = np.array(test[:, :32]).reshape(10, 10, 16, 32).transpose((1, 0, 2, 3))
plt.figure()
for i in range(100):
		plt.subplot(10,10,i+1)
		plt.imshow(test_show[i//10, i%10, :, :], 'gray')
		plt.xticks([])
		plt.yticks([])

plt.subplots_adjust(hspace=0)
plt.show()
'''

index = ([True] + [False] * 15) * 31 * 24
train_n = np.concatenate((train_n, train[:, 32:][index]), axis=1)

index = ([True] + [False] * 15) * 10 * 10
test_n = np.concatenate((test_n, test[:, 32:][index]), axis=1)

pd.DataFrame(train_n).to_csv('row_dataset_new/train.csv')
pd.DataFrame(test_n).to_csv('row_dataset_new/test.csv')








































