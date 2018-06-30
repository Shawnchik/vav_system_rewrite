import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel('input/building_v.xlsx').values
print(data[0])

lon_start = data[:, 5]
lat_start = data[:, 6]
lon_end = data[:, 7]
lat_end = data[:, 8]

# color = (data[:, 2] - min(data[:, 2])) / (max(data[:, 2]) - min(data[:, 2]))
color = list(np.random.rand(len(lon_start))) * 10


def lon2lon(lon):
	return float(lon[:3]) * 3600 + float(lon[4:6]) * 60 + float(lon[7:11])


def lat2lat(lat):
	return float(lat[:2]) * 3600 + float(lat[3:5]) * 60 + float(lat[6:10])

lon_start = [lon2lon(i) for i in lon_start]
lon_end = [lon2lon(i) for i in lon_end]

lat_start = [lat2lat(i) for i in lat_start]
lat_end = [lat2lat(i) for i in lat_end]

x_s = []
y_s = []
for i in range(len(lon_start)):
	x = np.linspace(lon_start[i], lon_end[i], 10)
	y = np.linspace(lat_start[i], lat_end[i], 10)
	x_s.extend(x)
	y_s.extend(y)


plt.scatter(x_s, y_s, c=color, cmap=plt.cm.autumn, edgecolor='none', s=[i*300 for i in color])
# plt.plot([lon_start[i], lon_end[i]], [lat_start[i], lat_end[i]], c=str(color[i]), linewidth=15)
plt.show()















