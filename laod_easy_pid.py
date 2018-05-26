# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


# input
# read_weather
weather_df = pd.read_excel('input/WeatherData.xlsx', header=[6]).drop([0])

# weather_pre
date_name_list = ['year', 'month', 'day', 'week', 'hour', ':', 'min', 'sec']
weather_df.columns = date_name_list + list(weather_df.columns[8:])
weather_df['time'] = pd.to_datetime(weather_df[['year', 'month', 'day', 'hour']])
weather_df = weather_df.drop(date_name_list, axis=1).dropna(axis=1).set_index('time').astype(float)
weather_df = weather_df.asfreq('min').interpolate()

# read_rooms
rooms_df = pd.read_excel('input/Rooms.xlsx', sheet_name=None).values()
[project, rooms_df, walls_df, windows_df, schedule, material_df] = [df.set_index(df.columns[0]) for df in rooms_df]
project = project.drop(['备注'], axis=1).to_dict()['values']
schedule = schedule.asfreq('min').interpolate()
outdoor_temp = weather_df.Temperature.values
outdoor_humidity = weather_df.Humidity.values


# 处理气象参数
# sun
def sun(weather_df):
	# 计算均时差 真太阳时 时角
	weather_df['hour'] = (weather_df.index - weather_df.index[0]).view('int64') / 3.6e12
	weather_df['b_r'] = np.deg2rad((weather_df.hour / 24 - 81) * 360 / 365)
	weather_df['e'] = (9.87*np.sin(2*weather_df.b_r) - 7.53*np.cos(weather_df.b_r) - 1.5*np.sin(weather_df.b_r)) / 60
	weather_df['tas'] = np.mod(weather_df.hour, 24) + weather_df.e + (project['longitude'] - 15*project['time_zone'])/15
	weather_df['omega'] = (weather_df.tas - 12) * 15

	# 计算太阳赤纬
	weather_df['sin_delta'] = 0.397949 * np.sin(weather_df.b_r)
	weather_df['cos_delta'] = np.cos(np.arcsin(weather_df.sin_delta))

	# 计算太阳高度角
	weather_df['sin_h'] = (np.sin(project['latitude']) * weather_df.sin_delta
	                       + np.cos(project['latitude']) * weather_df.cos_delta * np.cos(np.deg2rad(weather_df.omega)))
	weather_df['h'] = np.rad2deg(np.arcsin(weather_df.sin_h))
	weather_df['cos_h'] = np.cos(np.arcsin(weather_df.sin_h))

	# 计算太阳方位角
	weather_df['cos_A'] = ((weather_df.sin_h * np.sin(project['latitude']) - weather_df.sin_delta)
	                       / weather_df.cos_h / np.cos(project['latitude']))
	weather_df['A'] = np.rad2deg(np.sign(weather_df.omega) * np.arccos(weather_df.cos_A))
	weather_df['sin_A'] = np.sin(np.deg2rad(weather_df.A))

	# 日出日落
	weather_df.h[weather_df.h < 0] = 0
	weather_df.A[weather_df.h == 0] = 0

sun(weather_df)


# 建筑构成
# 围护结构
class Face(object):
	"""外立面，计算日射量"""
	def __init__(self, orientation, tilt):
		self.orientation = {'E': -90, 'S': 0, 'W': 90, 'N': -180, 0: 0}[orientation]
		self.tilt = float(tilt)

		# 墙面的三角函数
		self.sin_orientation = np.sin(np.deg2rad(self.orientation))
		self.cos_orientation = np.cos(np.deg2rad(self.orientation))
		self.sin_tilt = np.sin(np.deg2rad(self.tilt))
		self.cos_tilt = np.cos(np.deg2rad(self.tilt))

		# 计算入射角
		self.sh = weather_df.sin_h
		self.sw = np.multiply(weather_df.cos_h, weather_df.sin_A)
		self.ss = np.multiply(weather_df.cos_h, weather_df.cos_A)

		self.wz = self.cos_tilt
		self.ww = np.multiply(self.sin_tilt, self.sin_orientation)
		self.ws = np.multiply(self.sin_tilt, self.cos_orientation)

		self.cos_theta = np.multiply(self.sh, self.wz) + np.multiply(self.sw, self.ww) + np.multiply(self.ss, self.ws)
		self.cos_theta[self.cos_theta < 0] = 0

		# 计算日射量
		self.Fs = 1 / 2 + 1 / 2 * self.cos_tilt
		self.Fg = 1 - self.Fs
		self.I_D = np.multiply(weather_df['Direct-Solar-Radiation'], self.cos_theta)
		self.I_s = np.multiply(weather_df['Diffused-Solar-Radiation'], self.Fs)
		self.I_hol = (np.multiply(weather_df['Direct-Solar-Radiation'], weather_df.sin_h)
		              + weather_df['Diffused-Solar-Radiation'])
		self.I_r = np.multiply(self.I_hol, project['rho_g'] * self.Fg)
		self.I_w = self.I_D + self.I_s + self.I_r


# 日照得热在窗，差分法在墙
class Windows(Face):
	"""windows"""
	def __init__(self, window_df):
		self.window_id = window_df.name
		self.room_id = window_df.room_id
		self.area = self.window_area = float(window_df.window_area)
		self.glass_area = float(window_df.glass_area)
		self.tau = float(window_df.window_tau)
		self.bn = float(window_df.window_BN)
		self.k = float(window_df.window_K)
		super().__init__(window_df.orientation, window_df.tilt)

		self.alpha_0 = project['alpha_i']
		self.alpha_m = project['alpha_o']

		self.FI = (1 - self.k / self.alpha_0)
		self.FO = self.k / self.alpha_0
		self.anf = self.window_area * self.FI

		# 日射热取得
		self.CI_D = (3.4167 * self.cos_theta - 4.3890 * self.cos_theta ** 2 + 2.4948 *
		             self.cos_theta ** 3 - 0.5224 * self.cos_theta ** 4)
		self.GT = (self.glass_area * self.tau * (self.CI_D * self.I_D + 0.91 * (self.I_r + self.I_s))).values
		self.GA = self.glass_area * self.bn * (self.CI_D * self.I_D + 0.91 * (self.I_r + self.I_s))

		# 相当外气温度
		self.te_8760 = (self.GA / self.window_area / self.k - project['epsilon'] * self.Fs *
		                np.array(weather_df['Nocturnal-Radiation']) / self.alpha_m + weather_df.Temperature)


class Walls(Face):
	"""walls"""
	def __init__(self, wall_df):
		# for (k, v) in wall_df.to_dict().items():
		#     setattr(self, k, v)
		self.wall_id = wall_df.name
		self.room_id = wall_df.room_id
		self.wall_type = wall_df.wall_type
		self.area = self.wall_area = float(wall_df.wall_area)
		self.room_by_id = wall_df.room_by_id
		self.material = np.array(wall_df.material.split(', '))
		self.depth = np.array(wall_df.depth.split(', ')).astype(float)
		self.grid = np.array(wall_df.grid.split(', ')).astype(int)
		super().__init__(wall_df.orientation, wall_df.tilt)

		self.alpha_0 = project['alpha_i']
		self.alpha_m = {'outer_wall': project['alpha_o'], 'roof': project['alpha_o'],
		                'inner_wall': project['alpha_i'], 'floor': project['alpha_i'],
		                'ceiling': project['alpha_i'], 'ground': 9999}[self.wall_type]

		# 差分法
		material_lambda = material_df.to_dict()['lambda']
		material_c_rho = material_df.to_dict()['c_rho']

		self.r = [1 / self.alpha_0]
		self.cap = [0]
		for i in range(len(self.material)):
			self.r.extend([self.depth[i] / self.grid[i] / material_lambda[self.material[i]]] * self.grid[i])
			self.cap.extend([self.depth[i] / self.grid[i] * material_c_rho[self.material[i]] * 1000] * self.grid[i])
		self.r.append(1 / self.alpha_m)
		self.cap.append(0)

		self.ul = [project['dt'] * 2 / (self.cap[i] + self.cap[i + 1]) / self.r[i] for i in range(sum(self.grid) + 1)]
		self.ur = [project['dt'] * 2 / (self.cap[i] + self.cap[i + 1]) / self.r[i + 1] for i in range(sum(self.grid) + 1)]

		self.u = np.zeros((sum(self.grid) + 1, sum(self.grid) + 1))
		for i in range(sum(self.grid)):
			self.u[i][i] = 1 + self.ul[i] + self.ur[i]
			self.u[i + 1][i] = - self.ul[i + 1]
			self.u[i][i + 1] = - self.ur[i]
		self.u[-1][-1] = 1 + self.ul[-1] + self.ur[-1]
		self.ux = np.array(np.matrix(self.u).I)

		self.FI = self.ux[0][0] * self.ul[0]
		self.FO = self.ux[0][-1] * self.ur[-1]
		self.anf = self.wall_area * self.FI
		self.te = 0

		# 相当外气温度
		if self.wall_type in ('outer_wall', 'roof'):
			self.te_8760 = ((project['a_s'] * self.I_w - project['epsilon'] * self.Fs *
			                 np.array(weather_df['Nocturnal-Radiation'])) / self.alpha_m
			                + weather_df.Temperature)
		elif self.wall_type in 'ground':
			self.te_8760 = [10] * (8760 * 3600 // project['dt'])


# 墙
walls = [Walls(walls_df.loc[i]) for i in list(walls_df.index)]
windows = [Windows(windows_df.loc[i]) for i in list(windows_df.index)]


# 房间
class Rooms(object):
	def __init__(self, room_df):
		self.room_id = room_df.name
		self.volume = float(room_df.volume)
		self.CPF = float(room_df.CPF)
		self.n_air = float(room_df.n_air)
		self.HVAC_schedule = {0: schedule[0], 1: schedule[1]}[room_df.HVAC_schedule].values
		self.human_n = float(room_df.human_n)
		self.human_t = float(room_df.human_t)
		self.human_s24 = float(room_df.human_s24)
		self.human_d = float(room_df.human_d)
		self.human_schedule = {0: schedule[0], 1: schedule[1]}[room_df.human_schedule].values
		self.light_w = float(room_df.light_w)
		self.light_schedule = {0: schedule[0], 1: schedule[1]}[room_df.light_schedule].values
		self.equipment_ws = float(room_df.equipment_ws)
		self.equipment_wl = float(room_df.equipment_wl)
		self.equipment_schedule = {0: schedule[0], 1: schedule[1]}[room_df.equipment_schedule].values
		self.mode = 0
		self.rho = 1.2
		self.r = 2501000

		self.human_s = 0
		self.human_l = 0
		self.Q_HS = 0
		self.Q_HL = 0
		self.human_kc = 0.5
		self.human_kr = 1 - self.human_kc

		self.light_kc = 0.4
		self.light_kr = 1 - self.light_kc

		self.equipment_kc = 0.4
		self.equipment_kr = 1 - self.equipment_kc

		# 风管接口
		self.g = 0
		self.g_fresh = 0
		self.duct = 0
		self.gmax = 0
		self.gmin = 0
		self.g_set = 0
		self.load = 0

		# 定义变量
		self.load_sum_s = 0
		self.load_sum_w = 0
		self.load_max_s = 0
		self.load_max_w = 0

		self.HG_c = 0
		self.HG_r = 0
		self.HLG = 0
		self.GT = 0
		self.AFT = 0
		self.CA = 0
		self.BRM = 0
		self.BRC = 0
		self.BRMX = 0
		self.BRCX = 0
		self.T_mrt = 0
		self.capacity = 0

		self.e = 0
		self.es = 0
		self.de = 0

		# 房间构成
		self.windows = [x for x in windows if x.room_id == self.room_id]
		self.walls = [x for x in walls if x.room_id == self.room_id]
		self.envelope = self.windows + self.walls

		# 房间固有属性
		self.Arm = sum([x.area for x in self.envelope])
		self.ANF = sum([x.anf for x in self.envelope])
		self.SDT = self.Arm - project['kr'] * self.ANF
		self.AR = self.Arm * project['alpha_i'] * project['kc'] * (1 - project['kc'] * self.ANF / self.SDT)

		# 内表面吸收率
		for x in self.envelope:
			if "wall_type" in dir(x) and x.wall_type in ('floor', 'ground'):
				x.sn = 0.3 + 0.7 * x.area / self.Arm
			else:
				x.sn = 0.7 * x.area / self.Arm

		# 热容 湿容
		self.RMDT = (project['c_air'] * project['rho_air'] * self.volume + self.CPF * 1000) / project['dt']
		self.RMDTX = (project['rho_air'] * self.volume / project['dt'])

		# 换气量
		self.Go = project['rho_air'] * self.n_air * self.volume / 3600

		# 设定条件
		self.indoor_temp_set_point_summer = 26
		self.indoor_temp_set_point_winter = 20

		# 初始条件
		self.indoor_temp = 26
		self.indoor_temp_sensor = self.indoor_temp
		self.indoor_humidity = 0
		self.indoor_RH = 0
		self.ground_temp = 10
		for x in self.walls:
			x.tn = np.ones(np.array(x.ul).shape) * self.indoor_temp
			if x.wall_type is 'ground':
				x.tn = np.ones(np.array(x.ul).shape) * self.ground_temp

	# 循环接口
	# 需求
	def BR(self, step):
		self.HG_c = 0
		self.HG_r = 0
		self.HLG = 0
		self.GT = 0
		self.AFT = 0
		self.capacity = 0

		# 室内发热
		self.human_s = self.human_s24 - self.human_d * (self.indoor_temp - 24)
		self.human_l = self.human_t - self.human_s
		self.Q_HS = self.human_n * self.human_s
		self.Q_HL = self.human_n * self.human_l

		self.HG_c += self.human_kc * self.Q_HS * self.human_schedule[step]
		self.HG_r += self.human_kr * self.Q_HS * self.human_schedule[step]
		self.HLG += self.Q_HL * self.human_schedule[step]

		self.HG_c += self.light_kc * self.light_w * self.light_schedule[step]
		self.HG_r += self.light_kr * self.light_w * self.light_schedule[step]

		self.HG_c += self.equipment_kc * self.equipment_ws * self.equipment_schedule[step]
		self.HG_r += self.equipment_kr * self.equipment_ws * self.equipment_schedule[step]
		self.HLG += self.equipment_wl * self.equipment_schedule[step]

		# 蓄热
		for x in self.windows:
			x.cf = 0
		for x in self.walls:
			x.cf = np.dot(x.ux[0], x.tn)

		# 室内表面吸收的辐射热
		for x in self.windows:
			self.GT += x.GT[step]
		for x in self.envelope:
			x.rs = x.sn * (self.GT + self.HG_r) / x.area

		# 相当外气温度（注意邻室）
		for x in self.windows:
			x.te = x.te_8760[step]
		for x in self.walls:
			if x.wall_type in ('outer_wall', 'roof', 'ground'):
				x.te = x.te_8760[step]
			if x.wall_type in ('floor', 'ceiling'):
				x.te = self.indoor_temp
			if x.wall_type in 'inner_wall':
				if x.room_by_id:
					for y in rooms:
						if x.room_by_id == y.room_id:
							x.te = 0.7 * y.indoor_temp + 0.3 * outdoor_temp[step]
				else:
					x.te = 0.7 * self.indoor_temp + 0.3 * outdoor_temp[step]

		for x in self.envelope:
			x.aft = (x.FO * x.te + x.FI * x.rs / x.alpha_0 + x.cf) * x.area
			self.AFT += x.aft

		# BRM,BRC
		self.CA = self.Arm * project['alpha_i'] * project['kc'] * self.AFT / self.SDT
		self.BRM = self.RMDT + self.AR + project['c_air'] * self.Go
		self.BRC = (self.RMDT * self.indoor_temp + self.CA + project['c_air'] * self.Go * outdoor_temp[step] + self.HG_c)
		self.BRMX = self.RMDTX + self.Go
		self.BRCX = (self.RMDTX * self.indoor_humidity + self.Go * outdoor_humidity[step] + self.HLG / project['r'])


	# 后处理
	def after_cal(self, step, season):
		if self.mode:
			# capacity(summer)
			self.capacity = self.g * (26 - 15) * 1.005 * 1.2 / 3600
			# capacity(winter)
			# !!!!!!!!!!
			# indoor_temp
			if season == 'winter':
				self.indoor_temp = (self.BRC + self.capacity * 1000) / self.BRM
			elif season == 'summer':
				self.indoor_temp = (self.BRC - self.capacity * 1000) / self.BRM
			else:
				pass


		else:
			self.indoor_temp = self.BRC / self.BRM
			self.load = 0

		self.T_mrt = (project['kc'] * self.ANF * self.indoor_temp + self.AFT) / self.SDT
		for x in self.windows:
			x.T_sn = self.indoor_temp - (self.indoor_temp - outdoor_temp[step]) * x.k / x.alpha_0
		for x in self.walls:
			x.tn[0] += x.ul[0] * (project['kc'] * self.indoor_temp + project['kr'] * self.T_mrt + x.rs / x.alpha_0)
			x.tn[-1] += x.ur[-1] * x.te
			x.tn = np.dot(x.ux, x.tn)
			x.T_sn = x.tn[0]


rooms = [Rooms(rooms_df.loc[i]) for i in list(rooms_df.index)]


# pid control
# init e0,es, in ct0 ta0, out ct1
def pid_control(target, set_point, control0, p, i, d, e0, es, control_max=1, control_min=0, tf=1):
	e = target - set_point
	de = e - e0
	# if de * e <= 0:
	# 	de = 0
	es += e
	control = max(min(control0 - tf * (e * p + es * i + de * d), control_max), control_min)
	return control, e, es, de
	# control = max(min(control0 - tf * (e * p + es * i), control_max), control_min)
	# return control, e

# 阀门控风量
def room_control(room, step, season, method='flow'):
	# mode / schedule
	if room.HVAC_schedule[step]:
		# if ((room.indoor_temp > 22) & (season == 'winter')) or ((room.indoor_temp < 24) & (season == 'summer')):
		# 	room.mode = 0
		# else:
		# 	room.mode = 1
		room.mode = 1
	else:
		room.mode = 0

	if room.mode:
		# target / set_point / control0
		if method == 'flow':
			target = room.indoor_temp
			set_point = room.indoor_temp_set_point_summer if season == 'summer' else room.indoor_temp_set_point_winter
			tf = -1
			p = 1000
			i = 50
			# target = room.duct.g
			# set_point = room.ind
			# tf = 1
			# p = 0.02
			# i = 0.00005
			control0 = room.g
			d = 5000
		else:
			target = 0
			set_point = 0
			control0 = 0
			tf = 0
			p = 0
			i = 0
			d = 0

		# damper control by pid
		room.g, room.e, room.es, room.de = pid_control(target, set_point, control0, p, i, d, room.e, room.es, control_max=2000, control_min=200, tf=tf)
		# room.duct.damper.theta_run()

# 设定开始和结束的时间
start = pd.Timestamp('2001/08/27')
end = pd.Timestamp('2001/08/28')
output_time = pd.date_range(start, end, freq='min').values

output = []
dataset = []

stepdelta = int((start - pd.Timestamp('2001/01/01')).view('int64') / project['dt'] / 10e8)

# run
for cal_step in range(int((end - start).view('int64') / project['dt'] / 10e8)):
	# season
	month = str(output_time[cal_step])[5:7]
	if month in ['05','06','07','08','09','10']:
		season = 'summer'
	else:
		season = 'winter'

	# 打印进度
	if cal_step % 1000 == 0:
		print(cal_step)

	# 从range到schedule
	cal_step += stepdelta

	# load
	for room in rooms:
		room.BR(cal_step)

	# control
	for room in rooms:
		room_control(room, cal_step, season, method='flow')

	# duct_system_control(duct_system, method='flow')

	# g,p distribute
	# all_balanced()
	# print(duct_1.p)

	# # supply_air_temp
	# if duct_system.mode:
	# 	duct_system.air_state_cal(HE_ahu)

	# after
	for room in rooms:
		room.after_cal(cal_step, season)

	# dataset
	dataset.extend([room.indoor_temp for room in [rooms[0]]])
	dataset.extend([room.g for room in [rooms[0]]])
	dataset.extend([rooms[0].e])
	dataset.extend([rooms[0].es])
	dataset.extend([rooms[0].de])
	# dataset.extend([room.BRC for room in rooms])
	# dataset.extend([room.indoor_temp_set_point_summer for room in rooms])
	# dataset.extend([room.g_set for room in rooms])

dataset = np.array(dataset).reshape((-1, 5))
dataset = pd.DataFrame(dataset)
dataset['time'] = output_time[:-1]
dataset.set_index('time', inplace=True)
# dataset.to_csv('FDD_data_base/dataset_f1.csv')

print(dataset)

dataset = dataset.values
plt.figure()
plt.subplot(511)
plt.plot(dataset[:, 0])
# plt.xlim((420, 650))
plt.subplot(512)
plt.plot(dataset[:, 1])
# plt.xlim((420, 650))
plt.subplot(513)
plt.plot(dataset[:, 2])
# plt.xlim((420, 650))
plt.subplot(514)
plt.plot(dataset[:, 3])
# plt.xlim((420, 650))
plt.subplot(515)
plt.plot(dataset[:, 4])
# plt.xlim((420, 650))
plt.show()


# 打包测试PID




