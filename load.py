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
		                np.array(weather_df['Diffused-Solar-Radiation']) / self.alpha_m + weather_df.Temperature)


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

		self.co2_p = 400
		self.co2_dp = 0

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
		self.indoor_humidity = 0
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

	# CO2浓度模型
	def room_co2(self, step):
		# fresh air
		g_sum = sum([room.g for room in rooms])
		g_fresh_sum = duct_system.g_supply_air - duct_system.g_mix_air
		if g_sum:
			self.g_fresh = self.g * g_fresh_sum / g_sum
		else:
			self.g_fresh = 0
		# co2
		self.co2_dp = (0.02 * self.human_n * self.human_schedule[step] * project['dt'] / 3600 - (self.co2_p - 400)
		               * self.g_fresh * project['dt'] / 3600) / self.volume
		self.co2_p += self.co2_dp

	# 室内温度计算模型
	# 改
	def load_cal(self, step):
		if self.mode:
			# 空调开
			temp0 = self.BRC / self.BRM
			if temp0 <= 20:
				self.load = (self.BRC - self.BRM * 20) / 1000
				self.load_sum_w += self.load
				self.load_max_w = min(self.load_max_w, self.load)
			elif temp0 >= 26:
				self.load = (self.BRC - self.BRM * 26) / 1000
				self.load_sum_s += self.load
				self.load_max_s = max(self.load_max_s, self.load)
			else:
				self.load = 0
				self.indoor_temp = temp0
		else:
			# 空调关
			self.indoor_temp = self.BRC / self.BRM
			self.indoor_humidity = self.BRCX / self.BRMX
			self.load = 0


	# 后处理
	def after_cal(self, step, season):
		if self.mode:
			# capacity
			self.capacity = self.duct.g * 11 * 1.005 * 1.2 / 3600
			# indoor_temp
			if season == 'winter':
				self.indoor_temp = (self.BRC + self.capacity * 1000) / self.BRM
			elif season == 'summer':
				self.indoor_temp = (self.BRC - self.capacity * 1000) / self.BRM
			else:
				pass
		else:
			self.indoor_temp = self.BRC / self.BRM
			self.indoor_humidity = self.BRCX / self.BRMX
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


# HVAC
# 风阀
class Damper(object):
	def __init__(self, d, rho=1.2, theta0=0):
		self.d = d  # 管径(m)
		self.rho = rho  # 空气密度ρ(kg/m3)
		self.theta = theta0  # 开度θ（角度，0-90，默认0，既全开）
		self.l = 1  # 开度(百分比，0-1，默认1，全开)
		self.zeta = 0  # ζ，局部阻力系数
		self.s = 0  # p = SG2 的 S
		self.theta_run()  # 初始化l, ζ，S

	def theta_run(self, theta=False):
		# θ确定的情况下，算l, ζ，S
		if theta:
			self.theta = theta
		self.l = 1 - np.sin(np.deg2rad(self.theta))
		self.zeta = ((1 - self.l) * (1.5 - self.l) / self.l ** 2)
		self.s = (8 * self.rho * self.zeta) / (np.pi ** 2 * self.d ** 4)

	def plot(self):
		# 绘图 阀门特性曲线
		theta00 = self.theta
		x = np.linspace(1, 80)
		l = []
		zeta = []
		s = []
		for xi in x:
			self.theta_run(xi)
			l.append(self.l)
			zeta.append(self.zeta)
			s.append(self.s)
		#plt.plot(x, np.array(l) * 100, label=u"l/l_max")
		plt.plot(x, np.log10(zeta), label=u'ln(zeta)')
		plt.plot(x, np.log10([z+0.4 for z in zeta]), label=u'ln(zeta+)')
		x0 = [0, 10, 20, 30, 40, 50, 60, 70]
		y1 = [0.5, 0.65, 1.6, 4, 9.4, 24, 67]
		y2 = [0.52, 0.9, 1.6, 2.4, 5.2, 9.7]
		y3 = [0.24, 0.52, 1.54, 3.91, 10.8, 30.6, 188, 751]
		plt.plot(x0[:-1], np.log10(y1), label='ln(1)')
		plt.plot(x0[:-2], np.log10(y2), label='ln(2)')
		plt.plot(x0, np.log10(y3), label='ln(3)')
		#plt.plot(x, np.log(s), label=u'ln(s)')
		plt.legend()
		plt.grid(True)
		plt.show()
		self.theta_run(theta00)


# 风管类
class Duct(object):
	# 输入 流量，管长，局部阻力系数
	# 输出 管径，流速，S，压力损失
	def __init__(self, g, l, zeta, d=0., a=0., b=0., v=4., damper=None, show=False):
		self.g = g  # m3/h  # 流量
		self.g0 = g  # 定格流量
		self.l = l  # m  # 管长
		self.zeta = zeta  # 局部阻力系数ζ
		self.d = d  # 直径(当量直径 d = 2ab/(a+b))
		self.a = a  # 一边(方管可以指定一边)
		self.b = b  # 另一边
		self.v = v  # 流速
		self.damper = damper  # 管路中的可调部件，为了传递s
		self.A = self.g / 3600 / self.v  # 截面积
		self.s = 0

		# 管路树结构中的叶子节点
		self.left = None
		self.right = None
		self.root = 'duct'

		# 公称直径库
		nominal_diameter = [0.015, 0.02, 0.025, 0.032, 0.04, 0.05, 0.065, 0.08, 0.1, 0.125, 0.15, 0.2, 0.25,
		                    0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8]

		# 圆管
		if self.d == 0:  # 不指定管径
			self.d_target = (self.A / np.pi) ** 0.5 * 2  # 目标直径
			# print(self.d_target)
			dis = [abs(nd - self.d_target) for nd in nominal_diameter]  # 残差
			self.d = nominal_diameter[dis.index(min(dis))]  # 选定管径、

		# 方管
		if self.a != 0 and self.b == 0:  # 指定一条边
			self.d_target = (self.A / np.pi) ** 0.5 * 2  # 目标直径
			self.b_target = self.d_target * self.a / (2 * self.a - self.d_target)  # 湿周 计算零一条边的目标值
			dis = [abs(nd - self.b_target) for nd in nominal_diameter]  # 残差
			self.b = nominal_diameter[dis.index(min(dis))]  # 选定另一条边

		if self.a != 0:  # 指定两条边
			self.d = (2 * self.a * self.b) / (self.a + self.b)  # 当量直径

		# 反推
		self.A = 3.1415 * self.d ** 2 / 4
		self.v = self.g / 3600 / self.A

		# 阻力系数
		self.s_cal()
		# 额定压损
		self.p = self.s * (self.g / 3600) ** 2

		# 打印
		if show:
			if self.a:
				print('a, b, v, S, p')
				print(self.a, self.b, self.v, self.s, self.p)
			else:
				print('d, v, S, p')
				print(self.d, self.v, self.s, self.p)

	# 多态
	def g_cal(self, g):
		self.g = g
		self.p = self.s * self.g ** 2

	def s_cal(self):
		if self.damper:
			self.s = (0.02 * self.l / self.d + self.zeta + self.damper.zeta) * 8 * 1.2 / (3.1415 ** 2) / (self.d ** 4)
		else:
			self.s = (0.02 * self.l / self.d + self.zeta) * 8 * 1.2 / (3.1415 ** 2) / (self.d ** 4)


# 水泵 风机
# 性能曲线拟合
class Poly(object):
	"""多项式拟合 y是x的多项式 输出k是拟合好的系数"""
	def __init__(self, x, y, dim):
		self.x = np.array(x, dtype=float)
		self.y = np.array(y, dtype=float)
		self.dim = dim
		# please make sure len(self.x) == len(self.y)

		self.dim_ = True  # 是否能满足dim
		if len(self.x) < self.dim + 1:
			self.dim = len(self.x) - 1
			self.dim_ = False

		# 求x的多项式矩阵
		self.x_mat = np.mat([np.power(self.x, i) for i in range(self.dim + 1)]).T

		# 求解
		if len(self.x) == self.dim + 1:  # 精确解
			self.k = self.x_mat.I * np.mat(self.y).T
		elif len(self.x) > self.dim + 1:  # 最优解
			self.k = (self.x_mat.T * self.x_mat).I * self.x_mat.T * np.mat(self.y).T

		# 预测
		self.prediction = self.x_mat * self.k

	def plot(self):
		plt.scatter(self.x, self.y)
		plot_x = np.linspace(self.x.min(), self.x.max())
		plot_x_mat = np.mat([np.power(plot_x, i) for i in range(self.dim + 1)]).T
		plot_y = plot_x_mat * self.k
		plt.plot(plot_x, plot_y)
		plt.show()


class Fan(object):
	"""风机特性曲线 x是风量 z是频率 y是不同频率和风量下的压力"""
	def __init__(self, x, y, z=(50, 45, 40, 35, 30, 25, 20, 15), dim1=3, dim2=5, ideal=False):
		self.x = x  # 流量list
		self.y = y  # 压力matrix
		self.z = z  # 频率list
		self.dim1 = dim1  # 风机特性曲线的次数
		self.dim2 = dim2  # 曲线的系数的特性曲线的次数
		self.ideal = ideal  # 是否是理想风机(流量和频率成正比，压力和频率的平方成正比)
		self.g = 0
		self.p = 0
		self.inv = 50
		self.e = 0
		self.es = 0


		# 对不同频率下的风量和压力拟合，求出曲线的系数
		if ideal:
			self.x0 = [[xi * zi / 50 for xi in self.x] for zi in self.z]
			self.y0 = [[yi * (zi / 50) ** 2 for yi in self.y] for zi in self.z]
			self.h1 = [Poly(self.x0[i], self.y0[i], self.dim1) for i in range(len(self.x0))]
		else:
			self.h1 = [Poly(self.x[0:len(yi)], yi, self.dim1) for yi in self.y]
		self.k1 = np.array([hi.k for hi in self.h1]).reshape(-1, dim1 + 1)

		# 对不同频率下的曲线的系数拟合
		self.h2 = [Poly(self.z, k1i, dim2) for k1i in self.k1.T]
		self.k2 = np.array([hi.k for hi in self.h2]).reshape(-1, dim2 + 1)

		# check_k1
		self.k1_prediction = np.array([h2i.prediction for h2i in self.h2]).T.reshape(-1, dim1 + 1)
		# k1精度校核

		# check_y
		self.prediction = [np.array(self.k1_prediction[i] * self.h1[i].x_mat.T).flatten() for i in range(len(self.h1))]
		# y精度校核

	# 绘图
	def plot(self):
		for i in range(len(self.h1)):
			plt.scatter(self.h1[i].x, self.h1[i].y)
			plt.plot(self.h1[i].x, self.prediction[i])
		plt.grid(True)
		plt.show()

	# 预测(应用)
	def predict(self, g0, inv0):
		g0 = np.array(g0, dtype=float)
		inv0 = np.array(inv0, dtype=float)

		inv_mat = np.mat([np.power(inv0, i) for i in range(self.dim2 + 1)])
		k1_prediction = inv_mat * self.k2.T
		g_mat = np.mat([np.power(g0, i) for i in range(self.dim1 + 1)])

		return np.array(k1_prediction * g_mat.T).flatten()


# 风管的树结构
class Branch(object):
	# 支管
	def __init__(self, left, right):
		self.left = left
		self.right = right
		self.g = 0
		self.p = 0
		self.s = 0

		# 空枝
		if 'root' not in dir(self):
			self.root = None

		# 计算s
		self.s_cal()

	def s_cal(self):
		# 由上而下的堆栈，由下而上的递归
		self.left.s_cal()
		self.right.s_cal()
		if self.root == 'serial':
			self.s = self.left.s + self.right.s
		elif self.root == 'parallel':
			self.s = self.left.s * self.right.s / (self.left.s ** 0.5 + self.right.s ** 0.5) ** 2
		else:
			raise ValueError

	def g_cal(self, g):
		# 由上而下的递归
		self.g = g
		self.p = self.s * self.g ** 2
		if self.root == 'serial':
			self.left.g_cal(self.g)
			self.right.g_cal(self.g)
		elif self.root == 'parallel':
			self.left.g_cal(self.g * (self.s / self.left.s) ** 0.5)
			self.right.g_cal(self.g * (self.s / self.right.s) ** 0.5)


class Parallel(Branch):
	# 并联节点
	def __init__(self, left, right):
		self.root = 'parallel'
		super().__init__(left, right)


class Serial(Branch):
	# 串联节点
	def __init__(self, left, right):
		self.root = 'serial'
		super().__init__(left, right)


# 排风新风混风段 及 整个风管系统的物理模型
class DuctSystem(object):
	# 构筑风系统 包括，5段风管，两台风机，一个空调箱，空调箱作为定压源
	def __init__(self, duct_supply_air, duct_return_air, duct_exhaust_air, duct_fresh_air,
	             duct_mix_air, fan_s, fan_r, dp_ahu=200):
		self.duct_supply_air = duct_supply_air
		if isinstance(duct_return_air, Duct):  # 确保类型匹配
			self.duct_return_air = duct_return_air
		else:
			raise TypeError
		self.duct_exhaust_air = duct_exhaust_air
		self.duct_fresh_air = duct_fresh_air
		self.duct_mix_air = duct_mix_air
		if isinstance(fan_s, Fan):
			self.fan_s = fan_s
		else:
			raise TypeError
		self.fan_r = fan_r
		self.dp_ahu = dp_ahu
		self.mode = 0
		self.g_return_air = 0
		self.g_supply_air = 0
		self.g_mix_air = 0

	def balance(self):
		# 管路平衡计算
		#print(self.duct_return_air.s, self.duct_exhaust_air.s, self.duct_supply_air.s, self.duct_fresh_air.s, self.duct_mix_air.s)
		#print(self.fan_r.predict(4000, self.fan_r.inv), self.fan_s.predict(4000, self.fan_s.inv))

		# 用于scipy.optimize.fsolve解方程
		def f(x):
			x1 = float(x[0])
			x2 = float(x[1])
			x3 = float(x[2])
			return np.array([
				self.fan_r.predict(x1, self.fan_r.inv) - self.duct_return_air.s * (x1 / 3600) ** 2 -
				self.duct_exhaust_air.s * ((x1 - x3) / 3600) ** 2,
				self.fan_s.predict(x2, self.fan_s.inv) - self.duct_supply_air.s * (x2 / 3600) ** 2 -
				self.duct_fresh_air.s * ((x2 - x3) / 3600) ** 2 - self.dp_ahu,
				self.fan_r.predict(x1, self.fan_r.inv) - self.duct_return_air.s * (x1 / 3600) ** 2 -
				self.duct_mix_air.s * (x3 / 3600) ** 2 - self.duct_supply_air.s * (x2 / 3600) ** 2 +
				self.fan_s.predict(x2, self.fan_s.inv) - self.dp_ahu
			]).flatten()

		[self.g_return_air, self.g_supply_air, self.g_mix_air] = fsolve(f, np.array([3600, 3600, 3600]))
		#print(self.g_return_air, self.g_supply_air, self.g_mix_air)

	def balance_check(self):
		# 检查方程的解
		g1 = self.g_return_air
		g2 = self.g_supply_air
		p1 = self.fan_r.predict(g1, self.fan_r.inv)
		p2 = self.fan_s.predict(g2, self.fan_s.inv)
		ub = p1 - self.duct_return_air.s * (g1 / 3600) ** 2
		ua = - (p2 - self.duct_supply_air.s * (g2 / 3600) ** 2 - self.dp_ahu)
		g31 = g1 / 3600 - (ub / self.duct_exhaust_air.s) ** 0.5
		g32 = g2 / 3600 - (- ua / self.duct_fresh_air.s) ** 0.5
		g33 = ((ub - ua)/self.duct_mix_air.s) ** 0.5
		# 检查气流方向是否正确 ua<0, ub>0

		#print(g1/3600, g2/3600, p1, p2, ub, ua, g31*3600, g32*3600, g33*3600)


# HVAC设备构成
vav1 = Damper(0.45)
vav2 = Damper(0.4)
vav3 = Damper(0.45)
fresh_air_damper = Damper(0.7)
exhaust_air_damper = Damper(0.7)
mix_air_damper = Damper(0.7)

# 送风管段
duct_1 = Duct(2062, 10, 0.05+0.1+0.23+0.4+0.9+1.2+0.23, damper=vav1)
duct_2 = Duct(1819, 2.5, 0.3+0.1+0.4+0.23+1.2+0.9, damper=vav2)
duct_12 = Duct(2062 + 1819, 7.5, 0.05+0.1, a=0.6)
duct_3 = Duct(2062, 2.5, 0.3+0.1+0.4+0.23+1.2+0.9, damper=vav3)
duct_123 = Duct(5427, 4.3, 3.6+0.23, a=0.6)
# 回风管段
duct_return_air = Duct(5427, 1.75, 0.5+0.24, a=0.7)
duct_exhaust_air = Duct(5427, 0.95, 3.7+0.9+0.4+0.05, damper=exhaust_air_damper, a=0.7)
duct_fresh_air = Duct(5427, 0.78, 1.4+0.1+0.4, damper=fresh_air_damper, a=0.7)
duct_mix_air = Duct(5427, 2.2, 0.3+0.4+1.5, damper=mix_air_damper, a=0.7)

g = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800]
p = [[443, 383, 348, 305, 277, 249, 216, 172, 112, 30]]
p.append([355, 296, 256, 238, 207, 182, 148, 97, 21])
p.append([342, 284, 246, 217, 190, 161, 121, 62])
p.append([336, 278, 236, 206, 178, 145, 97, 38])
p.append([320, 264, 223, 189, 153, 109, 50])
p.append([300, 239, 194, 153, 110, 55])
p.append([260, 200, 152, 107, 52])
p.append([179, 129, 79, 24])
# 送回风机
g1 = list(map(lambda x: x * 5427 / 1200, g))
p1 = [[x * 80 / 216 for x in pi] for pi in p]
f1 = Fan(g1, p1)  # 回风机
g2 = list(map(lambda x: x * 5427 / 1200, g))
p2 = [[x * 320 / 216 for x in pi] for pi in p]
f2 = Fan(g2, p2)  # 送风机
# f1.plot()

# 送风管
duct_supply_air = Serial(duct_123, Parallel(duct_3, Serial(duct_12, Parallel(duct_1, duct_2))))


# 整合
def all_balanced():
	duct_1.s_cal()
	duct_2.s_cal()
	duct_3.s_cal()
	duct_fresh_air.s_cal()
	duct_return_air.s_cal()
	duct_mix_air.s_cal()
	duct_exhaust_air.s_cal()
	duct_supply_air.s_cal()
	duct_system.balance()
	duct_system.balance_check()
	duct_supply_air.g_cal(duct_system.g_supply_air)
	# print(duct_1.g, duct_2.g, duct_3.g)

# 风管系统
duct_system = DuctSystem(duct_supply_air, duct_return_air, duct_exhaust_air, duct_fresh_air, duct_mix_air, f2, f1)


# 风管和房间连接
def contract_room_duct(room, duct):
	room.duct = duct
	room.gmax = duct.g0
	room.gmin = duct.g0 / 10

contract_room_duct(rooms[0], duct_1)
contract_room_duct(rooms[1], duct_2)
contract_room_duct(rooms[2], duct_3)


# pid control
# init e0,es, in ct0 ta0, out ct1
def pid_control(target, set_point, control0, p, i, d, e0, es, control_max=1, control_min=0, tf=1):
	e = target - set_point
	de = e - e0
	if de * e <= 0:
		de = 0
	es += e
	control = max(min(control0 - tf * (e * p + es * i + de * d), control_max), control_min)
	return control, e, es
	# control = max(min(control0 - tf * (e * p + es * i), control_max), control_min)
	# return control, e


def deltatemp2flow(room, season):
	# deltatemp
	if season == 'summer':
		deltatemp = room.indoor_temp - room.indoor_temp_set_point_summer
	elif season == 'winter':
		deltatemp = room.indoor_temp - room.indoor_temp_set_point_winter
	else:
		deltatemp = 0

	# 2flow
	if deltatemp < -1:
		room.g_set = room.gmin
	elif deltatemp > 1:
		room.g_set = room.gmax
	else:
		room.g_set = room.gmin + (deltatemp + 1) * (room.gmax - room.gmin) / 2


def room_control(room, step, season, method='flow'):
	# mode / schedule
	if room.HVAC_schedule[step]:
		if ((room.indoor_temp > 22) & (season == 'winter')) or ((room.indoor_temp < 24) & (season == 'summer')):
			room.mode = 0
		else:
			room.mode = 1
		room.mode = 1
	else:
		room.mode = 0

	if room.mode:
		# target / set_point / control0
		if method == 'flow':
			deltatemp2flow(room, season)
			target = room.duct.g
			set_point = room.g_set
			control0 = room.duct.damper.theta
		elif method == 'pressure':
			target = room.indoor_temp
			if season == 'summer':
				set_point = room.indoor_temp_set_point_summer
			elif season == 'winter':
				set_point = room.indoor_temp_set_point_winter
			else:
				set_point = 0
			control0 = room.duct.damper.theta
		else:
			target = 0
			set_point = 0
			control0 = 0

		# damper control by pid
		room.duct.damper.theta, room.e, room.es = pid_control(target, set_point, control0, 0.002, 0, 0, room.e, room.es, control_max=70, control_min=0, tf=-1)
		room.duct.damper.theta_run()


def duct_system_control(system, method='flow'):
	# system_mode
	system.mode = sum([room.mode for room in rooms])

	if system.mode:
		# fan_s
		if method == 'flow':
			target_s = system.duct_supply_air.g
			target_r = system.duct_return_air.g
			set_point_s = sum([room.g_set for room in rooms])
			set_point_r = set_point_s + 20
			control0_s = system.fan_s.inv
			control0_r = system.fan_r.inv
		elif method == 'pressure':
			target_s = rooms[0].duct.p
			set_point_s = 30
			control0_s = system.fan_s.inv
			# r
		else:
			target_s = 0
			target_r = 0
			set_point_s = 0
			set_point_r = 0
			control0_s = 0
			control0_r = 0

		# fan_s control by pid
		system.fan_s.inv, system.fan_s.e, system.fan_s.es = pid_control(target_s, set_point_s, control0_s, 0.002, 0, 0, system.fan_s.e, system.fan_s.es, control_max=50, control_min=15)

		# fan_r control by pid
		system.fan_r.inv, system.fan_r.e, system.fan_r.es = pid_control(target_r, set_point_r, control0_r, 0.002, 0, 0, system.fan_r.e, system.fan_r.es, control_max=50, control_min=15)

		# ve, vm, vf 调节, theta_run

# 设定开始和结束的时间
start = pd.Timestamp('2001/08/29')
end = pd.Timestamp('2001/08/30')
output_time = pd.date_range(start, end, freq='min').values

output = []

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
		room.room_co2(cal_step)

	# control
	for room in rooms:
		deltatemp2flow(room, season)
		room_control(room, cal_step, season)

	duct_system_control(duct_system)

	# g,p distribute
	all_balanced()

	# after
	for room in rooms:
		room.after_cal(cal_step, season)

	output.extend([room.indoor_temp for room in rooms])
	output.extend([room.capacity for room in rooms])
	output.extend([vav1.theta, vav2.theta, vav3.theta, f1.inv, f2.inv, outdoor_temp[cal_step]])

output = np.array(output).reshape((-1, 12))
output = pd.DataFrame(output)
output['time'] = output_time[:-1]
output.set_index('time', inplace=True)
print(output)

# plt.plot(output)
# plt.show()

output.to_csv('output/load_control.csv')






















