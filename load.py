# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optm


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
        self.window_area = float(window_df.window_area)
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
        self.GT = self.glass_area * self.tau * (self.CI_D * self.I_D + 0.91 * (self.I_r + self.I_s))
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
        self.wall_area = float(wall_df.wall_area)
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

        self.ul = [project['dt'] * 2 / (self.cap[i] + self.cap[i+1]) / self.r[i] for i in range(sum(self.grid)+1)]
        self.ur = [project['dt'] * 2 / (self.cap[i] + self.cap[i+1]) / self.r[i+1] for i in range(sum(self.grid)+1)]

        self.u = np.zeros((sum(self.grid)+1, sum(self.grid)+1))
        for i in range(sum(self.grid)):
            self.u[i][i] = 1 + self.ul[i] + self.ur[i]
            self.u[i+1][i] = - self.ul[i+1]
            self.u[i][i+1] = - self.ur[i]
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
        # self.HVAC_schedule = schedule[str(int(room_df.HVAC_schedule))]


rooms = [Rooms(rooms_df.loc[i]) for i in list(rooms_df.index)]
print(rooms)
print(str(int(rooms_df.loc[1].HVAC_schedule)))
print(schedule.at[:,str(int(rooms_df.loc[1].HVAC_schedule))])




