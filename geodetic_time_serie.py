import numpy as np
from pyparsing import *
import math as m
import random
import pandas as pd
from pandas import DataFrame
from scipy import signal, stats
from matplotlib import pyplot as plt
from pandas import Timestamp as Ts
import os
from pathlib import Path
import datetime


class geodetic_time_serie(DataFrame):

    def lombescargle(self, axe_column, date_column, comb):
        if comb is "day_year":
            nout = 10000
            frequencies = np.linspace(1 / 12, 15, nout)
            w = frequencies
        if comb is 'hour_day':
            nout = 10000
            self[date_column].min()
            self[date_column].max()
            frequencies = np.linspace(1 / 12, 15, nout)
            w = frequencies
        y = self.loc[:, [axe_column]].to_numpy()
        y = np.squeeze(y)
        y = signal.detrend(y)
        x = pd.to_datetime(self[date_column], format="%Y/%m/%d %H:%M:%S.%f")
        x = np.squeeze(x)
        # A = 0.00000001
        # w0 = 10.  # rad/sec
        # nin = 150
        # nout = 100000
        # rng = np.random.default_rng()
        # x = rng.uniform(0, 10*np.pi, nin)
        #  y = A * np.cos(w0 * x)+0*np.cos(4*x)+0*np.sin(3*x)
        #   w = np.linspace(0.01, 12, nout)
        pgram = signal.lombscargle(x, y, w, normalize=True)
        fig, (ax_t, ax_w) = plt.subplots(2, 1, constrained_layout=True)
        ax_t.plot(x, y, 'b+')
        ax_t.set_xlabel('Time [s]')
        ax_w.plot(w, pgram)
        ax_w.set_xlabel('Angular frequency [rad/s]')
        ax_w.set_ylabel('Normalized amplitude')
        plt.show()
        return pgram

    def outliers_filtering(self, axe, type, whisker_width=1.5, kernel_size=3):
        if type is "iqr":  # надо проверять
            Q1 = self[axe].quantile(0.25)
            Q3 = self[axe].quantile(0.75)
            IQR = Q3 - Q1
            filtered = (self[axe] >= Q1 - whisker_width * IQR) & (
                        self[axe] <= Q3 + whisker_width * IQR)  # from here: https://stackoverflow.com/a/40341529
            filtered = self.loc[filtered]
            outliers_numb = self.shape[0] - filtered.shape[0]
            return filtered, outliers_numb
        if type is "3sigma":
            return
        if type is "median":
            filtered = signal.medfilt2d(self[axe], kernel_size=kernel_size)
            return filtered

            return

    def noisetype(axe_column):
        return

    def predict(axe_column, date, type):
        if type is "Linear":
            return
        if type is "Neural_Network_LTCM":
            return

    def test_coordinate_time_series_generator(period, random_from, random_after):
        x0 = 4.52260828455453e+05
        y0 = 3.63587749472371e+06
        z0 = 5.20345321431246e+06
        vx = -2.57928156083501e-02
        vy = 2.25331944516226e-03
        vz = -9.43054722599818e-04
        X_cosine_annual = -2.97770129341665e-03
        X_sine_annual = -1.52824655969895e-03
        Y_cosine_annual = -7.50343793311789e-03
        Y_sine_annual = -2.93987584804527e-03
        Z_cosine_annual = -1.13996274591522e-02
        Z_sine_annual = -5.53152026612332e-03
        X_cosine_semiannual = -2.97770129341665e-03
        X_sine_semiannual = -1.52824655969895e-03
        Y_cosine_semiannual = -7.50343793311789e-03
        Y_sine_semiannual = -2.93987584804527e-03
        Z_cosine_semiannual = -1.13996274591522e-02
        Z_sine_semiannual = -5.53152026612332e-03
        coordinate_time_series = np.zeros((period, 4))
        print(coordinate_time_series)
        for t in range(period):
            xt = x0 + vx * t / 365 + X_cosine_annual * m.cos(
                2 * m.pi * t / 365) + X_sine_annual * m.sin(2 * m.pi * t / 365) + X_cosine_semiannual * m.cos(
                4 * m.pi * t / 365) + X_sine_semiannual * m.sin(4 * m.pi * t / 365) + random.randint(random_from, random_after) / 10000
            yt = y0 + vy * t / 365 + Y_cosine_annual * m.cos(2 * m.pi * t / 365) + Y_sine_annual * m.sin(
                2 * m.pi * t / 365) + Y_cosine_semiannual * m.cos(4 * m.pi * t / 365) + Y_sine_semiannual * m.sin(
                4 * m.pi * t / 365) + random.randint(random_from, random_after) / 10000
            zt = z0 + vz * t / 365 + Z_cosine_annual * m.cos(2 * m.pi * t / 365) + Z_sine_annual * m.sin(
                2 * m.pi * t / 365) + Z_cosine_semiannual * m.cos(4 * m.pi * t / 365) + Z_sine_semiannual * m.sin(
                4 * m.pi * t / 365) + random.randint(random_from, random_after) / 10000


            coordinate_time_series[t][0] = xt
            coordinate_time_series[t][1] = yt
            coordinate_time_series[t][2] = zt
            coordinate_time_series[t][3] = t

        a = DataFrame(coordinate_time_series, columns=['X', 'Y', 'Z', 'Epoch'])
        return a

    # def kinematic_model_estimation(axe, date_column, type):
    #
    #     if type is "SimpleLS":
    #         A = np.zeros((self[axe].shape[0], 6))
    #         X = np.empty((3, 6))
    #         t = self[date_column]
    #         for i in range(self[date_column].shape[0]):
    #             A[i] = np.array(
    #                 [1, t, m.sin(2 * m.pi * t), m.cos(2 * m.pi * t), m.sin(4 * m.pi * t), m.cos(4 * m.pi * t)])
    #         N = -A.transpose().dot(A)
    #         for j in range(3):
    #             L = coordinate_time_series.transpose()[j]
    #             X[j] = - np.linalg.inv(N).dot(A.transpose().dot(L))
    #         print(X[0][0] - x0, X[0][1] - vx)
    #         return

    def pos2geodetic_time_serie(posfilepath, solution_selection_type, freqw, sol_freq):
        # to do: 1) дискриминатор временных рядов по разным станциям. Читать в заголовке приблизительные коорданаты и если координаты различаются более чем наперед заданное пользователем значение - создавать разные временные ряды
        pos_files_list = os.listdir(posfilepath)
        array = DataFrame()
        array3 = DataFrame()
        for files in pos_files_list:
            data = ''
            str = Path(posfilepath, files)
            with open(str, 'r') as file:
                data = file.read()
            data = data.split('\n')
            # ищем откуда начинаются решения
            start_data_ind = 0
            for i in data:
                if not i[0] == '%':
                    start_data_ind = data.index(i)
                    break
            # считываем название колонок
            name_col = ['0'] + data[start_data_ind - 1].split()[1:]
            data = data[start_data_ind:-2]
            data = [data[i].split() for i in range(len(data))]
            # загоняем в датафрейм. Так как у нас дата разбилась на дату и время их надо еще объединить в одну колонку
            df = pd.DataFrame(data, columns=name_col)
            df['GPST'] = df['0'] + ' ' + df['GPST']
            df.pop('0')
            # изменяем тип данных колонок
            df[df.columns[1:4]] = df[df.columns[1:4]].astype(float)
            df[df.columns[4:6]] = df[df.columns[4:6]].astype(int)
            df[df.columns[6:]] = df[df.columns[6:]].astype(float)
            df[df.columns[0]] = pd.to_datetime(df['GPST'], format="%Y/%m/%d %H:%M:%S.%f")
            array = pd.concat([array, df])
        if solution_selection_type is 'max_ratio':
            array = array.set_index('GPST')
            gap_filler2 = array.asfreq(freq=freqw)
            gap_filler2 = gap_filler2.reset_index()
            session_start_time = gap_filler2['GPST'].min(axis=0)
            sections = int((gap_filler2['GPST'].max(axis=0) - gap_filler2['GPST'].min(axis=0)) / sol_freq)
            gap_filler2 = gap_filler2.set_index(keys=['GPST'])
            for i in range(sections):
                array2 = gap_filler2.loc[session_start_time + i * sol_freq:session_start_time + (i + 1) * sol_freq]
                array2 = array2.reset_index()
                if array2['ratio'].min() > 0:
                    array3 = pd.concat([array3, array2.loc[array2['ratio'].argmax(axis=0), :]], axis=1)
            if solution_selection_type is 'a':
                array = pd.concat([array, df.loc[df['ratio'].argmax(), :]], axis=1)
        if solution_selection_type is "RMS_min":
            array = array.set_index('GPST')
            gap_filler2 = array.asfreq(freq=freqw)
            gap_filler2 = gap_filler2.reset_index()
            session_start_time = gap_filler2['GPST'].min(axis=0)
            sections = int((gap_filler2['GPST'].max(axis=0) - gap_filler2['GPST'].min(axis=0)) / sol_freq)
            RMS_array = gap_filler2.loc[:, 'sdn(m)'] ** 2 + gap_filler2.loc[:, 'sde(m)'] ** 2 + gap_filler2.loc[:,
                                                                                                'sdu(m)'] ** 2
            RMS_array = np.sqrt(RMS_array)
            gap_filler2 = gap_filler2.assign(RMS=RMS_array)
            gap_filler2 = gap_filler2.set_index(keys=['GPST'])
            for i in range(sections):
                array2 = gap_filler2.loc[session_start_time + i * sol_freq:session_start_time + (i + 1) * sol_freq]
                array2 = array2.reset_index()
                if array2['RMS'].min() > 0:
                    array3 = pd.concat([array3, array2.loc[array2['RMS'].argmin(), :]], axis=1)
        array3 = array3.transpose()
        array3 = array3.reset_index()
        return array3
