#in файлы train.log.
#out датасет(каждый раз создаётся новый, опционально обновить существующий)
import matplotlib.pyplot as plt
import array
from typing import Optional
import numpy as np
import re
import os

pt = r'\d{1,2}\.\d{1,3}'


class Train:
    def __init__(self, init_path : str, flag=True):
        self.lines = []
        self.init_path = init_path
        self.initialize() if flag != False else ''
        self.threshold = 3400
        self.eline = None

    def initialize(self):
        with open(self.init_path, 'r') as file:
            text = file.read().splitlines()
        lv = array.array('f', [])
        conc = -1
        temp = -1
        for line in text:
            if '*' in line:
                continue
            if 'Probe' in line:
                continue
            if 'Concentration' in  line:
                try:
                    arr = re.findall(pt, line)
                    conc = arr[0]
                    conc, temp = float(conc), -1
                except Exception:''
                continue
            if line == '':
                self.lines.append(Line(
                    lv, conc, temp
                ))

                lv = array.array('f', [])
                conc = None
                temp = None
            else:
                lv.extend(array.array('f', list(map(lambda x: float(x), line.split()))))

class Line:
    def __init__(self, lv : array.array[float], conc : float, temp : float, to_compare=None):
        self.lv = lv # засветка((128х1)of x where x in[0, 4096])
        self.conc = conc # концентрация (-1, +inf), если -1 то ошибка в определении концентрации.
        self.temp = temp # температура (-1, +inf), если -1 то ошибка в определении температуры.
        self.to_compare = to_compare # нулевая линия из train.log, чтобы получать относительные данные.

        self.k = None
        self.acs_px = None
        self.level_intersection = None

        self.o_acs_px = None
        self.o_level_intersection = None

        self.level = 3400
        self.create_features()

        if to_compare is not None:
            self.create_o()

    def find_level_intersection(self):
        for j in range(len(self.lv) - 1):
            y1, y2 = self.lv[j], self.lv[j + 1]

            if (y1 <= self.level <= y2) or (y2 <= self.level <= y1):
                if y2 != y1:
                    t = (self.level - y1) / (y2 - y1)
                    self.level_intersection = j + t
                    return
    def find_acs_px(self, window_size=2):
        slopes = np.diff(self.lv)
        max_slope_index = np.argmax(slopes)

        start_index = max(0, max_slope_index - window_size)
        end_index = min(len(slopes), max_slope_index + window_size + 1)

        weights_window = slopes[start_index:end_index]
        pixel_indices = np.arange(start_index, end_index) + 0.5

        sum_weights = np.sum(np.abs(weights_window))
        if sum_weights == 0:
            self.acs_px = float(max_slope_index + 0.5)
            return

        centroid = np.sum(pixel_indices * np.abs(weights_window)) / sum_weights

        self.acs_px = centroid

    def find_k(self):
        slopes = np.diff(self.lv)
        max_slope = np.max(np.abs(slopes))
        self.k =  float(max_slope)
    def create_features(self):
        self.find_k()
        self.find_acs_px()
        self.find_level_intersection()
    def create_o(self):
        self.o_acs_px = self.to_compare.acs_px - self.acs_px
        self.o_level_intersection = self.to_compare.level_intersection - self.level_intersection

    def execute_data(self):
        return [
            [self.o_level_intersection],
            [self.conc]
        ]

def generate():
    data = []

    for root, _, files in os.walk('trains'):
        for file in files:
            train = Train('trains/' + file)
            for line in train.lines:
                line.to_compare = train.lines[0]
                line.create_o()
                data.append(line.execute_data())
    return data

