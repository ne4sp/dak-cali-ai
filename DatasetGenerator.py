# in файлы train.log.
# out датасет(каждый раз создаётся новый, опционально обновить существующий)
import array
import re
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

pt = r'\d{1,2}\.\d{1,3}'


def intersection_at_level(lv: array.array, level: float) -> float | None:
    """Позиция пересечения кривой засветки с горизонталью level (индекс + дробь), или None."""
    level = float(level)
    for j in range(len(lv) - 1):
        y1, y2 = lv[j], lv[j + 1]
        if (y1 <= level <= y2) or (y2 <= level <= y1):
            if y2 != y1:
                t = (level - y1) / (y2 - y1)
                return float(j + t)
    return None


class Train:
    def __init__(
        self,
        init_path: str,
        flag=True,
        levels: Sequence[float] | None = None,
    ):
        self.lines = []
        self.init_path = init_path
        self.levels = tuple(float(x) for x in (levels if levels is not None else (3400.0,)))
        self.threshold = self.levels[-1] if self.levels else 3400.0
        self.eline = None
        self.initialize() if flag != False else None

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
            if 'Concentration' in line:
                try:
                    arr = re.findall(pt, line)
                    conc = arr[0]
                    conc, temp = float(conc), -1
                except Exception:
                    ''
                continue
            if line == '':
                self.lines.append(
                    Line(lv, conc, temp, levels=self.levels)
                )

                lv = array.array('f', [])
                conc = None
                temp = None
            else:
                lv.extend(array.array('f', list(map(lambda x: float(x), line.split()))))


class Line:
    def __init__(
        self,
        lv: array.array[float],
        conc: float,
        temp: float,
        to_compare=None,
        levels: Sequence[float] = (3400.0,),
    ):
        self.lv = lv
        self.conc = conc
        self.temp = temp
        self.to_compare = to_compare
        self.levels = tuple(float(x) for x in levels)

        self.k = None
        self.accuracy_of_line = None
        self.acs_px = None
        self.level_intersections: list[float | None] = []
        self.o_acs_px = None
        self.o_level_intersections: list[float] = []

        self.create_features()

        if to_compare is not None:
            self.create_o()

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
        self.k = float(max_slope)
    
    def find_accuracy_of_line(self):
        slopes = np.diff(self.lv)
        sign_changes = np.diff(np.sign(slopes))
        self.accuracy_of_line = np.sum(np.abs(sign_changes))

    def create_features(self):
        self.find_k()
        self.find_acs_px()
        self.level_intersections = [
            intersection_at_level(self.lv, L) for L in self.levels
        ]

    def create_o(self):
        self.o_acs_px = self.to_compare.acs_px - self.acs_px
        self.o_level_intersections = []
        for i, _ in enumerate(self.levels):
            ref_i = self.to_compare.level_intersections[i]
            s_i = self.level_intersections[i]
            if ref_i is not None and s_i is not None:
                self.o_level_intersections.append(float(ref_i - s_i))
            else:
                self.o_level_intersections.append(float("nan"))

    def execute_data(self):
        return [self.o_level_intersections, [self.conc]]


def list_training_log_relpaths(trains_dir: str) -> list[str]:
    root = Path(trains_dir)
    if not root.is_dir():
        return []
    out = []
    for p in sorted(root.rglob("*.log")):
        rel = p.relative_to(root)
        out.append(rel.as_posix())
    return out


def generate(
    trains_dir: str = "trains",
    include_relative: Optional[set[str]] = None,
    levels: Sequence[float] | None = None,
):
    """
    Собирает датасет из логов.
    levels: последовательность уровней засветки для признаков (относительно нулевой линии файла).
    """
    if levels is None:
        levels = (3400.0,)
    data = []
    root = Path(trains_dir)
    if not root.is_dir():
        return data

    paths = sorted(root.rglob("*.log"))
    for path in paths:
        rel = path.relative_to(root).as_posix()
        if include_relative is not None and rel not in include_relative:
            continue
        train = Train(str(path), levels=levels)
        for line in train.lines:
            line.to_compare = train.lines[0]
            line.create_o()
            data.append(line.execute_data())

    return data


def iter_labeled_lines(
    trains_dir: str,
    include_relative: Optional[set[str]],
    levels: Sequence[float],
) -> Iterable["Line"]:
    """Все строки из выбранных логов с привязкой к нулю файла (для calibration.json)."""
    root = Path(trains_dir)
    if not root.is_dir():
        return
    paths = sorted(root.rglob("*.log"))
    for path in paths:
        rel = path.relative_to(root).as_posix()
        if include_relative is not None and rel not in include_relative:
            continue
        train = Train(str(path), levels=levels)
        ref = train.lines[0]
        for line in train.lines:
            line.to_compare = ref
            line.create_o()
            yield line
