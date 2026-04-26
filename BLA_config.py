"""
Конфигурация CLI-классификатора «хорошие / плохие» линии (BLA_*).
Пути к логам и уровни засветки задаются здесь (без UI).
"""

from pathlib import Path

# Корень репозитория (родитель каталога с этим файлом — папка dak-cali-ai)
ROOT = Path(__file__).resolve().parent

# Лог(и) с пригодными кривыми (класс 1). Согласованы с разметкой BLA_test_pack (k_dips_rules).
GOOD_LOG_PATHS: tuple[Path, ...] = (
    ROOT / "BLA_test_pack" / "good" / "exported_good.log",
)

# Лог(и) с непригодными кривыми (класс 0).
BAD_LOG_PATHS: tuple[Path, ...] = (
    ROOT / "BLA_test_pack" / "bad" / "exported_bad.log",
)

# Уровни сигнала для Line (как в DatasetGenerator); первый уровень даёт третий признак.
LEVELS: tuple[float, ...] = (3400.0,)

# Параметры детектора «провалов» для accuracy_of_line
# dip_min_delta: минимальная «глубина» провала (разность с меньшим из соседей), чтобы считать провал.
# dip_min_y: минимальный уровень сигнала, ниже которого провалы игнорируются (борьба с шумом у нуля).
DIP_MIN_DELTA = 25.0
DIP_MIN_Y = 0.0
# Окно сглаживания для подсчёта провалов (moving average).
# 1 = без сглаживания. Рекомендуется нечётное 5..21.
DIP_SMOOTH_WINDOW = 7

# Куда сохранять обученную модель CatBoost
MODEL_OUT_PATH = ROOT / "BLA_model.cbm"

# Путь к «смешанному паку» для внешнего теста (две подпапки внутри).
# Структура:
#   TEST_PACK_ROOT/
#     good/   (файлы *.log с хорошими линиями)
#     bad/    (файлы *.log с плохими линиями)
TEST_PACK_ROOT = ROOT / "BLA_test_pack"
TEST_PACK_GOOD_SUBDIR = "good"
TEST_PACK_BAD_SUBDIR = "bad"

# Доля теста при train_test_split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Баланс классов в CatBoost: "Balanced" | "SqrtBalanced" | None
BLA_AUTO_CLASS_WEIGHTS: str | None = "Balanced"

# Экспорт old_data.pkl → good/*.log и bad/*.log (BLA_pkl_to_logs.py)
PKL_INPUT_PATH = ROOT / "BLA_test_pack" / "old_data.pkl"
PKL_EXPORT_GOOD_DIR = ROOT / "BLA_test_pack" / "good"
PKL_EXPORT_BAD_DIR = ROOT / "BLA_test_pack" / "bad"
PKL_EXPORT_GOOD_LOG_NAME = "exported_good.log"
PKL_EXPORT_BAD_LOG_NAME = "exported_bad.log"
# Режим экспорта (BLA_pkl_to_logs.py): dips | k_median | k_dips_rules
PKL_EXPORT_MODE = "k_dips_rules"
# Режим k_dips_rules: good, если k < PKL_RULE_K_GOOD_LT и число провалов <= PKL_RULE_DIPS_GOOD_MAX;
# иначе bad (k >= порога или провалов > PKL_RULE_DIPS_GOOD_MAX).
PKL_RULE_K_GOOD_LT = 140.0
PKL_RULE_DIPS_GOOD_MAX = 10
# (устар.) только для mode=dips
PKL_EXPORT_GOOD_MAX_DIPS = 2
# Сколько чисел засветки в одной строке файла (как в типичных train.log).
PKL_EXPORT_COLS_PER_ROW = 16
