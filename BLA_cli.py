"""
CLI: обучение бинарного классификатора линий засветки (хорошие / плохие).

Запуск из корня проекта:
    python BLA_cli.py

Пути к логам и гиперпараметры — в BLA_config.py (без интерфейса).
"""

from __future__ import annotations

import argparse

import BLA_config as cfg
from BLA_eval import eval_on_mixed_pack
from BLA_train import train_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description="BLA: train good/bad line classifier")
    parser.add_argument(
        "--iterations",
        type=int,
        default=500,
        help="CatBoost iterations",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="learning rate",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=6,
        help="tree depth",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="меньше логов CatBoost",
    )
    parser.add_argument(
        "--eval-pack",
        type=str,
        default="",
        help="путь к смешанному паку (подпапки good/ и bad/); если не задан — берём из конфига",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="не запускать оценку на смешанном паке после обучения",
    )
    args = parser.parse_args()

    result = train_classifier(
        cfg.GOOD_LOG_PATHS,
        cfg.BAD_LOG_PATHS,
        cfg.LEVELS,
        model_out=cfg.MODEL_OUT_PATH,
        dip_min_delta=cfg.DIP_MIN_DELTA,
        dip_min_y=cfg.DIP_MIN_Y,
        dip_smooth_window=cfg.DIP_SMOOTH_WINDOW,
        test_size=cfg.TEST_SIZE,
        random_state=cfg.RANDOM_STATE,
        iterations=args.iterations,
        learning_rate=args.lr,
        depth=args.depth,
        auto_class_weights=cfg.BLA_AUTO_CLASS_WEIGHTS,
        verbose=not args.quiet,
    )

    if args.no_eval:
        return

    pack_root = cfg.TEST_PACK_ROOT if not args.eval_pack else args.eval_pack
    try:
        eval_on_mixed_pack(
            result["model"],
            pack_root=pack_root,
            good_subdir=cfg.TEST_PACK_GOOD_SUBDIR,
            bad_subdir=cfg.TEST_PACK_BAD_SUBDIR,
            levels=cfg.LEVELS,
            dip_min_delta=cfg.DIP_MIN_DELTA,
            dip_min_y=cfg.DIP_MIN_Y,
            dip_smooth_window=cfg.DIP_SMOOTH_WINDOW,
            verbose=not args.quiet,
        )
    except FileNotFoundError as e:
        if not args.quiet:
            print()
            print("Внешний тест (mixed pack) пропущен:", e)


if __name__ == "__main__":
    main()
