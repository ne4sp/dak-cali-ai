import os
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple


def process_and_deduplicate_logs(directory: str, file_extension: str = '.log') -> Tuple[List[str], List[str]]:
    path = Path(directory)
    file_map: Dict[str, str] = {}
    files_to_delete: List[str] = []

    all_files = sorted(path.glob(f'*{file_extension}'))

    for file_path in all_files:
        if not file_path.is_file():
            continue

        try:
            # Вычисляем SHA256 хеш для надежного сравнения содержимого
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                # Читаем файл кусками для эффективности
                while chunk := f.read(4096):
                    hasher.update(chunk)
            file_hash = hasher.hexdigest()

            if file_hash in file_map:
                # Найден дубликат: отмечаем текущий файл для удаления
                files_to_delete.append(str(file_path))
                print(f"Дубликат найден: '{file_path}' (оригинал: '{file_map[file_hash]}'). Будет удален.")
            else:
                # Это первый встреченный файл с таким хешем
                file_map[file_hash] = str(file_path)

        except Exception as e:
            print(f"Ошибка при обработке файла {file_path}: {e}")

    # 2. Удаление дубликатов
    for file_path_str in files_to_delete:
        try:
            os.remove(file_path_str)
        except Exception as e:
            print(f"Не удалось удалить файл {file_path_str}: {e}")

    # 3. Переименование оставшихся файлов
    remaining_files = [Path(p) for p in file_map.values()]
    remaining_files.sort()  # Сортируем пути, чтобы нумерация была логичной

    new_names: List[str] = []

    for i, old_path in enumerate(remaining_files):
        new_name = f"train_{i}{file_extension}"
        new_path = old_path.with_name(new_name)

        try:
            old_path.rename(new_path)
            new_names.append(str(new_path))
        except Exception as e:
            print(f"Ошибка при переименовании {old_path.name} в {new_name}: {e}")

    return new_names, files_to_delete

process_and_deduplicate_logs(
    'trains/'
)