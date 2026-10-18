#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import sys
from pathlib import Path

def load_replacements(dict_file):
    """
    Загружает словарь замен из файла.
    Формат: исходное слово=слово замены
    """
    replacements = {}
    try:
        with open(dict_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' not in line:
                    print(f"Предупреждение: строка {line_num} не содержит '=', пропущена: {line}")
                    continue
                original, replacement = line.split('=', 1)
                original = original.strip()
                replacement = replacement.strip()
                if original:
                    replacements[original] = replacement
    except FileNotFoundError:
        print(f"Ошибка: файл словаря '{dict_file}' не найден")
        sys.exit(1)
    
    return replacements


def build_word_pattern(word):
    """
    Создает регулярное выражение для слова с учетом множественного числа.
    Учитывает формы: слово, словоs, словоes, словоies (для английского),
    а также русские окончания.
    """
    escaped = re.escape(word)
    
    # Проверяем окончание слова для корректной обработки множественного числа
    lower_word = word.lower()
    
    # Формируем альтернативы для множественного числа
    variants = [escaped]
    
    # Английские правила множественного числа
    if lower_word.endswith('y') and len(word) > 1 and lower_word[-2] not in 'aeiou':
        # city -> cities
        variants.append(re.escape(word[:-1]) + r'ies')
    elif lower_word.endswith(('s', 'x', 'z', 'ch', 'sh')):
        # box -> boxes
        variants.append(escaped + r'es')
    else:
        # cat -> cats
        variants.append(escaped + r's')
    
    # Русские окончания множественного числа
    # Убираем последнюю гласную и добавляем "ы", "и", "а", "я"
    if re.search(r'[а-яё]$', lower_word, re.IGNORECASE):
        stem = word[:-1]
        variants.append(re.escape(stem) + r'[ыиая]')
    # Для слов на согласную
    elif re.search(r'[бвгджзклмнпрстфхцчшщ]$', lower_word, re.IGNORECASE):
        variants.append(escaped + r'[ыиая]')
    
    # Собираем уникальные варианты
    pattern = r'(?<![A-Za-zА-Яа-яЁё0-9_])(' + '|'.join(variants) + r')(?![A-Za-zА-Яа-яЁё0-9_])'
    
    return pattern


def replace_in_text(text, replacements):
    """
    Выполняет замену всех слов в тексте согласно словарю.
    """
    for original, replacement in replacements.items():
        pattern = build_word_pattern(original)
        
        def repl_func(match):
            matched = match.group(0)
            # Пытаемся сохранить регистр
            if matched.isupper():
                return replacement.upper()
            elif matched[0].isupper():
                return replacement.capitalize()
            else:
                return replacement
        
        text = re.sub(pattern, repl_func, text)
    
    return text


def process_file(file_path, replacements, dry_run=False, backup=False):
    """
    Обрабатывает один файл.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_text = f.read()
    except UnicodeDecodeError:
        # Пробуем другую кодировку
        try:
            with open(file_path, 'r', encoding='cp1251') as f:
                original_text = f.read()
            encoding = 'cp1251'
        except Exception as e:
            print(f"  Ошибка чтения {file_path}: {e}")
            return False
    except Exception as e:
        print(f"  Ошибка чтения {file_path}: {e}")
        return False
    else:
        encoding = 'utf-8'
    
    new_text = replace_in_text(original_text, replacements)
    
    if new_text == original_text:
        return False
    
    if dry_run:
        print(f"  [DRY-RUN] Были бы изменены: {file_path}")
        return True
    
    try:
        if backup:
            backup_path = str(file_path) + '.bak'
            with open(backup_path, 'w', encoding=encoding) as f:
                f.write(original_text)
        
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(new_text)
        print(f"  ✓ Обработан: {file_path}")
        return True
    except Exception as e:
        print(f"  Ошибка записи {file_path}: {e}")
        return False


def process_directory(directory, replacements, extensions=None, dry_run=False, 
                     backup=False, recursive=False):
    """
    Обрабатывает все файлы в директории.
    """
    directory = Path(directory)
    
    if not directory.is_dir():
        print(f"Ошибка: '{directory}' не является директорией")
        return
    
    processed = 0
    modified = 0
    
    # Определяем итератор файлов
    if recursive:
        file_iterator = directory.rglob('*')
    else:
        file_iterator = directory.glob('*')
    
    for file_path in file_iterator:
        if not file_path.is_file():
            continue
        
        # Фильтр по расширению
        if extensions:
            if file_path.suffix.lower() not in extensions:
                continue
        
        # Пропускаем файлы бэкапов
        if file_path.suffix == '.bak':
            continue
        
        processed += 1
        if process_file(file_path, replacements, dry_run=dry_run, backup=backup):
            modified += 1
    
    print(f"\nВсего файлов: {processed}")
    print(f"Изменено файлов: {modified}")


def main():
    parser = argparse.ArgumentParser(
        description='Замена слов в файлах с учетом множественного числа'
    )
    parser.add_argument('directory', help='Директория с файлами')
    parser.add_argument('dictionary', help='Файл словаря замен (формат: слово=замена)')
    parser.add_argument('-e', '--extensions', nargs='+', 
                       help='Расширения файлов (например: .txt .md .py)')
    parser.add_argument('-r', '--recursive', action='store_true',
                       help='Обрабатывать рекурсивно поддиректории')
    parser.add_argument('-n', '--dry-run', action='store_true',
                       help='Тестовый режим без изменения файлов')
    parser.add_argument('-b', '--backup', action='store_true',
                       help='Создавать резервные копии (.bak)')
    
    args = parser.parse_args()
    
    # Загружаем словарь
    replacements = load_replacements(args.dictionary)
    if not replacements:
        print("Словарь пуст или не содержит валидных записей")
        sys.exit(1)
    
    print(f"Загружено правил замены: {len(replacements)}")
    for k, v in replacements.items():
        print(f"  {k} -> {v}")
    print()
    
    # Нормализуем расширения
    extensions = None
    if args.extensions:
        extensions = set()
        for ext in args.extensions:
            if not ext.startswith('.'):
                ext = '.' + ext
            extensions.add(ext.lower())
    
    # Обрабатываем
    process_directory(
        args.directory,
        replacements,
        extensions=extensions,
        dry_run=args.dry_run,
        backup=args.backup,
        recursive=args.recursive
    )


if __name__ == '__main__':
    main()