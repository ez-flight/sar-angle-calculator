#!/usr/bin/env python3
"""
Универсальный скрипт для создания PDF из Markdown
Пробует несколько методов конвертации
"""
import sys
import os
import subprocess

md_file = "Books/TerraSAR-X Staring Spotlight_ru.md"
html_file = "Books/TerraSAR-X Staring Spotlight_ru.html"
pdf_file = "Books/TerraSAR-X Staring Spotlight_ru.pdf"

print("Попытка конвертации Markdown в PDF...")
print(f"Входной файл: {md_file}")

# Метод 1: Попытка использовать weasyprint
try:
    from weasyprint import HTML
    print("\nМетод 1: Использование weasyprint...")
    if os.path.exists(html_file):
        HTML(html_file).write_pdf(pdf_file)
        if os.path.exists(pdf_file):
            print(f"✓ PDF успешно создан: {pdf_file}")
            sys.exit(0)
except Exception as e:
    print(f"  weasyprint недоступен: {e}")

# Метод 2: Попытка использовать wkhtmltopdf
try:
    print("\nМетод 2: Использование wkhtmltopdf...")
    if os.path.exists(html_file):
        result = subprocess.run(['wkhtmltopdf', html_file, pdf_file], 
                              capture_output=True, text=True)
        if os.path.exists(pdf_file):
            print(f"✓ PDF успешно создан: {pdf_file}")
            sys.exit(0)
        else:
            print(f"  wkhtmltopdf недоступен или ошибка: {result.stderr}")
except FileNotFoundError:
    print("  wkhtmltopdf не установлен")
except Exception as e:
    print(f"  Ошибка: {e}")

# Метод 3: Попытка использовать pandoc
try:
    print("\nМетод 3: Использование pandoc...")
    result = subprocess.run(['pandoc', md_file, '-o', pdf_file, '--pdf-engine=pdflatex'],
                          capture_output=True, text=True)
    if os.path.exists(pdf_file):
        print(f"✓ PDF успешно создан: {pdf_file}")
        sys.exit(0)
    else:
        print(f"  pandoc недоступен или ошибка: {result.stderr}")
except FileNotFoundError:
    print("  pandoc не установлен")
except Exception as e:
    print(f"  Ошибка: {e}")

# Если все методы не сработали
print("\n" + "="*60)
print("Автоматическая конвертация не удалась.")
print("="*60)
print(f"\nHTML файл создан: {html_file}")
print("\nДля создания PDF выполните одно из следующих действий:")
print("\n1. Откройте HTML файл в браузере и используйте:")
print("   Файл -> Печать -> Сохранить как PDF")
print("\n2. Установите один из инструментов:")
print("   - weasyprint: pip install weasyprint")
print("   - wkhtmltopdf: sudo apt-get install wkhtmltopdf")
print("   - pandoc: sudo apt-get install pandoc texlive-latex-base")
print("\n3. Используйте онлайн конвертеры HTML в PDF")


