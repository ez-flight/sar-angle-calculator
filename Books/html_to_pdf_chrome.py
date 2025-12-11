#!/usr/bin/env python3
"""
Конвертация HTML в PDF используя headless Chrome через playwright или selenium
"""
import sys
import os

html_file = "Books/TerraSAR-X Staring Spotlight_ru.html"
pdf_file = "Books/TerraSAR-X Staring Spotlight_ru.pdf"

# Попытка использовать playwright
try:
    from playwright.sync_api import sync_playwright
    print("Использование playwright...")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(f"file://{os.path.abspath(html_file)}")
        page.pdf(path=pdf_file, format='A4')
        browser.close()
    print(f"✓ PDF создан: {pdf_file}")
    sys.exit(0)
except ImportError:
    print("playwright не установлен")
except Exception as e:
    print(f"Ошибка playwright: {e}")

# Попытка использовать selenium
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    print("Использование selenium...")
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    driver = webdriver.Chrome(options=options)
    driver.get(f"file://{os.path.abspath(html_file)}")
    # Selenium не имеет прямого метода для PDF, нужен другой подход
    print("Selenium требует дополнительной настройки")
except ImportError:
    print("selenium не установлен")
except Exception as e:
    print(f"Ошибка selenium: {e}")

print("\nДля создания PDF из HTML:")
print("1. Откройте файл в браузере: " + html_file)
print("2. Используйте Файл -> Печать -> Сохранить как PDF")
print("\nИли установите playwright: pip install playwright && playwright install chromium")


