#!/usr/bin/env python3
"""
Скрипт для конвертации Markdown в HTML
"""
import sys
import os
import re

try:
    import markdown
except ImportError:
    print("Установите библиотеку markdown: pip install markdown")
    sys.exit(1)

def markdown_to_html(md_file, html_file):
    """Конвертирует Markdown файл в HTML"""
    
    # Читаем Markdown файл
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Сохраняем формулы перед конвертацией markdown
    # Заменяем $$...$$ на плейсхолдеры с HTML тегами, которые markdown не будет обрабатывать
    formula_placeholders = {}
    formula_counter = 0
    
    def save_formula(match):
        nonlocal formula_counter
        formula_counter += 1
        placeholder = f'<span class="formula-block-placeholder" data-id="{formula_counter}"></span>'
        formula_placeholders[formula_counter] = ('block', match.group(1).strip())
        return placeholder
    
    # Обрабатываем блочные формулы $$...$$
    md_content = re.sub(r'\$\$(.+?)\$\$', save_formula, md_content, flags=re.DOTALL)
    
    # Обрабатываем инлайн формулы $...$ (но не внутри уже обработанных блоков)
    def save_inline_formula(match):
        nonlocal formula_counter
        formula_counter += 1
        placeholder = f'<span class="formula-inline-placeholder" data-id="{formula_counter}"></span>'
        formula_placeholders[formula_counter] = ('inline', match.group(1).strip())
        return placeholder
    
    md_content = re.sub(r'(?<!\$)\$([^$\n]+?)\$(?!\$)', save_inline_formula, md_content)
    
    # Конвертируем Markdown в HTML
    html = markdown.markdown(md_content, extensions=['extra', 'tables', 'codehilite'])
    
    # Восстанавливаем формулы в формате MathJax
    for formula_id, (formula_type, formula) in formula_placeholders.items():
        if formula_type == 'block':
            replacement = f'\\[{formula}\\]'
        else:
            replacement = f'\\({formula}\\)'
        # Простая замена строки (не regex)
        placeholder = f'<span class="formula-{formula_type}-placeholder" data-id="{formula_id}"></span>'
        html = html.replace(placeholder, replacement)
    
    # Создаем полный HTML документ
    html_template = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TerraSAR-X Staring Spotlight</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script>
        window.MathJax = {{
            tex: {{
                inlineMath: [['\\\\(', '\\\\)']],
                displayMath: [['\\\\[', '\\\\]']],
                processEscapes: true,
                processEnvironments: true
            }},
            options: {{
                ignoreHtmlClass: '.*',
                processHtmlClass: 'arithmatex'
            }}
        }};
    </script>
    <style>
        body {{
            font-family: 'Times New Roman', serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }}
        h1 {{
            font-size: 24px;
            text-align: center;
            margin-top: 30px;
            margin-bottom: 20px;
        }}
        h2 {{
            font-size: 20px;
            margin-top: 25px;
            margin-bottom: 15px;
        }}
        h3 {{
            font-size: 18px;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        h4 {{
            font-size: 16px;
            margin-top: 15px;
            margin-bottom: 8px;
        }}
        p {{
            text-align: justify;
            margin-bottom: 10px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        pre {{
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        .arithmatex {{
            font-size: 1.1em;
        }}
        @media print {{
            body {{
                max-width: 100%;
                padding: 10px;
            }}
        }}
    </style>
</head>
<body class="arithmatex">
{html}
</body>
</html>"""
    
    # Сохраняем HTML
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print(f"HTML файл создан: {html_file}")
    print("Для конвертации в PDF откройте файл в браузере и используйте 'Печать' -> 'Сохранить как PDF'")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Использование: python convert_md_to_html.py <input.md> [output.html]")
        sys.exit(1)
    
    md_file = sys.argv[1]
    if not os.path.exists(md_file):
        print(f"Файл не найден: {md_file}")
        sys.exit(1)
    
    if len(sys.argv) >= 3:
        html_file = sys.argv[2]
    else:
        html_file = os.path.splitext(md_file)[0] + '.html'
    
    markdown_to_html(md_file, html_file)

