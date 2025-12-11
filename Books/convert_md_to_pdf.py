#!/usr/bin/env python3
"""
Скрипт для конвертации Markdown в PDF
"""
import sys
import os
from pathlib import Path

try:
    import markdown
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import re
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Установите необходимые библиотеки:")
    print("pip install markdown reportlab")
    sys.exit(1)

def markdown_to_pdf(md_file, pdf_file):
    """Конвертирует Markdown файл в PDF"""
    
    # Читаем Markdown файл
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Конвертируем Markdown в HTML
    html = markdown.markdown(md_content, extensions=['extra', 'tables', 'codehilite'])
    
    # Создаем PDF документ
    doc = SimpleDocTemplate(
        pdf_file,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    # Стили
    styles = getSampleStyleSheet()
    
    # Создаем кастомные стили
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor='black',
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=16,
        textColor='black',
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor='black',
        spaceAfter=10,
        spaceBefore=10,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        textColor='black',
        spaceAfter=6,
        alignment=TA_JUSTIFY,
        fontName='Helvetica'
    )
    
    # Парсим HTML и создаем элементы PDF
    story = []
    
    # Простой парсер для основных элементов
    lines = md_content.split('\n')
    in_code_block = False
    code_block_lines = []
    
    for line in lines:
        # Обработка кодовых блоков
        if line.strip().startswith('```'):
            if in_code_block:
                # Конец кодового блока
                if code_block_lines:
                    code_text = '\n'.join(code_block_lines)
                    story.append(Paragraph(f'<font face="Courier" size="9">{code_text}</font>', normal_style))
                    code_block_lines = []
                in_code_block = False
            else:
                in_code_block = True
            continue
        
        if in_code_block:
            code_block_lines.append(line)
            continue
        
        # Пропускаем пустые строки
        if not line.strip():
            story.append(Spacer(1, 6))
            continue
        
        # Заголовки
        if line.startswith('# '):
            text = line[2:].strip()
            story.append(Paragraph(text, title_style))
            story.append(Spacer(1, 12))
        elif line.startswith('## '):
            text = line[3:].strip()
            story.append(Paragraph(text, heading1_style))
            story.append(Spacer(1, 10))
        elif line.startswith('### '):
            text = line[4:].strip()
            story.append(Paragraph(text, heading2_style))
            story.append(Spacer(1, 8))
        elif line.startswith('#### '):
            text = line[5:].strip()
            story.append(Paragraph(f'<b>{text}</b>', normal_style))
            story.append(Spacer(1, 6))
        # Таблицы (простая обработка)
        elif '|' in line and line.strip().startswith('|'):
            # Пропускаем разделители таблиц
            if '---' in line or '===' in line:
                continue
            # Форматируем строку таблицы
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            table_text = ' | '.join(cells)
            story.append(Paragraph(f'<font size="10">{table_text}</font>', normal_style))
        # Обычный текст
        else:
            # Экранируем HTML символы
            text = line.strip()
            text = text.replace('&', '&amp;')
            text = text.replace('<', '&lt;')
            text = text.replace('>', '&gt;')
            # Обработка жирного текста
            text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
            # Обработка курсива
            text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
            # Обработка ссылок
            text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'<u>\1</u>', text)
            
            if text:
                story.append(Paragraph(text, normal_style))
    
    # Строим PDF
    doc.build(story)
    print(f"PDF файл создан: {pdf_file}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Использование: python convert_md_to_pdf.py <input.md> [output.pdf]")
        sys.exit(1)
    
    md_file = sys.argv[1]
    if not os.path.exists(md_file):
        print(f"Файл не найден: {md_file}")
        sys.exit(1)
    
    if len(sys.argv) >= 3:
        pdf_file = sys.argv[2]
    else:
        pdf_file = os.path.splitext(md_file)[0] + '.pdf'
    
    markdown_to_pdf(md_file, pdf_file)


