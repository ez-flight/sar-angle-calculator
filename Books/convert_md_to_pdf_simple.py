#!/usr/bin/env python3
"""
Простой скрипт для конвертации Markdown в PDF используя markdown и fpdf2
"""
import sys
import os
import re

try:
    import markdown
    try:
        from fpdf import FPDF
    except ImportError:
        from fpdf2 import FPDF
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Установите необходимые библиотеки:")
    print("pip install markdown fpdf2")
    sys.exit(1)

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, '', 0, 1, 'C')
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Страница {self.page_no()}', 0, 0, 'C')

def markdown_to_pdf(md_file, pdf_file):
    """Конвертирует Markdown файл в PDF"""
    
    # Читаем Markdown файл
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Создаем PDF
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font('Arial', '', 11)
    
    # Парсим Markdown
    lines = md_content.split('\n')
    in_code_block = False
    code_block_lines = []
    
    for line in lines:
        # Обработка кодовых блоков
        if line.strip().startswith('```'):
            if in_code_block:
                # Конец кодового блока
                if code_block_lines:
                    pdf.set_font('Courier', '', 9)
                    for code_line in code_block_lines:
                        pdf.cell(0, 5, code_line, 0, 1)
                    pdf.set_font('Arial', '', 11)
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
            pdf.ln(3)
            continue
        
        # Заголовки
        if line.startswith('# '):
            text = line[2:].strip()
            pdf.set_font('Arial', 'B', 18)
            pdf.ln(5)
            pdf.cell(0, 10, text, 0, 1)
            pdf.ln(3)
            pdf.set_font('Arial', '', 11)
        elif line.startswith('## '):
            text = line[3:].strip()
            pdf.set_font('Arial', 'B', 16)
            pdf.ln(5)
            pdf.cell(0, 8, text, 0, 1)
            pdf.ln(2)
            pdf.set_font('Arial', '', 11)
        elif line.startswith('### '):
            text = line[4:].strip()
            pdf.set_font('Arial', 'B', 14)
            pdf.ln(4)
            pdf.cell(0, 7, text, 0, 1)
            pdf.ln(2)
            pdf.set_font('Arial', '', 11)
        elif line.startswith('#### '):
            text = line[5:].strip()
            pdf.set_font('Arial', 'B', 12)
            pdf.ln(3)
            pdf.cell(0, 6, text, 0, 1)
            pdf.ln(1)
            pdf.set_font('Arial', '', 11)
        # Таблицы
        elif '|' in line and line.strip().startswith('|'):
            if '---' in line or '===' in line:
                continue
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            table_text = ' | '.join(cells)
            pdf.set_font('Arial', '', 10)
            pdf.cell(0, 5, table_text, 0, 1)
            pdf.set_font('Arial', '', 11)
        # Обычный текст
        else:
            text = line.strip()
            # Удаляем markdown форматирование для простоты
            text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # жирный
            text = re.sub(r'\*(.+?)\*', r'\1', text)  # курсив
            text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # ссылки
            text = re.sub(r'`([^`]+)`', r'\1', text)  # код
            
            if text:
                # Разбиваем длинные строки
                pdf.multi_cell(0, 5, text)
                pdf.ln(1)
    
    # Сохраняем PDF
    pdf.output(pdf_file)
    print(f"PDF файл создан: {pdf_file}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Использование: python convert_md_to_pdf_simple.py <input.md> [output.pdf]")
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

