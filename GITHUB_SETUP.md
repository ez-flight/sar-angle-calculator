# Инструкция по размещению проекта на GitHub

## Шаг 1: Создание репозитория на GitHub

1. Войдите в свой аккаунт GitHub
2. Нажмите кнопку "+" в правом верхнем углу и выберите "New repository"
3. Заполните форму:
   - **Repository name**: `sar-angle-calculator` (или другое название)
   - **Description**: "Расчет оптимальных периодов наблюдения для SAR-съемки с учетом геометрии орбиты и угловых ограничений"
   - **Visibility**: Public или Private (на ваше усмотрение)
   - **НЕ** добавляйте README, .gitignore или лицензию (они уже есть в проекте)
4. Нажмите "Create repository"

## Шаг 2: Подключение локального репозитория к GitHub

После создания репозитория GitHub покажет инструкции. Выполните следующие команды:

```bash
cd /home/ez/Dev/sar-angle-calculator

# Добавьте удаленный репозиторий (замените YOUR_USERNAME на ваш GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/sar-angle-calculator.git

# Или если используете SSH:
# git remote add origin git@github.com:YOUR_USERNAME/sar-angle-calculator.git

# Переименуйте ветку в main (если еще не сделано)
git branch -M main

# Загрузите код на GitHub
git push -u origin main
```

## Шаг 3: Проверка

Откройте ваш репозиторий на GitHub в браузере и убедитесь, что все файлы загружены.

## Дополнительные настройки (опционально)

### Добавление описания и тегов

На странице репозитория можно:
- Добавить описание проекта
- Добавить теги (topics): `sar`, `satellite`, `remote-sensing`, `orbital-mechanics`, `python`, `radar`
- Добавить лицензию (например, MIT или Apache 2.0)

### Настройка GitHub Pages (для документации)

Если хотите опубликовать документацию:
1. Перейдите в Settings → Pages
2. Выберите источник: "Deploy from a branch"
3. Выберите ветку `main` и папку `/docs` (или корневую)
4. Сохраните

## Структура проекта

Проект содержит:
- `README.md` - полная академическая статья и документация
- `continuous_angle_calc.py` - основной скрипт расчета
- `calc_cord.py` - преобразование координат
- `read_TBF.py` - загрузка TLE данных
- `requirements.txt` - зависимости Python
- `academic_paper.md` - академическая статья
- `USAGE.md` - инструкция по использованию

## Важные замечания

⚠️ **Не загружайте файл `.env`** - он содержит учетные данные для Space-Track API и должен оставаться локальным.

⚠️ **Папка `result/`** исключена из репозитория через `.gitignore`, так как содержит результаты расчетов.

