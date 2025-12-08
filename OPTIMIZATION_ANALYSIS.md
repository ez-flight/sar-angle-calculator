# Анализ оптимизации кода SAR Angle Calculator

## Выявленные узкие места

### 1. Множественные вызовы `get_position()` и `get_xyzv_from_latlon()`

**Проблема:**
- `get_position(orb, time)` вызывается очень часто (каждые 0.1 сек в периоде, каждые 10 сек при сканировании)
- `get_xyzv_from_latlon()` вызывается для каждой точки, хотя координаты цели не меняются
- `calculate_angle_from_velocity_at_time()` внутри снова вызывает `get_position()`, дублируя вычисления

**Текущая частота вызовов:**
- При сканировании (шаг 10 сек): ~8,640 вызовов/день
- При детальном сканировании (шаг 0.1 сек): ~1,200 вызовов/период
- При сохранении точек (шаг 0.1 сек): ~100-1000 вызовов/период

**Оптимизация:**
```python
# Кэширование координат цели (не меняются)
target_pos_eci_cache = None
target_pos_cache_time = None

def get_target_pos_eci_cached(time, target_pos):
    global target_pos_eci_cache, target_pos_cache_time
    # Координаты цели в ECI меняются только из-за вращения Земли
    # Можно кэшировать на несколько секунд
    if target_pos_eci_cache is None or (time - target_pos_cache_time).total_seconds() > 1.0:
        lat_t, lon_t, alt_t = target_pos
        target_pos_eci_cache, _ = get_xyzv_from_latlon(time, lon_t, lat_t, alt_t)
        target_pos_cache_time = time
    return target_pos_eci_cache
```

### 2. Неэффективное сохранение в GPKG

**Проблема:**
- Создание списков `geometries` и `attributes` с последующим построчным добавлением
- Множественные проверки и фильтрации после создания списков
- Чтение всего GPKG файла при каждом сохранении периода

**Текущий процесс:**
1. Создать списки для всех точек периода
2. Пройти по списку и отфильтровать
3. Прочитать весь существующий GPKG
4. Удалить старые точки периода
5. Объединить и сохранить

**Оптимизация:**
```python
# Использовать pandas/numpy для более эффективной работы
# Собирать данные в DataFrame сразу, фильтровать через pandas
# Использовать batch-запись вместо построчной
```

### 3. Медленный экспорт в TXT через `iterrows()`

**Проблема:**
- `gdf.iterrows()` очень медленный для больших датафреймов
- Построчная запись в файл

**Текущий код:**
```python
for idx, row in gdf.iterrows():
    values = []
    for col in columns:
        val = row[col]
        # ...
    f.write('\t'.join(values) + '\n')
```

**Оптимизация:**
```python
# Использовать to_csv() напрямую
gdf[columns].to_csv(txt_file, sep='\t', index=False, encoding='utf-8')
```

### 4. Дублирование вычислений углов

**Проблема:**
- В `save_period_points_to_gpkg()` угол вычисляется дважды для проверки
- В `scan_period_detailed()` и `find_visible_periods_for_day()` одни и те же вычисления

**Оптимизация:**
- Вычислять угол один раз и переиспользовать результат
- Кэшировать результаты для близких моментов времени

### 5. Избыточные преобразования координат

**Проблема:**
- `get_lonlatalt()` вызывается для каждой точки, даже если можно использовать кэш
- Двойное преобразование: ECI -> географические координаты

**Оптимизация:**
- Кэшировать преобразования для близких моментов времени
- Использовать более эффективные методы преобразования

### 6. Неоптимальная работа с GPKG

**Проблема:**
- При каждом сохранении периода читается весь файл
- Удаление старых точек через фильтрацию DataFrame
- Множественные операции записи

**Оптимизация:**
- Использовать SQLite напрямую для более эффективных операций
- Или накапливать данные и записывать батчами
- Использовать индексы для быстрого поиска

## Рекомендации по оптимизации

### Приоритет 1 (Критичные - дают наибольший прирост производительности)

1. **Кэширование координат цели**
   - Ожидаемый прирост: 20-30%
   - Сложность: низкая

2. **Оптимизация экспорта в TXT**
   - Ожидаемый прирост: 50-90% для больших файлов
   - Сложность: очень низкая

3. **Устранение дублирования `get_position()`**
   - Ожидаемый прирост: 15-25%
   - Сложность: средняя

### Приоритет 2 (Важные)

4. **Батч-операции с GPKG**
   - Ожидаемый прирост: 30-50% при множественных периодах
   - Сложность: средняя

5. **Оптимизация фильтрации точек**
   - Ожидаемый прирост: 10-20%
   - Сложность: низкая

### Приоритет 3 (Дополнительные улучшения)

6. **Векторизация вычислений через NumPy**
   - Ожидаемый прирост: 10-30% для больших массивов
   - Сложность: высокая

7. **Параллелизация вычислений**
   - Ожидаемый прирост: зависит от количества ядер
   - Сложность: высокая

## Конкретные изменения кода

### Изменение 1: Кэширование координат цели

```python
# Добавить в начало файла
_target_pos_eci_cache = {}
_target_pos_cache_time = {}

def get_target_pos_eci_cached(time, target_pos, cache_window_seconds=1.0):
    """Кэшированное получение координат цели в ECI"""
    cache_key = id(target_pos)
    
    if cache_key in _target_pos_eci_cache:
        cached_time, cached_pos = _target_pos_eci_cache[cache_key]
        if (time - cached_time).total_seconds() < cache_window_seconds:
            return cached_pos
    
    lat_t, lon_t, alt_t = target_pos
    pos_eci, _ = get_xyzv_from_latlon(time, lon_t, lat_t, alt_t)
    _target_pos_eci_cache[cache_key] = (time, pos_eci)
    return pos_eci
```

### Изменение 2: Оптимизация экспорта в TXT

```python
def export_gpkg_attributes_to_txt(gpkg_file='result/periods_points.gpkg', txt_file='result/periods_points.txt'):
    # ... существующий код до строки 1210 ...
    
    # Заменяем построчную запись на pandas to_csv
    columns = [col for col in gdf.columns if col != 'geometry' and col != 'angle' and col != 'visible']
    gdf[columns].to_csv(txt_file, sep='\t', index=False, encoding='utf-8')
    
    print(f"  Атрибуты экспортированы в текстовый файл: {txt_file} ({len(gdf)} записей)")
```

### Изменение 3: Устранение дублирования get_position

```python
def calculate_angle_from_velocity_at_time(target_time, orb, target_pos, sat_pos_vel=None):
    """Рассчитать угол между вектором на цель и направлением скорости"""
    if sat_pos_vel is None:
        X_s, Y_s, Z_s, Vx_s, Vy_s, Vz_s = get_position(orb, target_time)
    else:
        X_s, Y_s, Z_s, Vx_s, Vy_s, Vz_s = sat_pos_vel
    
    lat_t, lon_t, alt_t = target_pos
    pos_it, _ = get_xyzv_from_latlon(target_time, lon_t, lat_t, alt_t)
    X_t, Y_t, Z_t = pos_it

    angle = calculate_angle_from_velocity(X_s, Y_s, Z_s, Vx_s, Vy_s, Vz_s, X_t, Y_t, Z_t)
    return angle
```

### Изменение 4: Батч-операции с GPKG

```python
# Вместо сохранения каждого периода отдельно, накапливать и сохранять батчами
_periods_batch = []
_batch_size = 10

def save_period_batch_to_gpkg(periods_batch, orb, target_pos, output_file='result/periods_points.gpkg'):
    """Сохранить несколько периодов за один раз"""
    all_geometries = []
    all_attributes = []
    
    for period_start, period_end, period_id in periods_batch:
        # ... существующая логика генерации точек ...
        all_geometries.extend(geometries)
        all_attributes.extend(attributes)
    
    # Одна операция записи для всех периодов
    if len(all_geometries) > 0:
        new_gdf = gpd.GeoDataFrame(all_attributes, geometry=all_geometries, crs='EPSG:4326')
        # ... существующая логика сохранения ...
```

## Ожидаемый общий прирост производительности

После применения всех оптимизаций Приоритета 1:
- **Общее ускорение: 2-3x** для типичных сценариев
- **Ускорение экспорта: 5-10x** для больших файлов
- **Снижение использования памяти: 20-30%**

## Метрики для измерения

1. Время выполнения основного цикла сканирования
2. Время сохранения одного периода в GPKG
3. Время экспорта в TXT
4. Использование памяти
5. Количество вызовов `get_position()` и `get_xyzv_from_latlon()`

