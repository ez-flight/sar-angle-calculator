"""
Расчет непрерывной работы SAR с минимальными промежутками (10 мкс)
Фиксированный угол -2°, остальные рассчитываются
"""

from datetime import datetime, timedelta
from pyorbital.orbital import Orbital
from calc_cord import get_xyzv_from_latlon, get_lonlatalt, EARTH_EQUATORIAL_RADIUS
from read_TBF import read_tle_base_file
import math
import numpy as np
import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt

def get_position(orb: Orbital, utc_time: datetime) -> tuple:
    """Вычисляет положение и скорость спутника"""
    R_s, V_s = orb.get_position(utc_time, False)
    return (*R_s, *V_s)

def calculate_angle_from_velocity_at_time(target_time, orb, target_pos):
    """Рассчитать угол между вектором на цель и направлением скорости в заданный момент времени"""
    X_s, Y_s, Z_s, Vx_s, Vy_s, Vz_s = get_position(orb, target_time)
    lat_t, lon_t, alt_t = target_pos
    pos_it, _ = get_xyzv_from_latlon(target_time, lon_t, lat_t, alt_t)
    X_t, Y_t, Z_t = pos_it

    angle = calculate_angle_from_velocity(X_s, Y_s, Z_s, Vx_s, Vy_s, Vz_s, X_t, Y_t, Z_t)
    return angle

def calculate_traverse_angle_at_time(target_time, orb, target_pos):
    """Рассчитать угол отклонения в заданный момент времени"""
    X_s, Y_s, Z_s, Vx_s, Vy_s, Vz_s = get_position(orb, target_time)
    lat_t, lon_t, alt_t = target_pos
    pos_it, _ = get_xyzv_from_latlon(target_time, lon_t, lat_t, alt_t)
    X_t, Y_t, Z_t = pos_it

    angle = calculate_traverse_angle(X_s, Y_s, Z_s, Vx_s, Vy_s, Vz_s, X_t, Y_t, Z_t)
    return angle

def calculate_angle_from_velocity(X_s, Y_s, Z_s, Vx_s, Vy_s, Vz_s, X_t, Y_t, Z_t):
    """
    Вычисляет угол между вектором на цель и направлением скорости спутника
    
    Угол измеряется от направления скорости:
    - 0° = цель вдоль направления скорости (впереди)
    - 90° = цель перпендикулярно скорости (траверзная плоскость)
    - 180° = цель против направления скорости (сзади)
    
    В диапазоне 88-92° цель находится почти перпендикулярно скорости.
    
    Возвращает угол в градусах (0-180°)
    """
    import numpy as np

    V_s = np.array([Vx_s, Vy_s, Vz_s])
    V_s_norm = np.linalg.norm(V_s)

    if V_s_norm < 1e-3:
        return 90.0  # Если скорость нулевая, считаем 90°

    V_s_unit = V_s / V_s_norm
    R_target = np.array([X_t - X_s, Y_t - Y_s, Z_t - Z_s])
    R_target_norm = np.linalg.norm(R_target)

    if R_target_norm < 1e-3:
        return 90.0  # Если цель в той же точке, считаем 90°

    R_target_unit = R_target / R_target_norm
    
    # Проекция вектора на цель на направление скорости
    projection = np.dot(R_target_unit, V_s_unit)
    projection = np.clip(projection, -1.0, 1.0)

    # Угол между вектором на цель и направлением скорости (0-180°)
    # arccos возвращает 0° для projection=1 (цель впереди)
    #                   90° для projection=0 (цель перпендикулярно)
    #                   180° для projection=-1 (цель сзади)
    # БЕЗ np.abs() чтобы получить полный диапазон 0-180°
    angle_from_velocity_rad = np.arccos(projection)
    angle_deg = np.degrees(angle_from_velocity_rad)
    
    return angle_deg

def calculate_traverse_angle(X_s, Y_s, Z_s, Vx_s, Vy_s, Vz_s, X_t, Y_t, Z_t):
    """
    Вычисляет угол отклонения цели относительно траверзной плоскости КА
    
    Траверзная плоскость - плоскость, перпендикулярная вектору скорости спутника.
    Угол измеряется от траверзной плоскости:
    - 0° = цель в траверзной плоскости (перпендикулярно скорости)
    - Положительный угол = цель впереди по направлению движения
    - Отрицательный угол = цель сзади по направлению движения
    """
    import numpy as np

    V_s = np.array([Vx_s, Vy_s, Vz_s])
    V_s_norm = np.linalg.norm(V_s)

    if V_s_norm < 1e-3:
        return 0.0

    V_s_unit = V_s / V_s_norm
    R_target = np.array([X_t - X_s, Y_t - Y_s, Z_t - Z_s])
    R_target_norm = np.linalg.norm(R_target)

    if R_target_norm < 1e-3:
        return 0.0

    R_target_unit = R_target / R_target_norm
    
    # Проекция вектора на цель на направление скорости
    projection = np.dot(R_target_unit, V_s_unit)
    projection = np.clip(projection, -1.0, 1.0)

    # Угол между вектором на цель и направлением скорости
    angle_from_velocity_rad = np.arccos(np.abs(projection))
    
    # Угол относительно траверзной плоскости = 90° - угол от скорости
    # Если projection > 0, цель впереди (положительный угол)
    # Если projection < 0, цель сзади (отрицательный угол)
    angle_from_traverse_rad = np.pi/2.0 - angle_from_velocity_rad
    
    # Применяем знак в зависимости от направления
    if projection < 0:
        angle_from_traverse_rad = -angle_from_traverse_rad
    
    angle_deg = np.degrees(angle_from_traverse_rad)
    return angle_deg

def find_period_for_angle_at_time(target_time, orb, target_pos, known_periods):
    """
    Найти ближайший период любого угла, который начинается после target_time
    """
    # Ищем период любого угла, который начинается после target_time с минимальной задержкой
    best_period = None
    min_time_diff = None

    for angle in known_periods:
        for period_start, period_end in known_periods[angle]:
            if period_start >= target_time:
                time_diff = (period_start - target_time).total_seconds()
                if min_time_diff is None or time_diff < min_time_diff:
                    min_time_diff = time_diff
                    # Вычисляем реальный угол в момент начала периода
                    calculated_angle = calculate_angle_from_velocity_at_time(period_start, orb, target_pos)
                    best_period = (angle, period_start, period_end, calculated_angle)

    return best_period

def get_satellite_position(orb: Orbital, utc_time: datetime):
    """Получить географические координаты спутника в заданный момент времени"""
    lon, lat, alt = orb.get_lonlatalt(utc_time)
    return lon, lat, alt

def calculate_track_azimuth(orb: Orbital, utc_time: datetime, delta_seconds=1.0):
    """
    Вычислить азимут трассы КА (направление движения по поверхности Земли)
    
    Аргументы:
        orb: объект Orbital
        utc_time: время UTC
        delta_seconds: небольшой интервал времени для расчета направления (секунды)
    
    Возвращает:
        азимут в градусах (0-360°, где 0° = север, 90° = восток)
    """
    # Получаем координаты в двух близких моментах времени
    lon1, lat1, _ = orb.get_lonlatalt(utc_time)
    lon2, lat2, _ = orb.get_lonlatalt(utc_time + timedelta(seconds=delta_seconds))
    
    # Преобразуем в радианы
    lat1_rad = np.deg2rad(lat1)
    lon1_rad = np.deg2rad(lon1)
    lat2_rad = np.deg2rad(lat2)
    lon2_rad = np.deg2rad(lon2)
    
    # Вычисляем разность долгот
    dlon = lon2_rad - lon1_rad
    
    # Формула для расчета азимута (bearing) между двумя точками на сфере
    y = np.sin(dlon) * np.cos(lat2_rad)
    x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
    
    azimuth_rad = np.arctan2(y, x)
    azimuth_deg = np.degrees(azimuth_rad)
    
    # Нормализуем к диапазону 0-360°
    azimuth_deg = (azimuth_deg + 360) % 360
    
    return azimuth_deg

def move_point_by_distance_azimuth(lon, lat, distance_km, azimuth_deg):
    """
    Переместить точку на поверхности Земли на заданное расстояние по заданному азимуту
    
    Аргументы:
        lon: долгота начальной точки (градусы)
        lat: широта начальной точки (градусы)
        distance_km: расстояние в километрах
        azimuth_deg: азимут в градусах (0° = север, 90° = восток)
    
    Возвращает:
        tuple: (новая долгота, новая широта) в градусах
    """
    # Радиус Земли в км
    R = EARTH_EQUATORIAL_RADIUS
    
    # Преобразуем в радианы
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    azimuth_rad = np.deg2rad(azimuth_deg)
    
    # Угловое расстояние
    angular_distance = distance_km / R
    
    # Вычисляем новую широту
    new_lat_rad = np.arcsin(
        np.sin(lat_rad) * np.cos(angular_distance) +
        np.cos(lat_rad) * np.sin(angular_distance) * np.cos(azimuth_rad)
    )
    
    # Вычисляем новую долготу
    new_lon_rad = lon_rad + np.arctan2(
        np.sin(azimuth_rad) * np.sin(angular_distance) * np.cos(lat_rad),
        np.cos(angular_distance) - np.sin(lat_rad) * np.sin(new_lat_rad)
    )
    
    # Преобразуем обратно в градусы
    new_lat = np.degrees(new_lat_rad)
    new_lon = np.degrees(new_lon_rad)
    
    return new_lon, new_lat

def create_oriented_square_20km(center_lon, center_lat, track_azimuth):
    """
    Создать квадратную рамку 20x20 км, ориентированную вдоль трассы КА
    
    Аргументы:
        center_lon: долгота центра квадрата (градусы)
        center_lat: широта центра квадрата (градусы)
        track_azimuth: азимут трассы КА (градусы)
    
    Возвращает:
        Polygon: квадрат 20x20 км, одна сторона параллельна направлению движения КА
    """
    # Размер квадрата: 20 км
    half_size_km = 10.0  # Половина размера квадрата
    
    # Расстояние от центра до угла (диагональ квадрата)
    diagonal_km = half_size_km * np.sqrt(2)
    
    # Вычисляем координаты 4 углов квадрата
    # Углы квадрата относительно направления движения (track_azimuth):
    # - Вперед-вправо: track_azimuth + 45°
    # - Вперед-влево: track_azimuth - 45° (или track_azimuth + 315°)
    # - Назад-влево: track_azimuth + 135°
    # - Назад-вправо: track_azimuth + 225°
    
    # Вычисляем азимуты для каждого угла
    corner_azimuths = [
        (track_azimuth + 45) % 360,    # Вперед-вправо
        (track_azimuth - 45) % 360,    # Вперед-влево (или +315)
        (track_azimuth + 135) % 360,  # Назад-влево
        (track_azimuth + 225) % 360   # Назад-вправо
    ]
    
    # Вычисляем координаты углов
    corners = []
    for azimuth in corner_azimuths:
        corner_lon, corner_lat = move_point_by_distance_azimuth(
            center_lon, center_lat, diagonal_km, azimuth
        )
        corners.append((corner_lon, corner_lat))
    
    # Создаем полигон (порядок углов важен для правильной ориентации)
    # Порядок: вперед-вправо, вперед-влево, назад-влево, назад-вправо
    return Polygon(corners)

def is_target_visible(sat_pos, target_pos, utc_time):
    """
    Проверить, видна ли цель со спутника (не за горизонтом)
    
    Аргументы:
        sat_pos: кортеж (X, Y, Z) позиции спутника в ECI (км)
        target_pos: кортеж (X, Y, Z) позиции цели в ECI (км)
        utc_time: время UTC
    
    Возвращает:
        tuple: (bool видима, float расстояние в км, float максимальное расстояние видимости)
    """
    import numpy as np
    
    # Вектор от спутника к цели
    R_target = np.array([target_pos[0] - sat_pos[0], 
                        target_pos[1] - sat_pos[1], 
                        target_pos[2] - sat_pos[2]])
    distance = np.linalg.norm(R_target)
    
    # Расстояние от центра Земли до спутника
    R_sat = np.linalg.norm(sat_pos)
    
    # Расстояние от центра Земли до цели
    R_tgt = np.linalg.norm(target_pos)
    
    # Проверка видимости: цель видна, если линия спутник-цель не пересекает поверхность Земли
    # Более простой способ: проверить, что расстояние меньше максимального видимого расстояния
    
    # Вектор от центра Земли к спутнику
    vec_sat = np.array(sat_pos)
    # Вектор от центра Земли к цели
    vec_tgt = np.array(target_pos)
    
    R_sat = np.linalg.norm(vec_sat)
    R_tgt = np.linalg.norm(vec_tgt)
    
    # Максимальное расстояние видимости (расстояние до горизонта от спутника + от цели)
    if R_sat > EARTH_EQUATORIAL_RADIUS and R_tgt > EARTH_EQUATORIAL_RADIUS:
        # Расстояние до горизонта от спутника
        horizon_dist_sat = np.sqrt(R_sat**2 - EARTH_EQUATORIAL_RADIUS**2)
        # Расстояние до горизонта от цели
        horizon_dist_tgt = np.sqrt(R_tgt**2 - EARTH_EQUATORIAL_RADIUS**2)
        max_visible_distance = horizon_dist_sat + horizon_dist_tgt
    else:
        max_visible_distance = distance
    
    # Проверяем, что линия спутник-цель не пересекает Землю
    # Вектор направления от спутника к цели
    vec_dir = vec_tgt - vec_sat
    vec_dir_norm = np.linalg.norm(vec_dir)
    
    if vec_dir_norm < 1e-6:
        # Спутник и цель в одной точке
        is_visible = distance < max_visible_distance
        return is_visible, distance, max_visible_distance
    
    vec_dir_unit = vec_dir / vec_dir_norm
    
    # Минимальное расстояние от центра Земли до линии спутник-цель
    # Используем формулу для расстояния от точки до прямой
    # d = |vec_sat × vec_dir_unit| (но проще через проекцию)
    # Ближайшая точка на прямой к центру Земли
    t = -np.dot(vec_sat, vec_dir_unit)
    closest_point_on_line = vec_sat + t * vec_dir_unit
    min_distance_to_earth_center = np.linalg.norm(closest_point_on_line)
    
    # Проверяем, что ближайшая точка находится между спутником и целью
    # и что она дальше от центра, чем радиус Земли
    if t < 0 or t > vec_dir_norm:
        # Ближайшая точка вне отрезка спутник-цель, значит линия не пересекает Землю
        is_visible = distance <= max_visible_distance
    else:
        # Ближайшая точка на отрезке - проверяем расстояние до центра Земли
        is_visible = min_distance_to_earth_center >= EARTH_EQUATORIAL_RADIUS
    
    # Максимальное расстояние видимости (для справки)
    # Это расстояние до горизонта плюс расстояние от горизонта до цели
    if R_sat > EARTH_EQUATORIAL_RADIUS and R_tgt > EARTH_EQUATORIAL_RADIUS:
        max_visible_distance = np.sqrt(R_sat**2 - EARTH_EQUATORIAL_RADIUS**2) + \
                              np.sqrt(R_tgt**2 - EARTH_EQUATORIAL_RADIUS**2)
    else:
        max_visible_distance = distance  # Если одна из точек под поверхностью, используем фактическое расстояние
    
    return is_visible, distance, max_visible_distance

def save_period_points_to_gpkg(period_start, period_end, period_id, orb, target_pos, output_file='result/periods_points.gpkg'):
    """
    Сохранить все точки КА периода в GPKG файл
    Частота определения координат автоматически подстраивается под длительность периода,
    чтобы гарантировать минимум 10 точек на период
    
    Аргументы:
        period_start: datetime - начало периода
        period_end: datetime - конец периода
        period_id: int - ID периода
        orb: объект Orbital
        target_pos: кортеж (широта, долгота, высота) целевой точки
        output_file: путь к выходному GPKG файлу
    """
    # Создаем папку result, если её нет
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Вычисляем длительность периода
    period_duration_seconds = (period_end - period_start).total_seconds()
    
    # Минимальное количество точек на период
    min_points_per_period = 10
    
    # Начальный шаг: пытаемся получить достаточно точек с учетом высокой фильтрации
    # Из опыта видно, что фильтруется ~90-95% точек, поэтому нужен большой запас
    # Начинаем с шага, который даст минимум 300 точек для генерации (запас в 30 раз)
    # Это гарантирует, что даже при высокой фильтрации мы получим минимум 10 валидных точек
    initial_step = period_duration_seconds / (min_points_per_period * 30)  # Запас в 30 раз
    
    # Базовый шаг: не больше 5 секунд (для более частой выборки), не меньше 0.1 секунды
    point_step_seconds = min(5.0, max(0.1, initial_step))
    
    geometries = []
    attributes = []
    
    # Параметры фильтрации по углу
    angle_min = 88.0  # Минимальный угол (градусы)
    angle_max = 92.0  # Максимальный угол (градусы)
    
    current_time = period_start
    point_count = 0
    filtered_by_angle_count = 0  # Счетчик отфильтрованных точек
    valid_points_count = 0  # Счетчик валидных точек
    
    # Генерируем точки до тех пор, пока не наберем минимум валидных точек или не закончится период
    while current_time <= period_end:
        try:
            # Получаем координаты спутника в ECI для расчета расстояния и более точного преобразования
            # Важно: получаем координаты для ТОЧНОГО момента времени current_time
            X_s, Y_s, Z_s, Vx_s, Vy_s, Vz_s = get_position(orb, current_time)
            sat_pos_eci = (X_s, Y_s, Z_s)
            
            # Преобразуем координаты из ECI в географические для большей точности
            # Используем get_lonlatalt для более точного преобразования с учетом времени
            try:
                lon_sat, lat_sat, alt = get_lonlatalt(np.array([X_s, Y_s, Z_s]), current_time)
            except Exception as e:
                # Если не удалось через get_lonlatalt, используем orb.get_lonlatalt
                # Убеждаемся, что передаем правильное время
                lon_sat, lat_sat, alt = orb.get_lonlatalt(current_time)
            
            # Вычисляем угол между вектором на цель и направлением скорости
            angle = calculate_angle_from_velocity_at_time(current_time, orb, target_pos)
            
            # Проверяем, что угол не None и не NaN
            if angle is None or (isinstance(angle, float) and (np.isnan(angle) or np.isinf(angle))):
                filtered_by_angle_count += 1
                current_time += timedelta(seconds=point_step_seconds)
                continue
            
            # Фильтруем точки по диапазону угла (88-92°)
            if angle < angle_min or angle > angle_max:
                filtered_by_angle_count += 1
                current_time += timedelta(seconds=point_step_seconds)
                continue
            
            # Получаем координаты цели в ECI
            lat_t, lon_t, alt_t = target_pos
            pos_tgt_eci, _ = get_xyzv_from_latlon(current_time, lon_t, lat_t, alt_t)
            target_pos_eci = pos_tgt_eci
            
            # Проверяем видимость и расстояние
            is_visible, distance, max_distance = is_target_visible(sat_pos_eci, target_pos_eci, current_time)
            
            # Дополнительная проверка угла перед сохранением (на всякий случай)
            if angle < angle_min or angle > angle_max:
                filtered_by_angle_count += 1
                current_time += timedelta(seconds=point_step_seconds)
                continue
            
            # Создаем точку местоположения КА с высокой точностью
            sat_point = Point(lon_sat, lat_sat)
            geometries.append(sat_point)
            
            # Собираем атрибуты с увеличенной точностью координат
            attrs = {
                'period_id': period_id,
                'point_id': point_count,
                'time': current_time.strftime('%Y-%m-%d %H:%M:%S.%f'),
                'start_time': period_start.strftime('%Y-%m-%d %H:%M:%S.%f'),
                'end_time': period_end.strftime('%Y-%m-%d %H:%M:%S.%f'),
                'sat_lon': round(lon_sat, 9),  # Увеличена точность до 9 знаков после запятой
                'sat_lat': round(lat_sat, 9),  # Увеличена точность до 9 знаков после запятой
                'sat_alt': round(alt, 6),  # Увеличена точность высоты
                'angle_traverse': round(angle, 6),  # Угол между КА и объектом относительно траверса (88-92°)
                'distance': round(distance, 2),
                'max_dist': round(max_distance, 2),
                'visible': 1 if is_visible else 0
            }
            attributes.append(attrs)
            point_count += 1
            valid_points_count += 1
            
        except Exception as e:
            # Пропускаем точку при ошибке
            filtered_by_angle_count += 1
            pass
        
        # Переходим к следующей точке
        current_time += timedelta(seconds=point_step_seconds)
        
        # Адаптивное уменьшение шага: если валидных точек мало, уменьшаем шаг
        # Проверяем каждые 5 итераций, чтобы не делать это слишком часто
        if point_count > 0 and point_count % 5 == 0:
            # Если процент валидных точек меньше 10% и валидных точек меньше минимума
            valid_percentage = (valid_points_count / point_count) * 100 if point_count > 0 else 0
            if valid_percentage < 10.0 and valid_points_count < min_points_per_period and current_time <= period_end:
                # Уменьшаем шаг в 2 раза, но не меньше 0.1 секунды
                if point_step_seconds > 0.1:
                    old_step = point_step_seconds
                    point_step_seconds = max(0.1, point_step_seconds / 2.0)
                    # print(f"    Уменьшен шаг с {old_step:.2f} до {point_step_seconds:.2f} сек (валидных: {valid_points_count}/{point_count}, {valid_percentage:.1f}%)")
    
    # Сохраняем точки в GPKG файл
    if len(geometries) > 0:
        # Дополнительная проверка: фильтруем точки с невалидными углами
        valid_geometries = []
        valid_attributes = []
        
        for i, attr in enumerate(attributes):
            angle_val = attr.get('angle_traverse')
            # Проверяем, что угол валиден и в диапазоне 88-92
            if angle_val is not None and not (isinstance(angle_val, float) and (np.isnan(angle_val) or np.isinf(angle_val))):
                if angle_min <= angle_val <= angle_max:
                    valid_geometries.append(geometries[i])
                    valid_attributes.append(attr)
                else:
                    filtered_by_angle_count += 1
            else:
                filtered_by_angle_count += 1
        
        if len(valid_geometries) > 0:
            new_gdf = gpd.GeoDataFrame(valid_attributes, geometry=valid_geometries, crs='EPSG:4326')
            
            # Проверяем, существует ли уже файл
            if os.path.exists(output_file):
                # Читаем существующий файл и добавляем новые точки
                try:
                    existing_gdf = gpd.read_file(output_file, layer='periods_points')
                    # Дополнительная фильтрация существующих данных (на случай, если там есть невалидные)
                    if 'angle_traverse' in existing_gdf.columns:
                        existing_gdf = existing_gdf[
                            (existing_gdf['angle_traverse'] >= angle_min) & 
                            (existing_gdf['angle_traverse'] <= angle_max) &
                            (existing_gdf['angle_traverse'].notna())
                        ]
                    # Объединяем существующие и новые данные
                    combined_gdf = gpd.GeoDataFrame(pd.concat([existing_gdf, new_gdf], ignore_index=True), crs='EPSG:4326')
                    combined_gdf.to_file(output_file, driver='GPKG', layer='periods_points')
                except Exception as e:
                    # Если не удалось прочитать, создаем новый или перезаписываем
                    new_gdf.to_file(output_file, driver='GPKG', layer='periods_points')
            else:
                # Создаем новый файл
                new_gdf.to_file(output_file, driver='GPKG', layer='periods_points')
        
        # Создаем и сохраняем квадратную рамку 20x20 км для периода
        try:
            # Вычисляем среднее время периода для расчета азимута трассы
            mid_time = period_start + (period_end - period_start) / 2
            
            # Вычисляем азимут трассы КА
            track_azimuth = calculate_track_azimuth(orb, mid_time)
            
            # Получаем координаты целевой точки
            lat_t, lon_t, alt_t = target_pos
            
            # Создаем квадрат 20x20 км, ориентированный вдоль трассы КА
            square_polygon = create_oriented_square_20km(lon_t, lat_t, track_azimuth)
            
            # Создаем GeoDataFrame для квадрата
            square_attrs = {
                'period_id': period_id,
                'type': 'square_frame',
                'size_km': 20.0,
                'center_lon': round(lon_t, 9),
                'center_lat': round(lat_t, 9),
                'track_azimuth': round(track_azimuth, 6),
                'start_time': period_start.strftime('%Y-%m-%d %H:%M:%S.%f'),
                'end_time': period_end.strftime('%Y-%m-%d %H:%M:%S.%f')
            }
            square_gdf = gpd.GeoDataFrame([square_attrs], geometry=[square_polygon], crs='EPSG:4326')
            
            # Сохраняем квадрат в отдельный слой GPKG файла
            if os.path.exists(output_file):
                try:
                    # Пытаемся прочитать существующий слой с квадратами
                    try:
                        existing_squares_gdf = gpd.read_file(output_file, layer='periods_squares')
                        # Фильтруем квадраты текущего периода (если они уже есть)
                        existing_squares_gdf = existing_squares_gdf[existing_squares_gdf['period_id'] != period_id]
                        # Объединяем существующие и новые квадраты
                        combined_squares_gdf = gpd.GeoDataFrame(
                            pd.concat([existing_squares_gdf, square_gdf], ignore_index=True), 
                            crs='EPSG:4326'
                        )
                        combined_squares_gdf.to_file(output_file, driver='GPKG', layer='periods_squares')
                    except:
                        # Если слой не существует, создаем новый
                        square_gdf.to_file(output_file, driver='GPKG', layer='periods_squares')
                except Exception as e:
                    # Если не удалось, создаем новый слой
                    square_gdf.to_file(output_file, driver='GPKG', layer='periods_squares')
            else:
                # Если файл не существует, создаем новый (точки уже создали файл, но на всякий случай)
                square_gdf.to_file(output_file, driver='GPKG', layer='periods_squares')
        except Exception as e:
            print(f"    ⚠️  Предупреждение: не удалось создать квадрат для периода {period_id}: {e}")
        
        # Выводим информацию о сохранении
        period_duration_str = f"{period_duration_seconds:.1f} сек"
        if period_duration_seconds < 60:
            period_duration_str = f"{period_duration_seconds:.1f} сек"
        else:
            period_duration_str = f"{period_duration_seconds/60:.2f} мин"
        
        if filtered_by_angle_count > 0:
            print(f"  Период {period_id} (длительность: {period_duration_str}, шаг: {point_step_seconds:.2f} сек): сохранено {len(valid_geometries)} точек в GPKG (отфильтровано по углу: {filtered_by_angle_count})")
        else:
            print(f"  Период {period_id} (длительность: {period_duration_str}, шаг: {point_step_seconds:.2f} сек): сохранено {len(valid_geometries)} точек в GPKG")
        
        # Предупреждение, если точек меньше минимума
        if len(valid_geometries) < min_points_per_period:
            print(f"    ⚠️  Внимание: сохранено только {len(valid_geometries)} точек, что меньше минимума {min_points_per_period}")

def find_visible_periods_for_day(start_date, orb, target_pos, time_step_seconds=1.0, min_period_duration=30.0, R0_min=None, R0_max=None, num_days=1):
    """
    Найти все периоды видимости цели в течение заданного количества суток
    
    Фильтруются точки с углом отклонения вне диапазона ±2° от траверзной плоскости.
    
    Аргументы:
        start_date: datetime - начало периода
        orb: объект Orbital
        target_pos: кортеж (широта, долгота, высота) целевой точки
        time_step_seconds: шаг времени для сканирования (секунды)
        min_period_duration: минимальная длительность периода (секунды)
        R0_min: минимальное расстояние R0 (наклонная дальность) в км, None = без ограничения
        R0_max: максимальное расстояние R0 (наклонная дальность) в км, None = без ограничения
        num_days: количество суток для расчета (по умолчанию 1)
    
    Возвращает:
        список словарей с периодами видимости
    """
    sequence = []
    end_date = start_date + timedelta(days=num_days)
    
    current_time = start_date
    period_start = None
    period_end = None
    
    if num_days == 1:
        print(f"Сканирование суток: {start_date.strftime('%Y-%m-%d %H:%M:%S')} - {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print(f"Сканирование {num_days} суток: {start_date.strftime('%Y-%m-%d %H:%M:%S')} - {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Шаг сканирования: {time_step_seconds} сек")
    
    checked_count = 0
    visible_count = 0
    filtered_by_R0_count = 0  # Счетчик точек, отфильтрованных по R0
    filtered_by_angle_count = 0  # Счетчик точек, отфильтрованных по углу
    
    # Параметры фильтрации по углу между вектором на цель и направлением скорости
    # Угол измеряется от направления скорости: 0° = вдоль скорости, 90° = перпендикулярно
    angle_min = 88.0  # Минимальный угол (градусы)
    angle_max = 92.0  # Максимальный угол (градусы)
    
    while current_time < end_date:
        checked_count += 1
        
        # Получаем координаты спутника и цели в ECI
        X_s, Y_s, Z_s, _, _, _ = get_position(orb, current_time)
        sat_pos_eci = (X_s, Y_s, Z_s)
        
        lat_t, lon_t, alt_t = target_pos
        pos_tgt_eci, _ = get_xyzv_from_latlon(current_time, lon_t, lat_t, alt_t)
        target_pos_eci = pos_tgt_eci
        
        # Проверяем видимость
        is_visible, distance, max_distance = is_target_visible(sat_pos_eci, target_pos_eci, current_time)
        
        # R0 - это distance (наклонная дальность от спутника до цели)
        R0 = distance
        
        # Фильтруем точки по диапазону R0 (если заданы ограничения)
        if is_visible:
            if R0_min is not None and R0 < R0_min:
                is_visible = False
                filtered_by_R0_count += 1
            elif R0_max is not None and R0 > R0_max:
                is_visible = False
                filtered_by_R0_count += 1
        
        if is_visible:
            # Вычисляем угол между вектором на цель и направлением скорости
            angle = calculate_angle_from_velocity_at_time(current_time, orb, target_pos)
            
            # Фильтруем точки по диапазону угла (88-92°)
            if angle < angle_min or angle > angle_max:
                filtered_by_angle_count += 1
                current_time += timedelta(seconds=time_step_seconds)
                continue
            
            visible_count += 1
            
            if period_start is None:
                # Начало нового периода
                period_start = current_time
                period_angle = angle
            else:
                # Продолжение периода - обновляем угол (средний)
                period_angle = (period_angle + angle) / 2
        else:
            # Конец периода видимости
            if period_start is not None:
                period_end = current_time
                period_duration = (period_end - period_start).total_seconds()
                
                if period_duration >= min_period_duration:
                    # Вычисляем средний угол для периода
                    mid_time = period_start + (period_end - period_start) / 2
                    avg_angle = calculate_angle_from_velocity_at_time(mid_time, orb, target_pos)
                    
                    period_data = {
                        'angle': round(avg_angle, 1),  # Округляем до 0.1 градуса
                        'start_time': period_start,
                        'end_time': period_end,
                        'calculated_angle': avg_angle,
                        'period_duration': period_duration
                    }
                    sequence.append(period_data)
                    
                    # Сохраняем все точки периода в GPKG с частотой 10 секунд
                    try:
                        save_period_points_to_gpkg(period_start, period_end, len(sequence), orb, target_pos)
                    except Exception as e:
                        print(f"  Предупреждение: не удалось сохранить точки периода {len(sequence)} в GPKG: {e}")
                
                period_start = None
                period_end = None
        
        # Переходим к следующему моменту времени
        current_time += timedelta(seconds=time_step_seconds)
        
        # Прогресс каждые 1000 проверок
        if checked_count % 1000 == 0:
            print(f"  Проверено: {checked_count} точек, найдено видимых: {visible_count}, периодов: {len(sequence)}")
    
    # Обработка последнего периода, если он не закончился
    if period_start is not None:
        period_end = current_time
        period_duration = (period_end - period_start).total_seconds()
        
        if period_duration >= min_period_duration:
            mid_time = period_start + (period_end - period_start) / 2
            avg_angle = calculate_angle_from_velocity_at_time(mid_time, orb, target_pos)
            
            period_data = {
                'angle': round(avg_angle, 1),
                'start_time': period_start,
                'end_time': period_end,
                'calculated_angle': avg_angle,
                'period_duration': period_duration
            }
            sequence.append(period_data)
            
            # Сохраняем все точки периода в GPKG с частотой 10 секунд
            try:
                save_period_points_to_gpkg(period_start, period_end, len(sequence), orb, target_pos)
            except Exception as e:
                print(f"  Предупреждение: не удалось сохранить точки периода {len(sequence)} в GPKG: {e}")
    
    print(f"\nСканирование завершено:")
    print(f"  Всего проверено точек: {checked_count}")
    print(f"  Найдено видимых точек: {visible_count}")
    if filtered_by_R0_count > 0:
        print(f"  Отфильтровано по R0: {filtered_by_R0_count}")
    if filtered_by_angle_count > 0:
        print(f"  Отфильтровано по углу (вне диапазона {angle_min}° - {angle_max}°): {filtered_by_angle_count}")
    print(f"  Найдено периодов видимости: {len(sequence)}")
    
    return sequence

def continuous_angle_sequence():
    """
    Расчет непрерывной последовательности углов на заданное количество суток с фильтрацией по видимости
    """
    print("=" * 80)
    print("РАСЧЕТ НЕПРЕРЫВНОЙ ПОСЛЕДОВАТЕЛЬНОСТИ УГЛОВ SAR")
    print("Фильтрация по видимости: обязательная")
    print("=" * 80)

    # Загрузка данных
    s_name, tle_1, tle_2 = read_tle_base_file(56756)
    target_pos = (59.95, 30.316667, 12)
    orb = Orbital("N", line1=tle_1, line2=tle_2)

    # Начало периода для расчета
    start_date = datetime(2024, 3, 22, 0, 0, 0)
    
    # Количество суток для расчета
    num_days = 16
    
    # Параметры фильтрации по R0 (наклонная дальность)
    R0_min = 561  # Минимальное расстояние R0 в км
    R0_max = 964  # Максимальное расстояние R0 в км
    
    end_date = start_date + timedelta(days=num_days)
    
    print(f"Загружен спутник: {s_name}")
    print(f"Целевая точка: Широта {target_pos[0]}°, Долгота {target_pos[1]}°, Высота {target_pos[2]} км")
    print(f"Период расчета: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')} ({num_days} суток)")
    print(f"Фильтрация по R0: {R0_min} - {R0_max} км")
    print(f"Фильтрация по углу: 88° - 92° (угол между вектором на цель и направлением скорости)")

    # Находим все периоды видимости за указанный период
    sequence = find_visible_periods_for_day(
        start_date=start_date,
        orb=orb,
        target_pos=target_pos,
        time_step_seconds=10.0,  # Шаг 10 секунд для ускорения
        min_period_duration=30.0,  # Минимальная длительность периода 30 секунд
        R0_min=R0_min,
        R0_max=R0_max,
        num_days=num_days
    )

    if len(sequence) == 0:
        print("\n" + "=" * 80)
        print("ПЕРИОДЫ НЕ НАЙДЕНЫ")
        print("=" * 80)
        print(f"За {num_days} суток не найдено периодов видимости цели.")
        print("Возможные причины:")
        print("  - Спутник не пролетает над целевой точкой в этот день")
        print("  - Цель всегда находится за горизонтом")
        print("=" * 80)
        return

    # Вывод итоговой последовательности
    print("\n" + "=" * 80)
    print("ИТОГОВАЯ ПОСЛЕДОВАТЕЛЬНОСТЬ")
    print("=" * 80)

    total_duration = 0
    total_gaps = 0

    for i, item in enumerate(sequence, 1):
        print(f"\n{i}. Период: {item['start_time'].strftime('%Y-%m-%d %H:%M:%S')} - {item['end_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Длительность: {item['period_duration']:.3f} сек ({item['period_duration']/60:.2f} мин)")
        
        # Получаем координаты КА в начале и конце периода
        try:
            lon_start, lat_start, alt_start = orb.get_lonlatalt(item['start_time'])
            lon_end, lat_end, alt_end = orb.get_lonlatalt(item['end_time'])
            print(f"   КА в начале периода: Широта {lat_start:.6f}°, Долгота {lon_start:.6f}°, Высота {alt_start:.3f} км")
            print(f"   КА в конце периода: Широта {lat_end:.6f}°, Долгота {lon_end:.6f}°, Высота {alt_end:.3f} км")
        except Exception as e:
            # Если не удалось получить координаты через get_lonlatalt, используем get_position
            X_s_start, Y_s_start, Z_s_start, _, _, _ = get_position(orb, item['start_time'])
            X_s_end, Y_s_end, Z_s_end, _, _, _ = get_position(orb, item['end_time'])
            lon_start, lat_start, alt_start = get_lonlatalt(np.array([X_s_start, Y_s_start, Z_s_start]), item['start_time'])
            lon_end, lat_end, alt_end = get_lonlatalt(np.array([X_s_end, Y_s_end, Z_s_end]), item['end_time'])
            print(f"   КА в начале периода: Широта {lat_start:.6f}°, Долгота {lon_start:.6f}°, Высота {alt_start:.3f} км")
            print(f"   КА в конце периода: Широта {lat_end:.6f}°, Долгота {lon_end:.6f}°, Высота {alt_end:.3f} км")

        total_duration += item['period_duration']
        
        # Вычисляем промежуток до следующего периода
        if i < len(sequence):
            gap = (sequence[i]['start_time'] - item['end_time']).total_seconds()
            total_gaps += gap
            
            # Вычисляем пройденное расстояние и скорость КА за промежуток
            try:
                # Получаем координаты и скорость КА в конце текущего периода
                X_s_end, Y_s_end, Z_s_end, Vx_s_end, Vy_s_end, Vz_s_end = get_position(orb, item['end_time'])
                
                # Получаем координаты и скорость КА в начале следующего периода
                X_s_start, Y_s_start, Z_s_start, Vx_s_start, Vy_s_start, Vz_s_start = get_position(orb, sequence[i]['start_time'])
                
                # Вычисляем среднюю орбитальную скорость (используем среднее значение скоростей)
                V_end = np.array([Vx_s_end, Vy_s_end, Vz_s_end])
                V_start = np.array([Vx_s_start, Vy_s_start, Vz_s_start])
                
                # Средняя скорость (модуль вектора скорости)
                velocity_end = np.linalg.norm(V_end)
                velocity_start = np.linalg.norm(V_start)
                avg_velocity_km_s = (velocity_end + velocity_start) / 2.0
                
                # Вычисляем пройденное расстояние как произведение средней скорости на время
                if gap > 0:
                    distance_km = avg_velocity_km_s * gap
                    avg_velocity_km_h = avg_velocity_km_s * 3600
                else:
                    distance_km = 0.0
                    avg_velocity_km_s = 0.0
                    avg_velocity_km_h = 0.0
                
                print(f"   Промежуток до следующего периода: {gap:.1f} сек ({gap/60:.2f} мин)")
                print(f"   Пройденное расстояние КА: {distance_km:.2f} км")
                print(f"   Средняя скорость КА: {avg_velocity_km_s:.3f} км/с ({avg_velocity_km_h:.1f} км/ч)")
            except Exception as e:
                print(f"   Промежуток до следующего периода: {gap:.1f} сек ({gap/60:.2f} мин)")
                print(f"   (Не удалось вычислить расстояние и скорость: {e})")

    print("\nОБЩАЯ СТАТИСТИКА:")
    print(f"  Всего периодов видимости: {len(sequence)}")
    print(f"  Общая длительность съемки: {total_duration:.3f} сек ({total_duration/60:.2f} мин, {total_duration/3600:.2f} часов)")
    
    if len(sequence) > 1:
        print(f"  Общая длительность промежутков: {total_gaps:.1f} сек ({total_gaps/60:.2f} мин)")
        print(f"  Средний промежуток: {total_gaps/(len(sequence)-1):.1f} сек ({total_gaps/(len(sequence)-1)/60:.2f} мин)")
        total_time = total_duration + total_gaps
        efficiency = total_duration / total_time * 100 if total_time > 0 else 0
        print(f"  Эффективность использования времени: {efficiency:.1f}%")
    
    # Построение гистограммы продолжительности периодов обзора
    if len(sequence) > 0:
        try:
            # Собираем данные о продолжительности периодов
            durations = [item['period_duration'] for item in sequence]
            durations_minutes = [d / 60.0 for d in durations]  # Переводим в минуты для удобства
            
            # Вычисляем статистику
            avg_duration = np.mean(durations_minutes)
            median_duration = np.median(durations_minutes)
            min_duration = np.min(durations_minutes)
            max_duration = np.max(durations_minutes)
            std_duration = np.std(durations_minutes)
            
            print(f"\nСТАТИСТИКА ПО ПРОДОЛЖИТЕЛЬНОСТИ ПЕРИОДОВ:")
            print(f"  Средняя продолжительность: {avg_duration:.2f} мин ({avg_duration*60:.1f} сек)")
            print(f"  Медианная продолжительность: {median_duration:.2f} мин ({median_duration*60:.1f} сек)")
            print(f"  Минимальная продолжительность: {min_duration:.2f} мин ({min_duration*60:.1f} сек)")
            print(f"  Максимальная продолжительность: {max_duration:.2f} мин ({max_duration*60:.1f} сек)")
            print(f"  Стандартное отклонение: {std_duration:.2f} мин ({std_duration*60:.1f} сек)")
            
            # Создаем папку result, если её нет
            result_dir = 'result'
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            
            # Построение гистограммы
            plt.figure(figsize=(12, 6))
            
            # Определяем количество интервалов (bins) для гистограммы
            n_bins = min(30, max(10, len(durations) // 2))  # От 10 до 30 интервалов
            
            # Строим гистограмму
            n, bins, patches = plt.hist(durations_minutes, bins=n_bins, edgecolor='black', alpha=0.7, color='steelblue')
            
            # Добавляем вертикальную линию для среднего значения
            plt.axvline(avg_duration, color='red', linestyle='--', linewidth=2, label=f'Среднее: {avg_duration:.2f} мин')
            
            # Добавляем вертикальную линию для медианы
            plt.axvline(median_duration, color='green', linestyle='--', linewidth=2, label=f'Медиана: {median_duration:.2f} мин')
            
            # Настройка графика
            plt.xlabel('Продолжительность периода обзора (минуты)', fontsize=12)
            plt.ylabel('Количество периодов', fontsize=12)
            plt.title(f'Гистограмма продолжительности периодов обзора\nВсего периодов: {len(sequence)}', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.legend(fontsize=10)
            
            # Добавляем текстовую информацию о статистике
            stats_text = f'Среднее: {avg_duration:.2f} мин\nМедиана: {median_duration:.2f} мин\nМин: {min_duration:.2f} мин\nМакс: {max_duration:.2f} мин\nСт.откл.: {std_duration:.2f} мин'
            plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes,
                    fontsize=9, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            
            # Сохраняем гистограмму
            histogram_path = os.path.join(result_dir, 'duration_histogram.png')
            plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
            print(f"\nГистограмма сохранена: {histogram_path}")
            
            # Закрываем фигуру для освобождения памяти
            plt.close()
            
        except Exception as e:
            print(f"\nПредупреждение: не удалось построить гистограмму: {e}")
    
    print("=" * 80)

if __name__ == "__main__":
    continuous_angle_sequence()
