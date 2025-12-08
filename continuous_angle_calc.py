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

def calculate_spotlight_images_count(period_duration_seconds, 
                                      image_acquisition_time=10.0, 
                                      antenna_switch_time=2.0):
    """
    Рассчитывает количество снимков в детальном прожекторном режиме (ДПР) за период наблюдения
    
    Согласно руководству пользователя Кондор-ФКА, в детальном прожекторном режиме:
    - Время синтеза апертуры для одного кадра: ~10 секунд
    - Время переключения антенны между кадрами: ~2 секунды
    - Общий цикл на кадр: ~12 секунд
    
    Аргументы:
        period_duration_seconds: длительность периода наблюдения в секундах
        image_acquisition_time: время синтеза апертуры для одного кадра (секунды), по умолчанию 10.0
        antenna_switch_time: время переключения антенны между кадрами (секунды), по умолчанию 2.0
    
    Возвращает:
        кортеж (количество_снимков, общее_время_циклов, остаточное_время)
    """
    cycle_time = image_acquisition_time + antenna_switch_time
    
    if period_duration_seconds < image_acquisition_time:
        # Если период короче времени одного кадра, можно сделать только 0 снимков
        return 0, 0.0, period_duration_seconds
    
    # Количество полных циклов (каждый цикл = съемка + переключение)
    full_cycles = int(period_duration_seconds / cycle_time)
    
    # Остаточное время после полных циклов
    remaining_time = period_duration_seconds - (full_cycles * cycle_time)
    
    # Если остаточного времени достаточно для еще одного кадра (без переключения в конце)
    if remaining_time >= image_acquisition_time:
        images_count = full_cycles + 1
        total_cycle_time = full_cycles * cycle_time + image_acquisition_time
        residual_time = remaining_time - image_acquisition_time
    else:
        images_count = full_cycles
        total_cycle_time = full_cycles * cycle_time
        residual_time = remaining_time
    
    return images_count, total_cycle_time, residual_time

def calculate_doppler_frequency(X_s, Y_s, Z_s, Vx_s, Vy_s, Vz_s, X_t, Y_t, Z_t, wavelength=0.10):
    """
    Вычисляет доплеровскую частоту принимаемого сигнала согласно статье Зайцева В.В.
    
    Формула: f_D = -2/λ * dR_0/dt = -2/λ * (V_s · R_target) / R_0
    
    где:
    - λ - длина волны РСА (м)
    - V_s - вектор скорости космического аппарата
    - R_target - вектор от космического аппарата к цели
    - R_0 - наклонная дальность
    
    Аргументы:
        X_s, Y_s, Z_s: координаты космического аппарата в ГИСК (м)
        Vx_s, Vy_s, Vz_s: компоненты скорости космического аппарата в ГИСК (м/с)
        X_t, Y_t, Z_t: координаты цели в ГИСК (м)
        wavelength: длина волны РСА в метрах (по умолчанию 0.031 м для X-диапазона)
    
    Возвращает:
        доплеровская частота в Гц
    """
    # Вектор от КА к цели
    R_target = np.array([X_t - X_s, Y_t - Y_s, Z_t - Z_s])
    
    # Наклонная дальность
    R_0 = np.linalg.norm(R_target)
    
    if R_0 == 0:
        return 0.0
    
    # Вектор скорости КА
    V_s = np.array([Vx_s, Vy_s, Vz_s])
    
    # Скалярное произведение: V_s · R_target
    V_dot_R = np.dot(V_s, R_target)
    
    # Доплеровская частота: f_D = -2/λ * (V_s · R_target) / R_0
    f_D = -2.0 / wavelength * V_dot_R / R_0
    
    return f_D

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

def create_oriented_square_10km(center_lon, center_lat, track_azimuth):
    """
    Создать квадратную рамку 10x10 км для детального прожекторного режима (ДПР),
    ориентированную вдоль трассы КА согласно руководству пользователя Кондор-ФКА
    
    Аргументы:
        center_lon: долгота центра квадрата (градусы)
        center_lat: широта центра квадрата (градусы)
        track_azimuth: азимут трассы КА (градусы) - направление движения
    
    Возвращает:
        Polygon: квадрат 10x10 км, одна сторона параллельна направлению движения КА
    """
    # Размер квадрата: 10 км (согласно руководству пользователя Кондор-ФКА для детального прожекторного режима)
    half_size_km = 5.0  # Половина стороны квадрата (10 км / 2 = 5 км)
    
    # Для создания квадрата, где одна сторона параллельна трассе:
    # Используем диагональный подход, но с правильными углами
    # Расстояние от центра до угла = половина диагонали квадрата
    diagonal_half_km = half_size_km * np.sqrt(2)
    
    # Углы квадрата относительно направления движения (track_azimuth):
    # Для квадрата со стороной, параллельной трассе:
    # - Вперед-вправо: track_azimuth + 45° (диагональ вперед-вправо)
    # - Вперед-влево: track_azimuth - 45° (диагональ вперед-влево)
    # - Назад-влево: track_azimuth + 135° (диагональ назад-влево)
    # - Назад-вправо: track_azimuth + 225° (диагональ назад-вправо)
    
    # Вычисляем координаты 4 углов квадрата
    # Для квадрата со стороной, параллельной трассе, углы находятся на диагоналях
    # Порядок углов по часовой стрелке (для правильного полигона):
    # 1. Вперед-вправо (северо-восток, если track_azimuth = 0°)
    # 2. Назад-вправо (юго-восток)
    # 3. Назад-влево (юго-запад)
    # 4. Вперед-влево (северо-запад)
    
    corner1_lon, corner1_lat = move_point_by_distance_azimuth(
        center_lon, center_lat, diagonal_half_km, (track_azimuth + 45) % 360
    )
    
    corner2_lon, corner2_lat = move_point_by_distance_azimuth(
        center_lon, center_lat, diagonal_half_km, (track_azimuth + 135) % 360
    )
    
    corner3_lon, corner3_lat = move_point_by_distance_azimuth(
        center_lon, center_lat, diagonal_half_km, (track_azimuth + 225) % 360
    )
    
    corner4_lon, corner4_lat = move_point_by_distance_azimuth(
        center_lon, center_lat, diagonal_half_km, (track_azimuth - 45) % 360
    )
    
    # Создаем полигон в правильном порядке (по часовой стрелке)
    # Порядок: вперед-вправо, назад-вправо, назад-влево, вперед-влево
    corners = [
        (corner1_lon, corner1_lat),  # Вперед-вправо
        (corner2_lon, corner2_lat),  # Назад-вправо
        (corner3_lon, corner3_lat),  # Назад-влево
        (corner4_lon, corner4_lat)   # Вперед-влево
    ]
    
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
    Частота определения координат: фиксированный шаг 0.1 секунды внутри периода (10 точек в секунду)
    
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
    
    # Расчет количества снимков в детальном прожекторном режиме (ДПР)
    images_count, total_cycle_time, residual_time = calculate_spotlight_images_count(
        period_duration_seconds,
        image_acquisition_time=10.0,  # Время синтеза апертуры для одного кадра (сек)
        antenna_switch_time=2.0       # Время переключения антенны (сек)
    )
    
    # Параметры цикла съемки
    image_acquisition_time = 10.0  # Время синтеза апертуры для одного кадра (сек)
    antenna_switch_time = 2.0      # Время переключения антенны (сек)
    cycle_time = image_acquisition_time + antenna_switch_time  # 12 секунд
    
    # Фиксированный шаг 0.1 секунды внутри периода (10 точек в секунду)
    point_step_seconds = 0.1
    
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
            
            # Вычисляем доплеровскую частоту согласно статье Зайцева В.В.
            # Используем длину волны X-диапазона (0.031 м) по умолчанию
            # Для других диапазонов: C-диапазон ~0.055 м, L-диапазон ~0.24 м
            doppler_freq = calculate_doppler_frequency(
                X_s, Y_s, Z_s, Vx_s, Vy_s, Vz_s,
                pos_tgt_eci[0], pos_tgt_eci[1], pos_tgt_eci[2],
                wavelength=0.031  # X-диапазон (можно сделать параметром)
            )
            
            # Определяем, какому снимку соответствует эта точка
            # Явное присвоение номера снимка на основе временных интервалов
            image_number = None
            
            if images_count > 0:
                # Проверяем каждый снимок в порядке от первого к последнему
                # Используем ту же логику вычисления временных интервалов, что и при выводе
                for img_num in range(1, images_count + 1):
                    # Вычисляем время начала снимка абсолютное (datetime)
                    # Снимок 1 начинается в period_start
                    # Снимок 2 начинается через cycle_time после начала периода
                    # Снимок N начинается через (N-1) * cycle_time после начала периода
                    image_start_datetime = period_start + timedelta(seconds=(img_num - 1) * cycle_time)
                    
                    # Время окончания снимка (начало + время съемки)
                    image_end_datetime = image_start_datetime + timedelta(seconds=image_acquisition_time)
                    
                    # Явная проверка: попадает ли текущая точка во временной интервал этого снимка
                    # Интервал снимка: [image_start_datetime, image_end_datetime)
                    # Включаем момент начала, НЕ включаем момент окончания (момент окончания - это начало переключения)
                    if image_start_datetime <= current_time < image_end_datetime:
                        image_number = img_num
                        break  # Найден соответствующий снимок, прекращаем поиск
                
                # Если image_number остался None, значит точка попадает в интервал переключения антенны
                # Такие точки не сохраняются (фильтруются позже)
            
            # Создаем точку местоположения КА с высокой точностью
            sat_point = Point(lon_sat, lat_sat)
            geometries.append(sat_point)
            
            # Собираем атрибуты с увеличенной точностью координат
            attrs = {
                'period_id': period_id,
                'point_id': point_count,
                'image_number': image_number,  # Номер снимка, которому соответствует точка (None если время переключения)
                'time': current_time.strftime('%Y-%m-%d %H:%M:%S.%f'),
                'start_time': period_start.strftime('%Y-%m-%d %H:%M:%S.%f'),
                'end_time': period_end.strftime('%Y-%m-%d %H:%M:%S.%f'),
                'sat_lon': round(lon_sat, 9),  # Увеличена точность до 9 знаков после запятой
                'sat_lat': round(lat_sat, 9),  # Увеличена точность до 9 знаков после запятой
                'sat_alt': round(alt, 6),  # Увеличена точность высоты
                'angle_traverse': round(angle, 6),  # Угол между КА и объектом относительно траверса (88-92°)
                'distance': round(distance, 2),
                'max_dist': round(max_distance, 2),
                'doppler_freq': round(doppler_freq, 3)  # Доплеровская частота в Гц
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
    
    # Сохраняем точки в GPKG файл
    if len(geometries) > 0:
        # Дополнительная проверка: фильтруем точки с невалидными углами
        valid_geometries = []
        valid_attributes = []
        
        for i, attr in enumerate(attributes):
            angle_val = attr.get('angle_traverse')
            image_number_val = attr.get('image_number')
            
            # Проверяем, что угол валиден и в диапазоне 88-92
            angle_valid = angle_val is not None and not (isinstance(angle_val, float) and (np.isnan(angle_val) or np.isinf(angle_val)))
            angle_in_range = angle_valid and (angle_min <= angle_val <= angle_max)
            
            # Проверяем, что image_number не равен None (точка должна попадать в интервал активной съемки)
            image_number_valid = image_number_val is not None
            
            # Сохраняем только точки с валидным углом И валидным номером снимка
            if angle_in_range and image_number_valid:
                valid_geometries.append(geometries[i])
                valid_attributes.append(attr)
            else:
                if not angle_in_range:
                    filtered_by_angle_count += 1
                # Если image_number == None, точка во время переключения антенны - не сохраняем
        
        if len(valid_geometries) > 0:
            new_gdf = gpd.GeoDataFrame(valid_attributes, geometry=valid_geometries, crs='EPSG:4326')
            
            # Проверяем, существует ли уже файл
            if os.path.exists(output_file):
                # Читаем существующий файл
                try:
                    existing_gdf = gpd.read_file(output_file, layer='periods_points')
                    
                    # Удаляем все точки текущего периода, чтобы избежать дубликатов
                    if 'period_id' in existing_gdf.columns:
                        existing_gdf = existing_gdf[existing_gdf['period_id'] != period_id]
                    
                    # Дополнительная фильтрация существующих данных (на случай, если там есть невалидные)
                    # Фильтруем по углу и по image_number (должен быть не NULL)
                    filter_conditions = []
                    if 'angle_traverse' in existing_gdf.columns:
                        filter_conditions.append(
                            (existing_gdf['angle_traverse'] >= angle_min) & 
                            (existing_gdf['angle_traverse'] <= angle_max) &
                            (existing_gdf['angle_traverse'].notna())
                        )
                    if 'image_number' in existing_gdf.columns:
                        filter_conditions.append(existing_gdf['image_number'].notna())
                    
                    if filter_conditions:
                        # Объединяем все условия фильтрации
                        combined_filter = filter_conditions[0]
                        for condition in filter_conditions[1:]:
                            combined_filter = combined_filter & condition
                        existing_gdf = existing_gdf[combined_filter]
                    
                    # Объединяем существующие данные (без текущего периода) и новые данные
                    combined_gdf = gpd.GeoDataFrame(pd.concat([existing_gdf, new_gdf], ignore_index=True), crs='EPSG:4326')
                    
                    # Дополнительная проверка на дубликаты по уникальному ключу (period_id, point_id, time)
                    if 'period_id' in combined_gdf.columns and 'point_id' in combined_gdf.columns and 'time' in combined_gdf.columns:
                        initial_combined_count = len(combined_gdf)
                        combined_gdf = combined_gdf.drop_duplicates(subset=['period_id', 'point_id', 'time'], keep='first')
                        if len(combined_gdf) < initial_combined_count:
                            print(f"    ⚠️  Удалено {initial_combined_count - len(combined_gdf)} дубликатов при сохранении периода {period_id}")
                    
                    combined_gdf.to_file(output_file, driver='GPKG', layer='periods_points')
                except Exception as e:
                    # Если не удалось прочитать, создаем новый или перезаписываем
                    new_gdf.to_file(output_file, driver='GPKG', layer='periods_points')
            else:
                # Создаем новый файл
                new_gdf.to_file(output_file, driver='GPKG', layer='periods_points')
        
        # Создаем и сохраняем квадратные рамки 10x10 км для каждого снимка в детальном прожекторном режиме (ДПР)
        # Получаем координаты целевой точки (нужны для создания квадратов)
        lat_t, lon_t, alt_t = target_pos
        
        try:
            
            # Расчет количества снимков в детальном прожекторном режиме (ДПР)
            period_duration_seconds = (period_end - period_start).total_seconds()
            images_count, total_cycle_time, residual_time = calculate_spotlight_images_count(
                period_duration_seconds,
                image_acquisition_time=10.0,  # Время синтеза апертуры для одного кадра (сек)
                antenna_switch_time=2.0       # Время переключения антенны (сек)
            )
            
            # Создаем рамку для каждого снимка
            squares_list = []
            cycle_time = 10.0 + 2.0  # image_acquisition_time + antenna_switch_time
            
            for image_num in range(1, images_count + 1):
                # Вычисляем время начала снимка
                # Первый снимок начинается в начале периода
                # Каждый последующий снимок начинается после завершения предыдущего цикла (съемка + переключение)
                image_start_time = period_start + timedelta(seconds=(image_num - 1) * cycle_time)
                
                # Время окончания снимка (начало + время синтеза апертуры)
                image_end_time = image_start_time + timedelta(seconds=10.0)
                
                # Вычисляем азимут трассы КА в момент начала снимка (нужен для квадрата)
                track_azimuth = calculate_track_azimuth(orb, image_start_time)
                
                # Создаем квадрат 10x10 км для детального прожекторного режима, ориентированный вдоль трассы КА
                square_polygon = create_oriented_square_10km(lon_t, lat_t, track_azimuth)
                
                # Создаем атрибуты для рамки снимка
                square_attrs = {
                    'period_id': period_id,
                    'image_number': image_num,  # Номер снимка в периоде (начиная с 1)
                    'type': 'square_frame',
                    'size_km': 10.0,  # Размер кадра в детальном прожекторном режиме согласно руководству Кондор-ФКА
                    'center_lon': round(lon_t, 9),
                    'center_lat': round(lat_t, 9),
                    'track_azimuth': round(track_azimuth, 6),
                    'image_start_time': image_start_time.strftime('%Y-%m-%d %H:%M:%S.%f'),
                    'image_end_time': image_end_time.strftime('%Y-%m-%d %H:%M:%S.%f'),
                    'period_start_time': period_start.strftime('%Y-%m-%d %H:%M:%S.%f'),
                    'period_end_time': period_end.strftime('%Y-%m-%d %H:%M:%S.%f'),
                    'spotlight_images_count': images_count,  # Общее количество снимков в периоде
                    'spotlight_total_time': round(total_cycle_time, 1),  # Общее время съемки в ДПР (сек)
                    'spotlight_residual_time': round(residual_time, 1)  # Остаточное время периода после съемки (сек)
                }
                squares_list.append({'geometry': square_polygon, **square_attrs})
            
            # Создаем GeoDataFrame для всех квадратов периода
            if squares_list:
                squares_data = []
                squares_geometries = []
                for sq in squares_list:
                    geometry = sq.pop('geometry')
                    squares_geometries.append(geometry)
                    squares_data.append(sq)
                
                square_gdf = gpd.GeoDataFrame(squares_data, geometry=squares_geometries, crs='EPSG:4326')
                
                # Сохраняем квадраты в отдельный слой GPKG файла
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
            print(f"    ⚠️  Предупреждение: не удалось создать квадраты для периода {period_id}: {e}")
        
        # Выводим информацию о сохранении
        period_duration_str = f"{period_duration_seconds:.1f} сек"
        if period_duration_seconds < 60:
            period_duration_str = f"{period_duration_seconds:.1f} сек"
        else:
            period_duration_str = f"{period_duration_seconds/60:.2f} мин"
        
        # Статистика по image_number для отладки
        if len(valid_attributes) > 0:
            image_number_stats = {}
            for attr in valid_attributes:
                img_num = attr.get('image_number')
                if img_num not in image_number_stats:
                    image_number_stats[img_num] = 0
                image_number_stats[img_num] += 1
            
            image_stats_str = ", ".join([f"снимок {k}: {image_number_stats[k]}" for k in sorted([x for x in image_number_stats.keys() if x is not None])])
            if filtered_by_angle_count > 0:
                print(f"  Период {period_id} (длительность: {period_duration_str}, шаг: {point_step_seconds:.1f} сек): сохранено {len(valid_geometries)} точек в GPKG (отфильтровано по углу: {filtered_by_angle_count})")
                if image_stats_str:
                    print(f"    Распределение по снимкам: {image_stats_str}")
            else:
                print(f"  Период {period_id} (длительность: {period_duration_str}, шаг: {point_step_seconds:.1f} сек): сохранено {len(valid_geometries)} точек в GPKG")
                if image_stats_str:
                    print(f"    Распределение по снимкам: {image_stats_str}")

def find_exact_period_start(approx_start_time, orb, target_pos, time_step_seconds=1.0):
    """
    Точный поиск начала периода с шагом 1 секунда
    Возвращается назад на 10 секунд от приблизительного начала и ищет точное начало
    
    Аргументы:
        approx_start_time: datetime - приблизительное время начала периода (обнаружено с шагом 10 сек)
        orb: объект Orbital
        target_pos: кортеж (широта, долгота, высота) целевой точки
        time_step_seconds: шаг времени для точного поиска (секунды), по умолчанию 1.0
    
    Возвращает:
        datetime - точное время начала периода
    """
    # Отступаем на 10 секунд назад от приблизительного начала
    search_start = approx_start_time - timedelta(seconds=10.0)
    current_time = search_start
    
    angle_min = 88.0
    angle_max = 92.0
    period_start = None
    
    # Ищем точное начало периода с шагом 1 секунда
    while current_time <= approx_start_time:
        try:
            # Получаем координаты спутника и цели в ECI
            X_s, Y_s, Z_s, _, _, _ = get_position(orb, current_time)
            sat_pos_eci = (X_s, Y_s, Z_s)
            
            lat_t, lon_t, alt_t = target_pos
            pos_tgt_eci, _ = get_xyzv_from_latlon(current_time, lon_t, lat_t, alt_t)
            target_pos_eci = pos_tgt_eci
            
            # Проверяем видимость
            is_visible, distance, max_distance = is_target_visible(sat_pos_eci, target_pos_eci, current_time)
            
            if is_visible:
                # Вычисляем угол между вектором на цель и направлением скорости
                angle = calculate_angle_from_velocity_at_time(current_time, orb, target_pos)
                
                # Проверяем, что угол валиден
                if angle is not None and not (isinstance(angle, float) and (np.isnan(angle) or np.isinf(angle))):
                    if angle_min <= angle <= angle_max:
                        if period_start is None:
                            # Нашли начало периода
                            period_start = current_time
                    else:
                        # Если угол вышел за пределы, но период уже начался - это конец поиска
                        if period_start is not None:
                            break
        except Exception:
            # Пропускаем точку при ошибке
            pass
        
        current_time += timedelta(seconds=time_step_seconds)
    
    # Если не нашли точное начало, возвращаем приблизительное
    return period_start if period_start is not None else approx_start_time

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
                # Начало нового периода - делаем точный поиск начала с шагом 1 секунда
                period_start = find_exact_period_start(current_time, orb, target_pos, time_step_seconds=1.0)
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
                    
                    # Расчет количества снимков в детальном прожекторном режиме (ДПР)
                    images_count, total_cycle_time, residual_time = calculate_spotlight_images_count(
                        period_duration,
                        image_acquisition_time=10.0,  # Время синтеза апертуры для одного кадра (сек)
                        antenna_switch_time=2.0       # Время переключения антенны (сек)
                    )
                    
                    period_data = {
                        'angle': round(avg_angle, 1),  # Округляем до 0.1 градуса
                        'start_time': period_start,
                        'end_time': period_end,
                        'calculated_angle': avg_angle,
                        'period_duration': period_duration,
                        'spotlight_images_count': images_count,  # Количество снимков в ДПР
                        'spotlight_total_time': round(total_cycle_time, 1),  # Общее время съемки
                        'spotlight_residual_time': round(residual_time, 1)  # Остаточное время
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
            
            # Расчет количества снимков в детальном прожекторном режиме (ДПР)
            images_count, total_cycle_time, residual_time = calculate_spotlight_images_count(
                period_duration,
                image_acquisition_time=10.0,  # Время синтеза апертуры для одного кадра (сек)
                antenna_switch_time=2.0       # Время переключения антенны (сек)
            )
            
            period_data = {
                'angle': round(avg_angle, 1),
                'start_time': period_start,
                'end_time': period_end,
                'calculated_angle': avg_angle,
                'period_duration': period_duration,
                'spotlight_images_count': images_count,  # Количество снимков в ДПР
                'spotlight_total_time': round(total_cycle_time, 1),  # Общее время съемки
                'spotlight_residual_time': round(residual_time, 1)  # Остаточное время
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

def export_gpkg_attributes_to_txt(gpkg_file='result/periods_points.gpkg', txt_file='result/periods_points.txt'):
    """
    Экспортирует атрибуты из GPKG файла в текстовый файл для анализа
    
    Аргументы:
        gpkg_file: путь к GPKG файлу
        txt_file: путь к выходному текстовому файлу
    """
    try:
        if not os.path.exists(gpkg_file):
            print(f"  Предупреждение: GPKG файл {gpkg_file} не найден, экспорт пропущен")
            return
        
        # Читаем GPKG файл
        gdf = gpd.read_file(gpkg_file, layer='periods_points')
        
        if len(gdf) == 0:
            print(f"  Предупреждение: GPKG файл {gpkg_file} пуст, экспорт пропущен")
            return
        
        # Удаляем дубликаты по уникальному ключу: period_id, point_id, time
        if 'period_id' in gdf.columns and 'point_id' in gdf.columns and 'time' in gdf.columns:
            initial_count = len(gdf)
            # Удаляем дубликаты, оставляя первую запись
            gdf = gdf.drop_duplicates(subset=['period_id', 'point_id', 'time'], keep='first')
            duplicates_removed = initial_count - len(gdf)
            if duplicates_removed > 0:
                print(f"  Удалено дубликатов: {duplicates_removed} (было {initial_count}, стало {len(gdf)})")
        
        # Создаем папку result, если её нет
        output_dir = os.path.dirname(txt_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Получаем все колонки (кроме geometry, angle и visible)
        columns = [col for col in gdf.columns if col != 'geometry' and col != 'angle' and col != 'visible']
        
        # Открываем файл для записи
        with open(txt_file, 'w', encoding='utf-8') as f:
            # Записываем заголовок
            f.write('\t'.join(columns) + '\n')
            
            # Записываем данные
            for idx, row in gdf.iterrows():
                values = []
                for col in columns:
                    val = row[col]
                    if val is None:
                        values.append('')
                    elif isinstance(val, (int, float)):
                        values.append(str(val))
                    else:
                        values.append(str(val))
                f.write('\t'.join(values) + '\n')
        
        print(f"  Атрибуты экспортированы в текстовый файл: {txt_file} ({len(gdf)} записей)")
        
    except Exception as e:
        print(f"  Предупреждение: не удалось экспортировать атрибуты в текстовый файл: {e}")

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
    num_days = 1
    
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
        
        # Расчет количества снимков в детальном прожекторном режиме (ДПР) согласно руководству Кондор-ФКА
        # Параметры для S-диапазона Кондор-ФКА:
        # - Время синтеза апертуры для одного кадра: ~10 секунд
        # - Время переключения антенны между кадрами: ~2 секунды
        images_count, total_cycle_time, residual_time = calculate_spotlight_images_count(
            item['period_duration'],
            image_acquisition_time=10.0,  # Время синтеза апертуры для одного кадра (сек)
            antenna_switch_time=2.0       # Время переключения антенны (сек)
        )
        print(f"   Детальный прожекторный режим (ДПР):")
        print(f"     - Количество снимков: {images_count}")
        if images_count > 0:
            print(f"     - Общее время съемки: {total_cycle_time:.1f} сек")
            if residual_time > 0:
                print(f"     - Остаточное время: {residual_time:.1f} сек")
            
            # Выводим время начала и конца для каждого снимка
            image_acquisition_time = 10.0
            antenna_switch_time = 2.0
            cycle_time = image_acquisition_time + antenna_switch_time
            
            # Подсчитываем количество точек и сохраняем время точек для каждого снимка из GPKG файла
            points_count_by_image = {}
            points_times_by_image = {}  # Словарь для хранения времени всех точек каждого снимка
            gpkg_file = 'result/periods_points.gpkg'
            period_id = i  # period_id соответствует порядковому номеру периода в sequence
            
            try:
                if os.path.exists(gpkg_file):
                    gdf_points = gpd.read_file(gpkg_file, layer='periods_points')
                    # Фильтруем точки текущего периода
                    if 'period_id' in gdf_points.columns and 'image_number' in gdf_points.columns and 'time' in gdf_points.columns:
                        period_points = gdf_points[gdf_points['period_id'] == period_id]
                        # Подсчитываем количество точек и сохраняем время для каждого снимка
                        for img_num in range(1, images_count + 1):
                            image_points = period_points[period_points['image_number'] == img_num]
                            points_count_by_image[img_num] = len(image_points)
                            # Сохраняем время всех точек для этого снимка
                            if len(image_points) > 0:
                                # Извлекаем время точек и сортируем
                                times = image_points['time'].tolist()
                                points_times_by_image[img_num] = sorted(times)
                            else:
                                points_times_by_image[img_num] = []
            except Exception as e:
                # Если не удалось прочитать файл, просто не будем выводить количество точек
                pass
            
            print(f"     - Время снимков:")
            for image_num in range(1, images_count + 1):
                image_start_time = item['start_time'] + timedelta(seconds=(image_num - 1) * cycle_time)
                image_end_time = image_start_time + timedelta(seconds=image_acquisition_time)
                points_count = points_count_by_image.get(image_num, 0)
                print(f"       Снимок {image_num}: {image_start_time.strftime('%Y-%m-%d %H:%M:%S')} - {image_end_time.strftime('%Y-%m-%d %H:%M:%S')} ({points_count} точек)")
                
                # Диагностика: проверяем, почему точки могут обрываться
                period_end_time = item['end_time']
                if image_end_time > period_end_time:
                    # Время окончания снимка выходит за пределы периода
                    print(f"         ⚠️  Примечание: время окончания снимка ({image_end_time.strftime('%Y-%m-%d %H:%M:%S')}) выходит за пределы периода наблюдения ({period_end_time.strftime('%Y-%m-%d %H:%M:%S')})")
                    print(f"         ⚠️  Период заканчивается раньше, чем заканчивается снимок (вероятно, угол вышел за пределы 88-92°)")
                
                # Выводим время всех точек для этого снимка
                if image_num in points_times_by_image and len(points_times_by_image[image_num]) > 0:
                    print(f"         Время точек:")
                    for point_time_str in points_times_by_image[image_num]:
                        # Выводим время точки (формат: YYYY-MM-DD HH:MM:SS.microseconds)
                        # Пытаемся распарсить с микросекундами
                        try:
                            # Пробуем формат с микросекундами
                            point_time = datetime.strptime(point_time_str, '%Y-%m-%d %H:%M:%S.%f')
                            print(f"           {point_time.strftime('%Y-%m-%d %H:%M:%S.%f')}")
                        except:
                            try:
                                # Пробуем формат без микросекунд
                                point_time = datetime.strptime(point_time_str, '%Y-%m-%d %H:%M:%S')
                                print(f"           {point_time.strftime('%Y-%m-%d %H:%M:%S')}")
                            except:
                                # Если не удалось распарсить, выводим как есть
                                print(f"           {point_time_str}")
                    
                    # Диагностика: показываем, когда закончились точки относительно времени окончания снимка
                    if len(points_times_by_image[image_num]) > 0:
                        try:
                            last_point_time_str = points_times_by_image[image_num][-1]
                            last_point_time = datetime.strptime(last_point_time_str, '%Y-%m-%d %H:%M:%S.%f')
                        except:
                            try:
                                last_point_time = datetime.strptime(last_point_time_str, '%Y-%m-%d %H:%M:%S')
                            except:
                                last_point_time = None
                        
                        if last_point_time:
                            time_diff = (image_end_time - last_point_time).total_seconds()
                            if time_diff > 0.2:  # Если разница больше шага точек (0.1 сек), выводим предупреждение
                                print(f"         ⚠️  Последняя точка ({last_point_time.strftime('%Y-%m-%d %H:%M:%S.%f')}) на {time_diff:.1f} сек раньше времени окончания снимка")
                                print(f"         ⚠️  Возможные причины: угол вышел за пределы 88-92°, период наблюдения закончился, или точка не прошла фильтры")
                else:
                    # Если нет точек для этого снимка
                    print(f"         ⚠️  Нет точек для этого снимка (возможно, все точки были отфильтрованы или период закончился до начала снимка)")
        
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
            except Exception as e:
                print(f"   Промежуток до следующего периода: {gap:.1f} сек ({gap/60:.2f} мин)")
                print(f"   (Не удалось вычислить расстояние и скорость: {e})")

    # Подсчитываем общее количество снимков
    total_images = 0
    for item in sequence:
        if 'spotlight_images_count' in item:
            total_images += item['spotlight_images_count']

    print("\nОБЩАЯ СТАТИСТИКА:")
    print(f"  Всего периодов видимости: {len(sequence)}")
    print(f"  Общее количество снимков: {total_images}")
    print(f"  Общая длительность съемки: {total_duration:.3f} сек ({total_duration/60:.2f} мин, {total_duration/3600:.2f} часов)")
    
    if len(sequence) > 1:
        print(f"  Общая длительность промежутков: {total_gaps:.1f} сек ({total_gaps/60:.2f} мин)")
        print(f"  Средний промежуток: {total_gaps/(len(sequence)-1):.1f} сек ({total_gaps/(len(sequence)-1)/60:.2f} мин)")
        total_time = total_duration + total_gaps
        efficiency = total_duration / total_time * 100 if total_time > 0 else 0
        print(f"  Эффективность использования времени: {efficiency:.1f}%")
    
    # Статистика по продолжительности периодов обзора
    if len(sequence) > 0:
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
    
    # Экспорт атрибутов GPKG в текстовый файл для анализа
    export_gpkg_attributes_to_txt()
    
    # Проверка угла для указанного времени
    check_time = datetime(2024, 3, 22, 3, 58, 37)
    try:
        angle_at_time = calculate_angle_from_velocity_at_time(check_time, orb, target_pos)
        print("\n" + "=" * 80)
        print(f"ПРОВЕРКА УГЛА ДЛЯ УКАЗАННОГО ВРЕМЕНИ")
        print("=" * 80)
        print(f"Время: {check_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Угол между вектором на цель и направлением скорости: {angle_at_time:.6f}°")
        
        # Проверяем, попадает ли угол в допустимый диапазон
        if 88.0 <= angle_at_time <= 92.0:
            print(f"✓ Угол находится в допустимом диапазоне (88-92°)")
        else:
            print(f"✗ Угол НЕ в допустимом диапазоне (88-92°)")
        
        # Дополнительная информация
        try:
            X_s, Y_s, Z_s, Vx_s, Vy_s, Vz_s = get_position(orb, check_time)
            lat_t, lon_t, alt_t = target_pos
            pos_tgt_eci, _ = get_xyzv_from_latlon(check_time, lon_t, lat_t, alt_t)
            target_pos_eci = pos_tgt_eci
            sat_pos_eci = (X_s, Y_s, Z_s)
            is_visible, distance, max_distance = is_target_visible(sat_pos_eci, target_pos_eci, check_time)
            
            print(f"Видимость цели: {'Да' if is_visible else 'Нет'}")
            print(f"Наклонная дальность: {distance:.2f} км")
            if R0_min and R0_max:
                if R0_min <= distance <= R0_max:
                    print(f"✓ Наклонная дальность в допустимом диапазоне ({R0_min}-{R0_max} км)")
                else:
                    print(f"✗ Наклонная дальность НЕ в допустимом диапазоне ({R0_min}-{R0_max} км)")
        except Exception as e:
            print(f"Не удалось вычислить дополнительные параметры: {e}")
            
    except Exception as e:
        print(f"\n⚠️  Не удалось вычислить угол для времени {check_time.strftime('%Y-%m-%d %H:%M:%S')}: {e}")
    
    print("=" * 80)

if __name__ == "__main__":
    continuous_angle_sequence()
