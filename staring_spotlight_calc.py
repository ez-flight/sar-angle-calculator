"""
Расчет возможности работы в режиме Staring Spotlight для космических РСА

На основе документа: "Улучшение характеристик космических РСА в режимах высокого разрешения"
(Staring SpotLight.pdf)

Режим Staring Spotlight позволяет существенно повысить детальность и качество изображений
за счет расширенного сканирования луча антенны в азимутальной плоскости.
"""

from datetime import datetime, timedelta
from pyorbital.orbital import Orbital
from calc_cord import get_xyzv_from_latlon
import numpy as np

# Импортируем функции из continuous_angle_calc локально, чтобы избежать циклических зависимостей
# Эти импорты будут выполнены при первом вызове функций


def calculate_azimuth_sector(period_start, period_end, orb, target_pos, num_samples=10):
    """
    Вычисляет азимутальный сектор сканирования за период
    
    Азимутальный сектор - это изменение угла между вектором на цель и направлением скорости
    за период наблюдения. Для Staring Spotlight требуется сектор не менее 4.0-5.0°.
    
    Аргументы:
        period_start: datetime - начало периода
        period_end: datetime - конец периода
        orb: объект Orbital
        target_pos: кортеж (широта, долгота, высота) целевой точки
        num_samples: количество точек для расчета сектора (по умолчанию 10)
    
    Возвращает:
        словарь с параметрами азимутального сектора
    """
    # Локальный импорт для избежания циклических зависимостей
    from continuous_angle_calc import calculate_angle_from_velocity_at_time
    
    period_duration = (period_end - period_start).total_seconds()
    
    if period_duration <= 0:
        return {
            'azimuth_sector': 0.0,
            'min_angle': None,
            'max_angle': None,
            'angle_range': 0.0
        }
    
    # Вычисляем углы в нескольких точках периода
    angles = []
    time_step = period_duration / (num_samples - 1) if num_samples > 1 else 0
    
    for i in range(num_samples):
        sample_time = period_start + timedelta(seconds=i * time_step)
        if sample_time > period_end:
            sample_time = period_end
        
        angle = calculate_angle_from_velocity_at_time(sample_time, orb, target_pos)
        if angle is not None and not (isinstance(angle, float) and (np.isnan(angle) or np.isinf(angle))):
            angles.append(angle)
    
    if len(angles) == 0:
        return {
            'azimuth_sector': 0.0,
            'min_angle': None,
            'max_angle': None,
            'angle_range': 0.0
        }
    
    min_angle = min(angles)
    max_angle = max(angles)
    angle_range = max_angle - min_angle
    
    return {
        'azimuth_sector': angle_range,
        'min_angle': min_angle,
        'max_angle': max_angle,
        'angle_range': angle_range,
        'angles': angles
    }


def calculate_prf_requirements(doppler_freq, wavelength=0.031):
    """
    Рассчитывает требования к PRF (частоте повторения импульсов) для Staring Spotlight
    
    Согласно документации, необходимо найти компромисс между:
    - Неоднозначностью по азимуту
    - Неоднозначностью по дальности
    
    Для однозначности по доплеровской частоте: PRF >= 2 * |f_D_max|
    
    Аргументы:
        doppler_freq: доплеровская частота в Гц
        wavelength: длина волны РСА в метрах (по умолчанию 0.031 м для X-диапазона)
    
    Возвращает:
        словарь с требованиями к PRF
    """
    # Минимальная PRF для однозначности по доплеровской частоте
    min_prf_doppler = 2 * abs(doppler_freq)
    
    # Типичные значения PRF для космических РСА X-диапазона
    # COSMO-SkyMed: ~3000-5000 Гц
    # TerraSAR-X: ~3000-6500 Гц
    typical_prf_min = 3000.0
    typical_prf_max = 6500.0
    
    # Проверяем, попадает ли требуемая PRF в типичный диапазон
    prf_feasible = min_prf_doppler <= typical_prf_max
    
    return {
        'min_prf_doppler': min_prf_doppler,
        'doppler_freq': doppler_freq,
        'wavelength': wavelength,
        'typical_prf_range': (typical_prf_min, typical_prf_max),
        'prf_feasible': prf_feasible,
        'recommended_prf': max(min_prf_doppler, typical_prf_min) if prf_feasible else None
    }


def calculate_incidence_angle(sat_pos_eci, target_pos_eci, target_pos_geo):
    """
    Вычисляет угол падения (incidence angle) для точки на поверхности Земли
    
    Угол падения - это угол между направлением на цель и нормалью к поверхности Земли
    в точке цели.
    
    Аргументы:
        sat_pos_eci: кортеж (X_s, Y_s, Z_s) - координаты спутника в ECI (м)
        target_pos_eci: кортеж (X_t, Y_t, Z_t) - координаты цели в ECI (м)
        target_pos_geo: кортеж (lat, lon, alt) - географические координаты цели
    
    Возвращает:
        угол падения в градусах
    """
    # Вектор от спутника к цели
    R_target = np.array([
        target_pos_eci[0] - sat_pos_eci[0],
        target_pos_eci[1] - sat_pos_eci[1],
        target_pos_eci[2] - sat_pos_eci[2]
    ])
    R_target_norm = np.linalg.norm(R_target)
    
    if R_target_norm < 1e-3:
        return None
    
    R_target_unit = R_target / R_target_norm
    
    # Нормаль к поверхности Земли в точке цели (направлена от центра Земли)
    # Для упрощения используем направление от центра Земли к точке цели
    target_radius = np.linalg.norm(np.array(target_pos_eci))
    if target_radius < 1e-3:
        return None
    
    surface_normal = np.array(target_pos_eci) / target_radius
    
    # Угол между направлением на цель и нормалью к поверхности
    cos_incidence = np.dot(R_target_unit, surface_normal)
    cos_incidence = np.clip(cos_incidence, -1.0, 1.0)
    incidence_angle_rad = np.arccos(cos_incidence)
    incidence_angle_deg = np.degrees(incidence_angle_rad)
    
    return incidence_angle_deg


def calculate_staring_spotlight_feasibility(period_start, period_end, orb, target_pos):
    """
    Оценивает возможность работы в режиме Staring Spotlight для периода
    
    Критерии оценки на основе документации:
    1. Азимутальный сектор сканирования >= 4.0-5.0° (желательно)
    2. Длительность периода достаточна для расширенного сканирования
    3. Доплеровская частота в допустимых пределах
    4. Угол падения в оптимальном диапазоне (20-60° для большинства РСА)
    5. Угол между вектором на цель и направлением скорости в диапазоне 85.9-94.1°
    
    Аргументы:
        period_start: datetime - начало периода
        period_end: datetime - конец периода
        orb: объект Orbital
        target_pos: кортеж (широта, долгота, высота) целевой точки
    
    Возвращает:
        словарь с оценкой возможности и параметрами
    """
    # Локальные импорты для избежания циклических зависимостей
    from continuous_angle_calc import (
        get_position,
        calculate_angle_from_velocity_at_time,
        calculate_doppler_frequency,
        ANGLE_MIN,
        ANGLE_MAX
    )
    
    period_duration = (period_end - period_start).total_seconds()
    
    # 1. Вычисляем азимутальный сектор сканирования
    azimuth_sector_data = calculate_azimuth_sector(period_start, period_end, orb, target_pos)
    azimuth_sector = azimuth_sector_data['azimuth_sector']
    
    # 2. Вычисляем углы в начале и конце периода
    start_angle = calculate_angle_from_velocity_at_time(period_start, orb, target_pos)
    end_angle = calculate_angle_from_velocity_at_time(period_end, orb, target_pos)
    
    # 3. Вычисляем доплеровскую частоту в середине периода
    mid_time = period_start + (period_end - period_start) / 2
    X_s, Y_s, Z_s, Vx_s, Vy_s, Vz_s = get_position(orb, mid_time)
    sat_pos_eci = (X_s, Y_s, Z_s)
    
    lat_t, lon_t, alt_t = target_pos
    pos_tgt_eci, _ = get_xyzv_from_latlon(mid_time, lon_t, lat_t, alt_t)
    target_pos_eci = pos_tgt_eci
    
    doppler_freq = calculate_doppler_frequency(
        X_s, Y_s, Z_s, Vx_s, Vy_s, Vz_s,
        pos_tgt_eci[0], pos_tgt_eci[1], pos_tgt_eci[2],
        wavelength=0.031  # X-диапазон
    )
    
    # 4. Вычисляем угол падения
    incidence_angle = calculate_incidence_angle(sat_pos_eci, target_pos_eci, target_pos)
    
    # 5. Вычисляем требования к PRF
    prf_requirements = calculate_prf_requirements(doppler_freq, wavelength=0.031)
    
    # Критерии для Staring Spotlight
    min_azimuth_sector = 4.0  # Минимальный сектор сканирования (градусы) - желательно
    optimal_azimuth_sector = 5.0  # Оптимальный сектор сканирования (градусы)
    min_period_duration = 30.0  # Минимальная длительность периода (секунды)
    
    # Оптимальный диапазон угла падения (зависит от РСА, обычно 20-60°)
    optimal_incidence_min = 20.0
    optimal_incidence_max = 60.0
    
    # Проверка критериев
    criteria = {
        'azimuth_sector_ok': azimuth_sector >= min_azimuth_sector,
        'azimuth_sector_optimal': azimuth_sector >= optimal_azimuth_sector,
        'period_duration_ok': period_duration >= min_period_duration,
        'angle_in_range': (
            start_angle is not None and end_angle is not None and
            ANGLE_MIN <= start_angle <= ANGLE_MAX and
            ANGLE_MIN <= end_angle <= ANGLE_MAX
        ),
        'incidence_angle_ok': (
            incidence_angle is not None and
            optimal_incidence_min <= incidence_angle <= optimal_incidence_max
        ),
        'prf_feasible': prf_requirements['prf_feasible']
    }
    
    # Общая оценка возможности
    is_feasible = (
        criteria['azimuth_sector_ok'] and
        criteria['period_duration_ok'] and
        criteria['angle_in_range'] and
        criteria['prf_feasible']
    )
    
    # Оценка качества (от 0 до 100)
    quality_score = 0
    if criteria['azimuth_sector_ok']:
        quality_score += 30
        if criteria['azimuth_sector_optimal']:
            quality_score += 10
    if criteria['period_duration_ok']:
        quality_score += 20
    if criteria['angle_in_range']:
        quality_score += 20
    if criteria['incidence_angle_ok']:
        quality_score += 10
    if criteria['prf_feasible']:
        quality_score += 10
    
    # Рекомендации
    recommendations = []
    if not criteria['azimuth_sector_ok']:
        recommendations.append(
            f"Азимутальный сектор сканирования ({azimuth_sector:.2f}°) меньше минимального ({min_azimuth_sector}°). "
            f"Для Staring Spotlight желателен сектор >= {optimal_azimuth_sector}°"
        )
    if not criteria['period_duration_ok']:
        recommendations.append(
            f"Длительность периода ({period_duration:.1f} сек) меньше минимальной ({min_period_duration} сек)"
        )
    if not criteria['angle_in_range']:
        recommendations.append(
            f"Угол между вектором на цель и направлением скорости вне диапазона {ANGLE_MIN}-{ANGLE_MAX}°"
        )
    if not criteria['incidence_angle_ok'] and incidence_angle is not None:
        recommendations.append(
            f"Угол падения ({incidence_angle:.1f}°) вне оптимального диапазона "
            f"({optimal_incidence_min}-{optimal_incidence_max}°)"
        )
    if not criteria['prf_feasible']:
        recommendations.append(
            f"Требуемая PRF ({prf_requirements['min_prf_doppler']:.0f} Гц) превышает типичный диапазон "
            f"({prf_requirements['typical_prf_range'][0]:.0f}-{prf_requirements['typical_prf_range'][1]:.0f} Гц)"
        )
    
    if len(recommendations) == 0:
        recommendations.append("Все критерии выполнены. Режим Staring Spotlight возможен.")
    
    return {
        'is_feasible': is_feasible,
        'quality_score': quality_score,
        'azimuth_sector': azimuth_sector,
        'min_azimuth_sector_required': min_azimuth_sector,
        'optimal_azimuth_sector_required': optimal_azimuth_sector,
        'period_duration': period_duration,
        'start_angle': start_angle,
        'end_angle': end_angle,
        'angle_range': azimuth_sector_data['angle_range'],
        'doppler_freq': doppler_freq,
        'incidence_angle': incidence_angle,
        'prf_requirements': prf_requirements,
        'criteria': criteria,
        'recommendations': recommendations
    }


def analyze_periods_for_staring_spotlight(periods_data, orb, target_pos):
    """
    Анализирует список периодов на возможность работы в режиме Staring Spotlight
    
    Аргументы:
        periods_data: список словарей с данными периодов (из continuous_angle_calc)
        orb: объект Orbital
        target_pos: кортеж (широта, долгота, высота) целевой точки
    
    Возвращает:
        список словарей с результатами анализа для каждого периода
    """
    results = []
    
    for period_data in periods_data:
        period_start = period_data['start_time']
        period_end = period_data['end_time']
        period_id = period_data.get('period_id', len(results) + 1)
        
        # Оцениваем возможность работы в Staring Spotlight
        feasibility = calculate_staring_spotlight_feasibility(
            period_start, period_end, orb, target_pos
        )
        
        # Объединяем данные периода с результатами оценки
        result = {
            'period_id': period_id,
            'period_start': period_start,
            'period_end': period_end,
            'period_duration': period_data.get('period_duration', 0),
            'staring_spotlight_feasible': feasibility['is_feasible'],
            'quality_score': feasibility['quality_score'],
            'azimuth_sector': feasibility['azimuth_sector'],
            'doppler_freq': feasibility['doppler_freq'],
            'incidence_angle': feasibility['incidence_angle'],
            'min_prf_required': feasibility['prf_requirements']['min_prf_doppler'],
            'recommended_prf': feasibility['prf_requirements']['recommended_prf'],
            'recommendations': feasibility['recommendations']
        }
        
        results.append(result)
    
    return results


def print_staring_spotlight_analysis(results):
    """
    Выводит результаты анализа возможности работы в режиме Staring Spotlight
    
    Аргументы:
        results: список словарей с результатами анализа
    """
    print("\n" + "="*80)
    print("АНАЛИЗ ВОЗМОЖНОСТИ РАБОТЫ В РЕЖИМЕ STARING SPOTLIGHT")
    print("="*80)
    
    if len(results) == 0:
        print("Периоды не найдены для анализа.")
        return
    
    feasible_count = sum(1 for r in results if r['staring_spotlight_feasible'])
    total_count = len(results)
    
    print(f"\nВсего периодов проанализировано: {total_count}")
    print(f"Периодов, пригодных для Staring Spotlight: {feasible_count} ({feasible_count/total_count*100:.1f}%)")
    
    # Сортируем по качеству (quality_score)
    sorted_results = sorted(results, key=lambda x: x['quality_score'], reverse=True)
    
    print("\n" + "-"*80)
    print("ДЕТАЛЬНЫЙ АНАЛИЗ ПЕРИОДОВ:")
    print("-"*80)
    
    for i, result in enumerate(sorted_results, 1):
        print(f"\nПериод {result['period_id']}:")
        print(f"  Время: {result['period_start'].strftime('%Y-%m-%d %H:%M:%S')} - "
              f"{result['period_end'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Длительность: {result['period_duration']:.1f} сек")
        print(f"  Оценка качества: {result['quality_score']}/100")
        print(f"  Staring Spotlight возможен: {'✓ ДА' if result['staring_spotlight_feasible'] else '✗ НЕТ'}")
        print(f"  Азимутальный сектор сканирования: {result['azimuth_sector']:.2f}°")
        print(f"  Доплеровская частота: {result['doppler_freq']:.1f} Гц")
        if result['incidence_angle'] is not None:
            print(f"  Угол падения: {result['incidence_angle']:.1f}°")
        print(f"  Минимальная требуемая PRF: {result['min_prf_required']:.0f} Гц")
        if result['recommended_prf'] is not None:
            print(f"  Рекомендуемая PRF: {result['recommended_prf']:.0f} Гц")
        
        if result['recommendations']:
            print(f"  Рекомендации:")
            for rec in result['recommendations']:
                print(f"    - {rec}")
    
    # Статистика
    print("\n" + "-"*80)
    print("СТАТИСТИКА:")
    print("-"*80)
    
    if feasible_count > 0:
        feasible_results = [r for r in results if r['staring_spotlight_feasible']]
        avg_azimuth_sector = np.mean([r['azimuth_sector'] for r in feasible_results])
        avg_quality = np.mean([r['quality_score'] for r in feasible_results])
        
        print(f"  Средний азимутальный сектор (для пригодных периодов): {avg_azimuth_sector:.2f}°")
        print(f"  Средняя оценка качества (для пригодных периодов): {avg_quality:.1f}/100")
    
    print("="*80 + "\n")

