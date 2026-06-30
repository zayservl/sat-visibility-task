"""Оценка видимости космического аппарата с точки зрения наземного наблюдателя

Модуль содержит математические функции для:
- перевода времени между шкалами TT и UT1
- вычисления гринвичского среднего звёздного времени (GMST)
- преобразования координат между системами ECI, ECEF и SEZ
- расчёта угла возвышения КА над горизонтом

Используются только numpy и стандартная библиотека math
"""

import math
import numpy as np

# Параметры эллипсоида WGS84
_WGS84_A = 6378137.0                        # большая полуось, метры
_WGS84_F = 1.0 / 298.257223563              # сжатие
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)     # квадрат эксцентриситета

# Опорная эпоха J2000 в юлианских днях
_J2000_JD = 2451545.0


def jd_tt_to_ut1(jd_tt: float, delta_t: float = 69.29) -> float:
    """Перевод юлианской даты из шкалы TT в шкалу UT1

    Шкала TT (Terrestrial Time) — равномерная, используется в эфемеридах
    Шкала UT1 — неравномерная, привязана к вращению Земли Именно UT1 нужна
    для расчёта угла поворота Земли (GMST)

    Разница между ними обозначается ΔT = TT - UT1 и измеряется в секундах
    Для даты 18 февраля 2022 года ΔT = 69.29 секунды (по таблицам)

    Args:
        jd_tt: юлианская дата в шкале TT
        delta_t: разница TT - UT1 в секундах (по умолчанию 69.29)

    Returns:
        юлианская дата в шкале UT1
    """
    return jd_tt - delta_t / 86400.0


def compute_gmst(jd_ut1: float) -> float:
    """Вычисление гринвичского среднего звёздного времени (GMST)

    GMST — угол, на который Земля повернулась вокруг оси Z относительно
    точки весеннего равноденствия Используется для перевода координат
    между инерциальной системой ECI и земной системой ECEF

    Применяется формула IAU

    Args:
        jd_ut1: юлианская дата в шкале UT1

    Returns:
        угол GMST в радианах, приведённый к диапазону [0, 2π)
    """
    t_ut1 = (jd_ut1 - _J2000_JD) / 36525.0

    gmst_deg = (
        280.46061837
        + 360.98564736629 * (jd_ut1 - _J2000_JD)
        + 0.000387933 * t_ut1 ** 2
        - t_ut1 ** 3 / 38710000.0
    )

    gmst_deg = gmst_deg % 360.0
    return math.radians(gmst_deg)


def eci_to_ecef(position_eci: np.ndarray, gmst_rad: float) -> np.ndarray:
    """Преобразование координат из инерциальной системы ECI в земную ECEF

    ECI зафиксирована относительно звёзд, а ECEF вращается вместе с Землёй
    Переход — поворот вокруг оси Z на угол GMST

    Args:
        position_eci: вектор координат КА в системе ECI, метры
        gmst_rad: угол GMST в радианах

    Returns:
        numpy.ndarray: вектор координат КА в системе ECEF, метры
    """
    x, y, z = np.asarray(position_eci, dtype=float)
    cos_t = math.cos(gmst_rad)
    sin_t = math.sin(gmst_rad)

    x_ecef = x * cos_t + y * sin_t
    y_ecef = -x * sin_t + y * cos_t
    z_ecef = z

    return np.array([x_ecef, y_ecef, z_ecef])


def observer_ecef(lat_deg: float, lon_deg: float) -> tuple[float, float, float]:
    """Координаты наземного наблюдателя в системе ECEF на эллипсоиде WGS84

    Высота над уровнем моря
    считается равной нулю (наблюдатель на поверхности эллипсоида)

    Args:
        lat_deg: географическая широта наблюдателя, градусы
        lon_deg: географическая долгота наблюдателя, градусы
            (западная — отрицательная, восточная — положительная)

    Returns:
        кортеж координат (x, y, z) в системе ECEF, метры
    """
    if not math.isfinite(lat_deg):
        raise ValueError(f"lat_deg должно быть конечным числом, получено: {lat_deg}")
    if not -90.0 <= lat_deg <= 90.0:
        raise ValueError(f"lat_deg должна быть в диапазоне [-90, 90], получено: {lat_deg}")
    if not math.isfinite(lon_deg):
        raise ValueError(f"lon_deg должно быть конечным числом, получено: {lon_deg}")
    if not -180.0 <= lon_deg <= 180.0:
        raise ValueError(f"lon_deg должна быть в диапазоне [-180, 180], получено: {lon_deg}")

    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)

    n = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * math.sin(lat) ** 2)

    x = n * math.cos(lat) * math.cos(lon)
    y = n * math.cos(lat) * math.sin(lon)
    z = n * (1.0 - _WGS84_E2) * math.sin(lat)

    return (x, y, z)


def ecef_to_sez(vector_ecef: np.ndarray, lat_deg: float, lon_deg: float) -> np.ndarray:
    """Преобразование вектора из ECEF в топоцентрическую систему SEZ

    Система SEZ (South-East-Zenith) привязана к наблюдателю:
    - ось S (South) — на юг
    - ось E (East) — на восток
    - ось Z (Zenith) — вверх, в зенит

    Компонента Z показывает, насколько объект «над горизонтом» —
    именно она определяет угол возвышения

    Args:
        vector_ecef: вектор в системе ECEF, метры
        lat_deg: широта наблюдателя, градусы
        lon_deg: долгота наблюдателя, градусы

    Returns:
        numpy.ndarray: вектор (S, E, Z) в топоцентрической системе, метры
    """
    rx, ry, rz = np.asarray(vector_ecef, dtype=float)
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)

    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)

    s = sin_lat * cos_lon * rx + sin_lat * sin_lon * ry - cos_lat * rz
    e = -sin_lon * rx + cos_lon * ry
    z = cos_lat * cos_lon * rx + cos_lat * sin_lon * ry + sin_lat * rz

    return np.array([s, e, z])


def compute_elevation(
    position_eci: np.ndarray,
    jd_tt: float,
    lat_deg: float,
    lon_deg: float,
    delta_t: float = 69.29,
) -> float:
    """Вычисление угла возвышения КА над горизонтом наблюдателя

    Объединяет все шаги:
    1. Перевод времени TT -> UT1
    2. Вычисление угла поворота Земли (GMST)
    3. Преобразование координат КА из ECI в ECEF
    4. Определение положения наблюдателя в ECEF (WGS84)
    5. Вектор «наблюдатель - КА» и перевод в систему SEZ
    6. Расчёт угла возвышения

    Args:
        position_eci: координаты КА в ECI J2000, метры
        jd_tt: юлианская дата в шкале TT
        lat_deg: широта наблюдателя, градусы
        lon_deg: долгота наблюдателя, градусы
        delta_t: разница TT - UT1 в секундах (по умолчанию 69.29)

    Returns:
        угол возвышения КА над горизонтом, градусы
    """
    pos = np.asarray(position_eci, dtype=float)
    if pos.shape != (3,):
        raise ValueError(f"position_eci должна быть вектором формы (3,), получена форма: {pos.shape}")
    if not np.all(np.isfinite(pos)):
        raise ValueError("position_eci содержит NaN или Inf")
    if np.linalg.norm(pos) <= _WGS84_A:
        raise ValueError(
            f"КА должен находиться над поверхностью Земли (норма > {_WGS84_A}), "
            f"получена норма: {np.linalg.norm(pos)}"
        )

    if not math.isfinite(jd_tt):
        raise ValueError(f"jd_tt должно быть конечным числом, получено: {jd_tt}")
    if jd_tt <= 0:
        raise ValueError(f"jd_tt должно быть положительным, получено: {jd_tt}")

    if not math.isfinite(lat_deg):
        raise ValueError(f"lat_deg должно быть конечным числом, получено: {lat_deg}")
    if not -90.0 <= lat_deg <= 90.0:
        raise ValueError(f"lat_deg должна быть в диапазоне [-90, 90], получено: {lat_deg}")
    if not math.isfinite(lon_deg):
        raise ValueError(f"lon_deg должно быть конечным числом, получено: {lon_deg}")
    if not -180.0 <= lon_deg <= 180.0:
        raise ValueError(f"lon_deg должна быть в диапазоне [-180, 180], получено: {lon_deg}")

    if not math.isfinite(delta_t):
        raise ValueError(f"delta_t должно быть конечным числом, получено: {delta_t}")

    jd_ut1 = jd_tt_to_ut1(jd_tt, delta_t=delta_t)
    gmst_rad = compute_gmst(jd_ut1)
    sat_ecef = eci_to_ecef(position_eci, gmst_rad)
    obs_ecef = np.array(observer_ecef(lat_deg, lon_deg))
    rel_ecef = sat_ecef - obs_ecef
    sez = ecef_to_sez(rel_ecef, lat_deg, lon_deg)

    distance = np.linalg.norm(sez)
    if distance == 0.0:
        return 0.0
    elevation_rad = math.asin(np.clip(sez[2] / distance, -1.0, 1.0))
    return math.degrees(elevation_rad)


def is_visible(elevation_deg: float, min_elevation: float = 15.0) -> bool:
    """Проверка видимости КА по углу возвышения

    КА считается видимым, если его угол возвышения превышает заданный порог

    Args:
        elevation_deg: угол возвышения КА, градусы
        min_elevation: минимальный порог видимости, градусы

    Returns:
        True, если КА виден, иначе False
    """
    if not math.isfinite(elevation_deg):
        raise ValueError(f"elevation_deg должно быть конечным числом, получено: {elevation_deg}")
    if not math.isfinite(min_elevation):
        raise ValueError(f"min_elevation должно быть конечным числом, получено: {min_elevation}")
    if not 0.0 <= min_elevation <= 90.0:
        raise ValueError(f"min_elevation должна быть в диапазоне [0, 90], получено: {min_elevation}")
    return elevation_deg > min_elevation