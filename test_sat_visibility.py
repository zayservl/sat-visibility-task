"""Тесты для модуля sat_visibility
"""

import math

import numpy as np
import pytest

import sat_visibility as sv

# Входные данные задачи — для интеграционных тестов
POSITION_ECI = [4435144.0, -2137297.0, 4670064.0]  # метры, ECI J2000
JD_TT = 2451545.0 + 8084.185608609847               # полная юлианская дата (TT)
OBS_LAT = 45.920266                                 # северная широта, градусы
OBS_LON = -63.342286                                # западная долгота, градусы


def test_jd_tt_to_ut1_subtract_delta():
    """UT1 должна быть меньше TT на величину delta_t в днях"""
    jd_tt = 2459629.185608609847
    delta_t = 69.29
    jd_ut1 = sv.jd_tt_to_ut1(jd_tt, delta_t=delta_t)
    assert jd_ut1 == pytest.approx(jd_tt - delta_t / 86400.0)


def test_jd_tt_to_ut1_zero_delta():
    """Если delta_t = 0, UT1 совпадает с TT"""
    jd_tt = 2459629.0
    assert sv.jd_tt_to_ut1(jd_tt, delta_t=0.0) == jd_tt


def test_jd_tt_to_ut1_round_trip():
    """Обратный перевод должен вернуть исходное TT"""
    jd_tt = 2459629.185608609847
    jd_ut1 = sv.jd_tt_to_ut1(jd_tt, delta_t=69.29)
    jd_tt_back = jd_ut1 + 69.29 / 86400.0
    assert jd_tt_back == pytest.approx(jd_tt)


def test_gmst_in_range():
    """GMST — угол, должен лежать в диапазоне [0, 2π) радиан"""
    gmst = sv.compute_gmst(JD_TT - 69.29 / 86400.0)
    assert 0.0 <= gmst < 2.0 * math.pi


def test_gmst_at_j2000_epoch():
    """На эпоху J2000 (UT1 = 2451545.0) GMST ≈ 280.46° от начального смещения"""
    gmst_rad = sv.compute_gmst(2451545.0)
    expected_deg = 280.46061837 % 360.0
    assert math.degrees(gmst_rad) == pytest.approx(expected_deg, abs=1e-6)


def test_gmst_monotonic():
    """Земля вращается в одну сторону: при росте времени GMST растёт"""
    g1 = math.degrees(sv.compute_gmst(2451545.0))
    g2 = math.degrees(sv.compute_gmst(2451545.0 + 0.01))
    assert g2 > g1


def test_eci_to_ecef_zero_angle():
    """При GMST = 0 координаты не меняются"""
    pos = np.array([100.0, 200.0, 300.0])
    out = sv.eci_to_ecef(pos, 0.0)
    assert np.allclose(out, pos)


def test_eci_to_ecef_90_degrees():
    """Поворот на 90°: ось X переходит в -Y, Z не меняется"""
    pos = np.array([1.0, 0.0, 5.0])
    out = sv.eci_to_ecef(pos, math.pi / 2.0)
    assert out == pytest.approx(np.array([0.0, -1.0, 5.0]))


def test_eci_to_ecef_preserves_length():
    """Поворот сохраняет длину вектора"""
    pos = np.array(POSITION_ECI)
    out = sv.eci_to_ecef(pos, 1.23)
    assert np.linalg.norm(out) == pytest.approx(np.linalg.norm(pos))


def test_observer_ecef_equator():
    """На экваторе z = 0, x ≈ большая полуось"""
    x, y, z = sv.observer_ecef(0.0, 0.0)
    assert z == pytest.approx(0.0, abs=1e-3)
    assert x == pytest.approx(6378137.0, rel=1e-6)
    assert y == pytest.approx(0.0, abs=1e-3)


def test_observer_ecef_north_pole():
    """На северном полюсе x = y = 0, z ≈ полярный радиус"""
    x, y, z = sv.observer_ecef(90.0, 0.0)
    assert x == pytest.approx(0.0, abs=1e-3)
    assert y == pytest.approx(0.0, abs=1e-3)
    expected_z = 6378137.0 * (1.0 - 1.0 / 298.257223563)
    assert z == pytest.approx(expected_z, rel=1e-6)


def test_observer_ecef_west_longitude():
    """Западная долгота отрицательная — y должно быть отрицательным на 90°W"""
    x, y, z = sv.observer_ecef(0.0, -90.0)
    assert x == pytest.approx(0.0, abs=1e-3)
    assert y == pytest.approx(-6378137.0, rel=1e-6)
    assert z == pytest.approx(0.0, abs=1e-3)


def test_ecef_to_sez_zenith_vector():
    """Вектор в зенит даёт S = E = 0, Z > 0"""
    up = np.array([1.0, 0.0, 0.0])
    s, e, z = sv.ecef_to_sez(up, 0.0, 0.0)
    assert s == pytest.approx(0.0, abs=1e-12)
    assert e == pytest.approx(0.0, abs=1e-12)
    assert z > 0


def test_ecef_to_sez_east_vector():
    """На экваторе, нулевой меридиан: направление на восток = +Y в ECEF"""
    up = np.array([0.0, 1.0, 0.0])
    s, e, z = sv.ecef_to_sez(up, 0.0, 0.0)
    assert e == pytest.approx(1.0)
    assert s == pytest.approx(0.0, abs=1e-12)
    assert z == pytest.approx(0.0, abs=1e-12)


def test_ecef_to_sez_preserves_length():
    """Поворот в SEZ сохраняет длину вектора"""
    vec = np.array([1e6, -2e6, 3e6])
    out = sv.ecef_to_sez(vec, 45.0, -30.0)
    assert np.linalg.norm(out) == pytest.approx(np.linalg.norm(vec))


def test_compute_elevation_integration():
    """Интеграционный тест: угол возвышения для данных задачи"""
    elev = sv.compute_elevation(POSITION_ECI, JD_TT, OBS_LAT, OBS_LON)
    assert -90.0 <= elev <= 90.0
    assert elev > 0.0


def test_compute_elevation_zenith():
    """КА прямо над наблюдателем — угол возвышения близок к 90°"""
    jd_ut1 = 2451545.0
    gmst_rad = sv.compute_gmst(jd_ut1)
    lon_deg = -math.degrees(gmst_rad)
    # приводим долготу к диапазону [-180, 180]
    lon_deg = ((lon_deg + 180.0) % 360.0) - 180.0
    sat_eci = np.array([7000000.0, 0.0, 0.0])
    elev = sv.compute_elevation(sat_eci, jd_ut1 + 69.29 / 86400.0, 0.0, lon_deg)
    assert elev == pytest.approx(90.0, abs=1e-3)


def test_is_visible_above_threshold():
    """Угол выше порога — КА виден"""
    assert sv.is_visible(30.0, min_elevation=15.0) is True


def test_is_visible_below_threshold():
    """Угол ниже порога — КА не виден"""
    assert sv.is_visible(5.0, min_elevation=15.0) is False


def test_is_visible_at_threshold():
    """На самом пороге используем строгое неравенство — не виден"""
    assert sv.is_visible(15.0, min_elevation=15.0) is False


def test_is_visible_negative():
    """КА под горизонтом — не виден"""
    assert sv.is_visible(-10.0) is False