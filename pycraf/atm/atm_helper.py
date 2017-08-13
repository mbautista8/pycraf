#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from astropy.utils.data import get_pkg_data_filename
from astropy import units as apu
from .. import utils
from .. import conversions as cnv


__all__ = [
    'RESONANCES_OXYGEN', 'RESONANCES_WATER',
    'refractive_index', 'saturation_water_pressure',
    'pressure_water_from_humidity', 'humidity_from_pressure_water',
    'pressure_water_from_rho_water', 'rho_water_from_pressure_water',
    'opacity_from_atten', 'atten_from_opacity',
    ]


fname_oxygen = get_pkg_data_filename(
    '../itudata/p.676-10/R-REC-P.676-10-201309_table1.csv'
    )
fname_water = get_pkg_data_filename(
    '../itudata/p.676-10/R-REC-P.676-10-201309_table2.csv'
    )

oxygen_dtype = np.dtype([
    (str(s), np.float64) for s in ['f0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    ])
water_dtype = np.dtype([
    (str(s), np.float64) for s in ['f0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6']
    ])
RESONANCES_OXYGEN = np.genfromtxt(
    fname_oxygen, dtype=oxygen_dtype, delimiter=';'
    )
RESONANCES_WATER = np.genfromtxt(
    fname_water, dtype=water_dtype, delimiter=';'
    )


@utils.ranged_quantity_input(
    atten=(1.000000000001, None, cnv.dimless), elev=(-90, 90, apu.deg),
    strip_input_units=True, output_unit=cnv.dimless
    )
def opacity_from_atten(atten, elev):
    '''
    Atmospheric opacity derived from attenuation.

    Parameters
    ----------
    atten : `~astropy.units.Quantity`
        Atmospheric attenuation [dB or dimless]
    elev : `~astropy.units.Quantity`
        Elevation [deg]
        This is used to calculate the so-called Airmass, AM = 1 / sin(elev)

    Returns
    -------
    tau : `~astropy.units.Quantity`
        Atmospheric opacity [dimless aka neper]
    '''

    AM_inv = np.sin(np.radians(elev))

    return AM_inv * np.log(atten)


@utils.ranged_quantity_input(
    tau=(0.000000000001, None, cnv.dimless), elev=(-90, 90, apu.deg),
    strip_input_units=True, output_unit=cnv.dB
    )
def atten_from_opacity(tau, elev):
    '''
    Atmospheric attenuation derived from opacity.

    Parameters
    ----------
    tau : `~astropy.units.Quantity`
        Atmospheric opacity [dimless aka neper]
    elev : `~astropy.units.Quantity`
        Elevation [deg]

        This is used to calculate the so-called Airmass, AM = 1 / sin(elev)

    Returns
    -------
    atten : `~astropy.units.Quantity`
        Atmospheric attenuation [dB or dimless]
    '''

    AM = 1. / np.sin(np.radians(elev))

    return 10 * np.log10(np.exp(tau * AM))


@utils.ranged_quantity_input(
    temp=(1.e-30, None, apu.K),
    press=(1.e-30, None, apu.hPa),
    press_w=(1.e-30, None, apu.hPa),
    strip_input_units=True, output_unit=cnv.dimless
    )
def refractive_index(temp, press, press_w):
    '''
    Refractive index according to `ITU-R P.453-12
    <https://www.itu.int/rec/R-REC-P.453-12-201609-I/en>`_.

    Parameters
    ----------
    temp : `~astropy.units.Quantity`
        Air temperature [K]
    press : `~astropy.units.Quantity`
        Total air pressure [hPa]
    press_w : `~astropy.units.Quantity`
        Water vapor partial pressure [hPa]

    Returns
    -------
    n_index : `~astropy.units.Quantity`
        Refractive index [dimless]
    '''

    return (
        1 + 1e-6 / temp * (
            77.6 * press - 5.6 * press_w +
            3.75e5 * press_w / temp
            )
        )


@utils.ranged_quantity_input(
    temp=(1.e-30, None, apu.K),
    press=(1.e-30, None, apu.hPa),
    strip_input_units=True, output_unit=apu.hPa
    )
def saturation_water_pressure(temp, press, wet_type='water'):
    '''
    Saturation water pressure according to `ITU-R P.453-12
    <https://www.itu.int/rec/R-REC-P.453-12-201609-I/en>`_.

    Parameters
    ----------
    temp : `~astropy.units.Quantity`
        Air temperature [K]
    press : `~astropy.units.Quantity`
        Total air pressure [hPa]
    wet_type : str, optional
        Type of wet material: 'water', 'ice'

    Returns
    -------
    press_sat : `~astropy.units.Quantity`
        Saturation water vapor pressure, e_s [hPa]
    '''

    return _saturation_water_pressure(
        temp, press, wet_type=wet_type
        )


def _saturation_water_pressure(temp, press, wet_type):
    # temp_C is temperature in Celcius
    temp_C = temp - 273.15

    assert wet_type in ['water', 'ice']

    EF = (
        1. + 1.e-4 * (7.2 + press * (0.0320 + 5.9e-6 * temp_C ** 2))
        if wet_type == 'water' else
        1. + 1.e-4 * (2.2 + press * (0.0382 + 6.4e-6 * temp_C ** 2))
        )

    a, b, c, d = (
        (6.1121, 18.678, 257.14, 234.5)
        if wet_type == 'water' else
        (6.1115, 23.036, 279.82, 333.7)
        )
    e_s = EF * a * np.exp((b - temp_C / d) * temp_C / (c + temp_C))

    return e_s


@utils.ranged_quantity_input(
    temp=(1.e-30, None, apu.K),
    press=(1.e-30, None, apu.hPa),
    humidity=(0, 100, apu.percent),
    strip_input_units=True, output_unit=apu.hPa
    )
def pressure_water_from_humidity(
        temp, press, humidity, wet_type='water'
        ):
    '''
    Water pressure according to `ITU-R P.453-12
    <https://www.itu.int/rec/R-REC-P.453-12-201609-I/en>`_.

    Parameters
    ----------
    temp : `~astropy.units.Quantity`
        Air temperature [K]
    press : `~astropy.units.Quantity`
        Total air pressure [hPa]
    humidity : `~astropy.units.Quantity`
        Relative humidity [%]
    wet_type : str, optional
        Type of wet material: 'water', 'ice'

    Returns
    -------
    press_w : `~astropy.units.Quantity`
        Water vapor partial pressure [hPa]
    '''

    e_s = _saturation_water_pressure(
        temp, press, wet_type=wet_type
        )

    press_w = humidity / 100. * e_s

    return press_w


@utils.ranged_quantity_input(
    temp=(1.e-30, None, apu.K),
    press=(1.e-30, None, apu.hPa),
    press_w=(1.e-30, None, apu.hPa),
    strip_input_units=True, output_unit=apu.percent
    )
def humidity_from_pressure_water(
        temp, press, press_w, wet_type='water'
        ):
    '''
    Relative humidity according to `ITU-R P.453-12
    <https://www.itu.int/rec/R-REC-P.453-12-201609-I/en>`_.

    wet_type - 'water' or 'ice'

    Parameters
    ----------
    temp : `~astropy.units.Quantity`
        Air temperature [K]
    press : `~astropy.units.Quantity`
        Total air pressure [hPa]
    press_w : `~astropy.units.Quantity`
        Water vapor partial pressure [hPa]
    wet_type : str, optional
        Type of wet material: 'water', 'ice'

    Returns
    -------
    humidity : `~astropy.units.Quantity`
        Relative humidity [%]
    '''

    e_s = _saturation_water_pressure(
        temp, press, wet_type=wet_type
        )

    humidity = 100. * press_w / e_s

    return humidity


@utils.ranged_quantity_input(
    temp=(1.e-30, None, apu.K),
    rho_w=(1.e-30, None, apu.g / apu.m ** 3),
    strip_input_units=True, output_unit=apu.hPa
    )
def pressure_water_from_rho_water(temp, rho_w):
    '''
    Water pressure according to `ITU-R P.453-12
    <https://www.itu.int/rec/R-REC-P.453-12-201609-I/en>`_.

    Parameters
    ----------
    temp : `~astropy.units.Quantity`
        Air temperature [K]
    rho_w : `~astropy.units.Quantity`
        Water vapor density [g / m**3]

    Returns
    -------
    press_w : `~astropy.units.Quantity`
        Water vapor partial pressure [hPa]
    '''

    press_w = rho_w * temp / 216.7

    return press_w


@utils.ranged_quantity_input(
    temp=(1.e-30, None, apu.K),
    press_w=(1.e-30, None, apu.hPa),
    strip_input_units=True, output_unit=apu.g / apu.m ** 3
    )
def rho_water_from_pressure_water(temp, press_w):
    '''
    Water density according to `ITU-R P.453-12
    <https://www.itu.int/rec/R-REC-P.453-12-201609-I/en>`_.

    Parameters
    ----------
    temp : `~astropy.units.Quantity`
        Air temperature [K]
    press_w : `~astropy.units.Quantity`
        Water vapor partial pressure [hPa]

    Returns
    -------
    rho_w : `~astropy.units.Quantity`
        Water vapor density [g / m**3]
    '''

    rho_w = press_w * 216.7 / temp

    return rho_w
