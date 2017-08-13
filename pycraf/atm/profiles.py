#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from astropy import units as apu
from .. import utils
from .atm_helper import refractive_index, humidity_from_pressure_water


__all__ = [
    'profile_standard', 'profile_lowlat',
    'profile_midlat_summer', 'profile_midlat_winter',
    'profile_highlat_summer', 'profile_highlat_winter',
    ]


@utils.ranged_quantity_input(
    height=(0, 84.99999999, apu.km),
    strip_input_units=True, output_unit=None
    )
def profile_standard(height):
    '''
    Standard height profiles according to `ITU-R P.835-5
    <https://www.itu.int/rec/R-REC-P.835-5-201202-I/en>`_, Annex 1.

    Parameters
    ----------
    height : `~astropy.units.Quantity`
        Height above ground [km]

    Returns
    -------
    temp : `~astropy.units.Quantity`
        Temperature [K]
    press : `~astropy.units.Quantity`
        Total pressure [hPa]
    rho_w : `~astropy.units.Quantity`
        Water vapor density [g / m**3]
    press_w : `~astropy.units.Quantity`
        Water vapor partial pressure [hPa]
    n_index : `~astropy.units.Quantity`
        Refractive index [dimless]
    humidity_water : `~astropy.units.Quantity`
        Relative humidity if water vapor was in form of liquid water [%]
    humidity_ice : `~astropy.units.Quantity`
        Relative humidity if water vapor was in form of ice [%]

    Notes
    -----
    For convenience, derived quantities like water density/pressure
    and refraction indices are also returned.
    '''

    # height = np.asarray(height)  # this is not sufficient for masking :-(
    height = np.atleast_1d(height)
    _height = height.flatten()

    # to make this work with numpy arrays
    # lets first find the correct index for every height

    layer_heights = np.array([0., 11., 20., 32., 47., 51., 71., 85.])
    indices = np.zeros(_height.size, dtype=np.int32)
    for i, lh in enumerate(layer_heights[0:-1]):
        indices[_height > lh] = i

    T0 = 288.15  # K
    P0 = 1013.25  # hPa
    rho0 = 7.5  # g / m^3
    h0 = 2.  # km
    layer_temp_gradients = np.array([-6.5, 0., 1., 2.8, 0., -2.8, -2.])
    layer_start_temperatures = [T0]
    layer_start_pressures = [P0]
    for i in range(1, len(layer_heights) - 1):

        dh = layer_heights[i] - layer_heights[i - 1]
        Ti = layer_start_temperatures[i - 1]
        Li = layer_temp_gradients[i - 1]
        Pi = layer_start_pressures[i - 1]
        # print(i, Ti, Li, Pi, i - 1 not in [1, 4])
        layer_start_temperatures.append(Ti + dh * Li)
        layer_start_pressures.append(
            Pi * (
                (Ti / (Ti + Li * dh)) ** (34.163 / Li)
                if i - 1 not in [1, 4] else
                np.exp(-34.163 * dh / Ti)
                )
            )
    layer_start_temperatures = np.array(layer_start_temperatures)
    layer_start_pressures = np.array(layer_start_pressures)

    temperatures = (
        layer_start_temperatures[indices] +
        (_height - layer_heights[indices]) * layer_temp_gradients[indices]
        )

    pressures = np.empty_like(temperatures)

    # gradient zero
    mask = np.in1d(indices, [1, 4])
    indm = indices[mask]
    dhm = (_height[mask] - layer_heights[indm])
    pressures[mask] = (
        layer_start_pressures[indm] *
        np.exp(-34.163 * dhm / layer_start_temperatures[indm])
        )

    # gradient non-zero
    mask = np.logical_not(mask)
    indm = indices[mask]
    dhm = (_height[mask] - layer_heights[indm])
    Lim = layer_temp_gradients[indm]
    Tim = layer_start_temperatures[indm]
    pressures[mask] = (
        layer_start_pressures[indm] *
        (Tim / (Tim + Lim * dhm)) ** (34.163 / Lim)
        )

    rho_water = rho0 * np.exp(-_height / h0)
    pressures_water = rho_water * temperatures / 216.7
    mask = (pressures_water / pressures) < 2.e-6
    pressures_water[mask] = pressures[mask] * 2.e-6
    rho_water[mask] = pressures_water[mask] / temperatures[mask] * 216.7

    temperatures = apu.Quantity(temperatures.reshape(height.shape), apu.K)
    pressures = apu.Quantity(pressures.reshape(height.shape), apu.hPa)
    pressures_water = apu.Quantity(
        pressures_water.reshape(height.shape), apu.hPa
        )
    rho_water = apu.Quantity(
        rho_water.reshape(height.shape), apu.g / apu.m ** 3
        )

    ref_indices = refractive_index(temperatures, pressures, pressures_water)
    humidities_water = humidity_from_pressure_water(
        temperatures, pressures, pressures_water, wet_type='water'
        )
    humidities_ice = humidity_from_pressure_water(
        temperatures, pressures, pressures_water, wet_type='ice'
        )

    result = (
        temperatures.squeeze(),
        pressures.squeeze(),
        rho_water.squeeze(),
        pressures_water.squeeze(),
        ref_indices.squeeze(),
        humidities_water.squeeze(),
        humidities_ice.squeeze(),
        )

    # return tuple(v.reshape(height.shape) for v in result)
    return result


@apu.quantity_input(height=apu.km)
def _profile_helper(
        height,
        temp_heights, temp_funcs,
        press_heights, press_funcs,
        rho_heights, rho_funcs,
        ):
    '''
    Helper function for specialized profiles.

    Parameters
    ----------
    height : `~astropy.units.Quantity`
        Height above ground [km]
    {temp,press,rho}_heights : list of floats
        Height steps for which piece-wise functions are defined
    {temp,press,rho}_funcs - list of functions
        Functions that return the desired quantity for a given height interval

    Returns
    -------
    temp : `~astropy.units.Quantity`
        Temperature [K]
    press : `~astropy.units.Quantity`
        Total pressure [hPa]
    rho_w : `~astropy.units.Quantity`
        Water vapor density [g / m**3]
    press_w : `~astropy.units.Quantity`
        Water vapor partial pressure [hPa]
    n_index : `~astropy.units.Quantity`
        Refractive index [dimless]
    humidity_water : `~astropy.units.Quantity`
        Relative humidity if water vapor was in form of liquid water [%]
    humidity_ice : `~astropy.units.Quantity`
        Relative humidity if water vapor was in form of ice [%]
    '''

    height = np.atleast_1d(height)
    _height = height.to(apu.km).value

    assert np.all(_height < temp_heights[-1]), (
        'profile only defined below {} km height!'.format(temp_heights[-1])
        )
    assert np.all(_height >= temp_heights[0]), (
        'profile only defined above {} km height!'.format(temp_heights[0])
        )

    temperature = np.empty(height.shape, dtype=np.float64)
    pressure = np.empty(height.shape, dtype=np.float64)
    pressure_water = np.empty(height.shape, dtype=np.float64)
    rho_water = np.empty(height.shape, dtype=np.float64)

    Pstarts = [None]
    for i in range(1, len(press_heights) - 1):
        Pstarts.append(press_funcs[i - 1](Pstarts[-1], press_heights[i]))

    # calculate temperature profile
    for i in range(len(temp_heights) - 1):
        hmin, hmax = temp_heights[i], temp_heights[i + 1]
        mask = (_height >= hmin) & (_height < hmax)
        temperature[mask] = (temp_funcs[i])(_height[mask])

    # calculate pressure profile
    for i in range(len(press_heights) - 1):
        hmin, hmax = press_heights[i], press_heights[i + 1]
        mask = (_height >= hmin) & (_height < hmax)
        pressure[mask] = (press_funcs[i])(Pstarts[i], _height[mask])

    # calculate rho profile
    for i in range(len(rho_heights) - 1):
        hmin, hmax = rho_heights[i], rho_heights[i + 1]
        mask = (_height >= hmin) & (_height < hmax)
        rho_water[mask] = (rho_funcs[i])(_height[mask])

    # calculate pressure_water profile
    pressure_water = rho_water * temperature / 216.7
    mask = (pressure_water / pressure) < 2.e-6
    pressure_water[mask] = pressure[mask] * 2.e-6
    rho_water[mask] = pressure_water[mask] / temperature[mask] * 216.7

    temperature = apu.Quantity(temperature.reshape(height.shape), apu.K)
    pressure = apu.Quantity(pressure.reshape(height.shape), apu.hPa)
    pressure_water = apu.Quantity(
        pressure_water.reshape(height.shape), apu.hPa
        )
    rho_water = apu.Quantity(
        rho_water.reshape(height.shape), apu.g / apu.m ** 3
        )

    ref_index = refractive_index(temperature, pressure, pressure_water)
    humidity_water = humidity_from_pressure_water(
        temperature, pressure, pressure_water, wet_type='water'
        )
    humidity_ice = humidity_from_pressure_water(
        temperature, pressure, pressure_water, wet_type='ice'
        )

    return (
        temperature.squeeze(),
        pressure.squeeze(),
        rho_water.squeeze(),
        pressure_water.squeeze(),
        ref_index.squeeze(),
        humidity_water.squeeze(),
        humidity_ice.squeeze(),
        )


def profile_lowlat(height):
    '''
    Low latitude height profiles according to `ITU-R P.835-5
    <https://www.itu.int/rec/R-REC-P.835-5-201202-I/en>`_.

    Valid for geographic latitudes :math:`\\vert \\phi\\vert < 22^\\circ`.

    Parameters
    ----------
    height : `~astropy.units.Quantity`
        Height above ground [km]

    Returns
    -------
    temp : `~astropy.units.Quantity`
        Temperature [K]
    press : `~astropy.units.Quantity`
        Total pressure [hPa]
    rho_w : `~astropy.units.Quantity`
        Water vapor density [g / m**3]
    press_w : `~astropy.units.Quantity`
        Water vapor partial pressure [hPa]
    n_index : `~astropy.units.Quantity`
        Refractive index [dimless]
    humidity_water : `~astropy.units.Quantity`
        Relative humidity if water vapor was in form of liquid water [%]
    humidity_ice : `~astropy.units.Quantity`
        Relative humidity if water vapor was in form of ice [%]
    '''

    temp_heights = [0., 17., 47., 52., 80., 100.]
    temp_funcs = [
        lambda h: 300.4222 - 6.3533 * h + 0.005886 * h ** 2,
        lambda h: 194 + (h - 17.) * 2.533,
        lambda h: 270.,
        lambda h: 270. - (h - 52.) * 3.0714,
        lambda h: 184.,
        ]

    press_heights = [0., 10., 72., 100.]
    press_funcs = [
        lambda Pstart, h: 1012.0306 - 109.0338 * h + 3.6316 * h ** 2,
        lambda Pstart, h: Pstart * np.exp(-0.147 * (h - 10)),
        lambda Pstart, h: Pstart * np.exp(-0.165 * (h - 72)),
        ]

    rho_heights = [0., 15., 100.]
    rho_funcs = [
        lambda h: 19.6542 * np.exp(
            -0.2313 * h -
            0.1122 * h ** 2 +
            0.01351 * h ** 3 -
            0.0005923 * h ** 4
            ),
        lambda h: 0.,
        ]

    return _profile_helper(
        height,
        temp_heights, temp_funcs,
        press_heights, press_funcs,
        rho_heights, rho_funcs,
        )


def profile_midlat_summer(height):
    '''
    Mid latitude summer height profiles according to `ITU-R P.835-5
    <https://www.itu.int/rec/R-REC-P.835-5-201202-I/en>`_.

    Valid for geographic latitudes :math:`22^\\circ < \\vert \\phi\\vert < 45^\\circ`.

    Parameters
    ----------
    height : `~astropy.units.Quantity`
        Height above ground [km]

    Returns
    -------
    temp : `~astropy.units.Quantity`
        Temperature [K]
    press : `~astropy.units.Quantity`
        Total pressure [hPa]
    rho_w : `~astropy.units.Quantity`
        Water vapor density [g / m**3]
    press_w : `~astropy.units.Quantity`
        Water vapor partial pressure [hPa]
    n_index : `~astropy.units.Quantity`
        Refractive index [dimless]
    humidity_water : `~astropy.units.Quantity`
        Relative humidity if water vapor was in form of liquid water [%]
    humidity_ice : `~astropy.units.Quantity`
        Relative humidity if water vapor was in form of ice [%]
    '''

    temp_heights = [0., 13., 17., 47., 53., 80., 100.]
    temp_funcs = [
        lambda h: 294.9838 - 5.2159 * h - 0.07109 * h ** 2,
        lambda h: 215.15,
        lambda h: 215.15 * np.exp(0.008128 * (h - 17.)),
        lambda h: 275.,
        lambda h: 275. + 20. * (1. - np.exp(0.06 * (h - 53.))),
        lambda h: 175.,
        ]

    press_heights = [0., 10., 72., 100.]
    press_funcs = [
        lambda Pstart, h: 1012.8186 - 111.5569 * h + 3.8646 * h ** 2,
        lambda Pstart, h: Pstart * np.exp(-0.147 * (h - 10)),
        lambda Pstart, h: Pstart * np.exp(-0.165 * (h - 72)),
        ]

    rho_heights = [0., 15., 100.]
    rho_funcs = [
        lambda h: 14.3542 * np.exp(
            -0.4174 * h - 0.02290 * h ** 2 + 0.001007 * h ** 3
            ),
        lambda h: 0.,
        ]

    return _profile_helper(
        height,
        temp_heights, temp_funcs,
        press_heights, press_funcs,
        rho_heights, rho_funcs,
        )


def profile_midlat_winter(height):
    '''
    Mid latitude winter height profiles according to `ITU-R P.835-5
    <https://www.itu.int/rec/R-REC-P.835-5-201202-I/en>`_.

    Valid for geographic latitudes :math:`22^\\circ < \\vert \\phi\\vert < 45^\\circ`.

    Parameters
    ----------
    height : `~astropy.units.Quantity`
        Height above ground [km]

    Returns
    -------
    temp : `~astropy.units.Quantity`
        Temperature [K]
    press : `~astropy.units.Quantity`
        Total pressure [hPa]
    rho_w : `~astropy.units.Quantity`
        Water vapor density [g / m**3]
    press_w : `~astropy.units.Quantity`
        Water vapor partial pressure [hPa]
    n_index : `~astropy.units.Quantity`
        Refractive index [dimless]
    humidity_water : `~astropy.units.Quantity`
        Relative humidity if water vapor was in form of liquid water [%]
    humidity_ice : `~astropy.units.Quantity`
        Relative humidity if water vapor was in form of ice [%]
    '''

    temp_heights = [0., 10., 33., 47., 53., 80., 100.]
    temp_funcs = [
        lambda h: 272.7241 - 3.6517 * h - 0.1759 * h ** 2,
        lambda h: 218.,
        lambda h: 218. + 3.3571 * (h - 33.),
        lambda h: 265.,
        lambda h: 265. - 2.0370 * (h - 53.),
        lambda h: 210.,
        ]

    press_heights = [0., 10., 72., 100.]
    press_funcs = [
        lambda Pstart, h: 1018.8627 - 124.2954 * h + 4.8307 * h ** 2,
        lambda Pstart, h: Pstart * np.exp(-0.147 * (h - 10)),
        lambda Pstart, h: Pstart * np.exp(-0.155 * (h - 72)),
        ]

    rho_heights = [0., 10., 100.]
    rho_funcs = [
        lambda h: 3.4742 * np.exp(
            -0.2697 * h - 0.03604 * h ** 2 + 0.0004489 * h ** 3
            ),
        lambda h: 0.,
        ]

    return _profile_helper(
        height,
        temp_heights, temp_funcs,
        press_heights, press_funcs,
        rho_heights, rho_funcs,
        )


def profile_highlat_summer(height):
    '''
    High latitude summer height profiles according to `ITU-R P.835-5
    <https://www.itu.int/rec/R-REC-P.835-5-201202-I/en>`_.

    Valid for geographic latitudes :math:`\\vert \\phi\\vert > 45^\\circ`.

    Parameters
    ----------
    height : `~astropy.units.Quantity`
        Height above ground [km]

    Returns
    -------
    temp : `~astropy.units.Quantity`
        Temperature [K]
    press : `~astropy.units.Quantity`
        Total pressure [hPa]
    rho_w : `~astropy.units.Quantity`
        Water vapor density [g / m**3]
    press_w : `~astropy.units.Quantity`
        Water vapor partial pressure [hPa]
    n_index : `~astropy.units.Quantity`
        Refractive index [dimless]
    humidity_water : `~astropy.units.Quantity`
        Relative humidity if water vapor was in form of liquid water [%]
    humidity_ice : `~astropy.units.Quantity`
        Relative humidity if water vapor was in form of ice [%]
    '''

    temp_heights = [0., 10., 23., 48., 53., 79., 100.]
    temp_funcs = [
        lambda h: 286.8374 - 4.7805 * h - 0.1402 * h ** 2,
        lambda h: 225.,
        lambda h: 225. * np.exp(0.008317 * (h - 23.)),
        lambda h: 277.,
        lambda h: 277. - 4.0769 * (h - 53.),
        lambda h: 171.,
        ]

    press_heights = [0., 10., 72., 100.]
    press_funcs = [
        lambda Pstart, h: 1008.0278 - 113.2494 * h + 3.9408 * h ** 2,
        lambda Pstart, h: Pstart * np.exp(-0.147 * (h - 10)),
        lambda Pstart, h: Pstart * np.exp(-0.165 * (h - 72)),
        ]

    rho_heights = [0., 15., 100.]
    rho_funcs = [
        lambda h: 8.988 * np.exp(
            -0.3614 * h - 0.005402 * h ** 2 - 0.001955 * h ** 3
            ),
        lambda h: 0.,
        ]

    return _profile_helper(
        height,
        temp_heights, temp_funcs,
        press_heights, press_funcs,
        rho_heights, rho_funcs,
        )


def profile_highlat_winter(height):
    '''
    High latitude winter height profiles according to `ITU-R P.835-5
    <https://www.itu.int/rec/R-REC-P.835-5-201202-I/en>`_.

    Valid for geographic latitudes :math:`\\vert \\phi\\vert > 45^\\circ`.

    Parameters
    ----------
    height : `~astropy.units.Quantity`
        Height above ground [km]

    Returns
    -------
    temp : `~astropy.units.Quantity`
        Temperature [K]
    press : `~astropy.units.Quantity`
        Total pressure [hPa]
    rho_w : `~astropy.units.Quantity`
        Water vapor density [g / m**3]
    press_w : `~astropy.units.Quantity`
        Water vapor partial pressure [hPa]
    n_index : `~astropy.units.Quantity`
        Refractive index [dimless]
    humidity_water : `~astropy.units.Quantity`
        Relative humidity if water vapor was in form of liquid water [%]
    humidity_ice : `~astropy.units.Quantity`
        Relative humidity if water vapor was in form of ice [%]
    '''

    temp_heights = [0., 8.5, 30., 50., 54., 100.]
    temp_funcs = [
        lambda h: 257.4345 + 2.3474 * h - 1.5479 * h ** 2 + 0.08473 * h ** 3,
        lambda h: 217.5,
        lambda h: 217.5 + 2.125 * (h - 30.),
        lambda h: 260.,
        lambda h: 260. - 1.667 * (h - 54.),
        ]

    press_heights = [0., 10., 72., 100.]
    press_funcs = [
        lambda Pstart, h: 1010.8828 - 122.2411 * h + 4.554 * h ** 2,
        lambda Pstart, h: Pstart * np.exp(-0.147 * (h - 10)),
        lambda Pstart, h: Pstart * np.exp(-0.150 * (h - 72)),
        ]

    rho_heights = [0., 10., 100.]
    rho_funcs = [
        lambda h: 1.2319 * np.exp(
            0.07481 * h - 0.0981 * h ** 2 + 0.00281 * h ** 3
            ),
        lambda h: 0.,
        ]

    return _profile_helper(
        height,
        temp_heights, temp_funcs,
        press_heights, press_funcs,
        rho_heights, rho_funcs,
        )
