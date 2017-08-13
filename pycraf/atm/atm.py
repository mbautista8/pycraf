#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

from functools import partial
import numbers
import numpy as np
from astropy import units as apu
from .. import conversions as cnv
from .. import utils
from .atm_helper import RESONANCES_OXYGEN, RESONANCES_WATER


__all__ = [
    'atten_specific_annex1',
    'atten_terrestrial', 'atten_slant_annex1',
    'atten_specific_annex2',
    'atten_slant_annex2',
    'equivalent_height_dry', 'equivalent_height_wet',
    ]


F0_O2, A1_O2, A2_O2, A3_O2, A4_O2, A5_O2, A6_O2 = (
    RESONANCES_OXYGEN[k] for k in ['f0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    )
F0_H2O, B1_H2O, B2_H2O, B3_H2O, B4_H2O, B5_H2O, B6_H2O = (
    RESONANCES_WATER[k] for k in ['f0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6']
    )


def _S_oxygen(press_dry, temp):
    '''
    Line strengths of all oxygen resonances according to `ITU-R P.676-10
    <https://www.itu.int/rec/R-REC-P.676-10-201309-S/en>`_, Eq (3).

    Parameters
    ----------
    press_dry : `numpy.ndarray`, float
        Dry air (=Oxygen) pressure [hPa]
    temp : `numpy.ndarray`, float
        Temperature [K]

    Returns
    -------
    I : `numpy.ndarray`, float
        Line strength

    Notes
    -----
    Total pressure: `press = press_dry + press_w`
    '''

    theta = 300. / temp
    factor = 1.e-7 * press_dry * theta ** 3

    return A1_O2 * factor * np.exp(A2_O2 * (1. - theta))


def _S_water(press_w, temp):
    '''
    Line strengths of all water resonances according to `ITU-R P.676-10
    <https://www.itu.int/rec/R-REC-P.676-10-201309-S/en>`_, Eq (3).

    Parameters
    ----------
    press_w : `numpy.ndarray`, float
        Water vapor partial pressure [hPa]
    temp : `numpy.ndarray`, float
        Temperature [K]

    Returns
    -------
    I : `numpy.ndarray`, float
        Line strength

    Notes
    -----
    Total pressure: `press = press_dry + press_w`
    '''

    theta = 300. / temp
    factor = 1.e-1 * press_w * theta ** 3.5

    return B1_H2O * factor * np.exp(B2_H2O * (1. - theta))


def _Delta_f_oxygen(press_dry, press_w, temp):
    '''
    Line widths for all oxygen resonances according to `ITU-R P.676-10
    <https://www.itu.int/rec/R-REC-P.676-10-201309-S/en>`_, Eq (6a/b).

    Parameters
    ----------
    press_dry : `numpy.ndarray`, float
        Dry air (=Oxygen) pressure [hPa]
    press_w : `numpy.ndarray`, float
        Water vapor partial pressure [hPa]
    temp : `numpy.ndarray`, float
        Temperature [K]

    Returns
    -------
    W : `numpy.ndarray`, float
        Line widths for all oxygen resonances

    Notes
    -----
    Oxygen resonance line widths also depend on wet-air pressure.
    '''

    theta = 300. / temp
    df = A3_O2 * 1.e-4 * (
        press_dry * theta ** (0.8 - A4_O2) +
        1.1 * press_w * theta
        )
    return np.sqrt(df ** 2 + 2.25e-6)


def _Delta_f_water(press_dry, press_w, temp):
    '''
    Line widths for all water resonances according to
    `ITU-R P.676-10 <https://www.itu.int/rec/R-REC-P.676-10-201309-S/en>`_,
    Eq (6a/b).

    Parameters
    ----------
    press_dry : `numpy.ndarray`, float
        Dry air (=Oxygen) pressure [hPa]
    press_w : `numpy.ndarray`, float
        Water vapor partial pressure [hPa]
    temp : `numpy.ndarray`, float
        Temperature [K]

    Returns
    -------
    W : `numpy.ndarray`, float
        Line widths for all water resonances [GHz]

    Notes
    -----
    Water resonance line widths also depend on dry-air pressure.
    '''

    theta = 300. / temp
    df = B3_H2O * 1.e-4 * (
        press_dry * theta ** B4_H2O +
        B5_H2O * press_w * theta ** B6_H2O
        )

    return 0.535 * df + np.sqrt(
        0.217 * df ** 2 + 2.1316e-12 * F0_H2O ** 2 / theta
        )


def _delta_oxygen(press_dry, press_w, temp):
    '''
    Shape correction for all oxygen resonances according to
    `ITU-R P.676-10 <https://www.itu.int/rec/R-REC-P.676-10-201309-S/en>`_,
    Eq (7).

    Parameters
    ----------
    press_dry : `numpy.ndarray`, float
        Dry air (=Oxygen) pressure [hPa]
    press_w : `numpy.ndarray`, float
        Water vapor partial pressure [hPa]
    temp : `numpy.ndarray`, float
        Temperature [K]

    Returns
    -------
    delta : `numpy.ndarray`, float
        Profile shape correction factors for all oxygen resonances

    Notes
    -----
    This function accounts for interference effects in oxygen lines.
    '''

    theta = 300. / temp
    return (
        (A5_O2 + A6_O2 * theta) * 1.e-4 *
        (press_dry + press_w) * theta ** 0.8
        )


def _delta_water():
    '''
    Shape correction factor for all water vapor resonances (all-zero).

    Returns
    -------
    delta : `numpy.ndarray`, float
        Profile shape correction factors for all water resonances.

    Notes
    -----
    This is only introduced to be able to use the same machinery that is
    working with oxygen, in the `_delta_oxygen` function.
    '''

    return np.zeros(len(F0_H2O), dtype=np.float64)


def _F(freq_grid, f_i, Delta_f, delta):
    '''
    Line-profiles for all resonances at the freq_grid positions according to
    `ITU-R P.676-10 <https://www.itu.int/rec/R-REC-P.676-10-201309-S/en>`_,
    Eq (5).

    Parameters
    ----------
    freq_grid : `numpy.ndarray` of float
        Frequencies at which to calculate line-width shapes [GHz]
    f_i : `numpy.ndarray` of float
        Resonance line frequencies [GHz]
    Delta_f : `numpy.ndarray` of float
        Line widths of all resonances [GHz]
    delta : `numpy.ndarray` of float
        Correction factors to account for interference effects in oxygen
        lines

    Returns
    -------
    S : `numpy.ndarray` of float (m, n)
        Line-shape values, (`n = len(freq_grid)`, `m = len(Delta_f)`)

    Notes
    -----
    No integration is done between `freq_grid` positions, so if you're
    interested in high accuracy near resonance lines, make your `freq_grid`
    sufficiently fine.
    '''

    _freq_grid = freq_grid[np.newaxis]
    _f_i = f_i[:, np.newaxis]
    _Delta_f = Delta_f[:, np.newaxis]
    _delta = delta[:, np.newaxis]

    _df_plus, _df_minus = _f_i + _freq_grid, _f_i - _freq_grid

    sum_1 = (_Delta_f - _delta * _df_minus) / (_df_minus ** 2 + _Delta_f ** 2)
    sum_2 = (_Delta_f - _delta * _df_plus) / (_df_plus ** 2 + _Delta_f ** 2)

    return _freq_grid / _f_i * (sum_1 + sum_2)


def _N_D_prime2(freq_grid, press_dry, press_w, temp):
    '''
    Dry air continuum absorption (Debye spectrum) according to
    `ITU-R P.676-10 <https://www.itu.int/rec/R-REC-P.676-10-201309-S/en>`_,
    Eq (8/9).

    Parameters
    ----------
    freq_grid : `numpy.ndarray`, float
        Frequencies at which to calculate line-width shapes [GHz]
    press_dry : `numpy.ndarray`, float
        Dry air (=Oxygen) pressure [hPa]
    press_w : `numpy.ndarray`, float
        Water vapor partial pressure [hPa]
    temp : `numpy.ndarray`, float
        Temperature [K]

    Returns
    -------
    deb_spec : `numpy.ndarray`, float
       Debye absorption spectrum [dB / km]
    '''

    theta = 300. / temp
    d = 5.6e-4 * (press_dry + press_w) * theta ** 0.8

    sum_1 = 6.14e-5 / d / (1 + (freq_grid / d) ** 2)
    sum_2 = 1.4e-12 * press_dry * theta ** 1.5 / (
        1 + 1.9e-5 * freq_grid ** 1.5
        )

    return freq_grid * press_dry * theta ** 2 * (sum_1 + sum_2)


def _atten_specific_annex1(
        freq_grid, press_dry, press_w, temp
        ):

    freq_grid = np.atleast_1d(freq_grid)

    if not isinstance(press_dry, numbers.Real):
        raise TypeError('press_dry must be a scalar float')
    if not isinstance(press_w, numbers.Real):
        raise TypeError('press_w must be a scalar float')
    if not isinstance(temp, numbers.Real):
        raise TypeError('temp must be a scalar float')

    # first calculate dry attenuation (oxygen lines + N_D_prime2)
    S_o2 = _S_oxygen(press_dry, temp)
    Delta_f = _Delta_f_oxygen(press_dry, press_w, temp)
    delta = _delta_oxygen(press_dry, press_w, temp)
    F_o2 = _F(freq_grid, F0_O2, Delta_f, delta)

    atten_o2 = np.sum(S_o2[:, np.newaxis] * F_o2, axis=0)
    atten_o2 += _N_D_prime2(
        freq_grid, press_dry, press_w, temp
        )

    # now, wet contribution
    S_h2o = _S_water(press_w, temp)
    Delta_f = _Delta_f_water(press_dry, press_w, temp)
    delta = _delta_water()
    F_h2o = _F(freq_grid, F0_H2O, Delta_f, delta)

    atten_h2o = np.sum(S_h2o[:, np.newaxis] * F_h2o, axis=0)

    return atten_o2 * 0.182 * freq_grid, atten_h2o * 0.182 * freq_grid


@utils.ranged_quantity_input(
    freq_grid=(1.e-30, 1000, apu.GHz),
    press_dry=(1.e-30, None, apu.hPa),
    press_w=(1.e-30, None, apu.hPa),
    temp=(1.e-30, None, apu.K),
    strip_input_units=True, output_unit=(cnv.dB / apu.km, cnv.dB / apu.km)
    )
def atten_specific_annex1(
        freq_grid, press_dry, press_w, temp
        ):
    '''
    Specific (one layer) atmospheric attenuation according to `ITU-R P.676-10
    <https://www.itu.int/rec/R-REC-P.676-10-201309-S/en>`_, Annex 1.

    Parameters
    ----------
    freq_grid : `~astropy.units.Quantity`
        Frequencies at which to calculate line-width shapes [GHz]
    press_dry : `~astropy.units.Quantity`
        Dry air (=Oxygen) pressure [hPa]
    press_w : `~astropy.units.Quantity`
        Water vapor partial pressure [hPa]
    temp : `~astropy.units.Quantity`
        Temperature [K]

    Returns
    -------
    atten_dry : `~astropy.units.Quantity`
        Dry-air specific attenuation [dB / km]
    atten_wet : `~astropy.units.Quantity`
        Wet-air specific attenuation [dB / km]

    Notes
    -----
    No integration is done between `freq_grid` positions, so if you're
    interested in high accuracy near resonance lines, make your `freq_grid`
    sufficiently fine.
    '''

    return _atten_specific_annex1(
        freq_grid, press_dry, press_w, temp
        )


@utils.ranged_quantity_input(
    specific_atten=(1.e-30, None, cnv.dB / apu.km),
    path_length=(1.e-30, None, apu.km),
    strip_input_units=True, output_unit=cnv.dB
    )
def atten_terrestrial(specific_atten, path_length):
    '''
    Total path attenuation for a path close to the ground (i.e., one layer),
    according to `ITU-R P.676-10
    <https://www.itu.int/rec/R-REC-P.676-10-201309-S/en>`_, Annex 1 + 2.

    Parameters
    ----------
    specific_atten : `~astropy.units.Quantity`
        Specific attenuation (dry + wet) [dB / km]
    path_length : `~astropy.units.Quantity`
        Length of path [km]

    Returns
    -------
    total_atten : `~astropy.units.Quantity`
        Total attenuation along path [dB]
    '''

    return specific_atten * path_length


def _prepare_path(elev, obs_alt, profile_func, max_path_length=1000.):
    '''
    Helper function to construct the path parameters; see `ITU-R P.676-10
    <https://www.itu.int/rec/R-REC-P.676-10-201309-S/en>`_, Annex 1.

    Parameters
    ----------
    elev : float
        (Apparent) elevation of source as seen from observer [deg]
    obs_alt : float
        Height of observer above sea-level [m]
    profile_func : func
        A height profile function having the same signature as
        `~pycraf.atm.profile_standard`
    max_path_length : float, optional
        Maximal length of path before stopping iteration [km]
        (default: 1000 km; useful for terrestrial paths)

    Returns
    -------
    layer_params : list
        List of tuples with the following quantities for each height layer `n`:

        0) press_n - Total pressure [hPa]
        1) press_w_n - Water vapor partial pressure [hPa]
        2) temp_n - Temperature [K]
        3) a_n - Path length [km]
        4) r_n - Radius (i.e., distance to Earth center) [km]
        5) alpha_n - exit angle [rad]
        6) delta_n - angle between current normal vector and first normal
           vector (aka projected angular distance to starting point) [rad]
        7) beta_n - entry angle [rad]
        8) h_n - height above sea-level [km]

    Refraction : float
        Offset with respect to a hypothetical straight path, i.e., the
        correction between real and apparent source elevation [deg]

    Notes
    -----
    Although the `profile_func` must have the same signature as
    `~pycraf.atm.profile_standard`, which is one of the standardized
    atmospheric height profiles, only temperature, total pressure and
    water vapor pressure are needed here, i.e., `profile_func` may return
    dummy values for the rest.
    '''

    # construct height layers
    # deltas = 0.0001 * np.exp(np.arange(922) / 100.)
    # atm profiles only up to 80 km...
    deltas = 0.0001 * np.exp(np.arange(899) / 100.)
    heights = np.cumsum(deltas)

    # radius calculation
    # TODO: do we need to account for non-spherical Earth?
    # probably not - some tests suggest that the relative error is < 1e-6
    earth_radius = 6371. + obs_alt / 1000.
    radii = earth_radius + heights  # distance Earth-center to layers

    (
        temperature,
        pressure,
        rho_water,
        pressure_water,
        ref_index,
        humidity_water,
        humidity_ice
        ) = profile_func(apu.Quantity(heights, apu.km))
    # handle units
    temperature = temperature.to(apu.K).value
    pressure = pressure.to(apu.hPa).value
    rho_water = rho_water.to(apu.g / apu.m ** 3).value
    pressure_water = pressure_water.to(apu.hPa).value
    ref_index = ref_index.to(cnv.dimless).value
    humidity_water = humidity_water.to(apu.percent).value
    humidity_ice = humidity_ice.to(apu.percent).value

    def fix_arg(arg):
        '''
        Ensure argument is in [-1., +1.] for arcsin, arccos functions.
        '''

        if arg < -1.:
            return -1.
        elif arg > 1.:
            return 1.
        else:
            return arg

    # calculate layer path lengths (Equation 17 to 19)
    # all angles in rad
    beta_n = beta_0 = np.radians(90. - elev)  # initial value

    # we will store a_n, gamma_n, and temperature for each layer, to allow
    # Tebb calculation
    path_params = []
    # angle of the normal vector (r_n) at current layer w.r.t. zenith (r_1):
    delta_n = 0
    path_length = 0

    # TODO: this is certainly a case for cython
    for i in range(len(heights) - 1):

        r_n = radii[i]
        d_n = deltas[i]
        a_n = -r_n * np.cos(beta_n) + 0.5 * np.sqrt(
            4 * r_n ** 2 * np.cos(beta_n) ** 2 + 8 * r_n * d_n + 4 * d_n ** 2
            )
        alpha_n = np.pi - np.arccos(fix_arg(
            (-a_n ** 2 - 2 * r_n * d_n - d_n ** 2) / 2. / a_n / (r_n + d_n)
            ))
        delta_n += beta_n - alpha_n
        beta_n = np.arcsin(
            fix_arg(ref_index[i] / ref_index[i + 1] * np.sin(alpha_n))
            )

        h_n = 0.5 * (heights[i] + heights[i + 1])
        press_n = 0.5 * (pressure[i] + pressure[i + 1])
        press_w_n = 0.5 * (pressure_water[i] + pressure_water[i + 1])
        temp_n = 0.5 * (temperature[i] + temperature[i + 1])

        path_length += a_n
        if path_length > max_path_length:
            break

        path_params.append((
            press_n, press_w_n, temp_n,
            a_n, r_n, alpha_n, delta_n, beta_n, h_n
            ))

    refraction = - np.degrees(beta_n + delta_n - beta_0)

    return path_params, refraction


@utils.ranged_quantity_input(
    freq_grid=(1.e-30, 1000, apu.GHz),
    elevation=(-90, 90, apu.deg),
    obs_alt=(1.e-30, None, apu.m),
    t_bg=(1.e-30, None, apu.K),
    max_path_length=(1.e-30, None, apu.km),
    strip_input_units=True, output_unit=(cnv.dB, apu.deg, apu.K)
    )
def atten_slant_annex1(
        freq_grid, elevation, obs_alt, profile_func,
        t_bg=2.73 * apu.K, max_path_length=1000. * apu.km
        ):
    '''
    Path attenuation for a slant path through full atmosphere according to
    `ITU-R P.676-10 <https://www.itu.int/rec/R-REC-P.676-10-201309-S/en>`_
    Eq (17-20).

    Parameters
    ----------
    freq_grid : `~astropy.units.Quantity`
        Frequencies at which to calculate line-width shapes [GHz]
    elev : `~astropy.units.Quantity`, scalar
        (Apparent) elevation of source as seen from observer [deg]
    obs_alt : `~astropy.units.Quantity`, scalar
        Height of observer above sea-level [m]
    profile_func : func
        A height profile function having the same signature as
        `~pycraf.atm.profile_standard`
    t_bg : `~astropy.units.Quantity`, scalar, optional
        Background temperature, i.e. temperature just after the outermost
        layer (default: 2.73 K)

        This is needed for accurate `t_ebb` calculation, usually this is the
        temperature of the CMB (if Earth-Space path), but at lower
        frequencies, Galactic foreground contribution might play a role.
    max_path_length : `~astropy.units.Quantity`, scalar
        Maximal length of path before stopping iteration [km]
        (default: 1000 km; useful for terrestrial paths)

    Returns
    -------
    total_atten : `~astropy.units.Quantity`
        Total attenuation along path [dB]
    Refraction : `~astropy.units.Quantity`
        Offset with respect to a hypothetical straight path, i.e., the
        correction between real and apparent source elevation [deg]
    t_ebb (K) : `~astropy.units.Quantity`
        Equivalent black body temperature of the atmosphere (accounting
        for any outside contribution, e.g., from CMB) [K]

    Notes
    -----
    Although the `profile_func` must have the same signature as
    `~pycraf.atm.profile_standard`, which is one of the standardized
    atmospheric height profiles, only temperature, total pressure and
    water vapor pressure are needed here, i.e., `profile_func` may return
    dummy values for the rest.
    '''

    if not isinstance(elevation, numbers.Real):
        raise TypeError('elevation must be a scalar float')
    if not isinstance(obs_alt, numbers.Real):
        raise TypeError('obs_alt must be a scalar float')
    if not isinstance(t_bg, numbers.Real):
        raise TypeError('t_bg must be a scalar float')
    if not isinstance(max_path_length, numbers.Real):
        raise TypeError('max_path_length must be a scalar float')

    freq_grid = np.atleast_1d(freq_grid)
    _freq = freq_grid
    _elev = elevation
    _alt = obs_alt
    _t_bg = t_bg
    _max_plen = max_path_length

    total_atten_db = np.zeros(freq_grid.shape, dtype=np.float64)
    tebb = np.ones(freq_grid.shape, dtype=np.float64) * _t_bg

    path_params, refraction = _prepare_path(
        _elev, _alt, profile_func, max_path_length=_max_plen
        )

    # do backward raytracing (to allow tebb calculation)

    for press_n, press_w_n, temp_n, a_n, _, _, _, _, _ in path_params[::-1]:

        atten_dry, atten_wet = _atten_specific_annex1(
            _freq, press_n, press_w_n, temp_n
        )
        gamma_n = atten_dry + atten_wet
        total_atten_db += gamma_n * a_n

        # need to calculate (linear) atten per layer for tebb
        gamma_n_lin = 10 ** (-gamma_n * a_n / 10.)
        tebb *= gamma_n_lin
        tebb += (1. - gamma_n_lin) * temp_n

    return total_atten_db, refraction, tebb


def _phi_helper(r_p, r_t, args):
    '''
    Helper function according to `ITU-R P.676-10
    <https://www.itu.int/rec/R-REC-P.676-10-201309-S/en>`_ Eq (22u).
    '''

    phi0, a, b, c, d = args
    return phi0 * r_p ** a * r_t ** b * np.exp(
        c * (1. - r_p) + d * (1. - r_t)
        )


_HELPER_PARAMS = {
    'xi1': (1., 0.0717, -1.8132, 0.0156, -1.6515),
    'xi2': (1., 0.5146, -4.6368, -0.1921, -5.7416),
    'xi3': (1., 0.3414, -6.5851, 0.2130, -8.5854),
    'xi4': (1., -0.0112, 0.0092, -0.1033, -0.0009),
    'xi5': (1., 0.2705, -2.7192, -0.3016, -4.1033),
    'xi6': (1., 0.2445, -5.9191, 0.0422, -8.0719),
    'xi7': (1., -0.1833, 6.5589, -0.2402, 6.131),
    'gamma54': (2.192, 1.8286, -1.9487, 0.4051, -2.8509),
    'gamma58': (12.59, 1.0045, 3.5610, 0.1588, 1.2834),
    'gamma60': (15., 0.9003, 4.1335, 0.0427, 1.6088),
    'gamma62': (14.28, 0.9886, 3.4176, 0.1827, 1.3429),
    'gamma64': (6.819, 1.4320, 0.6258, 0.3177, -0.5914),
    'gamma66': (1.908, 2.0717, -4.1404, 0.4910, -4.8718),
    'delta': (-0.00306, 3.211, -14.94, 1.583, -16.37),
    }


_helper_funcs = dict(
    (k, partial(_phi_helper, args=v))
    for k, v in _HELPER_PARAMS.items()
    )


def _atten_specific_annex2(freq_grid, press, rho_w, temp):

    freq_grid = np.atleast_1d(freq_grid)

    if not isinstance(press, numbers.Real):
        raise TypeError('press must be a scalar float')
    if not isinstance(rho_w, numbers.Real):
        raise TypeError('rho_w must be a scalar float')
    if not isinstance(temp, numbers.Real):
        raise TypeError('temp must be a scalar float')

    _freq = freq_grid
    _press = press
    _rho_w = rho_w
    _temp = temp

    r_p = _press / 1013.
    r_t = 288. / (_temp - 0.15)

    atten_dry = np.empty_like(_freq)
    atten_wet = np.zeros_like(_freq)

    h = dict(
        (k, func(r_p, r_t))
        for k, func in _helper_funcs.items()
        )

    # calculate dry attenuation, depending on frequency
    mask = _freq <= 54
    f = _freq[mask]
    atten_dry[mask] = f ** 2 * r_p ** 2 * 1.e-3 * (
        7.2 * r_t ** 2.8 / (f ** 2 + 0.34 * r_p ** 2 * r_t ** 1.6) +
        0.62 * h['xi3'] / ((54. - f) ** (1.16 * h['xi1']) + 0.83 * h['xi2'])
        )

    mask = (_freq > 54) & (_freq <= 60)
    f = _freq[mask]
    atten_dry[mask] = np.exp(
        np.log(h['gamma54']) / 24. * (f - 58.) * (f - 60.) -
        np.log(h['gamma58']) / 8. * (f - 54.) * (f - 60.) +
        np.log(h['gamma60']) / 12. * (f - 54.) * (f - 58.)
        )

    mask = (_freq > 60) & (_freq <= 62)
    f = _freq[mask]
    atten_dry[mask] = (
        h['gamma60'] + (h['gamma62'] - h['gamma60']) * (f - 60.) / 2.
        )

    mask = (_freq > 62) & (_freq <= 66)
    f = _freq[mask]
    atten_dry[mask] = np.exp(
        np.log(h['gamma62']) / 8. * (f - 64.) * (f - 66.) -
        np.log(h['gamma64']) / 4. * (f - 62.) * (f - 66.) +
        np.log(h['gamma66']) / 8. * (f - 62.) * (f - 64.)
        )

    mask = (_freq > 66) & (_freq <= 120)
    f = _freq[mask]
    atten_dry[mask] = f ** 2 * r_p ** 2 * 1.e-3 * (
        3.02e-4 * r_t ** 3.5 +
        0.283 * r_t ** 3.8 / (
            (f - 118.75) ** 2 + 2.91 * r_p ** 2 * r_t ** 1.6
            ) +
        0.502 * h['xi6'] * (1. - 0.0163 * h['xi7'] * (f - 66.)) / (
            (f - 66.) ** (1.4346 * h['xi4']) + 1.15 * h['xi5']
            )
        )

    mask = (_freq > 120) & (_freq <= 350)
    f = _freq[mask]
    atten_dry[mask] = h['delta'] + f ** 2 * r_p ** 3.5 * 1.e-3 * (
        3.02e-4 / (1. + 1.9e-5 * f ** 1.5) +
        0.283 * r_t ** 0.3 / (
            (f - 118.75) ** 2 + 2.91 * r_p ** 2 * r_t ** 1.6
            )
        )

    # calculate wet attenuation, depending on frequency

    eta_1 = 0.955 * r_p * r_t ** 0.68 + 0.006 * _rho_w
    eta_2 = 0.735 * r_p * r_t ** 0.5 + 0.0353 * r_t ** 4 * _rho_w

    f = _freq

    def g(f, f_i):

        return 1. + ((f - f_i) / (f + f_i)) ** 2

    def _helper(a, eta, b, c, d, do_g):

        return (
            a * eta * np.exp(b * (1 - r_t)) /
            ((f - c) ** 2 + d * eta ** 2) *
            (g(f, int(c + 0.5)) if do_g else 1.)
            )

    for args in [
            (3.98, eta_1, 2.23, 22.235, 9.42, True),
            (11.96, eta_1, 0.7, 183.31, 11.14, False),
            (0.081, eta_1, 6.44, 321.226, 6.29, False),
            (3.66, eta_1, 1.6, 325.153, 9.22, False),
            (25.37, eta_1, 1.09, 380, 0, False),
            (17.4, eta_1, 1.46, 448, 0, False),
            (844.6, eta_1, 0.17, 557, 0, True),
            (290., eta_1, 0.41, 752, 0, True),
            (83328., eta_2, 0.99, 1780, 0, True),
            ]:

        atten_wet += _helper(*args)

    atten_wet *= f ** 2 * r_t ** 2.5 * _rho_w * 1.e-4

    return atten_dry, atten_wet


@utils.ranged_quantity_input(
    freq_grid=(1.e-30, 350., apu.GHz),
    press=(1.e-30, None, apu.hPa),
    rho_w=(1.e-30, None, apu.g / apu.m ** 3),
    temp=(1.e-30, None, apu.K),
    strip_input_units=True, output_unit=(cnv.dB / apu.km, cnv.dB / apu.km)
    )
def atten_specific_annex2(freq_grid, press, rho_w, temp):
    '''
    Specific (one layer) atmospheric attenuation based on a simplified
    algorithm according to `ITU-R P.676-10
    <https://www.itu.int/rec/R-REC-P.676-10-201309-S/en>`_, Annex 2.1.

    Parameters
    ----------
    freq_grid : `~astropy.units.Quantity`
        Frequencies at which to calculate line-width shapes [GHz]
    press : `~astropy.units.Quantity`
        Total air pressure (dry + wet) [hPa]
    rho_w : `~astropy.units.Quantity`
        Water vapor density [g / m^3]
    temp : `~astropy.units.Quantity`
        Temperature [K]

    Returns
    -------
    atten_dry : `~astropy.units.Quantity`
        Dry-air specific attenuation [dB / km]
    atten_wet : `~astropy.units.Quantity`
        Wet-air specific attenuation [dB / km]

    Notes
    -----
    In contrast to Annex 1, the method in Annex 2 is only valid below 350 GHz.
    '''

    return _atten_specific_annex2(
        freq_grid, press, rho_w, temp
        )


@utils.ranged_quantity_input(
    freq_grid=(1.e-30, 350., apu.GHz),
    press=(1.e-30, None, apu.hPa),
    strip_input_units=True, output_unit=apu.km
    )
def equivalent_height_dry(freq_grid, press):
    '''
    Equivalent height for dry air according to `ITU-R P.676-10
    <https://www.itu.int/rec/R-REC-P.676-10-201309-S/en>`_, Annex 2.2.

    Parameters
    ----------
    freq_grid : `~astropy.units.Quantity`
        Frequenciesat which to calculate line-width shapes [GHz]
    press : `~astropy.units.Quantity`
        Total air pressure (dry + wet) [hPa]

    Returns
    -------
    h_dry : `~astropy.units.Quantity`
        Equivalent height for dry air [km]
    '''

    r_p = press / 1013.

    f = np.atleast_1d(freq_grid).astype(dtype=np.float64, copy=False)

    t_1 = 4.64 / (1. + 0.066 * r_p ** -2.3) * np.exp(
        - ((f - 59.7) / (2.87 + 12.4 * np.exp(-7.9 * r_p))) ** 2
        )

    t_2 = 0.14 * np.exp(2.12 * r_p) / (
        (f - 118.75) ** 2 + 0.031 * np.exp(2.2 * r_p)
        )

    t_3 = 0.0114 * f / (1. + 0.14 * r_p ** -2.6) * (
        (-0.0247 + 0.0001 * f + 1.61e-6 * f ** 2) /
        (1. - 0.0169 * f + 4.1e-5 * f ** 2 + 3.2e-7 * f ** 3)
        )

    h_0 = 6.1 * (1. + t_1 + t_2 + t_3) / (1. + 0.17 * r_p ** -1.1)

    h_0[(h_0 > 10.8 * r_p ** 0.3) & (f < 70.)] = 10.8 * r_p ** 0.3

    return h_0.squeeze()


@utils.ranged_quantity_input(
    freq_grid=(1.e-30, 350., apu.GHz),
    press=(1.e-30, None, apu.hPa),
    strip_input_units=True, output_unit=apu.km
    )
def equivalent_height_wet(freq_grid, press):
    '''
    Equivalent height for wet air according to `ITU-R P.676-10
    <https://www.itu.int/rec/R-REC-P.676-10-201309-S/en>`_, Annex 2.2.

    Parameters
    ----------
    freq_grid : `~astropy.units.Quantity`
        Frequenciesat which to calculate line-width shapes [GHz]
    press : `~astropy.units.Quantity`
        Total air pressure (dry + wet) [hPa]

    Returns
    -------
    h_wet : `~astropy.units.Quantity`
        Equivalent height for wet air [km]
    '''

    r_p = press / 1013.

    f = np.atleast_1d(freq_grid).astype(dtype=np.float64, copy=False)

    s_w = 1.013 / (1. + np.exp(-8.6 * (r_p - 0.57)))

    def _helper(a, b, c):
        return a * s_w / ((f - b) ** 2 + c * s_w)

    h_w = 1.66 * (
        1. +
        _helper(1.39, 22.235, 2.56) +
        _helper(3.37, 183.31, 4.69) +
        _helper(1.58, 325.1, 2.89)
        )

    return h_w.squeeze()


@utils.ranged_quantity_input(
    atten_dry=(1.e-30, None, cnv.dB / apu.km),
    atten_wet=(1.e-30, None, cnv.dB / apu.km),
    h_dry=(1.e-30, None, apu.km),
    h_wet=(1.e-30, None, apu.km),
    elev=(-90, 90, apu.deg),
    strip_input_units=True, output_unit=cnv.dB
    )
def atten_slant_annex2(atten_dry, atten_wet, h_dry, h_wet, elev):
    '''
    Simple path attenuation for slant path through full atmosphere according to
    `ITU-R P.676-10 <https://www.itu.int/rec/R-REC-P.676-10-201309-S/en>`_,
    Eq (28).

    Parameters
    ----------
    atten_dry : `~astropy.units.Quantity`
        Specific attenuation for dry air [dB / km]
    atten_wet : `~astropy.units.Quantity`
        Specific attenuation for wet air [dB / km]
    h_dry : `~astropy.units.Quantity`
        Equivalent height for dry air [km]
    h_wet : `~astropy.units.Quantity`
        Equivalent height for wet air [km]
    elev : `~astropy.units.Quantity`
        (Apparent) elevation of source as seen from observer [deg]

    Returns
    -------
    total_atten : `~astropy.units.Quantity`
        Total attenuation along path [dB]

    Notes
    -----
    You can use the helper functions `~pycraf.atm.equivalent_height_dry` and
    `~pycraf.atm.equivalent_height_wet` to infer the equivalent heights from
    the total (wet+dry) air pressure.
    '''

    AM = 1. / np.sin(np.radians(elev))

    return AM * (atten_dry * h_dry + atten_wet * h_wet)
