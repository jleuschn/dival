# -*- coding: utf-8 -*-
"""
Provides physical and numerical constants.
"""

MU_WATER = 20  # M^-1
"""
Linear attenuation of water in SI unit ``m^-1``.
Source:
`nist.gov <https://www.nist.gov/pml/x-ray-mass-attenuation-coefficients>`_.
"""

MU_AIR = 0.02  # M^-1
"""
Linear attenuation of air in SI unit ``m^-1``.
Source:
`nist.gov <https://www.nist.gov/pml/x-ray-mass-attenuation-coefficients>`_.
"""

MU_MAX = 3071 * (MU_WATER - MU_AIR) / 1000 + MU_WATER
"""
Maximum linear attenuation representable with standard 12 bit HU value format.
The value is in SI unit ``m^-1``.
"""
