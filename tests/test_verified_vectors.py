"""
Regression against eight fixed Virtuoso stimulus vectors.

Disabled until software dequant units ↔ measured V_OA (mV) mapping is agreed.
See background_info/HWA_CIM_Required_Changes.md and AgDR-0003.
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="Virtuoso V_OA ↔ c2c_mac unit mapping not finalized; do not ship guessed mV scale"
)


def test_8_verified_vectors_within_calibration_uncertainty():
    from hwa_cim.evaluate import parity_c2c_against_verified_vectors  # noqa: F401 — future API

    raise NotImplementedError("Enable when parity_c2c_against_verified_vectors is implemented")
