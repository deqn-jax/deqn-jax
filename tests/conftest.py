"""Suite-wide fixtures and settings.

64-bit precision is switched on here, before any test module is imported.
Fourteen test modules set it at import time, and several package modules
compute numerics at import (steady states, linearizations); which precision
those import-time values carry used to depend on whichever test module
pytest happened to collect first. Retiring the alphabetically first module
in September 2026 made `test_jacrev_kf_rows_match_linearization` fail at
fp32 tolerance in the full run while passing alone. One switch, one place.
"""

import jax

jax.config.update("jax_enable_x64", True)
