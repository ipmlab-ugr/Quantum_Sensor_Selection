# -*- coding: utf-8 -*-
"""Central configuration for the sensor-selection case studies.

To switch case study, change only ``CASE_STUDY`` below.
All four scripts import the selected configuration from this file.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


# =============================================================================
# SELECT THE CASE STUDY
# =============================================================================

CASE_STUDY = 1  # Use 1 for Case Study I or 2 for Case Study II.


# =============================================================================
# CASE-STUDY PARAMETERS
# =============================================================================

CASE_STUDY_CONFIGS: Dict[int, Dict[str, Any]] = {
    1: {
        "sampling_rate_hz": 1000.0,
        "number_of_sensors_to_select": 7,
        "symmetry_coordinate": "y",
        "symmetry_axis_value": None,
        "geometry_length_scale": 1.5,
        "geometry_weight": 1.0,
        "symmetry_weight": 1.0,
        "qubo_lambda": 5.0,
        "qubo_alpha": 1.0,
    },
    2: {
        # Verify these values against the final Case Study II dataset/paper.
        "sampling_rate_hz": 1000.0,
        "number_of_sensors_to_select": 3,
        "symmetry_coordinate": "x",
        "symmetry_axis_value": None,
        "geometry_length_scale": 20.0,
        "geometry_weight": 1.0,
        "symmetry_weight": 1.0,
        "qubo_lambda": 5.0,
        "qubo_alpha": 1.0,
    },
}


def get_case_study_config(case_study: int = CASE_STUDY) -> Dict[str, Any]:
    """Return a copy of the configuration for the requested case study."""
    if case_study not in CASE_STUDY_CONFIGS:
        raise ValueError(
            f"Unknown case study {case_study}. Valid values are: "
            f"{sorted(CASE_STUDY_CONFIGS)}."
        )
    return deepcopy(CASE_STUDY_CONFIGS[case_study])
