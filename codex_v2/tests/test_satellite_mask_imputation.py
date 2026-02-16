from __future__ import annotations

import numpy as np
import pandas as pd

from codex_v2.src.data.satellite_alignment_v2 import impute_satellite_by_state_mean


def test_imputed_active_steps_marked_valid() -> None:
    # 4 districts, 3 steps, 1 feature.
    sat_x = np.array(
        [
            [[10.0], [20.0], [0.0]],  # state A, valid
            [[0.0], [0.0], [0.0]],    # state A, missing active steps
            [[5.0], [0.0], [0.0]],    # state B, partially valid
            [[0.0], [0.0], [0.0]],    # state B, missing where state mean available only at t0
        ],
        dtype=np.float32,
    )
    sat_mask = np.array(
        [
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    district_df = pd.DataFrame(
        {
            "state_name": ["A", "A", "B", "B"],
            "district_id": ["d1", "d2", "d3", "d4"],
        }
    )
    active_steps = np.array([1.0, 1.0, 0.0], dtype=np.float32)

    out_x, out_m = impute_satellite_by_state_mean(
        sat_x=sat_x,
        sat_mask=sat_mask,
        district_df=district_df,
        active_steps=active_steps,
        set_mask_valid=True,
    )

    # State A district 2 should be imputed at active steps and mask set to valid.
    assert out_m[1, 0] == 1.0
    assert out_m[1, 1] == 1.0
    assert out_x[1, 0, 0] == 10.0
    assert out_x[1, 1, 0] == 20.0

    # Inactive step remains untouched.
    assert out_m[1, 2] == 0.0

    # State B at step=1 has no valid source in the state, so it stays missing.
    assert out_m[3, 1] == 0.0
