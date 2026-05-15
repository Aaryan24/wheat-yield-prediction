from __future__ import annotations

import datetime as dt
from typing import List


PROFILE_MANUAL = "manual"
PROFILE_FIVE_DAY_DEC1_APR30 = "five_day_dec1_apr30"

SUPPORTED_OPDATE_PROFILES = {
    PROFILE_MANUAL,
    PROFILE_FIVE_DAY_DEC1_APR30,
}


def build_five_day_dec1_apr30_labels() -> List[str]:
    labels: List[str] = []
    cur = dt.date(2001, 12, 1)
    end = dt.date(2002, 4, 30)
    while cur <= end:
        labels.append(cur.strftime("%m-%d"))
        cur += dt.timedelta(days=5)
    return labels


def opdates_for_profile(profile: str) -> List[str]:
    key = str(profile).strip().lower()
    if key == PROFILE_FIVE_DAY_DEC1_APR30:
        return build_five_day_dec1_apr30_labels()
    raise ValueError(
        f"Unknown opdate profile={profile}. Supported profiles={sorted(SUPPORTED_OPDATE_PROFILES)}"
    )
