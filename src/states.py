"""The activity states, in one place so every module uses the same names."""

from enum import Enum


class State(str, Enum):
    LYING_IN_BED = "LYING_IN_BED"
    SITTING_ON_BED = "SITTING_ON_BED"
    SITTING_OUTSIDE_BED = "SITTING_OUTSIDE_BED"
    STANDING = "STANDING"
    WALKING = "WALKING"
    OUT_OF_BED = "OUT_OF_BED"            # not visible after a confirmed bed exit (set by the state machine)
    LYING_OUTSIDE_BED = "LYING_OUTSIDE_BED"  # extra state: possible fall
    UNKNOWN = "UNKNOWN"


IN_BED_STATES = {State.LYING_IN_BED, State.SITTING_ON_BED}
