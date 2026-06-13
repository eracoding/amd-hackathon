"""
All configuration for the attention room should be defined here.
"""

from dataclasses import dataclass, field

@dataclass
class GazeConfig:
    """
    Configuration for the gaze modality.
    """
    enabled: bool = True

    # estimator: how eye + head combine into a gaze angle in degrees
    eye_gain_deg: float      = 25.0
    eye_h_sign: float        = 1.0
    eye_v_sign: float        = 1.0
    head_forward_sign: float = -1.0
    head_yaw_sign: float     = 1.0
    head_pitch_sign: float   = 1.0

    # target classifier: coarse zones from gaze angles
    screen_yaw_deg: float  = 15.0
    screen_pitch_lo: float = -35.0
    screen_pitch_hi: float = 12.0
    away_yaw_deg: float    = 22.0
    up_pitch_deg: float    = 20.0
    blink_thresh: float    = 0.55

    # fusion
    emit_interval_s: float = 0.5
    stale_timeout_s: float = 2.0

    shared_targets: frozenset[str] = field(
        default_factory=lambda: frozenset({"left", "right", "up_away", "elsewhere"})
    )
