"""
Typed messagaes that flow through the system
"""

import time
from dataclasses import dataclass, field, asdict
from typing import Optional, Any

@dataclass
class Observation:
    """
    A single observation from a modality, with a timestamp and optional metadata.
    """
    modality: str
    source_id : str
    payload: dict[str, Any]
    subject_id: Optional[str] = None
    t_wall: float = field(default_factory=time.time)
    t_mono: float = field(default_factory=time.monotonic)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Observation':
        return cls(**data)
    
@dataclass
class SubjectAttention:
    """
    A single observation of a subject's attention state, with a timestamp and optional metadata.
    """
    subject_id: str
    target: str
    confidence: float
    source_id: str
    t_wall: float
    gaze: dict[str, Any] = field(default_factory=dict)
    
@dataclass
class AttentionState:
    """
    The current attention state of all subjects, with a timestamp and optional metadata.
    """
    subjects: dict[str, SubjectAttention]
    t_wall: float
    joint_attention: dict[str, Any] = field(default_factory=dict)
