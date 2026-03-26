
from pydantic import BaseModel
from typing import Optional

class DeliveryAction(BaseModel):
    action_type: str

class DeliveryObservation(BaseModel):
    done: bool
    reward: Optional[float] = None
    otp_attempts: float = 0.0
    recipient_available: float = 0.0
    alternate_available: float = 0.0
    package_value: float = 0.0
    package_fragile: float = 0.0
    time_remaining: float = 1.0
    weather_risk: float = 0.0
    location_safety: float = 1.0
    locker_nearby: float = 0.0
    message: str = ""
    task_id: str = "task_1"
    score: float = 0.0

class DeliveryState(BaseModel):
    episode_id: Optional[str] = None
    step_count: int = 0
    difficulty: str = "easy"
    outcome: str = "in_progress"
    task_id: str = "task_1"
