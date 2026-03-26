
import random, uuid
from pydantic import BaseModel
from typing import Optional

VALID_ACTIONS = ["retry","safe_drop","locker_drop","neighbor_handoff","contact_support","return"]

TASKS = {
    "task_1": {
        "name": "Standard Delivery",
        "difficulty": "easy",
        "description": "Deliver a low-value package. Locker is nearby. Good weather.",
        "scenario": {
            "otp_attempts": 0.2,
            "recipient_available": 0.0,
            "alternate_available": 0.8,
            "package_value": 0.2,
            "package_fragile": 0.0,
            "time_remaining": 0.9,
            "weather_risk": 0.1,
            "location_safety": 0.9,
            "locker_nearby": 1.0,
        }
    },
    "task_2": {
        "name": "Fragile Package in Bad Weather",
        "difficulty": "medium",
        "description": "Fragile package, heavy rain, no locker nearby.",
        "scenario": {
            "otp_attempts": 0.3,
            "recipient_available": 0.0,
            "alternate_available": 0.4,
            "package_value": 0.6,
            "package_fragile": 1.0,
            "time_remaining": 0.5,
            "weather_risk": 0.9,
            "location_safety": 0.5,
            "locker_nearby": 0.0,
        }
    },
    "task_3": {
        "name": "High-Value Critical Delivery",
        "difficulty": "hard",
        "description": "Expensive item, unsafe area, no locker, no neighbor, time running out.",
        "scenario": {
            "otp_attempts": 0.7,
            "recipient_available": 0.0,
            "alternate_available": 0.1,
            "package_value": 0.95,
            "package_fragile": 1.0,
            "time_remaining": 0.15,
            "weather_risk": 0.7,
            "location_safety": 0.2,
            "locker_nearby": 0.0,
        }
    },
}

class DeliveryState(BaseModel):
    episode_id: Optional[str] = None
    step_count: int = 0
    difficulty: str = "easy"
    outcome: str = "in_progress"
    task_id: str = "task_1"

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

class DeliveryDecisionEnvironment:
    def __init__(self, task_id: str = "task_1"):
        self._task_id = task_id
        self._state = DeliveryState(task_id=task_id)
        self._scenario = {}
        self._rng = random.Random()
        self._total_reward = 0.0

    def reset(self) -> DeliveryObservation:
        task = TASKS[self._task_id]
        self._state = DeliveryState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            difficulty=task["difficulty"],
            outcome="in_progress",
            task_id=self._task_id
        )
        self._scenario = dict(task["scenario"])
        self._total_reward = 0.0
        return self._obs(False, None, f"Task: {task['name']}. {task['description']}", 0.0)

    def step(self, action_type: str) -> DeliveryObservation:
        self._state.step_count += 1
        act = action_type.lower().strip()

        if act not in VALID_ACTIONS:
            return self._obs(False, -1.0, f"Invalid. Choose: {VALID_ACTIONS}", 0.0)

        outcome, terminal, reward, msg = self._resolve(act)
        self._state.outcome = outcome
        self._total_reward += reward
        self._scenario = self._transition(self._scenario, act)

        if self._state.step_count >= 10 and not terminal:
            terminal = True
            reward = -15.0
            msg = "Max steps reached. Package returned."

        score = self.grade()
        return self._obs(terminal, reward, msg, score)

    def grade(self) -> float:
        outcome = self._state.outcome
        steps = self._state.step_count
        task_id = self._task_id
        if outcome == "success":
            speed_bonus = max(0.0, (5 - steps) * 0.05)
            base = {"task_1": 0.8, "task_2": 0.85, "task_3": 0.9}[task_id]
            return min(1.0, round(base + speed_bonus, 2))
        elif outcome == "failed":
            return 0.0
        else:
            return round(max(0.0, 0.3 - self._state.step_count * 0.03), 2)

    @property
    def state(self) -> DeliveryState:
        return self._state

    def get_tasks(self):
        return [{"task_id": k, "name": v["name"], "difficulty": v["difficulty"],
                 "description": v["description"]} for k, v in TASKS.items()]

    def _transition(self, s, action):
        s = dict(s)
        s["time_remaining"] = max(0.0, s["time_remaining"] - 0.15)
        if action == "retry":
            s["otp_attempts"] = min(1.0, s["otp_attempts"] + 0.25)
        elif action == "contact_support":
            s["recipient_available"] = min(1.0, s["recipient_available"] + 0.3)
        return s

    def _resolve(self, action):
        s = self._scenario
        step = self._state.step_count
        bonus = 2.0 if step <= 3 else 0.0
        if action == "retry":
            if s["otp_attempts"] >= 0.9:
                return "in_progress", False, -3.0, "OTP maxed out."
            p = 0.3 + (1 - s["otp_attempts"]) * 0.4
            if self._rng.random() < p:
                return "success", True, 10.0+bonus, "Delivered successfully!"
            return "in_progress", False, -0.5, "No answer. Try another option."
        elif action == "safe_drop":
            ok = (s["location_safety"]>0.5 and s["weather_risk"]<0.6
                  and s["package_fragile"]<0.5 and s["package_value"]<0.7)
            penalty = (-8 if s["location_safety"]<0.4 else 0) +                       (-6 if s["package_fragile"]>0.5 else 0) +                       (-5 if s["weather_risk"]>0.6 else 0)
            if ok:
                return "success", True, 10.0+bonus, "Package left safely!"
            return "in_progress", False, -0.5+penalty, "Unsafe to drop here."
        elif action == "locker_drop":
            if s["locker_nearby"]>0.5:
                return "success", True, 6.0+bonus, "Secured in locker!"
            return "in_progress", False, -0.5, "No locker nearby."
        elif action == "neighbor_handoff":
            if s["alternate_available"]>0.5:
                return "success", True, 7.0+bonus, "Neighbor accepted!"
            return "in_progress", False, -0.5, "No neighbor available."
        elif action == "contact_support":
            return "in_progress", False, -0.5, "Support contacted."
        elif action == "return":
            return "failed", True, -15.0, "Returned to warehouse."
        return "in_progress", False, -0.5, "Unknown action."

    def _obs(self, done, reward, message, score):
        s = self._scenario
        return DeliveryObservation(
            done=done, reward=reward, message=message,
            score=score, task_id=self._task_id,
            otp_attempts=s.get("otp_attempts",0),
            recipient_available=s.get("recipient_available",0),
            alternate_available=s.get("alternate_available",0),
            package_value=s.get("package_value",0),
            package_fragile=s.get("package_fragile",0),
            time_remaining=s.get("time_remaining",1),
            weather_risk=s.get("weather_risk",0),
            location_safety=s.get("location_safety",1),
            locker_nearby=s.get("locker_nearby",0),
        )
