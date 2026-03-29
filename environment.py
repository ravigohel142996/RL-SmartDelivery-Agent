import random, uuid
from models import DeliveryObservation, DeliveryState

TASKS = {
    "task_1": {
        "name": "Standard Delivery",
        "difficulty": "easy",
        "description": "Low-value package. Locker nearby, good weather.",
        "scenario": {"otp_attempts":0.3,"recipient_available":0.0,
            "alternate_available":0.8,"package_value":0.2,"package_fragile":0.0,
            "time_remaining":0.9,"weather_risk":0.1,"location_safety":0.9,"locker_nearby":1.0}
    },
    "task_2": {
        "name": "Fragile Package in Bad Weather",
        "difficulty": "medium",
        "description": "Fragile item, bad weather, no locker.",
        "scenario": {"otp_attempts":0.5,"recipient_available":0.0,
            "alternate_available":0.6,"package_value":0.5,"package_fragile":1.0,
            "time_remaining":0.5,"weather_risk":0.8,"location_safety":0.4,"locker_nearby":0.0}
    },
    "task_3": {
        "name": "High-Value Critical Delivery",
        "difficulty": "hard",
        "description": "High-value, unsafe area, time running out.",
        "scenario": {"otp_attempts":0.8,"recipient_available":0.0,
            "alternate_available":0.2,"package_value":0.95,"package_fragile":1.0,
            "time_remaining":0.1,"weather_risk":0.7,"location_safety":0.2,"locker_nearby":0.0}
    }
}

VALID_ACTIONS = ["retry","safe_drop","locker_drop",
                 "neighbor_handoff","contact_support","return"]

class DeliveryDecisionEnvironment:
    def __init__(self, task_id="task_1"):
        self.task_id = task_id
        self._state  = DeliveryState(task_id=task_id)
        self._scene  = {}
        self._score  = 0.0

    def reset(self):
        self._state = DeliveryState(episode_id=str(uuid.uuid4()),
            step_count=0, task_id=self.task_id, outcome="in_progress")
        self._scene = dict(TASKS[self.task_id]["scenario"])
        self._score = 0.0
        return self._obs(False, None, "Package arrived. Recipient unavailable. What do you do?")

    def step(self, action_type):
        self._state.step_count += 1
        act = action_type.lower().strip()
        if act not in VALID_ACTIONS:
            return self._obs(False, -1.0, f"Invalid. Choose: {VALID_ACTIONS}")
        outcome, terminal, reward, msg = self._resolve(act)
        self._state.outcome = outcome
        self._scene["time_remaining"] = max(0.0, self._scene["time_remaining"]-0.15)
        if act == "retry":
            self._scene["otp_attempts"] = min(1.0, self._scene["otp_attempts"]+0.25)
        elif act == "contact_support":
            self._scene["recipient_available"] = min(1.0, self._scene["recipient_available"]+0.3)
        if self._state.step_count >= 10 and not terminal:
            terminal=True; reward=-15.0
            self._state.outcome="failed"
            msg="Max steps. Returned to warehouse."
        if terminal:
            self._score = self._calc(outcome, reward)
        return self._obs(terminal, reward, msg)

    def grade(self): return round(self._score, 3)

    @property
    def state(self): return self._state

    def _resolve(self, action):
        s=self._scene; bon=2.0 if self._state.step_count<=3 else 0.0
        if action=="retry":
            if s["otp_attempts"]>=0.9: return "in_progress",False,-3.0,"OTP maxed."
            if random.random()<0.3+(1-s["otp_attempts"])*0.4:
                return "success",True,10.0+bon,"Delivered successfully!"
            return "in_progress",False,-0.5,"No answer. Try another."
        elif action=="safe_drop":
            ok=(s["location_safety"]>0.5 and s["weather_risk"]<0.6
                and s["package_fragile"]<0.5 and s["package_value"]<0.7)
            pen=(-8 if s["location_safety"]<0.4 else 0)+                (-6 if s["package_fragile"]>0.5 else 0)+                (-5 if s["weather_risk"]>0.6 else 0)
            if ok: return "success",True,10.0+bon,"Safe drop done!"
            return "in_progress",False,-0.5+pen,"Unsafe to drop here."
        elif action=="locker_drop":
            if s["locker_nearby"]>0.5: return "success",True,6.0+bon,"Secured in locker!"
            return "in_progress",False,-0.5,"No locker nearby."
        elif action=="neighbor_handoff":
            if s["alternate_available"]>0.5: return "success",True,7.0+bon,"Neighbor accepted!"
            return "in_progress",False,-0.5,"No neighbor available."
        elif action=="contact_support": return "in_progress",False,-0.5,"Support contacted."
        elif action=="return": return "failed",True,-15.0,"Returned to warehouse."
        return "in_progress",False,-0.5,"Unknown."

    def _calc(self, outcome, reward):
        if outcome=="success":
            return min(1.0, 0.7+max(0,(10-self._state.step_count)/10)*0.3)
        return 0.0 if outcome=="failed" else max(0.0,(reward+15)/30)

    def _obs(self, done, reward, message):
        s=self._scene
        return DeliveryObservation(done=done,reward=reward,message=message,
            otp_attempts=s.get("otp_attempts",0),
            recipient_available=s.get("recipient_available",0),
            alternate_available=s.get("alternate_available",0),
            package_value=s.get("package_value",0),
            package_fragile=s.get("package_fragile",0),
            time_remaining=s.get("time_remaining",1),
            weather_risk=s.get("weather_risk",0),
            location_safety=s.get("location_safety",1),
            locker_nearby=s.get("locker_nearby",0))
