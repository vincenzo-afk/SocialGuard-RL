from pydantic import BaseModel
from typing import Literal, Optional

class ObservationModel(BaseModel):
    observation: list[float]
    reward: float
    terminated: bool
    truncated: bool
    info: dict

class ActionModel(BaseModel):
    action_id: Literal[0, 1, 2, 3, 4]
    action_name: str

class RewardModel(BaseModel):
    total: float
    correctness: float
    fp_cost: float
    collateral_damage: float
    speed_bonus: float
    escalation_penalty: float

class StepRequest(BaseModel):
    task: str
    action: int

class ResetRequest(BaseModel):
    task: str
    seed: Optional[int] = None
