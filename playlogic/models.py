from pydantic import BaseModel

class Observation(BaseModel):
    text: str    # JSON string: {"player_id": "local obs text", ...}

class Action(BaseModel):
    text: str    # JSON string: {"player_id": "action_string", ...}