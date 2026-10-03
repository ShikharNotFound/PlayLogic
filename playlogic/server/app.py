from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import json, os
import numpy as np
from starlette.middleware.base import BaseHTTPMiddleware
from playlogic.server.environment import MultiAgentFootball
from playlogic.models import Action

class RemoveCSPMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if "Content-Security-Policy" in response.headers:
            del response.headers["Content-Security-Policy"]
        return response

env = MultiAgentFootball()
app = FastAPI()
app.add_middleware(RemoveCSPMiddleware)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def convert_state(state):
    if isinstance(state, dict):
        return {k: convert_state(v) for k, v in state.items()}
    elif isinstance(state, list):
        return [convert_state(v) for v in state]
    elif isinstance(state, np.ndarray):
        return state.tolist()
    elif isinstance(state, np.generic):
        return state.item()
    else:
        return state

app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")
app.mount("/web/static", StaticFiles(directory=BASE_DIR), name="web-static")

@app.get("/", response_class=HTMLResponse)
@app.get("/web", response_class=HTMLResponse)
@app.get("/web/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(BASE_DIR, "index.html")
    with open(html_path, "r") as f:
        return f.read()

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/reset")
@app.post("/web/reset")
async def reset():
    obs = env.reset()
    state = convert_state(env.state())
    return {"observation": obs.text, "state": state}

@app.post("/step")
@app.post("/web/step")
async def step(request: Request):
    data = await request.json()
    action_text = data["action"]
    action = Action(text=action_text)
    obs, rewards, done, info = env.step(action)
    state = convert_state(env.state())
    return {"observation": obs.text, "reward": rewards, "done": done, "info": info, "state": state}
