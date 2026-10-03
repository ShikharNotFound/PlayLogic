import json
from playlogic.server.environment import MultiAgentFootball
from playlogic.models import Action

env = MultiAgentFootball()
obs = env.reset()
print("Initial observations (first 2 players):", list(json.loads(obs.text).items())[:2])

# Create random actions for all 22 players
actions = {}
for pid in range(22):
    actions[pid] = "MOVE 1.0 0.0"
joint_action = Action(text=json.dumps(actions))
obs, rewards, done, info = env.step(joint_action)
print("Rewards:", rewards)
print("Done:", done)