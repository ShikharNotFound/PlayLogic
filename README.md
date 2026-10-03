---
title: Playlogic Environment Server
emoji: 🏉
colorFrom: indigo
colorTo: red
sdk: docker
pinned: false
app_port: 7860
base_path: /web
app_file: server/app.py
tags:
  - openenv
---

# ⚽ PlayLogic - Multi-Agent Football

**Train 22 AI agents (11v11) to play football using deep multi‑agent reinforcement learning.**  
Built on **OpenEnv**, with **GRPO** (TRL) training, hybrid physics, and a shaped reward function.  
Discover emergent tactics, formations, and set‑piece routines – a sandbox for MARL and football analytics.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arjune4dev/PlayLogic/blob/main/training/grpo_football.ipynb)
[![GitHub Repo](https://img.shields.io/badge/GitHub-PlayLogic-blue?logo=github)](https://github.com/arjune4dev/PlayLogic)
[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Model-playlogic--football--grpo-orange)](https://huggingface.co/ShikharNotFound/playlogic-football-grpo)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🧠 Overview

This **Hugging Face Space** runs a live 11‑vs‑11 football simulation where every player is an **AI agent** powered by a fine‑tuned language model.  
The agents decide their actions in real time using a shared policy trained with **GRPO (Group Relative Policy Optimization)** – no hard‑coded behaviours, only reinforcement learning from the game state.

Watch how the agents spontaneously learn:
- Passing triangles
- Offside traps
- Pressing triggers
- Counter‑attack patterns

---

## 🤖 How It Works

| Component | Detail |
|-----------|--------|
| **Base model** | `Qwen/Qwen2-0.5B-Instruct` (0.5B parameters) |
| **Fine‑tuning** | GRPO via `trl`, on a custom football environment |
| **Input** | Text description of the game state (positions, ball, teammates, opponents, action history) |
| **Output** | One discrete action per player, from a set of 15 football actions |
| **Simulation** | Hybrid physics engine with realistic speed, acceleration, drag, and friction |
| **Decision frequency** | ~2.4 seconds per 50‑step training iteration on a T4 GPU |

The entire project runs in a **free Google Colab** for training and in this **Hugging Face Space** for inference.

---

## 🎮 Agent Actions

Each player can select exactly one action per tick:
IDLE
MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT
SHORT_PASS, LONG_PASS, HIGH_PASS
SHOOT_LOW, SHOOT_MID, SHOOT_HIGH
TACKLE, SLIDE_TACKLE
DRIBBLE, DRIBBLE_SPRINT


Movement magnitudes are clipped to **[-8, 8]** in each direction per step.

---

## ⚡ Physics & Dynamics

The environment runs a **hybrid continuous‑step physics** model with realistic constraints.

| Parameter | Value |
|-----------|-------|
| Max speed (free) | 8.0 m/s |
| Max speed (dribble) | 6.5 m/s |
| Acceleration (free) | 5.0 m/s² |
| Acceleration (dribble) | 4.0 m/s² |
| Agility (max turn rate) | 10.0 rad/s |
| Ball mass | 0.45 kg |
| Drag coefficient | 0.25 |
| Rolling friction | 0.85 per step |
| Max ball speed | 35.0 m/s |
| Pass speed | 22.0 m/s |
| Shot speed | 28.0 m/s |
| Tackle distance | 1.0 m |
| Tackle angle threshold | 60° |
| Tackle success (base) | 70% |

Pass and shot accuracy degrade with defender pressure and distance:

accuracy = base * (1 - 0.3 * pressure/3) * (1 - 0.005 * distance)

---

## 🎯 Reward Function

The agents are trained to maximise a **shaped reward** at each step:

R_total = R_goal + R_pass + R_intercept + R_turnover + R_formation + R_spacing


| Component | Description | Value |
|-----------|-------------|-------|
| **R_goal** | Team scored / conceded | +5.0 / -5.0 |
| **R_pass** | Successful / intercepted pass | +0.5 / -0.3 |
| **R_turnover** | Lost possession | -0.1 |
| **R_formation** | Positional adherence | 0.02 * Σ(1 - tanh(‖player_pos - ideal_pos‖)) |
| **R_spacing** | Teammates too close (<2m for >10 steps) | -0.01 |

This encourages attacking play, structured formations, and proper spacing – hallmarks of real football tactics.

---

## 🚀 Training

You can train your own agents from scratch using the provided notebook.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arjune4dev/PlayLogic/blob/main/training/grpo_football.ipynb)

**Quick start (memory‑safe):**
```bash
!python training/train_grpo.py \
  --model-name Qwen/Qwen2-0.5B-Instruct \
  --output-dir ./grpo_football_final \
  --num-resets 8 \
  --max-prompts 64 \
  --max-steps 25 \
  --num-generations 2 \
  --per-device-train-batch-size 1 \
  --push-to-hub \
  --hub-model-id YOUR_USERNAME/playlogic-football-grp

```  

  The trained model is automatically pushed to the Hugging Face Hub.


## 🤗 Model
The fine‑tuned model powering this Space is:

Repo: ShikharNotFound/playlogic-football-grpo

Base: Qwen/Qwen2-0.5B-Instruct

Method: GRPO (Group Relative Policy Optimization)

Training environment: 11v11 football simulation with physics and rewards as above

You can load it in two lines:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("ShikharNotFound/playlogic-football-grpo")
tokenizer = AutoTokenizer.from_pretrained("ShikharNotFound/playlogic-football-grpo")
```

## 🖥 Using This Space
The Space runs a Gradio interface that shows a simplified 2D top‑down view of the pitch.
You can:

Click "Start Match" to begin a new 11v11 game.

Watch the agents move, pass, and shoot in real time.

Use the scoreboard and possession bar to follow the action.

Hit "Reset" to start a new match.

The agents adapt to the evolving game state – no two matches are the same.

## 📁 Repository Structure
```text
PlayLogic/
├── playlogic/                 # Core environment & physics engine
│   ├── env.py                 # OpenEnv football environment
│   ├── physics.py             # Hybrid physics (ball, player dynamics)
│   └── rewards.py             # Reward shaping
├── training/
│   ├── train_grpo.py          # GRPO training script
│   ├── requirements-training.txt
│   └── grpo_football.ipynb    # Colab notebook
├── app.py                     # Gradio Space entrypoint
├── requirements.txt
└── README.md

```

## 🔧 Local Development
```bash
git clone https://github.com/arjune4dev/PlayLogic.git
cd PlayLogic
pip install -r requirements.txt -r training/requirements-training.txt
export PYTHONPATH=$PWD
python app.py            # run the Gradio app locally
```

