"""Train 22 agents with REINFORCE – TRUE MARL with individual rewards."""
import os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.distributions import Bernoulli
import sys, matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from server.environment import FootballTacticsMARL
from training.agent_policy import SoccerAgent
from training.agent_utils import get_local_obs

base_dir = os.path.dirname(__file__)
plot_path = os.path.join(base_dir, "..", "assets", "plots")
os.makedirs(plot_path, exist_ok=True)

# Training hyperparameters
NUM_EPISODES = 5
MAX_STEPS = 2000
LR = 1e-3
HIDDEN_DIM = 128
GAMMA = 0.99
EPS_START = 0.5
EPS_END = 0.02
EPS_DECAY = 0.95
MODEL_DIR_A = "../trained_agents_A"
MODEL_DIR_B = "../trained_agents_B"
os.makedirs(MODEL_DIR_A, exist_ok=True)
os.makedirs(MODEL_DIR_B, exist_ok=True)

# Initialize environment and agents
env = FootballTacticsMARL()
agents_a = [SoccerAgent(input_dim=53, hidden_dim=HIDDEN_DIM) for _ in range(11)]
agents_b = [SoccerAgent(input_dim=53, hidden_dim=HIDDEN_DIM) for _ in range(11)]
opt_a = [optim.Adam(ag.parameters(), lr=LR) for ag in agents_a]
opt_b = [optim.Adam(ag.parameters(), lr=LR) for ag in agents_b]
epsilon = EPS_START

def heuristic_action(sim, player, team):
    ball = sim.ball
    opp_goal_x = 105 if team == 'A' else 0
    own_goal_x = 0 if team == 'A' else 105
    attack_right = (team == 'A')

    if player.has_ball:
        # ---------- ATTACKING (with crowding awareness) ----------
        opp_team = sim.team_b if team == 'A' else sim.team_a
        pressure = sum(1 for opp in opp_team
                       if np.hypot(player.x - opp.x, player.y - opp.y) < 5.0)

        # Shoot if close and not swarmed
        dist_to_goal = abs(opp_goal_x - player.x)
        if dist_to_goal < 20 and pressure < 3:
            return ("shoot", None)

        # Find best forward and safe pass options
        mates = sim.team_a if team == 'A' else sim.team_b
        best_forward = None
        best_forward_score = -1
        best_safe = None
        best_safe_score = -1

        for mate in mates:
            if mate.id == player.id or mate.has_ball:
                continue
            dx = mate.x - player.x
            dy = mate.y - player.y
            dist = np.hypot(dx, dy)
            if dist < 2 or dist > 40:
                continue

            # Check passing lane
            opps = sim.team_b if team == 'A' else sim.team_a
            blocked = any(point_to_segment_distance(opp.x, opp.y, player.x, player.y, mate.x, mate.y) < 2.5
                          for opp in opps)
            if blocked:
                continue

            is_forward = (dx > 0) if attack_right else (dx < 0)
            score = (abs(dx) / (dist + 1)) * (1.5 if is_forward else 0.5)

            if is_forward and score > best_forward_score:
                best_forward_score = score
                best_forward = mate
            if score > best_safe_score:
                best_safe_score = score
                best_safe = mate

        # Decision based on pressure
        if pressure >= 2:
            target = best_forward if best_forward else best_safe
            if target and np.random.random() < 0.95:
                return ("pass", target.id)
        elif pressure == 1:
            if best_forward and np.random.random() < 0.75:
                return ("pass", best_forward.id)
            if best_safe and np.random.random() < 0.5:
                return ("pass", best_safe.id)
        else:
            if best_forward and np.random.random() < 0.85:
                return ("pass", best_forward.id)

        # Dribble (slower if crowded)
        tx = opp_goal_x
        ty = np.clip(player.y + np.random.uniform(-10, 10), 8, 60)
        dx, dy = tx - player.x, ty - player.y
        mag = np.hypot(dx, dy)
        if mag > 0:
            speed = 6.0 if pressure >= 2 else 7.5 if pressure == 1 else 8.5
            return ("move", [dx/mag * speed, dy/mag * speed])
        return ("hold", None)

    else:
        # ========== DEFENSIVE BEHAVIOUR ==========
        # 1. Loose ball chase
        if ball.possessor is None:
            dist = np.hypot(player.x - ball.x, player.y - ball.y)
            if dist < 20:
                dx, dy = ball.x - player.x, ball.y - player.y
                mag = np.hypot(dx, dy)
                if mag > 0:
                    return ("move", [dx/mag * 9.0, dy/mag * 9.0])

        # 2. Opponent has the ball
        if ball.possessor and ball.possessor.team != team:
            carrier = ball.possessor
            dist_to_carrier = np.hypot(player.x - carrier.x, player.y - carrier.y)

            # ---- Goalkeeper ----
            if player.is_goalkeeper:
                target_x = 5 if team == 'A' else 100
                target_y = np.clip(34 + (ball.y - 34) * 0.5, 28, 40)
                dx, dy = target_x - player.x, target_y - player.y
                mag = np.hypot(dx, dy)
                if mag > 0.5:
                    return ("move", [dx/mag * 6.0, dy/mag * 6.0])
                return ("hold", None)

            # ---- Defenders (ID 1-4) ----
            if 1 <= player.id <= 4:
                ball_dist_to_own_goal = abs(ball.x - own_goal_x)
                in_danger = ball_dist_to_own_goal < 35

                if in_danger:
                    target_x = 15 if team == 'A' else 90
                    y_positions = [34, 20, 48, 28, 40]
                    target_y = y_positions[player.id - 1]
                    dx, dy = target_x - player.x, target_y - player.y
                    mag = np.hypot(dx, dy)
                    if mag > 0.5:
                        return ("move", [dx/mag * 8.0, dy/mag * 8.0])
                    return ("hold", None)
                else:
                    if dist_to_carrier < 3.0:
                        dx, dy = carrier.x - player.x, carrier.y - player.y
                        mag = np.hypot(dx, dy)
                        if mag > 0:
                            return ("move", [dx/mag * 9.5, dy/mag * 9.5])
                    opps = sim.team_b if team == 'A' else sim.team_a
                    if team == 'A':
                        dangerous_opp = min(opps, key=lambda o: o.x)
                    else:
                        dangerous_opp = max(opps, key=lambda o: o.x)
                    mark_x = dangerous_opp.x - 5 if team == 'A' else dangerous_opp.x + 5
                    mark_y = dangerous_opp.y
                    mark_x = max(5, min(100, mark_x))
                    dx, dy = mark_x - player.x, mark_y - player.y
                    mag = np.hypot(dx, dy)
                    if mag > 0.5:
                        return ("move", [dx/mag * 7.0, dy/mag * 7.0])
                    return ("hold", None)

            # ---- Midfielders (ID 5-7) ----
            if 5 <= player.id <= 7:
                in_our_half = (ball.x < 52.5) if attack_right else (ball.x > 52.5)
                if in_our_half and dist_to_carrier < 10:
                    dx, dy = carrier.x - player.x, carrier.y - player.y
                    mag = np.hypot(dx, dy)
                    if mag > 0:
                        return ("move", [dx/mag * 8.5, dy/mag * 8.5])
                from server.formation import get_ideal_position
                phase = 'defend'
                tx, ty = get_ideal_position(player, (ball.x, ball.y), phase, team)
                dx, dy = tx - player.x, ty - player.y
                mag = np.hypot(dx, dy)
                if mag > 0.5:
                    return ("move", [dx/mag * 7.0, dy/mag * 7.0])
                return ("hold", None)

            # ---- Forwards (ID 8-10) ----
            if player.id >= 8:
                if (ball.x > 70 if attack_right else ball.x < 35):
                    if dist_to_carrier < 12:
                        dx, dy = carrier.x - player.x, carrier.y - player.y
                        mag = np.hypot(dx, dy)
                        if mag > 0:
                            return ("move", [dx/mag * 8.0, dy/mag * 8.0])
                from server.formation import get_ideal_position
                phase = 'attack'
                tx, ty = get_ideal_position(player, (ball.x, ball.y), phase, team)
                dx, dy = tx - player.x, ty - player.y
                mag = np.hypot(dx, dy)
                if mag > 0.5:
                    return ("move", [dx/mag * 6.5, dy/mag * 6.5])
                return ("hold", None)

        # 3. Default – return to formation
        from server.formation import get_ideal_position
        phase = 'attack' if (ball.x > 52.5 if attack_right else ball.x < 52.5) else 'defend'
        tx, ty = get_ideal_position(player, (ball.x, ball.y), phase, team)
        dx, dy = tx - player.x, ty - player.y
        mag = np.hypot(dx, dy)
        if mag > 0.5:
            return ("move", [dx/mag * 7.0, dy/mag * 7.0])
        return ("hold", None)

def point_to_segment_distance(px, py, ax, ay, bx, by):
    """Distance from point to line segment"""
    ABx, ABy = bx - ax, by - ay
    if ABx == 0 and ABy == 0:
        return np.hypot(px - ax, py - ay)
    APx, APy = px - ax, py - ay
    t = (APx * ABx + APy * ABy) / (ABx*ABx + ABy*ABy)
    t = max(0, min(1, t))
    closest_x = ax + t * ABx
    closest_y = ay + t * ABy
    return np.hypot(px - closest_x, py - closest_y)

# Training metrics
all_team_rewards = []
goals_a_hist, goals_b_hist = [], []
individual_rewards_a_hist = {i: [] for i in range(11)}
individual_rewards_b_hist = {i: [] for i in range(11)}

print("=" * 60)
print("STARTING MARL TRAINING - 22 Independent Agents")
print("=" * 60)

for ep in range(NUM_EPISODES):
    print(f"\n{'='*40}")
    print(f"Episode {ep+1}/{NUM_EPISODES} (epsilon={epsilon:.3f})")
    print(f"{'='*40}")
    
    env.reset()
    done = False
    step = 0
    
    # Per-agent storage for this episode
    log_probs_a = [[] for _ in range(11)]  # (log_prob_pass, log_prob_shoot)
    log_probs_b = [[] for _ in range(11)]
    rewards_a = [[] for _ in range(11)]   # Individual rewards per agent
    rewards_b = [[] for _ in range(11)]
    
    score_a_prev, score_b_prev = 0, 0
    episode_reward_components = defaultdict(float)

    while not done and step < MAX_STEPS:
        actions_a, actions_b = {}, {}
        
        # ===========================================
        # TEAM A - 11 Independent Agents
        # ===========================================
        for i, p in enumerate(env.sim.team_a):
            # Get observation
            state = get_local_obs(env.sim, p, 'A')
            state_t = torch.from_numpy(state).float().unsqueeze(0)  # (1, 46)
            
            # Get actions from agent
            move, plog, slog = agents_a[i](state_t)
            heur_act, heur_param = heuristic_action(env.sim, p, 'A')
            
            if np.random.random() < epsilon:
                # Use heuristic
                if heur_act == "pass":
                    actions_a[p.id] = {"pass": heur_param}
                elif heur_act == "shoot":
                    actions_a[p.id] = {"shoot": None}
                elif heur_act == "move":
                    actions_a[p.id] = {"move": list(heur_param)}
                else:
                    actions_a[p.id] = {"hold": None}
                log_probs_a[i].append((None, None))
            else:
                # Use policy network
                pass_prob = torch.sigmoid(plog)
                shoot_prob = torch.sigmoid(slog)
                
                if p.has_ball:
                    # Decide between pass, shoot, or dribble
                    m_pass = Bernoulli(pass_prob)
                    sample_pass = m_pass.sample()
                    
                    if sample_pass.item() == 1:
                        # Try to pass forward
                        mates = [t for t in env.sim.team_a 
                                if t.id != p.id and t.x > p.x]  # Forward only
                        if mates:
                            target = np.random.choice(mates)
                            actions_a[p.id] = {"pass": target.id}
                            lp_pass = m_pass.log_prob(sample_pass)
                        else:
                            # No forward pass option, dribble
                            actions_a[p.id] = {"move": [8.0, np.random.uniform(-3, 3)]}
                            lp_pass = torch.tensor(0.0)
                        log_probs_a[i].append((lp_pass, torch.tensor(0.0)))
                    else:
                        # Try to shoot or dribble
                        m_shoot = Bernoulli(shoot_prob)
                        sample_shoot = m_shoot.sample()
                        
                        if sample_shoot.item() == 1:
                            actions_a[p.id] = {"shoot": None}
                        else:
                            # Dribble towards opponent goal
                            actions_a[p.id] = {"move": [8.0, np.random.uniform(-3, 3)]}
                        
                        lp_shoot = m_shoot.log_prob(sample_shoot)
                        log_probs_a[i].append((torch.tensor(0.0), lp_shoot))
                else:
                    # Without ball - use movement from policy
                    move_list = move.squeeze(0).tolist()
                    actions_a[p.id] = {"move": move_list}
                    log_probs_a[i].append((torch.tensor(0.0), torch.tensor(0.0)))
        
        # ===========================================
        # TEAM B - 11 Independent Agents
        # ===========================================
        for i, p in enumerate(env.sim.team_b):
            # Get observation
            state = get_local_obs(env.sim, p, 'B')
            state_t = torch.from_numpy(state).float().unsqueeze(0)  # (1, 46)
            
            # Get actions from agent
            move, plog, slog = agents_b[i](state_t)
            heur_act, heur_param = heuristic_action(env.sim, p, 'B')
            
            if np.random.random() < epsilon:
                # Use heuristic
                if heur_act == "pass":
                    actions_b[p.id] = {"pass": heur_param}
                elif heur_act == "shoot":
                    actions_b[p.id] = {"shoot": None}
                elif heur_act == "move":
                    actions_b[p.id] = {"move": list(heur_param)}
                else:
                    actions_b[p.id] = {"hold": None}
                log_probs_b[i].append((None, None))
            else:
                # Use policy network
                pass_prob = torch.sigmoid(plog)
                shoot_prob = torch.sigmoid(slog)
                
                if p.has_ball:
                    m_pass = Bernoulli(pass_prob)
                    sample_pass = m_pass.sample()
                    
                    if sample_pass.item() == 1:
                        # Forward pass only
                        mates = [t for t in env.sim.team_b 
                                if t.id != p.id and t.x < p.x]  # Forward for Team B
                        if mates:
                            target = np.random.choice(mates)
                            actions_b[p.id] = {"pass": target.id}
                            lp_pass = m_pass.log_prob(sample_pass)
                        else:
                            actions_b[p.id] = {"move": [-8.0, np.random.uniform(-3, 3)]}
                            lp_pass = torch.tensor(0.0)
                        log_probs_b[i].append((lp_pass, torch.tensor(0.0)))
                    else:
                        m_shoot = Bernoulli(shoot_prob)
                        sample_shoot = m_shoot.sample()
                        
                        if sample_shoot.item() == 1:
                            actions_b[p.id] = {"shoot": None}
                        else:
                            actions_b[p.id] = {"move": [-8.0, np.random.uniform(-3, 3)]}
                        
                        lp_shoot = m_shoot.log_prob(sample_shoot)
                        log_probs_b[i].append((torch.tensor(0.0), lp_shoot))
                else:
                    move_list = move.squeeze(0).tolist()
                    actions_b[p.id] = {"move": move_list}
                    log_probs_b[i].append((torch.tensor(0.0), torch.tensor(0.0)))

        # ===========================================
        # STEP THE ENVIRONMENT
        # ===========================================
        _, rewards_dict, done, info = env.step_multi_both(actions_a, actions_b)
        
        # Store individual rewards
        for i in range(11):
            rewards_a[i].append(rewards_dict['totals_a'][i])
            rewards_b[i].append(rewards_dict['totals_b'][i])
        
        # Track events
        if env.sim.score_a > score_a_prev:
            print(f"  ⚽ GOAL! Team A scores! ({env.sim.score_a}-{env.sim.score_b})")
        if env.sim.score_b > score_b_prev:
            print(f"  ⚽ GOAL! Team B scores! ({env.sim.score_a}-{env.sim.score_b})")
        
        score_a_prev, score_b_prev = env.sim.score_a, env.sim.score_b
        step += 1

    # ===========================================
    # UPDATE EACH AGENT WITH ITS OWN REWARDS
    # ===========================================
    
    # Team A - Update each of the 11 agents independently
    for i in range(11):
        if len(rewards_a[i]) == 0:
            continue
        
        # Compute returns (discounted cumulative reward)
        R = 0
        returns = []
        for r in reversed(rewards_a[i]):
            R = r + GAMMA * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Normalize returns for stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute policy gradient loss
        agent = agents_a[i]
        opt = opt_a[i]
        opt.zero_grad()
        loss = 0.0
        
        for t in range(min(len(log_probs_a[i]), len(returns))):
            lp_pass, lp_shoot = log_probs_a[i][t]
            if lp_pass is None:  # Heuristic step, no gradient
                continue
            
            R_t = returns[t]
            if lp_pass.requires_grad:
                loss += -lp_pass * R_t
            if lp_shoot.requires_grad:
                loss += -lp_shoot * R_t
        
        if loss != 0.0:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            opt.step()
    
    # Team B - Update each of the 11 agents independently
    for i in range(11):
        if len(rewards_b[i]) == 0:
            continue
        
        R = 0
        returns = []
        for r in reversed(rewards_b[i]):
            R = r + GAMMA * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        agent = agents_b[i]
        opt = opt_b[i]
        opt.zero_grad()
        loss = 0.0
        
        for t in range(min(len(log_probs_b[i]), len(returns))):
            lp_pass, lp_shoot = log_probs_b[i][t]
            if lp_pass is None:
                continue
            
            R_t = returns[t]
            if lp_pass.requires_grad:
                loss += -lp_pass * R_t
            if lp_shoot.requires_grad:
                loss += -lp_shoot * R_t
        
        if loss != 0.0:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            opt.step()

    # ===========================================
    # TRACK METRICS
    # ===========================================
    team_a_total = sum(sum(rewards_a[i]) for i in range(11))
    team_b_total = sum(sum(rewards_b[i]) for i in range(11))
    all_team_rewards.append(team_a_total)
    
    goals_a_hist.append(env.sim.score_a)
    goals_b_hist.append(env.sim.score_b)
    
    for i in range(11):
        individual_rewards_a_hist[i].append(sum(rewards_a[i]))
        individual_rewards_b_hist[i].append(sum(rewards_b[i]))
    
    # Decay epsilon
    epsilon = max(EPS_END, epsilon * EPS_DECAY)
    
    # Print summary
    avg_reward_a = np.mean([sum(rewards_a[i]) for i in range(11)])
    avg_reward_b = np.mean([sum(rewards_b[i]) for i in range(11)])
    
    print(f"\nEpisode {ep+1} Summary:")
    print(f"  Score: A {env.sim.score_a} - B {env.sim.score_b}")
    print(f"  Steps: {step}")
    print(f"  Avg Reward Agent A: {avg_reward_a:.2f}")
    print(f"  Avg Reward Agent B: {avg_reward_b:.2f}")
    print(f"  Top Agent A: #{np.argmax([sum(rewards_a[i]) for i in range(11)])} "
          f"({max([sum(rewards_a[i]) for i in range(11)]):.2f})")
    print(f"  Top Agent B: #{np.argmax([sum(rewards_b[i]) for i in range(11)])} "
          f"({max([sum(rewards_b[i]) for i in range(11)]):.2f})")

# ===========================================
# SAVE TRAINED AGENTS
# ===========================================
print("\n" + "=" * 60)
print("SAVING TRAINED AGENTS")
print("=" * 60)

for i, ag in enumerate(agents_a):
    save_path = os.path.join(MODEL_DIR_A, f"agent_{i}.pt")
    torch.save(ag.state_dict(), save_path)  # Save ONLY the state dict
    print(f"  Team A Agent {i} saved to {save_path}")

for i, ag in enumerate(agents_b):
    save_path = os.path.join(MODEL_DIR_B, f"agent_{i}.pt")
    torch.save({
        'model_state_dict': ag.state_dict(),
        'agent_id': i,
        'team': 'B',
        'role': ['GK', 'DEF', 'DEF', 'DEF', 'DEF', 'MID', 'MID', 'MID', 'FWD', 'FWD', 'FWD'][i]
    }, save_path)
    print(f"  Team B Agent {i} saved to {save_path}")

print("\nAll agents saved successfully!")

# ===========================================
# PLOT TRAINING RESULTS
# ===========================================
print("\nGenerating training plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Team A total reward per episode
axes[0, 0].plot(all_team_rewards, marker='o', markersize=3, color='blue')
axes[0, 0].set_title('Team A Total Reward per Episode')
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Total Reward')
axes[0, 0].grid(True)

# Plot 2: Goals scored
axes[0, 1].plot(goals_a_hist, label='Team A', color='blue', marker='o', markersize=3)
axes[0, 1].plot(goals_b_hist, label='Team B', color='red', marker='x', markersize=3)
axes[0, 1].set_title('Goals per Episode')
axes[0, 1].set_xlabel('Episode')
axes[0, 1].set_ylabel('Goals')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Plot 3: Average reward by role - Team A
roles = {
    'GK': [0],
    'DEF': [1, 2, 3, 4],
    'MID': [5, 6, 7],
    'FWD': [8, 9, 10]
}
colors = {'GK': 'black', 'DEF': 'blue', 'MID': 'green', 'FWD': 'red'}

for role_name, ids in roles.items():
    avg_reward = np.mean([individual_rewards_a_hist[i] for i in ids], axis=0)
    axes[1, 0].plot(avg_reward, label=role_name, color=colors[role_name], marker='o', markersize=2)
axes[1, 0].set_title('Team A - Avg Individual Reward by Role')
axes[1, 0].set_xlabel('Episode')
axes[1, 0].set_ylabel('Avg Reward')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Plot 4: Average reward by role - Team B
for role_name, ids in roles.items():
    avg_reward = np.mean([individual_rewards_b_hist[i] for i in ids], axis=0)
    axes[1, 1].plot(avg_reward, label=role_name, color=colors[role_name], marker='o', markersize=2)
axes[1, 1].set_title('Team B - Avg Individual Reward by Role')
axes[1, 1].set_xlabel('Episode')
axes[1, 1].set_ylabel('Avg Reward')
axes[1, 1].legend()
axes[1, 1].grid(True)

# Plot 5: Goal difference
goal_diff = [a - b for a, b in zip(goals_a_hist, goals_b_hist)]
bar_colors = ['blue' if d > 0 else 'red' if d < 0 else 'gray' for d in goal_diff]
axes[1, 2].bar(range(len(goal_diff)), goal_diff, color=bar_colors)
axes[1, 2].axhline(y=0, color='black', linestyle='-', lw=1)
axes[1, 2].set_title('Goal Difference (A - B) per Episode')
axes[1, 2].set_xlabel('Episode')
axes[1, 2].set_ylabel('Goal Difference')

# Plot 6: Individual agent performance (final episode)
final_rewards_a = [individual_rewards_a_hist[i][-1] for i in range(11)]
final_rewards_b = [individual_rewards_b_hist[i][-1] for i in range(11)]
x = range(11)
axes[0, 2].bar(x, final_rewards_a, alpha=0.7, label='Team A', color='blue')
axes[0, 2].bar(x, final_rewards_b, alpha=0.7, label='Team B', color='red')
axes[0, 2].set_title('Final Episode - Per-Agent Reward')
axes[0, 2].set_xlabel('Agent ID')
axes[0, 2].set_ylabel('Total Reward')
axes[0, 2].set_xticks(x)
axes[0, 2].set_xticklabels(['GK','D1','D2','D3','D4','M1','M2','M3','F1','F2','F3'])
axes[0, 2].legend()
axes[0, 2].grid(True, axis='y')

plt.tight_layout()
plot_file = os.path.join(plot_path, "marl_training_results.png")
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {plot_file}")
plt.show()

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)