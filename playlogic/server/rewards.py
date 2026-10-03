import numpy as np
from playlogic.server.physics import SPACING_THRESHOLD, FIELD_WIDTH
from playlogic.server.formation import get_ideal_position

def compute_graph_density(players, team, threshold=20.0):
    """Graph density for passing network."""
    team_players = [p for p in players if p['team'] == team]
    n = len(team_players)
    if n < 2:
        return 0.0
    edges = 0
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(team_players[i]['pos'] - team_players[j]['pos']) < threshold:
                edges += 1
    return edges / (n * (n-1) / 2.0)

def compute_reward_for_team(state, prev_state, team):
    """Compute reward for specified team ('A' or 'B')."""
    reward = 0.0
    breakdown = {}
    players = state['players']
    ball = state['ball']
    score = state['score']
    my_team = [p for p in players if p['team'] == team]
    opp_team = [p for p in players if p['team'] != team]

    # Goal
    if prev_state:
        if team == 'A' and score['A'] > prev_state['score']['A']:
            return 5.0, {'goal_scored': 5.0}
        if team == 'B' and score['B'] > prev_state['score']['B']:
            return 5.0, {'goal_scored': 5.0}
        if team == 'A' and score['B'] > prev_state['score']['B']:
            return -5.0, {'goal_conceded': -5.0}
        if team == 'B' and score['A'] > prev_state['score']['A']:
            return -5.0, {'goal_conceded': -5.0}

    # Turnover
    if prev_state and state.get('possession_team') != prev_state.get('possession_team'):
        if prev_state.get('possession_team') == team and state.get('possession_team') != team:
            reward -= 0.1

    # Pass result
    if 'pass_result' in state and state['pass_result'] == 'success':
        pass_team = None
        for p in players:
            if p['id'] == state['pass_from']:
                pass_team = p['team']
                break
        if pass_team == team:
            reward += 0.5
        elif pass_team is not None:
            reward -= 0.3
    if 'pass_result' in state and state['pass_result'] == 'intercepted':
        if state.get('possession_team') == team:
            reward += 0.3
        else:
            reward -= 0.1

    # Formation cohesion – using the new formation signature
    cohesion = 0.0
    ball_pos = ball['pos']
    # Prepare lists of own and opponent positions (the formation uses own_players and opponent_players)
    own_players = my_team
    opponent_players = opp_team
    for p in my_team:
        if p['id'] == 0:   # skip GK
            continue
        ideal = get_ideal_position(team, p['id'], ball_pos, own_players, opponent_players)
        dist = np.linalg.norm(p['pos'] - ideal)
        cohesion += 1.0 - np.tanh(dist)
    reward += 0.02 * cohesion

    # Spacing penalty (simple bunching)
    too_close = False
    for i, p1 in enumerate(my_team):
        for j, p2 in enumerate(my_team):
            if i < j and np.linalg.norm(p1['pos'] - p2['pos']) < SPACING_THRESHOLD:
                too_close = True
                break
        if too_close:
            break
    if too_close:
        bunch_key = 'bunch_counter_' + team
        if bunch_key not in state:
            state[bunch_key] = 0
        state[bunch_key] += 1
        if state[bunch_key] > 10:
            reward -= 0.01
    else:
        bunch_key = 'bunch_counter_' + team
        state[bunch_key] = 0

    return reward, breakdown