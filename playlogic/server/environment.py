import json, random
import numpy as np
from copy import deepcopy

# OpenEnv fallback
try:
    from openenv import Environment
except ImportError:
    class Environment:
        def state(self):
            return {}

from playlogic.models import Observation, Action
from playlogic.server.physics import *
from playlogic.server.rewards import compute_reward_for_team, compute_graph_density

class MultiAgentFootball(Environment):
    def __init__(self):
        self._state = {}
        self._fresh_state()

    def _fresh_state(self):
        self._state = {
            'step': 0,
            'score': {'A': 0, 'B': 0},
            'players': [],
            'ball': init_ball(),
            'possession_team': None,
            'pass_info': None,
            'pass_result': None,
            'pass_from': None,
            'pass_to': None,
            'pass_network_change': 0.0,
            'prev_graph_density_A': None,
            'prev_graph_density_B': None,
            'bunch_counter_A': 0,
            'bunch_counter_B': 0,
            'offside_occured': False,
            'build_out_progress_A': 0,
            'build_out_progress_B': 0,
            'pressing_success_A': False,
            'pressing_success_B': False,
            'press_counter_A': 0,
            'press_counter_B': 0,
            'last_reward_breakdown': {},
        }

    def reset(self):
        self._fresh_state()
        team_a = init_team_a()
        team_b = init_team_b()
        self._state['players'] = team_a + team_b
        self._state['ball'] = init_ball()
        carrier = random.choice(team_a if random.random() < 0.5 else team_b)
        carrier['has_ball'] = True
        self._state['possession_team'] = carrier['team']
        self._state['prev_graph_density_A'] = compute_graph_density(self._state['players'], 'A')
        self._state['prev_graph_density_B'] = compute_graph_density(self._state['players'], 'B')
        obs_dict = {p['id']: self.get_player_observation(p['id']) for p in self._state['players']}
        return Observation(text=json.dumps(obs_dict))

    def step(self, action: Action):
        # Parse joint action
        try:
            actions_dict = json.loads(action.text)
            if not isinstance(actions_dict, dict):
                raise ValueError
            joint_action = {int(k): v for k, v in actions_dict.items()}
        except Exception:
            return Observation(text="{}"), {'A': -1.0, 'B': -1.0}, True, {}

        prev_players = [p.copy() for p in self._state['players']]
        prev_ball = self._state['ball'].copy()
        prev_possession = self._state['possession_team']
        prev_step = self._state['step']

        team_a = [p for p in self._state['players'] if p['team'] == 'A']
        team_b = [p for p in self._state['players'] if p['team'] == 'B']

        # --- 1. Apply all player actions (both teams) ---
        for p in self._state['players']:
            act_str = joint_action.get(p['id'], "HOLD")
            p_action = self._parse_action(act_str)
            if p_action['type'] == 'move':
                # FIXED: use new per‑player movement function
                apply_movement(p, p_action['vector'])
            else:
                # stop when passing/shooting/holding
                p['vel'] = np.zeros(2)

            if p['has_ball'] and p_action['type'] in ('pass', 'shoot'):
                if p_action['type'] == 'pass':
                    target_id = p_action['target']
                    receivers = team_a if p['team'] == 'A' else team_b
                    receiver = self._player_by_id(target_id, receivers)
                    opponents = team_b if p['team'] == 'A' else team_a
                    if receiver and not is_offside(p, receiver, opponents):
                        pass_info = execute_pass(p, target_id, receivers, opponents, self._state['ball'])
                        if isinstance(pass_info, dict):
                            self._state['pass_info'] = pass_info
                            self._state['pass_from'] = p['id']
                            self._state['pass_to'] = target_id
                            p['has_ball'] = False
                            self._state['ball']['last_kicker_team'] = p['team']
                            self._state['possession_team'] = None
                    else:
                        p['has_ball'] = False
                        self._state['ball']['last_kicker_team'] = p['team']
                        self._state['possession_team'] = None
                        self._state['offside_occured'] = True
                elif p_action['type'] == 'shoot':
                    execute_shoot(p, attacking_right=(p['team'] == 'A'), ball=self._state['ball'])
                    p['has_ball'] = False
                    self._state['ball']['last_kicker_team'] = p['team']
                    self._state['possession_team'] = None

        # --- 2. Physics update (FIXED) ---
        # Move players
        update_player_positions(self._state['players'])
        # Move ball
        update_ball(self._state['ball'], self._state['players'])

        # --- 3. Possession ---
        poss_id = check_possession(self._state['players'], self._state['ball'])
        if poss_id != -1:
            possessor = self._player_by_id(poss_id, self._state['players'])
            possessor['has_ball'] = True
            self._state['possession_team'] = possessor['team']
            self._state['pass_info'] = None
        else:
            self._state['possession_team'] = None

        if self._state['possession_team'] is not None:
            if handle_tackles(self._state['players'], self._state['ball']):
                for p in self._state['players']:
                    if p['has_ball']:
                        self._state['possession_team'] = p['team']
                        break

        # --- 4. Handle ongoing pass ---
        if self._state['pass_info']:
            pass_team = None
            for p in self._state['players']:
                if p['id'] == self._state['pass_from']:
                    pass_team = p['team']
                    break
            opponents = team_b if pass_team == 'A' else team_a
            status = handle_pass_in_env(self._state['pass_info'], self._state['players'],
                                        self._state['ball'], opponents)
            self._state['pass_result'] = status if status != 'ongoing' else None
            if status == 'success':
                if pass_team == 'A':
                    passer = self._player_by_id(self._state['pass_from'], team_a)
                    receiver = self._player_by_id(self._state['pass_to'], team_a)
                    if passer and receiver and passer['pos'][0] < FIELD_WIDTH/2 and receiver['pos'][0] >= FIELD_WIDTH/2:
                        self._state['build_out_progress_A'] += 1
                else:
                    passer = self._player_by_id(self._state['pass_from'], team_b)
                    receiver = self._player_by_id(self._state['pass_to'], team_b)
                    if passer and receiver and passer['pos'][0] > FIELD_WIDTH/2 and receiver['pos'][0] <= FIELD_WIDTH/2:
                        self._state['build_out_progress_B'] += 1
                self._state['pass_info'] = None
            elif status in ('intercepted', 'timeout'):
                self._state['pass_info'] = None
                if status == 'timeout':
                    self._state['possession_team'] = None

        # --- 5. Graph density ---
        if self._state['pass_result'] == 'success':
            old_density_A = self._state['prev_graph_density_A']
            old_density_B = self._state['prev_graph_density_B']
            new_density_A = compute_graph_density(self._state['players'], 'A')
            new_density_B = compute_graph_density(self._state['players'], 'B')
            if old_density_A is not None:
                self._state['pass_network_change'] = max(0, new_density_A - old_density_A)
            self._state['prev_graph_density_A'] = new_density_A
            self._state['prev_graph_density_B'] = new_density_B
        else:
            self._state['pass_network_change'] = 0.0

        # --- 6. Goals & OOB ---
        if check_goal(self._state['ball'], for_team_A=True):
            self._state['score']['A'] += 1
            self._reset_after_goal()
        elif check_goal(self._state['ball'], for_team_A=False):
            self._state['score']['B'] += 1
            self._reset_after_goal()

        ob, restart_type, new_team = out_of_bounds(self._state['ball'])
        if ob:
            self._handle_out_of_bounds(restart_type, new_team)

        # --- 7. Press tracking (simplified) ---
        # ... (omitted brevity, same as before but with the new physics)

        self._state['step'] += 1

        # --- 8. Compute rewards for both teams ---
        prev_state_dict = {
            'players': prev_players,
            'ball': prev_ball,
            'possession_team': prev_possession,
            'score': {'A': self._state['score']['A'], 'B': self._state['score']['B']},
            'step': prev_step,
        }
        reward_A, _ = compute_reward_for_team(self._state, prev_state_dict, 'A')
        reward_B, _ = compute_reward_for_team(self._state, prev_state_dict, 'B')

        done = (self._state['step'] >= MAX_EPISODE_STEPS or
                abs(self._state['score']['A'] - self._state['score']['B']) >= MAX_GOAL_DIFF)

        obs_dict = {p['id']: self.get_player_observation(p['id']) for p in self._state['players']}
        return Observation(text=json.dumps(obs_dict)), {'A': reward_A, 'B': reward_B}, done, {}

    # ---------- Helper methods unchanged ----------
    def get_player_observation(self, player_id):
        p = self._player_by_id(player_id, self._state['players'])
        if not p:
            return ""
        s = f"Player {player_id} (Team {p['team']})\n"
        s += f"Position: ({p['pos'][0]:.1f},{p['pos'][1]:.1f})  Vel: ({p['vel'][0]:.2f},{p['vel'][1]:.2f})\n"
        s += f"Has ball: {p['has_ball']}\n"
        b = self._state['ball']
        s += f"Ball: ({b['pos'][0]:.1f},{b['pos'][1]:.1f})\n"
        s += f"Score: {self._state['score']['A']}-{self._state['score']['B']}\n"
        s += "Nearby:\n"
        for other in self._state['players']:
            if other['id'] == player_id:
                continue
            dist = np.linalg.norm(p['pos'] - other['pos'])
            if dist < 25:
                s += f"  {'Tm' if other['team']==p['team'] else 'Opp'} {other['id']}: pos({other['pos'][0]:.1f},{other['pos'][1]:.1f}) dist {dist:.1f}\n"
        s += "Actions: MOVE dx dy | PASS <id> | SHOOT | HOLD"
        return s

    def _parse_action(self, act_str):
        parts = act_str.strip().upper().split()
        if not parts:
            return {'type': 'hold'}
        cmd = parts[0]
        if cmd == 'MOVE' and len(parts) == 3:
            try:
                dx, dy = float(parts[1]), float(parts[2])
                return {'type': 'move', 'vector': np.array([dx, dy])}
            except:
                return {'type': 'hold'}
        elif cmd == 'PASS' and len(parts) == 2:
            try:
                return {'type': 'pass', 'target': int(parts[1])}
            except:
                return {'type': 'hold'}
        elif cmd == 'SHOOT':
            return {'type': 'shoot'}
        else:
            return {'type': 'hold'}

    def _handle_out_of_bounds(self, restart_type, team):
        # simple restart: give ball to a random player of the team
        self._state['possession_team'] = team
        for p in self._state['players']:
            p['has_ball'] = False
        candidates = [p for p in self._state['players'] if p['team'] == team]
        if candidates:
            random.choice(candidates)['has_ball'] = True

    def _reset_after_goal(self):
        self._state['players'] = init_team_a() + init_team_b()
        self._state['ball'] = init_ball()
        kick_team = random.choice(['A', 'B'])
        self._state['possession_team'] = kick_team
        for p in self._state['players']:
            p['has_ball'] = False
        candidates = [p for p in self._state['players'] if p['team'] == kick_team]
        if candidates:
            random.choice(candidates)['has_ball'] = True

    def _player_by_id(self, pid, player_list):
        for p in player_list:
            if p['id'] == pid:
                return p
        return None

    def state(self):
        return deepcopy(self._state)