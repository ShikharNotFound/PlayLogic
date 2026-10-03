import numpy as np
from playlogic.server.physics import FIELD_WIDTH, FIELD_HEIGHT, clamp

# Role‑based ideal positions using dynamic phases and opponent pressure.

class Formation:
    """
    Defines a team's tactical shape.
    Provide a dict of role -> list of indices, e.g.:
    {'GK': [0], 'LB': [1], 'CB': [2,3], ... }
    Then call get_ideal_position(player_id, ball_pos, own_players, opponent_players) 
    to get the target (x,y) for that player.
    """
    def __init__(self, team, roles_dict, width_attack=40, width_defend=35, depth_attack=30, depth_defend=20):
        self.team = team
        self.roles = roles_dict
        self.player_role = {}
        for role, ids in roles_dict.items():
            for pid in ids:
                self.player_role[pid] = role
        self.width_attack = width_attack   # horizontal spread when attacking
        self.width_defend = width_defend   # when defending
        self.depth_attack = depth_attack   # distance from ball to defensive line in attack
        self.depth_defend = depth_defend   # in defense

    def get_ideal_position(self, player_id, ball_pos, own_players, opponent_players):
        role = self.player_role.get(player_id, 'unknown')
        ball_x, ball_y = ball_pos
        # Determine phase
        if self.team == 'A':
            attacking = ball_x > FIELD_WIDTH/2
        else:
            attacking = ball_x < FIELD_WIDTH/2
        phase = 'attack' if attacking else 'defend'

        # Base reference point: ball position, but shifted for some lines
        if self.team == 'A':
            if attacking:
                def_line_x = ball_x - self.depth_attack
                mid_line_x = ball_x - 0.6 * self.depth_attack
                forw_line_x = ball_x + 0.3 * self.depth_attack
            else:
                def_line_x = ball_x - self.depth_defend
                mid_line_x = ball_x - 0.7 * self.depth_defend
                forw_line_x = ball_x - 0.2 * self.depth_defend
        else:   # Team B attacks left
            if attacking:
                def_line_x = ball_x + self.depth_attack
                mid_line_x = ball_x + 0.6 * self.depth_attack
                forw_line_x = ball_x - 0.3 * self.depth_attack
            else:
                def_line_x = ball_x + self.depth_defend
                mid_line_x = ball_x + 0.7 * self.depth_defend
                forw_line_x = ball_x + 0.2 * self.depth_defend

        # Width factor depends on phase
        width = self.width_attack if phase == 'attack' else self.width_defend

        # Adjust for opponent pressure: if many opponents are near the ball, drop the defensive line back
        pressure = sum(1 for op in opponent_players if np.linalg.norm(op['pos'] - ball_pos) < 20.0)
        if pressure >= 4 and phase == 'defend':
            shift = 5.0 * (pressure - 3)   # drop deeper
            if self.team == 'A':
                def_line_x -= shift
                mid_line_x -= shift
                forw_line_x -= shift
            else:
                def_line_x += shift
                mid_line_x += shift
                forw_line_x += shift

        # Position by role
        if role == 'GK':
            if self.team == 'A':
                return np.array([5.0, FIELD_HEIGHT/2])
            else:
                return np.array([FIELD_WIDTH-5.0, FIELD_HEIGHT/2])

        # Defenders
        if role in ['LB', 'RB', 'CB', 'LWB', 'RWB']:
            line_x = def_line_x
            # distribute along width
            def_order = [r for r in ['LB','CB','CB','RB','LWB','RWB'] if r in self.roles]
            idx = def_order.index(role) if role in def_order else 0
            total = len(def_order)
            spacing = width / (total - 1) if total > 1 else 0
            y_min = max(0, FIELD_HEIGHT/2 - width/2)
            # arrange from leftmost to rightmost
            y = y_min + idx * spacing
            # for specific roles like CB, they might be more central, but we can just evenly spread
            return np.array([clamp(line_x, 0, FIELD_WIDTH), y])

        # Midfielders
        if role in ['CDM', 'CM', 'CAM', 'LM', 'RM']:
            line_x = mid_line_x
            mf_roles = [r for r in self.roles if r in ['CDM','CM','CAM','LM','RM']]
            idx = mf_roles.index(role) if role in mf_roles else 0
            total = len(mf_roles)
            spacing = width / (total - 1) if total > 1 else 0
            y_min = max(0, FIELD_HEIGHT/2 - width/2)
            y = y_min + idx * spacing
            return np.array([clamp(line_x, 0, FIELD_WIDTH), y])

        # Forwards
        if role in ['LW', 'ST', 'RW', 'CF']:
            line_x = forw_line_x
            fwd_roles = [r for r in self.roles if r in ['LW','ST','RW','CF']]
            idx = fwd_roles.index(role) if role in fwd_roles else 0
            total = len(fwd_roles)
            # Wingers stay wide, striker central
            if role == 'LW':
                y = max(0, FIELD_HEIGHT/2 - width/2)
            elif role == 'RW':
                y = min(FIELD_HEIGHT, FIELD_HEIGHT/2 + width/2)
            else:
                # striker or CF centre
                y = FIELD_HEIGHT/2
            return np.array([clamp(line_x, 0, FIELD_WIDTH), y])

        # Default: near ball
        return np.array([ball_x, ball_y])


# Example configuration for Team A (4-3-3)
default_roles_A = {
    'GK': [0],
    'LB': [1], 'CB': [2,3], 'RB': [4],
    'CM': [5,6,7],
    'LW': [8], 'ST': [9], 'RW': [10]
}
teamA_formation = Formation('A', default_roles_A)

# Team B (4-4-2)
default_roles_B = {
    'GK': [0],
    'RB': [1], 'CB': [2,3], 'LB': [4],
    'RM': [5], 'CM': [6], 'LM': [7],
    'RW': [8], 'ST': [9], 'LW': [10]
}
teamB_formation = Formation('B', default_roles_B)

# Convenience function for reward calculation
def get_ideal_position(team, player_id, ball_pos, own_players, opponent_players):
    if team == 'A':
        return teamA_formation.get_ideal_position(player_id, ball_pos, own_players, opponent_players)
    else:
        return teamB_formation.get_ideal_position(player_id, ball_pos, own_players, opponent_players)