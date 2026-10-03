import math
import random
import numpy as np

# ---------- Constants ----------
FIELD_WIDTH  = 105.0
FIELD_HEIGHT = 68.0
GOAL_WIDTH   = 7.32
GOAL_HEIGHT  = 2.44   # not used in 2D, but for reference
GOAL_Y_CENTER = 34.0
GOAL_Y_MIN   = GOAL_Y_CENTER - GOAL_WIDTH / 2
GOAL_Y_MAX   = GOAL_Y_CENTER + GOAL_WIDTH / 2
SPACING_THRESHOLD = 2.0   # distance (m) below which two teammates are considered "bunched"

DT = 0.1

# Player dynamics
MAX_SPEED_FREE  = 8.0    # m/s
MAX_SPEED_DRIBBLE = 6.5  # slower with ball
ACCELERATION_FREE = 5.0  # m/s²
ACCELERATION_DRIBBLE = 4.0
AGILITY = 10.0            # max angular velocity (rad/s) for velocity change

# Ball dynamics
PLAYER_RADIUS = 0.5
BALL_RADIUS   = 0.11
AIR_DENSITY = 1.225
BALL_MASS = 0.45           # kg
BALL_RADIUS = 0.11
BALL_CROSS_SECTION = math.pi * BALL_RADIUS**2
DRAG_COEFF = 0.25          # typical for smooth sphere
ROLLING_FRICTION = 0.85    # factor per step when on ground
SPIN_FACTOR = 0.0003       # very small, for realistic curve
MAX_BALL_SPEED = 35.0      # hard shot

# Pass & shot parameters
PASS_SPEED = 22.0
SHOT_SPEED = 28.0
PASS_ACCURACY = 0.8        # base accuracy, reduced by distance & pressure
SHOT_ACCURACY = 0.6
PASS_TIME_STEPS = 5         # max steps for a pass to be considered "completed"
PASS_ARRIVE_DIST = 1.5
INTERCEPT_DIST = 1.2       # distance an opponent must be from ball path to attempt interception
REACTION_TIME = 0.2        # seconds, simulated as steps delay

# Tackle
TACKLE_DIST = 1.0
TACKLE_ANGLE_THRESH = 60   # degrees; attacker must be within this angle of approach
TACKLE_SUCCESS_BASE = 0.7

# Offside
OFFSIDE_LINE_BUF = 0.0     # level is onside

# Episode
MAX_EPISODE_STEPS = 2000
MAX_GOAL_DIFF = 5

# Helper
def clamp(val, low, high):
    return max(low, min(high, val))

# ---------- Initialisation ----------
def init_player(id, team, x, y, role=''):
    return {
        'id': id,
        'team': team,
        'role': role,
        'pos': np.array([x, y], dtype=float),
        'vel': np.zeros(2, dtype=float),
        'has_ball': False,
        'max_speed': MAX_SPEED_FREE,
    }

def init_team_a():
    # 4-3-3
    base = [
        (5.0, 34.0, 'GK'),
        (15.0, 10.0, 'LB'),
        (15.0, 25.0, 'CB'),
        (15.0, 43.0, 'CB'),
        (15.0, 58.0, 'RB'),
        (30.0, 15.0, 'CM'),
        (30.0, 34.0, 'CM'),
        (30.0, 53.0, 'CM'),
        (45.0, 10.0, 'LW'),
        (50.0, 34.0, 'ST'),
        (45.0, 58.0, 'RW'),
    ]
    players = []
    for i, (x, y, role) in enumerate(base):
        x += random.uniform(-1, 1)
        y += random.uniform(-1, 1)
        players.append(init_player(i, 'A', x, y, role))
    return players

def init_team_b():
    # 4-4-2 mirror
    base = [
        (100.0, 34.0, 'GK'),
        (90.0, 10.0, 'RB'),
        (90.0, 25.0, 'CB'),
        (90.0, 43.0, 'CB'),
        (90.0, 58.0, 'LB'),
        (75.0, 15.0, 'RM'),
        (75.0, 34.0, 'CM'),
        (75.0, 53.0, 'LM'),
        (60.0, 20.0, 'RW'),
        (55.0, 34.0, 'ST'),
        (60.0, 48.0, 'LW'),
    ]
    players = []
    for i, (x, y, role) in enumerate(base):
        x += random.uniform(-1, 1)
        y += random.uniform(-1, 1)
        players.append(init_player(i, 'B', x, y, role))
    return players

def init_ball():
    return {
        'pos': np.array([FIELD_WIDTH/2, FIELD_HEIGHT/2], dtype=float),
        'vel': np.zeros(2, dtype=float),
        'last_kicker_team': None,
        'spin': np.zeros(2),   # spin vector (arbitrary)
    }

# ---------- Player Movement ----------
def apply_movement(player, desired_vel):
    """Smoothly accelerate towards desired velocity with agility constraints."""
    current = player['vel']
    max_acc = ACCELERATION_DRIBBLE if player['has_ball'] else ACCELERATION_FREE
    max_speed = MAX_SPEED_DRIBBLE if player['has_ball'] else player['max_speed']

    # Clamp desired velocity to max speed
    desired = np.array(desired_vel)
    speed = np.linalg.norm(desired)
    if speed > max_speed:
        desired = desired / speed * max_speed

    # Direction change limit (agility)
    if np.linalg.norm(current) > 0.1 and np.linalg.norm(desired) > 0.1:
        current_dir = current / np.linalg.norm(current)
        desired_dir = desired / np.linalg.norm(desired)
        angle = math.acos(np.clip(np.dot(current_dir, desired_dir), -1, 1))
        max_angle = AGILITY * DT
        if angle > max_angle:
            # Rotate desired direction towards current direction
            rotation_axis = np.cross(np.append(current_dir, 0), np.append(desired_dir, 0))[2]
            if rotation_axis > 0:
                rot_matrix = np.array([[math.cos(max_angle), -math.sin(max_angle)],
                                       [math.sin(max_angle), math.cos(max_angle)]])
            else:
                rot_matrix = np.array([[math.cos(max_angle), math.sin(max_angle)],
                                       [-math.sin(max_angle), math.cos(max_angle)]])
            desired = speed * (rot_matrix @ desired_dir)

    # Acceleration limit
    vel_diff = desired - current
    diff_mag = np.linalg.norm(vel_diff)
    max_diff = max_acc * DT
    if diff_mag > max_diff:
        vel_diff = vel_diff / diff_mag * max_diff
    player['vel'] = current + vel_diff

def update_player_positions(players):
    """Move players according to their velocities, clamp inside field."""
    for p in players:
        p['pos'] += p['vel'] * DT
        p['pos'][0] = clamp(p['pos'][0], 0, FIELD_WIDTH)
        p['pos'][1] = clamp(p['pos'][1], 0, FIELD_HEIGHT)

# ---------- Ball Physics ----------
def update_ball(ball, players):
    """Apply drag, roll, spin, and collision with players. No goal check here."""
    # Air drag
    speed = np.linalg.norm(ball['vel'])
    if speed > 0.01:
        drag_force = 0.5 * AIR_DENSITY * BALL_CROSS_SECTION * DRAG_COEFF * speed**2
        drag_acc = drag_force / BALL_MASS
        drag_vec = -ball['vel'] / speed * drag_acc * DT
        ball['vel'] += drag_vec
    # Rolling friction (simulate ground contact)
    ball['vel'] *= ROLLING_FRICTION
    # Spin (Magnus effect, simplified)
    if np.linalg.norm(ball['spin']) > 0:
        perp = np.array([ball['spin'][1], -ball['spin'][0]])
        ball['vel'] += SPIN_FACTOR * perp * np.linalg.norm(ball['vel']) * DT
        ball['spin'] *= 0.99
    ball['pos'] += ball['vel'] * DT

    # Check collision with players (bounce)
    for p in players:
        dist = np.linalg.norm(ball['pos'] - p['pos'])
        min_dist = PLAYER_RADIUS + BALL_RADIUS
        if dist < min_dist and np.linalg.norm(ball['vel']) > 1.0:
            # Elastic bounce with player (player mass = infinite)
            normal = (ball['pos'] - p['pos']) / (dist + 1e-6)
            v_rel = ball['vel'] - p['vel']
            vn = np.dot(v_rel, normal)
            if vn < 0:   # ball moving towards player
                # reflect
                ball['vel'] -= 2 * vn * normal
                # add player velocity influence
                ball['vel'] += p['vel'] * 0.3
                # separate
                ball['pos'] = p['pos'] + normal * (min_dist + 0.01)

    # Out-of-bounds will be handled in environment

# ---------- Possession & Tackling ----------
def check_possession(players, ball):
    """Give possession to nearest player of the opposite team from last kicker."""
    min_dist = float('inf')
    possessor = -1
    for p in players:
        if p['team'] == ball['last_kicker_team']:
            continue
        d = np.linalg.norm(p['pos'] - ball['pos'])
        if d < PLAYER_RADIUS + BALL_RADIUS + 0.2 and d < min_dist:
            min_dist = d
            possessor = p['id']
    return possessor

def handle_tackles(players, ball):
    """Realistic tackle logic: successful if attacker is within tackle distance, 
    approaching from a suitable angle, and close enough to the carrier."""
    carrier = None
    for p in players:
        if p['has_ball']:
            carrier = p
            break
    if not carrier:
        return False

    for p in players:
        if p['team'] == carrier['team'] or p['id'] == carrier['id']:
            continue
        dist = np.linalg.norm(p['pos'] - carrier['pos'])
        if dist < TACKLE_DIST:
            # Angle check: defender should be facing the carrier or at least moving toward them
            carrier_to_def = p['pos'] - carrier['pos']
            if np.linalg.norm(carrier_to_def) < 0.01:
                continue
            angle = math.degrees(math.acos(np.dot(p['vel']/max(np.linalg.norm(p['vel']),1e-6), 
                                            carrier_to_def/np.linalg.norm(carrier_to_def))))
            if angle < TACKLE_ANGLE_THRESH:
                prob = TACKLE_SUCCESS_BASE * (1 - dist/TACKLE_DIST)
                if random.random() < prob:
                    carrier['has_ball'] = False
                    p['has_ball'] = True
                    ball['last_kicker_team'] = carrier['team']
                    return True
    return False

# ---------- Passing ----------
def execute_pass(passer, target_id, team_players, opponents, ball):
    """Launch a pass towards the target's position, with error based on pressure and distance."""
    target = None
    for p in team_players:
        if p['id'] == target_id:
            target = p
            break
    if not target:
        return 'invalid'

    target_pos = target['pos'].copy()
    dist = np.linalg.norm(target_pos - passer['pos'])
    # Error: reduces accuracy
    pressure = 0
    for opp in opponents:
        if np.linalg.norm(opp['pos'] - passer['pos']) < 2.0:
            pressure += 1
    accuracy = PASS_ACCURACY * (1 - 0.3*min(pressure,3)/3) * (1 - 0.005*dist)   # distance penalty
    error = (1 - accuracy) * random.uniform(-1,1) * 2.0   # meters of deviation
    # Add lateral error to target position
    dir_vec = target_pos - passer['pos']
    perp = np.array([-dir_vec[1], dir_vec[0]]) / (np.linalg.norm(dir_vec)+1e-6)
    target_pos += perp * error

    direction = target_pos - passer['pos']
    dir_len = np.linalg.norm(direction)
    if dir_len == 0:
        return 'invalid'
    vel = direction / dir_len * PASS_SPEED
    ball['vel'] = vel
    ball['pos'] = passer['pos'].copy()
    ball['spin'] = np.array([0.0, random.uniform(-0.5,0.5)])   # slight random spin
    return {
        'from': passer['id'],
        'to': target_id,
        'steps_left': PASS_TIME_STEPS,
        'intercepted': False,
        'arrived': False,
    }

def check_pass_interception(ball, opponents):
    """Check if any opponent is within interception distance of the ball,
    and is facing/reacting to the ball."""
    for opp in opponents:
        dist = np.linalg.norm(opp['pos'] - ball['pos'])
        # Reaction time: if opponent is stationary or not looking, they might miss
        # Simplified: just distance check
        if dist < INTERCEPT_DIST:
            return True
    return False

def handle_pass_in_env(pass_info, players, ball, opponents):
    """Update the pass state each step, using realistic interception."""
    pass_info['steps_left'] -= 1
    target = next((p for p in players if p['id'] == pass_info['to']), None)
    if not target:
        return 'timeout'

    if np.linalg.norm(ball['pos'] - target['pos']) < PASS_ARRIVE_DIST:
        target['has_ball'] = True
        ball['vel'] = np.zeros(2)
        ball['spin'] = np.zeros(2)
        return 'success'

    if check_pass_interception(ball, opponents):
        pass_info['intercepted'] = True
        closest = min(opponents, key=lambda o: np.linalg.norm(o['pos'] - ball['pos']))
        closest['has_ball'] = True
        ball['vel'] = np.zeros(2)
        ball['spin'] = np.zeros(2)
        return 'intercepted'

    if pass_info['steps_left'] <= 0:
        return 'timeout'
    return 'ongoing'

# ---------- Shooting ----------
def execute_shoot(player, attacking_right, ball):
    """Shoot with accuracy decreasing with distance and pressure. Include chance of post."""
    if attacking_right:
        goal_center = np.array([FIELD_WIDTH, GOAL_Y_CENTER])
    else:
        goal_center = np.array([0.0, GOAL_Y_CENTER])
    dist = np.linalg.norm(goal_center - player['pos'])
    pressure = sum(1 for p in opponents(player['team']) if np.linalg.norm(p['pos'] - player['pos']) < 2.0)
    accuracy = SHOT_ACCURACY * (1 - 0.5*min(pressure,3)/3) * (1 - 0.01*dist)
    error_y = random.uniform(-1,1) * (1-accuracy) * 3.0   # metres off target
    target_y = clamp(GOAL_Y_CENTER + error_y, GOAL_Y_MIN+0.3, GOAL_Y_MAX-0.3)
    target_pos = np.array([goal_center[0], target_y])
    direction = target_pos - player['pos']
    shot_speed = SHOT_SPEED * random.uniform(0.9, 1.1)
    vel = direction / np.linalg.norm(direction) * shot_speed
    ball['vel'] = vel
    ball['pos'] = player['pos'].copy()
    ball['spin'] = np.array([0.0, random.uniform(-1,1)])
    ball['last_kicker_team'] = player['team']

def opponents(team):
    """Helper placeholder for team B logic – we don't have global access, so this is for conceptual use.
    In the environment, you'll pass the opponent list directly."""
    pass

# ---------- Goal & Offside ----------
def check_goal(ball, for_team_A):
    if for_team_A:
        return ball['pos'][0] >= FIELD_WIDTH and GOAL_Y_MIN <= ball['pos'][1] <= GOAL_Y_MAX
    else:
        return ball['pos'][0] <= 0 and GOAL_Y_MIN <= ball['pos'][1] <= GOAL_Y_MAX

def is_offside(passer, receiver, defenders):
    """FIFA rule: offside if receiver is nearer to opponent's goal line than both the ball and the second-last opponent
    (excluding the goalkeeper) at the moment the ball is played."""
    if passer['team'] == 'A':
        opponent_goal_line = FIELD_WIDTH
        # defenders is list of Team B players
        field_players = [p for p in defenders if p.get('role') != 'GK']
        if len(field_players) < 2:
            return False   # need at least two outfield opponents
        # Sort by x coordinate (for Team B defending left, smaller x = closer to own goal)
        sorted_x = sorted([p['pos'][0] for p in field_players])
        second_last_x = sorted_x[-2]   # second last outfield player (penultimate)
        ball_x = passer['pos'][0]      # ball is at passer's feet
        # Offside conditions: receiver is ahead of the ball AND ahead of second-last defender
        if receiver['pos'][0] > ball_x and receiver['pos'][0] > second_last_x:
            return True
        return False
    else:   # Team B attacking left
        opponent_goal_line = 0.0
        field_players = [p for p in defenders if p.get('role') != 'GK']
        if len(field_players) < 2:
            return False
        sorted_x = sorted([p['pos'][0] for p in field_players])
        second_last_x = sorted_x[1]   # second last (closer to B's goal is larger x, so second last is index 1)
        ball_x = passer['pos'][0]
        if receiver['pos'][0] < ball_x and receiver['pos'][0] < second_last_x:
            return True
        return False

# ---------- Out-of-bounds (realistic restarts) ----------
def out_of_bounds(ball):
    """Return (True, restart_type, team_possession) based on where ball went out.
    restart_type: 'throw-in', 'goal-kick', 'corner', 'kick-off'
    team_possession: 'A' or 'B' who gets the ball."""
    x, y = ball['pos']
    # L/R touchlines
    if x < 0:
        if GOAL_Y_MIN <= y <= GOAL_Y_MAX:
            # Ball went into left goal (already handled by goal checker)
            return False, None, None
        elif y < 0 or y > FIELD_HEIGHT:
            # corner area? simplified
            return True, 'corner' if y < FIELD_HEIGHT/2 else 'corner', 'A' if x < 0 else 'B'
        else:
            return True, 'throw-in', 'A' if x < 0 else 'B'
    if x > FIELD_WIDTH:
        if GOAL_Y_MIN <= y <= GOAL_Y_MAX:
            return False, None, None
        elif y < 0 or y > FIELD_HEIGHT:
            return True, 'corner', 'B' if x > FIELD_WIDTH else 'A'
        else:
            return True, 'throw-in', 'B' if x > FIELD_WIDTH else 'A'
    # End lines (goal lines)
    if y < 0 or y > FIELD_HEIGHT:
        return True, 'throw-in', 'A' if y < 0 else 'B'   # simple throw-in
    return False, None, None