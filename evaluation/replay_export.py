"""
Export a logged match to the action format of the browser demo
(github.com/TagalpallewarAbhishek/wallz-custom-game, js/game.js executeAITurn):

    {type: 'MOVE', x, y, isWallHop}   {type: 'WALL', gx, gy, orientation}   {type: 'BLOCK', x, y}

Coordinate mapping (engine -> demo): engine player 0 starts on row 0 and moves first;
the demo's player 1 starts at y = 8 and moves first, so player p -> demo player p + 1,
cell (r, c) -> (x = c, y = N-1-r) and wall (r, c, o) -> (gx = c, gy = N-2-r, o).
The demo has no replay loader, is fixed at 9x9, uses FIFO wall recycling with no
REPLACE, a 1-cell "wall hop" instead of the engine's 2-cell HOP, and a lock that lasts
3 plies. Events are always exported; ``demo_compatible`` says whether the demo's own
rules would reproduce the game.
"""
from evaluation.match_runner import action_from_json

DEMO_BOARD_SIZE = 9
DEMO_LOCK_PLIES = 3


def to_demo_event(action, player, n):
    kind, val = action
    event = {'player': player + 1}
    if kind in ('MOVE', 'HOP'):
        r, c = val
        event.update({'type': 'MOVE', 'x': c, 'y': n - 1 - r, 'isWallHop': False})
        if kind == 'HOP':
            event['engineHop'] = True
    elif kind in ('WALL', 'REPLACE'):
        r, c, o = val
        event.update({'type': 'WALL', 'gx': c, 'gy': n - 2 - r, 'orientation': o})
        if kind == 'REPLACE':
            event['engineReplace'] = True
    elif kind == 'BLOCK':
        r, c = val
        event.update({'type': 'BLOCK', 'x': c, 'y': n - 1 - r})
    else:
        raise ValueError(f"Unknown action {action}")
    return event


def export_demo_replay(result):
    if 'replay' not in result:
        raise ValueError("Match was logged without a replay.")
    cfg = result['config']
    n = cfg['board_size']
    rules = cfg['rules']
    issues = []
    if n != DEMO_BOARD_SIZE:
        issues.append(f"board is {n}x{n}; demo is fixed at 9x9")
    if rules['C'] and rules['K'] != DEMO_LOCK_PLIES:
        issues.append(f"K={rules['K']}; demo locks last {DEMO_LOCK_PLIES} plies")
    if rules['M'] > 10:
        issues.append("wall budget above the demo's 10-wall queue")
    events = []
    for step in result['replay']:
        action = action_from_json(step['action'])
        if action[0] == 'HOP':
            issues.append(f"ply {step['ply']}: engine 2-cell HOP has no demo equivalent")
        if action[0] == 'REPLACE':
            issues.append(f"ply {step['ply']}: REPLACE (latest-wall repositioning) is not in the demo")
        events.append(to_demo_event(action, step['player'], n))
    return {
        'format': 'wallz-custom-game/executeAITurn-actions',
        'match_id': result['match_id'],
        'source': result['source'],
        'outcome': result['outcome'],
        'demo_compatible': not issues,
        'incompatibilities': issues,
        'events': events,
    }
