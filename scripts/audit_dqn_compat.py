#!/usr/bin/env python3
"""Stage 2 DQN audit at scale (Member 4, Tasks 1-2).

1. mask audit: over random positions on 3x3, 4x4 and 5x5 and every rule cell in
   scripts/final_dqn_lib.RULE_CELLS, decode EVERY action id and compare
   ``codec.legal_mask`` with the engine's validator (apply on a clone), and the mask's
   decoded set with ``get_legal_actions()``.
2. alias audit: group the same positions by their observation bytes and report pairs of
   different positions that share an observation but differ in legal ids or successors;
   count positions whose successors differ between the real history and a fresh history
   (repetition is invisible to the observation); and build owner-swapped variants of
   positions with fixed walls to measure the wall-ownership alias.

Writes experiments/results/final_dqn/compat_mask_audit.json and encoding_alias_audit.json.
Example: python scripts/audit_dqn_compat.py --seeds 3:200 4:100 5:40
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.dqn.encoding import encode_observation  # noqa: E402
from scripts.final_dqn_lib import (RULE_CELLS, engine_accepts, provenance,  # noqa: E402
                                   random_positions, restore, write_result)
try:
    from modified_wallz.actions import ActionCodec  # noqa: E402
except ImportError:
    from src.modified_wallz.actions import ActionCodec  # noqa: E402


def obs_hash(engine, mover=None):
    return hashlib.sha1(encode_observation(engine, mover=mover).tobytes()).hexdigest()


def outcome_of(engine, mover):
    if engine.winner is not None:
        return 'win' if engine.winner == mover else 'loss'
    if engine.is_draw:
        return 'draw_' + str(engine.terminal_reason)
    return 'open'


def successor_signature(engine, codec):
    """(legal ids, per-id (successor observation, outcome for the mover))."""
    mover = engine.current_player
    ids = sorted(codec.legal_ids(engine))
    succ = []
    for aid in ids:
        child = engine.clone()
        child.apply_action(codec.decode(aid, engine), update_clock=False)
        succ.append((aid, obs_hash(child), outcome_of(child, mover)))
    return tuple(ids), tuple(succ)


def owner_swapped_variants(engine):
    """Raw keys that swap the owners of one fixed P0 wall and one fixed P1 wall
    (inventories and latest walls unchanged, so the observation is unchanged)."""
    key = engine._get_state_hashable_key()
    (p0, p1, w0, w1, *rest) = key
    l0, l1 = key[6], key[7]
    f0 = [w for w in w0 if w != l0] if engine.rules['A'] else list(w0)
    f1 = [w for w in w1 if w != l1] if engine.rules['A'] else list(w1)
    for a in f0:
        for b in f1:
            nw0 = tuple(b if w == a else w for w in w0)
            nw1 = tuple(a if w == b else w for w in w1)
            yield (p0, p1, nw0, nw1, *rest)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--seeds', nargs='+', default=['3:200', '4:100', '5:40'],
                    help='board:number_of_playout_seeds per board')
    ap.add_argument('--plies', type=int, default=40)
    args = ap.parse_args()
    plan = {int(s.split(':')[0]): int(s.split(':')[1]) for s in args.seeds}

    t0 = time.time()
    mask = {'positions': 0, 'ids_checked': 0, 'mismatches': [], 'collisions': 0,
            'legal_action_types': Counter(), 'per_board': {}}
    alias = {'positions': 0, 'distinct_positions': 0, 'obs_groups_with_several_positions': 0,
             'pairs_legal_differs': 0, 'pairs_successor_differs': 0, 'examples': [],
             'history_dependent_positions': 0, 'history_examples': [],
             'owner_variants_checked': 0, 'owner_variants_legal_differs': 0,
             'owner_variants_by_board': {}, 'owner_examples': []}

    for n, n_seeds in sorted(plan.items()):
        codec = ActionCodec(n)
        board_pos = 0
        for cell, rules in RULE_CELLS.items():
            groups = defaultdict(dict)   # obs hash -> {position_key: (signature, raw key)}
            for seed in range(n_seeds):
                for biased in (True, False):
                    for e in random_positions(n, rules, seed * 2 + int(biased), args.plies, biased):
                        board_pos += 1
                        # --- mask audit
                        m = codec.legal_mask(e)
                        legal = e.get_legal_actions()
                        mask['legal_action_types'].update(a[0] for a in legal)
                        if sum(m) != len(legal):
                            mask['collisions'] += 1
                        dec = set()
                        for aid in range(codec.num_actions):
                            act = codec.decode(aid, e)
                            ok = engine_accepts(e, act)
                            mask['ids_checked'] += 1
                            if ok != m[aid] and len(mask['mismatches']) < 20:
                                mask['mismatches'].append({'N': n, 'cell': cell, 'id': aid,
                                                           'action': act, 'mask': m[aid],
                                                           'engine': ok,
                                                           'key': repr(e._get_state_hashable_key())})
                            if m[aid]:
                                dec.add(act)
                        if dec != set(legal) and len(mask['mismatches']) < 20:
                            mask['mismatches'].append({'N': n, 'cell': cell, 'set_mismatch': True})
                        mask['positions'] += 1
                        # --- alias audit
                        alias['positions'] += 1
                        sig_hist = successor_signature(e, codec)
                        raw = e._get_state_hashable_key()
                        fresh = restore(n, rules, raw)
                        sig_fresh = successor_signature(fresh, codec)
                        if sig_hist != sig_fresh:
                            alias['history_dependent_positions'] += 1
                            if len(alias['history_examples']) < 3:
                                diff = [(a, b) for a, b in zip(sig_hist[1], sig_fresh[1]) if a != b]
                                alias['history_examples'].append({'N': n, 'cell': cell, 'key': repr(raw),
                                                                  'with_history_vs_fresh': diff[:2]})
                        groups[obs_hash(e)][e.position_key()] = (sig_fresh, raw)
                        # owner-swap variants
                        if rules['B'] and e.hop_available[e.current_player]:
                            base_ids = sig_fresh[0]
                            for vk in owner_swapped_variants(e):
                                try:
                                    v = restore(n, rules, vk)
                                except ValueError:
                                    continue
                                if v.is_terminal() or obs_hash(v) != obs_hash(e):
                                    continue
                                alias['owner_variants_checked'] += 1
                                ob = alias['owner_variants_by_board'].setdefault(str(n), [0, 0])
                                ob[0] += 1
                                if tuple(sorted(codec.legal_ids(v))) != base_ids:
                                    alias['owner_variants_legal_differs'] += 1
                                    ob[1] += 1
                                    if len(alias['owner_examples']) < 3:
                                        alias['owner_examples'].append({'N': n, 'cell': cell, 'key': repr(raw),
                                                                        'variant': repr(vk)})
            for h, members in groups.items():
                alias['distinct_positions'] += len(members)
                if len(members) < 2:
                    continue
                alias['obs_groups_with_several_positions'] += 1
                items = list(members.values())
                for i in range(len(items)):
                    for j in range(i + 1, len(items)):
                        (ids_a, succ_a), ka = items[i]
                        (ids_b, succ_b), kb = items[j]
                        if ids_a != ids_b:
                            alias['pairs_legal_differs'] += 1
                        elif succ_a != succ_b:
                            alias['pairs_successor_differs'] += 1
                        else:
                            continue
                        if len(alias['examples']) < 5:
                            alias['examples'].append({'N': n, 'cell': cell, 'a': repr(ka), 'b': repr(kb),
                                                      'legal_differs': ids_a != ids_b})
        mask['per_board'][str(n)] = board_pos
        print(f"N={n}: {board_pos} positions, {time.time() - t0:.0f}s", flush=True)

    mask['legal_action_types'] = dict(mask['legal_action_types'])
    cfg = {'boards_seeds': plan, 'plies': args.plies, 'rule_cells': RULE_CELLS,
           'playouts': 'per seed s: biased type-first playout (seed 2s+1) and uniform playout (seed 2s)'}
    prov = provenance(cfg, seeds=None)
    mask['verdict'] = 'held' if not mask['mismatches'] and mask['collisions'] == 0 else 'FAILED'
    mask['elapsed_seconds'] = round(time.time() - t0, 1)
    write_result('compat_mask_audit.json', {'provenance': prov, 'mask_audit': mask})
    alias['note'] = ('pairs_* compare different canonical positions (fresh history) with equal '
                     'observations. history_dependent_positions: positions whose successor '
                     'outcomes differ between the real game history and a fresh history '
                     '(repetition draws), which the observation cannot show. owner_variants_*: '
                     'positions rebuilt with the owners of two fixed walls swapped; they encode '
                     'identically by construction.')
    write_result('encoding_alias_audit.json', {'provenance': prov, 'alias_audit': alias})
    print('mask verdict:', mask['verdict'], '| positions', mask['positions'], '| ids', mask['ids_checked'])
    print('alias:', {k: v for k, v in alias.items() if not isinstance(v, (list, dict))})


if __name__ == '__main__':
    main()
