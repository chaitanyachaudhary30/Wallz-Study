#!/usr/bin/env python3
"""DQN Training Entry Point (P1.11).

Executes self-play training with periodic evaluation against greedy and random baselines.
Outputs checkpoints conforming to C9 and logs metrics to train_log.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Dict
import yaml

from agents.dqn.agent import DQNAgent, save_checkpoint
from agents.dqn.self_play import SelfPlayManager
from agents.dqn.trainer import DQNTrainer, DQNConfig
from evaluation.match_runner import MatchConfig, run_match, make_agent


def load_yaml_config(config_path: str, board_size: int | None = None, block_name: str | None = None) -> Dict[str, Any]:
    """Loads configuration and merges default block with board-specific block."""
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    cfg = dict(data.get('default', {}))

    # Determine target block
    target_block = block_name
    if target_block is None and board_size is not None:
        target_block = f"board_{board_size}"

    if target_block and target_block in data:
        cfg.update(data[target_block])

    return cfg


def evaluate_agent(
    model,
    board_size: int,
    rules: dict,
    num_matches: int = 10,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Evaluates the current online network against greedy and random baselines."""
    eval_results = {}
    opponents = ['greedy', 'random']

    for opp in opponents:
        wins = 0
        draws = 0
        losses = 0
        for i in range(num_matches):
            p0_is_dqn = (i % 2 == 0)
            dqn_agent = DQNAgent(
                player_id=0 if p0_is_dqn else 1,
                model=model,
                board_size=board_size,
                rules=rules,
                device=device,
            )

            p0_spec = "dqn_in_memory" if p0_is_dqn else opp
            p1_spec = opp if p0_is_dqn else "dqn_in_memory"

            match_cfg = MatchConfig(
                board_size=board_size,
                rules=rules,
                p0=p0_spec,
                p1=p1_spec,
                max_plies=150,
                opening_seed=i * 1001,
            )

            opp_seat = 1 if p0_is_dqn else 0
            opp_agent, opp_identity = make_agent(
                opp,
                player_id=opp_seat,
                config=match_cfg,
                solutions=None,
                match_seed=i * 1001,
            )

            agents = {
                (0 if p0_is_dqn else 1): dqn_agent,
                opp_seat: opp_agent,
            }
            identities = {
                (0 if p0_is_dqn else 1): {'name': 'dqn', 'spec': 'in_memory', 'params': {}, 'codec_version': 1},
                opp_seat: opp_identity,
            }

            res = run_match(match_cfg, agents=agents, identities=identities)
            outcome = res.get('outcome', {})
            cat = outcome.get('category', '')

            if cat == 'goal':
                if outcome.get('winner') == (0 if p0_is_dqn else 1):
                    wins += 1
                else:
                    losses += 1
            elif 'draw' in cat or cat == 'truncated':
                draws += 1
            else:
                losses += 1

        eval_results[f"vs_{opp}"] = {
            'matches': num_matches,
            'win_rate': round(wins / num_matches, 3),
            'draw_rate': round(draws / num_matches, 3),
            'loss_rate': round(losses / num_matches, 3),
        }

    return eval_results


def main():
    parser = argparse.ArgumentParser(description="Train DQN agent on Modified Wallz.")
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Path to config YAML")
    parser.add_argument("--board-size", type=int, default=3, help="Board size (3, 4, 5, 9)")
    parser.add_argument("--steps", type=int, default=None, help="Override total env steps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu' or 'cuda')")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for checkpoints and logs")
    parser.add_argument("--eval-freq", type=int, default=None, help="Env steps between evaluations")
    parser.add_argument("--smoke", action="store_true", help="Run quick 2000-step smoke test")
    args = parser.parse_args()

    cfg_dict = load_yaml_config(args.config, board_size=args.board_size)

    # CLI Overrides
    if args.steps is not None:
        cfg_dict['total_steps'] = args.steps
    if args.seed is not None:
        cfg_dict['seed'] = args.seed
    if args.device is not None:
        cfg_dict['device'] = args.device
    if args.eval_freq is not None:
        cfg_dict['eval_freq'] = args.eval_freq

    if args.smoke:
        cfg_dict['total_steps'] = 2000
        cfg_dict['batch_size'] = 32
        cfg_dict['buffer_capacity'] = 5000
        cfg_dict['min_buffer_size'] = 64
        cfg_dict['epsilon_decay_steps'] = 1000
        cfg_dict['eval_freq'] = 1000
        cfg_dict['eval_matches'] = 4
        cfg_dict['snapshot_interval'] = 500

    board_size = cfg_dict['board_size']
    rules = cfg_dict.get('rules', {'A': True, 'B': True, 'C': True, 'K': 1, 'M': 1})
    total_steps = cfg_dict.get('total_steps', 200_000)
    seed = cfg_dict.get('seed', 42)

    run_id = f"dqn_N{board_size}_s{seed}_{int(time.time())}"
    output_dir = Path(args.output_dir if args.output_dir else f"checkpoints/{run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "train_log.jsonl"

    print(f"============================================================")
    print(f"Starting DQN Training: N={board_size}, steps={total_steps}, seed={seed}")
    print(f"Output Directory: {output_dir}")
    print(f"============================================================")

    # Construct DQNConfig
    dqn_config = DQNConfig(
        board_size=board_size,
        rules=rules,
        lr=cfg_dict.get('lr', 1e-4),
        gamma=cfg_dict.get('gamma', 0.99),
        batch_size=cfg_dict.get('batch_size', 128),
        target_update_freq=cfg_dict.get('target_update_freq', 1000),
        train_freq=cfg_dict.get('train_freq', 4),
        buffer_capacity=cfg_dict.get('buffer_capacity', 100_000),
        min_buffer_size=cfg_dict.get('min_buffer_size', 1_000),
        epsilon_start=cfg_dict.get('epsilon_start', 1.0),
        epsilon_end=cfg_dict.get('epsilon_end', 0.05),
        epsilon_decay_steps=cfg_dict.get('epsilon_decay_steps', 20_000),
        grad_clip=cfg_dict.get('grad_clip', 10.0),
        huber_delta=cfg_dict.get('huber_delta', 1.0),
        double_dqn=cfg_dict.get('double_dqn', True),
        dueling=cfg_dict.get('dueling', False),
        channels=cfg_dict.get('channels', 64),
        blocks=cfg_dict.get('blocks', 3),
        hidden=cfg_dict.get('hidden', 256),
        mirror_prob=cfg_dict.get('mirror_prob', 0.5 if board_size % 2 == 1 else 0.0),
        seed=seed,
        device=cfg_dict.get('device', 'cpu'),
    )

    trainer = DQNTrainer(dqn_config)
    sp_manager = SelfPlayManager(
        trainer=trainer,
        snapshot_interval=cfg_dict.get('snapshot_interval', 20_000),
        snapshot_pool_size=cfg_dict.get('snapshot_pool_size', 10),
        mixture=tuple(cfg_dict.get('mixture', [0.5, 0.4, 0.1])),
    )

    eval_freq = cfg_dict.get('eval_freq', 10_000)
    eval_matches = cfg_dict.get('eval_matches', 10)
    last_eval_step = 0
    episode = 0

    with open(log_file, 'a', encoding='utf-8') as f_log:
        while trainer.total_env_steps < total_steps:
            ep_res = sp_manager.run_episode()
            episode += 1

            # Log episode metric
            metric_entry = {
                'type': 'episode',
                'episode': episode,
                'env_steps': trainer.total_env_steps,
                'grad_steps': trainer.total_grad_steps,
                'epsilon': round(trainer.get_epsilon(), 4),
                'length': ep_res['episode_length'],
                'learner_won': ep_res['learner_won'],
                'opponent': ep_res['opponent_type'],
                'avg_loss': round(ep_res['avg_loss'], 5) if ep_res['avg_loss'] is not None else None,
                'terminal_reason': ep_res['terminal_reason'],
                'timestamp': time.time(),
            }
            f_log.write(json.dumps(metric_entry) + "\n")
            f_log.flush()

            if episode % 20 == 0 or trainer.total_env_steps >= total_steps:
                print(
                    f"Ep {episode:5d} | Steps: {trainer.total_env_steps:7d}/{total_steps} | "
                    f"Eps: {trainer.get_epsilon():.3f} | Opp: {ep_res['opponent_type']:8s} | "
                    f"Len: {ep_res['episode_length']:3d} | Loss: {ep_res['avg_loss']}"
                )

            # Periodic Evaluation & Checkpoint
            if trainer.total_env_steps - last_eval_step >= eval_freq or trainer.total_env_steps >= total_steps:
                last_eval_step = trainer.total_env_steps
                print(f"--- Running Evaluation at step {trainer.total_env_steps} ---")
                eval_out = evaluate_agent(
                    model=trainer.online_net,
                    board_size=board_size,
                    rules=rules,
                    num_matches=eval_matches,
                    device=trainer.config.device,
                )
                print(f"Eval results: {eval_out}")

                eval_entry = {
                    'type': 'eval',
                    'env_steps': trainer.total_env_steps,
                    'grad_steps': trainer.total_grad_steps,
                    'eval': eval_out,
                    'timestamp': time.time(),
                }
                f_log.write(json.dumps(eval_entry) + "\n")
                f_log.flush()

                # Save intermediate checkpoint
                meta = {
                    'N': board_size,
                    'rules': rules,
                    'gamma': dqn_config.gamma,
                    'double': dqn_config.double_dqn,
                    'dueling': dqn_config.dueling,
                    'channels': dqn_config.channels,
                    'blocks': dqn_config.blocks,
                    'hidden': dqn_config.hidden,
                    'seed': seed,
                    'step': trainer.total_env_steps,
                }
                save_checkpoint(
                    save_dir=output_dir,
                    model=trainer.online_net,
                    meta=meta,
                    filename=f"step_{trainer.total_env_steps}.pt",
                )

    # Final Checkpoint
    final_meta = {
        'N': board_size,
        'rules': rules,
        'gamma': dqn_config.gamma,
        'double': dqn_config.double_dqn,
        'dueling': dqn_config.dueling,
        'channels': dqn_config.channels,
        'blocks': dqn_config.blocks,
        'hidden': dqn_config.hidden,
        'seed': seed,
        'step': trainer.total_env_steps,
    }
    save_checkpoint(
        save_dir=output_dir,
        model=trainer.online_net,
        meta=final_meta,
        filename="final.pt",
    )
    # Also save model.pt as canonical default
    save_checkpoint(
        save_dir=output_dir,
        model=trainer.online_net,
        meta=final_meta,
        filename="model.pt",
    )

    print(f"Training complete. Final checkpoint saved to {output_dir}")


if __name__ == "__main__":
    main()
