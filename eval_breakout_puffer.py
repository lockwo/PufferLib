"""Evaluate a trained JAX PPO model on PufferLib's C Breakout.

Tests that policies trained in the JAX environment transfer to the
original C environment (same observation encoding, same game rules).

Usage:
    python eval_breakout_puffer.py
    python eval_breakout_puffer.py --model breakout_jax_model.eqx --episodes 200
    python eval_breakout_puffer.py --render          # watch it play
    python eval_breakout_puffer.py --stochastic      # sample actions instead of argmax

PufferLib's own training command (for comparison):
    puffer train puffer_breakout
    puffer eval puffer_breakout --model-path <checkpoint>
"""

import argparse
import time

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from ppo_jax import PPOPolicy


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate JAX model on PufferLib C Breakout"
    )
    p.add_argument("--model", type=str, default="breakout_jax_model.eqx")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-envs", type=int, default=1, help="Parallel eval envs")
    p.add_argument("--render", action="store_true")
    p.add_argument(
        "--stochastic", action="store_true", help="Sample actions instead of argmax"
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    from pufferlib.ocean.breakout.breakout import Breakout

    # Breakout obs: 10 + 6*18 = 118
    obs_size = 118
    num_actions = 3

    # Load trained model
    template = PPOPolicy(obs_size, args.hidden_size, num_actions, jax.random.key(0))
    model = eqx.tree_deserialise_leaves(args.model, template)
    print(f"Loaded model from {args.model}")

    # JIT compile inference
    if args.num_envs == 1:

        @jax.jit
        def get_actions_greedy(obs):
            logits, _ = model(obs)
            return jnp.argmax(logits)[None]

        @jax.jit
        def get_actions_stochastic(obs, key):
            logits, _ = model(obs)
            return jax.random.categorical(key, logits)[None]
    else:

        @jax.jit
        def get_actions_greedy(obs):
            logits, _ = jax.vmap(model)(obs)
            return jnp.argmax(logits, axis=-1)

        @jax.jit
        def get_actions_stochastic(obs, key):
            logits, _ = jax.vmap(model)(obs)
            keys = jax.random.split(key, obs.shape[0])
            return jax.vmap(jax.random.categorical)(keys, logits)

    # Create PufferLib C Breakout
    num_envs = args.num_envs
    env = Breakout(
        num_envs=num_envs,
        render_mode="human" if args.render else None,
    )
    env.reset()

    # Episode tracking (per-env)
    episode_returns = []
    episode_lengths = []
    current_returns = np.zeros(num_envs)
    current_lengths = np.zeros(num_envs, dtype=int)
    key = jax.random.key(args.seed)

    max_steps = args.episodes * 5000  # safety limit
    t0 = time.perf_counter()

    for step in range(max_steps):
        if num_envs == 1:
            obs = jnp.array(env.observations[0])
        else:
            obs = jnp.array(env.observations)

        if args.stochastic:
            key, action_key = jax.random.split(key)
            actions = get_actions_stochastic(obs, action_key)
        else:
            actions = get_actions_greedy(obs)

        env.step(np.array(actions, dtype=np.int32))

        if args.render:
            env.render()

        current_returns += env.rewards
        current_lengths += 1

        for i in range(num_envs):
            if env.terminals[i]:
                episode_returns.append(float(current_returns[i]))
                episode_lengths.append(int(current_lengths[i]))
                current_returns[i] = 0.0
                current_lengths[i] = 0

        if len(episode_returns) >= args.episodes:
            break

    elapsed = time.perf_counter() - t0
    total_steps = (step + 1) * num_envs

    if len(episode_returns) == 0:
        print("No episodes completed.")
        env.close()
        return

    returns = np.array(episode_returns)
    lengths = np.array(episode_lengths)

    print(f"\nEvaluated {len(returns)} episodes on PufferLib C Breakout")
    print(f"  Mean return:   {returns.mean():>8.1f} +/- {returns.std():.1f}")
    print(f"  Median return: {np.median(returns):>8.1f}")
    print(f"  Max return:    {returns.max():>8.1f}")
    print(f"  Min return:    {returns.min():>8.1f}")
    print(f"  Mean length:   {lengths.mean():>8.0f} +/- {lengths.std():.0f}")
    print(f"  Steps/sec:     {total_steps / elapsed:>8,.0f}")

    env.close()


if __name__ == "__main__":
    main()
