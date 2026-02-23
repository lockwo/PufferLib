"""MLP PPO training with adapted PufferLib hyperparams for 1024 envs.

PufferLib uses 8192 agents with ent_coef=0.000569, lr=0.045.
For 1024 envs, we scale LR down and increase ent_coef to prevent entropy collapse.

Usage:
    python train_breakout_jax_debug.py
    python train_breakout_jax_debug.py --total-timesteps 90000000
"""
import argparse
import time

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from breakout_jax import BreakoutJAX
from ppo_jax import PPOPolicy, Trajectory, compute_vtrace_advantages, ppo_loss


def parse_args():
    p = argparse.ArgumentParser(description="MLP PPO on JAX Breakout")
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--num-steps", type=int, default=64)
    p.add_argument("--total-timesteps", type=int, default=90_000_000)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--output", type=str, default="breakout_jax_model_mlp.eqx")
    # Allow easy tuning
    p.add_argument("--lr", type=float, default=None, help="Override LR")
    p.add_argument("--ent-coef", type=float, default=None, help="Override ent_coef")
    return p.parse_args()


def main():
    args = parse_args()

    # Adapted hyperparams for 1024 envs
    # Keep PufferLib's structural hyperparams but use standard PPO lr/ent_coef
    lr = args.lr if args.lr is not None else 2.5e-4
    ent_coef = args.ent_coef if args.ent_coef is not None else 0.01

    # Use PufferLib's gamma/lambda (critical for Breakout's long episodes)
    # Standard PPO values for everything else
    adam_b1 = 0.9
    adam_b2 = 0.999
    adam_eps = 1e-5
    gamma = 0.9997  # PufferLib: long-horizon (episodes ~350-500 steps)
    gae_lambda = 0.95
    clip_coef = 0.1
    vf_clip_coef = 0.1
    vf_coef = 0.5
    max_grad_norm = 0.5
    update_epochs = 4
    num_minibatches = 4

    # Disable V-trace/prio for clean PPO baseline
    vtrace_rho_clip = 1.0
    vtrace_c_clip = 1.0

    num_envs = args.num_envs
    num_steps = args.num_steps
    steps_per_update = num_envs * num_steps
    num_updates = args.total_timesteps // steps_per_update
    minibatch_envs = num_envs // num_minibatches

    print(f"Config: {num_envs} envs, {num_steps} steps, {num_updates} updates, {num_minibatches} mb, MLP")
    print(f"LR: {lr:.6f}, ent_coef: {ent_coef:.6f}")
    print(f"update_epochs={update_epochs}, gamma={gamma}, gae_lambda={gae_lambda}")
    print(f"Total timesteps: {num_updates * steps_per_update:,}")

    key = jax.random.key(args.seed)
    env = BreakoutJAX()
    obs_size = env.observation_size

    key, model_key = jax.random.split(key)
    model = PPOPolicy(obs_size, args.hidden_size, env.num_actions, model_key)

    total_opt_steps = num_updates * update_epochs * num_minibatches
    lr_schedule = optax.cosine_decay_schedule(lr, total_opt_steps)
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(lr_schedule, b1=adam_b1, b2=adam_b2, eps=adam_eps),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    key, init_key = jax.random.split(key)
    init_keys = jax.random.split(init_key, num_envs)
    states, _, _ = jax.vmap(env.reset)(init_keys)

    vmapped_step = jax.vmap(env.step)
    vmapped_observe = jax.vmap(env._observe)

    @eqx.filter_jit
    def collect_rollout(model, states, key):
        def step_fn(carry, step_key):
            states = carry
            obs = vmapped_observe(states)
            logits, values = jax.vmap(model)(obs)

            action_key, env_key = jax.random.split(step_key)
            actions = jax.random.categorical(action_key, logits)
            log_probs_all = jax.nn.log_softmax(logits)
            log_probs = jnp.take_along_axis(
                log_probs_all, actions[:, None], axis=1
            ).squeeze(-1)

            env_keys = jax.random.split(env_key, num_envs)
            new_states, _, rewards, terminated, truncated, _ = vmapped_step(
                states, actions, env_keys
            )
            dones = terminated | truncated
            return new_states, (obs, actions, log_probs, values, rewards, dones)

        keys = jax.random.split(key, num_steps)
        final_states, transitions = jax.lax.scan(step_fn, states, keys)
        return final_states, Trajectory(*transitions)

    @jax.jit
    def compute_gae(rewards, values, dones):
        importance = jnp.ones_like(rewards)
        return compute_vtrace_advantages(
            rewards, values, dones, importance,
            gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip,
        )

    @eqx.filter_jit
    def train_step(model, opt_state, obs, actions, old_log_probs, old_values,
                   advantages, returns, prio_weights):
        (loss, aux), grads = eqx.filter_value_and_grad(ppo_loss, has_aux=True)(
            model, obs, actions, old_log_probs, old_values,
            advantages, returns, prio_weights,
            clip_coef, vf_clip_coef, vf_coef, ent_coef,
        )
        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, loss, aux

    # Track episode returns
    episode_returns = []
    env_episode_returns = jnp.zeros(num_envs)

    total_steps = 0
    best_ep_ret = -float("inf")

    for update in range(num_updates):
        t0 = time.perf_counter()

        key, rollout_key = jax.random.split(key)
        states, trajectory = collect_rollout(model, states, rollout_key)

        # Track episode returns
        for t in range(num_steps):
            env_episode_returns = env_episode_returns + trajectory.rewards[t]
            done_mask = trajectory.dones[t]
            if bool(done_mask.any()):
                done_returns = env_episode_returns[done_mask]
                episode_returns.extend(done_returns.tolist())
            env_episode_returns = jnp.where(done_mask, 0.0, env_episode_returns)

        clipped_rewards = jnp.clip(trajectory.rewards, -1.0, 1.0)

        advantages = compute_gae(clipped_rewards, trajectory.values, trajectory.dones)
        returns = advantages + trajectory.values

        for _epoch in range(update_epochs):
            # Shuffle envs
            key, perm_key = jax.random.split(key)
            perm = jax.random.permutation(perm_key, num_envs)

            for mb_start in range(0, num_envs, minibatch_envs):
                env_idx = perm[mb_start:mb_start + minibatch_envs]

                mb_obs = trajectory.obs[:, env_idx].reshape(-1, obs_size)
                mb_actions = trajectory.actions[:, env_idx].reshape(-1).astype(jnp.int32)
                mb_log_probs = trajectory.log_probs[:, env_idx].reshape(-1)
                mb_old_values = trajectory.values[:, env_idx].reshape(-1)
                mb_adv = advantages[:, env_idx].reshape(-1)
                mb_ret = returns[:, env_idx].reshape(-1)
                mb_prio = jnp.ones_like(mb_adv)

                model, opt_state, loss, aux = train_step(
                    model, opt_state, mb_obs, mb_actions, mb_log_probs,
                    mb_old_values, mb_adv, mb_ret, mb_prio,
                )

        pg_loss, v_loss, entropy, ratio, _, _ = aux
        total_steps += steps_per_update
        elapsed = time.perf_counter() - t0
        sps = steps_per_update / elapsed

        if update % args.log_interval == 0 or update == num_updates - 1:
            num_episodes = int(trajectory.dones.sum())
            if episode_returns:
                recent = episode_returns[-200:]
                ep_ret = sum(recent) / len(recent)
            else:
                ep_ret = 0.0

            print(
                f"Update {update:>4}/{num_updates} | "
                f"Steps {total_steps:>10,} | "
                f"SPS {sps:>8,.0f} | "
                f"Loss {float(loss):>7.3f} | "
                f"PG {float(pg_loss):>7.4f} | "
                f"VF {float(v_loss):>7.4f} | "
                f"Ent {float(entropy):>6.4f} | "
                f"EpRet {ep_ret:>7.1f} | "
                f"Eps {num_episodes}"
            )
            if ep_ret > best_ep_ret:
                best_ep_ret = ep_ret

    # Save model
    eqx.tree_serialise_leaves(args.output, model)
    print(f"\nModel saved to {args.output}")
    print(f"Best mean episode return: {best_ep_ret:.1f}")
    print(f"Total steps: {total_steps:,}")
    if episode_returns:
        final = episode_returns[-200:]
        print(f"Final mean episode return (last 200): {sum(final)/len(final):.1f}")
        print(f"Total episodes: {len(episode_returns)}")


if __name__ == "__main__":
    main()
