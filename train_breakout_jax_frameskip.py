"""Train PPO with frameskip (action repeat) on JAX Breakout.

Each policy decision is repeated for --frameskip frames, accumulating rewards.
This reduces effective episode length and makes credit assignment easier.
Uses the MLP policy (no LSTM).

Usage:
    python train_breakout_jax_frameskip.py
    python train_breakout_jax_frameskip.py --frameskip 4
    python train_breakout_jax_frameskip.py --num-envs 2048 --total-timesteps 20000000
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
    p = argparse.ArgumentParser(description="Train PPO+frameskip on JAX Breakout")
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument(
        "--num-steps", type=int, default=64, help="Rollout length (policy steps)"
    )
    p.add_argument("--frameskip", type=int, default=4, help="Action repeat frames")
    p.add_argument("--total-timesteps", type=int, default=10_000_000)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="breakout_jax_model.eqx")
    p.add_argument("--log-interval", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()

    # --- Hyperparameters from PufferLib's Protein-swept breakout.ini ---
    lr = 1e-3
    adam_b1 = 0.895
    adam_b2 = 0.9
    adam_eps = 1e-4
    gamma = 0.9997
    gae_lambda = 0.748
    clip_coef = 0.197
    vf_clip_coef = 2.178
    vf_coef = 1.683
    ent_coef = 5.69e-4
    max_grad_norm = 2.236
    vtrace_rho_clip = 0.788
    vtrace_c_clip = 2.878
    prio_alpha = 0.990
    prio_beta0 = 0.1
    update_epochs = 1
    minibatch_envs = min(512, args.num_envs)

    num_envs = args.num_envs
    num_steps = args.num_steps
    frameskip = args.frameskip
    steps_per_update = num_envs * num_steps  # policy steps
    env_steps_per_update = steps_per_update * frameskip  # actual env frames
    num_updates = args.total_timesteps // env_steps_per_update

    num_minibatches = num_envs // minibatch_envs

    print(
        f"Config: {num_envs} envs, {num_steps} policy steps/rollout, "
        f"frameskip={frameskip}, {num_updates} updates, {num_minibatches} mb/epoch"
    )
    print(f"Total env frames: {num_updates * env_steps_per_update:,}")

    # --- Initialize ---
    key = jax.random.key(args.seed)
    env = BreakoutJAX()
    obs_size = env.observation_size

    key, model_key = jax.random.split(key)
    model = PPOPolicy(obs_size, args.hidden_size, env.num_actions, model_key)

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(lr, b1=adam_b1, b2=adam_b2, eps=adam_eps),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Reset all environments
    key, init_key = jax.random.split(key)
    init_keys = jax.random.split(init_key, num_envs)
    states, _, _ = jax.vmap(env.reset)(init_keys)

    # Pre-vmap environment functions
    vmapped_step = jax.vmap(env.step)
    vmapped_observe = jax.vmap(env._observe)

    # --- JIT-compiled core functions ---

    @eqx.filter_jit
    def collect_rollout(model, states, key):
        """Collect num_steps of experience with frameskip action repeat."""

        def step_fn(carry, step_key):
            states = carry
            obs = vmapped_observe(states)
            logits, values = jax.vmap(model)(obs)

            action_key, skip_key = jax.random.split(step_key)
            actions = jax.random.categorical(action_key, logits)
            log_probs_all = jax.nn.log_softmax(logits)
            log_probs = jnp.take_along_axis(
                log_probs_all, actions[:, None], axis=1
            ).squeeze(-1)

            # Repeat action for frameskip frames, accumulate rewards
            skip_keys = jax.random.split(skip_key, frameskip)

            def skip_fn(carry, sk):
                states, total_reward, any_done = carry
                env_keys = jax.random.split(sk, num_envs)
                new_states, _, rewards, terminated, truncated, _ = vmapped_step(
                    states, actions, env_keys
                )
                new_done = (terminated | truncated).astype(jnp.float32)
                # Don't accumulate rewards after first done (env auto-resets)
                total_reward = total_reward + rewards * (1.0 - any_done)
                any_done = jnp.maximum(any_done, new_done)
                return (new_states, total_reward, any_done), None

            (new_states, total_rewards, dones_float), _ = jax.lax.scan(
                skip_fn,
                (states, jnp.zeros(num_envs), jnp.zeros(num_envs)),
                skip_keys,
            )
            dones = dones_float > 0.5

            return new_states, (obs, actions, log_probs, values, total_rewards, dones)

        keys = jax.random.split(key, num_steps)
        final_states, transitions = jax.lax.scan(step_fn, states, keys)
        return final_states, Trajectory(*transitions)

    @jax.jit
    def compute_advantages(rewards, values, dones, importance):
        return compute_vtrace_advantages(
            rewards,
            values,
            dones,
            importance,
            gamma,
            gae_lambda,
            vtrace_rho_clip,
            vtrace_c_clip,
        )

    @eqx.filter_jit
    def train_step(
        model,
        opt_state,
        obs,
        actions,
        old_log_probs,
        old_values,
        advantages,
        returns,
        prio_weights,
    ):
        (loss, aux), grads = eqx.filter_value_and_grad(ppo_loss, has_aux=True)(
            model,
            obs,
            actions,
            old_log_probs,
            old_values,
            advantages,
            returns,
            prio_weights,
            clip_coef,
            vf_clip_coef,
            vf_coef,
            ent_coef,
        )
        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, loss, aux

    # --- Training loop ---

    total_env_steps = 0
    best_mean_reward = -float("inf")

    for update in range(num_updates):
        t0 = time.perf_counter()

        # 1. Collect rollout
        key, rollout_key = jax.random.split(key)
        states, trajectory = collect_rollout(model, states, rollout_key)

        # Reward clipping
        clipped_rewards = jnp.clip(trajectory.rewards, -1.0, 1.0)

        # 2. Initialize retracking state
        ratios = jnp.ones_like(clipped_rewards)
        values = trajectory.values

        # Anneal priority beta
        anneal_beta = prio_beta0 + (1 - prio_beta0) * prio_alpha * update / max(
            num_updates - 1, 1
        )

        # 3. PPO epochs with ratio/value retracking
        for _epoch in range(update_epochs):
            for _mb in range(num_minibatches):
                advantages = compute_advantages(
                    clipped_rewards, values, trajectory.dones, ratios
                )
                returns = advantages + values

                adv_per_env = jnp.abs(advantages).sum(axis=0)
                weights = jnp.nan_to_num(adv_per_env**prio_alpha, 0.0, 0.0, 0.0)
                probs = (weights + 1e-6) / (weights.sum() + 1e-6)

                key, mb_key = jax.random.split(key)
                env_idx = jax.random.choice(
                    mb_key,
                    num_envs,
                    shape=(minibatch_envs,),
                    p=probs,
                    replace=False,
                )

                mb_prio = (num_envs * probs[env_idx]) ** -anneal_beta
                mb_prio_flat = jnp.repeat(mb_prio, num_steps)

                mb_obs = trajectory.obs[:, env_idx].reshape(-1, obs_size)
                mb_actions = (
                    trajectory.actions[:, env_idx].reshape(-1).astype(jnp.int32)
                )
                mb_log_probs = trajectory.log_probs[:, env_idx].reshape(-1)
                mb_old_values = values[:, env_idx].reshape(-1)
                mb_adv = advantages[:, env_idx].reshape(-1)
                mb_ret = returns[:, env_idx].reshape(-1)

                model, opt_state, loss, aux = train_step(
                    model,
                    opt_state,
                    mb_obs,
                    mb_actions,
                    mb_log_probs,
                    mb_old_values,
                    mb_adv,
                    mb_ret,
                    mb_prio_flat,
                )

                pg_loss, v_loss, entropy, new_ratio, _, new_values = aux
                ratios = ratios.at[:, env_idx].set(
                    new_ratio.reshape(num_steps, minibatch_envs)
                )
                values = values.at[:, env_idx].set(
                    new_values.reshape(num_steps, minibatch_envs)
                )

        total_env_steps += env_steps_per_update
        elapsed = time.perf_counter() - t0
        sps = env_steps_per_update / elapsed

        # Logging
        if update % args.log_interval == 0 or update == num_updates - 1:
            mean_reward = float(trajectory.rewards.sum(axis=0).mean())
            num_episodes = int(trajectory.dones.sum())
            print(
                f"Update {update:>4}/{num_updates} | "
                f"Env frames {total_env_steps:>10,} | "
                f"SPS {sps:>8,.0f} | "
                f"Loss {float(loss):>7.3f} | "
                f"PG {float(pg_loss):>7.4f} | "
                f"VF {float(v_loss):>7.4f} | "
                f"Ent {float(entropy):>6.4f} | "
                f"Rew {mean_reward:>6.1f} | "
                f"Eps {num_episodes}"
            )
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward

    # Save model
    eqx.tree_serialise_leaves(args.output, model)
    print(f"\nModel saved to {args.output}")
    print(f"Best mean reward per rollout: {best_mean_reward:.1f}")
    print(f"Total env frames: {total_env_steps:,}")


if __name__ == "__main__":
    main()
