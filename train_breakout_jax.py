"""Train PPO+LSTM on JAX Breakout with PufferLib's Big Three techniques.

Uses V-trace, ratio retracking, and prioritized sampling for better
sample efficiency. LSTM policy for ball trajectory memory.

PufferLib reaches 430 at ~107M steps with 8192 agents.
We use 1024 envs with adapted hyperparams.

Usage:
    python train_breakout_jax.py
    python train_breakout_jax.py --total-timesteps 500000000
"""

import argparse
import time

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from breakout_jax import BreakoutJAX
from ppo_jax import PPOPolicyLSTM, Trajectory, compute_vtrace_advantages, ppo_loss_lstm


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO+LSTM on JAX Breakout")
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--num-steps", type=int, default=64)
    p.add_argument("--total-timesteps", type=int, default=500_000_000)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="breakout_jax_model.eqx")
    p.add_argument("--log-interval", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()

    # Hyperparams: PufferLib's Protein-swept values adapted for 1024 envs
    # PufferLib batch=524K (8192 agents), ours=65K (1024 agents) = 8x smaller
    # LR scaled down, ent_coef up slightly, keep other PufferLib params
    lr = 0.001  # PufferLib 0.045 / ~45 (conservative to prevent entropy collapse)
    ent_coef = 0.003  # PufferLib 0.000569; higher for 1024 envs

    # PufferLib's Protein-swept structural hyperparams
    adam_b1 = 0.8946507418260217
    adam_b2 = 0.9
    adam_eps = 1e-4
    gamma = 0.9997053654668936
    gae_lambda = 0.747650023961198
    clip_coef = 0.19696765958267629
    vf_clip_coef = 2.178492167689251
    vf_coef = 1.6832989594296321
    max_grad_norm = 2.2356112188495723
    vtrace_rho_clip = 0.7876748061547312
    vtrace_c_clip = 2.878171091654008
    prio_alpha = 0.98967001208896
    prio_beta0 = 0.09999999999999998
    update_epochs = 1
    minibatch_envs = 512  # 2 minibatches per epoch

    num_envs = args.num_envs
    num_steps = args.num_steps
    hidden_size = args.hidden_size
    steps_per_update = num_envs * num_steps
    num_updates = args.total_timesteps // steps_per_update
    num_minibatches = num_envs // minibatch_envs

    print(f"Config: {num_envs} envs, {num_steps} steps, {num_updates} updates, "
          f"{num_minibatches} mb, LSTM h={hidden_size}")
    print(f"LR: {lr}, ent_coef: {ent_coef}, gamma: {gamma:.4f}")
    print(f"V-trace: rho_clip={vtrace_rho_clip:.3f}, c_clip={vtrace_c_clip:.3f}")
    print(f"Total timesteps: {num_updates * steps_per_update:,}")

    key = jax.random.key(args.seed)
    env = BreakoutJAX()
    obs_size = env.observation_size

    key, model_key = jax.random.split(key)
    model = PPOPolicyLSTM(obs_size, hidden_size, env.num_actions, model_key)

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

    hidden = (
        jnp.zeros((num_envs, hidden_size)),
        jnp.zeros((num_envs, hidden_size)),
    )

    @eqx.filter_jit
    def collect_rollout(model, states, hidden, key):
        def step_fn(carry, step_key):
            states, hidden = carry
            obs = vmapped_observe(states)
            logits, values, new_hidden = jax.vmap(model)(obs, hidden)

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

            h, c = new_hidden
            h = jnp.where(dones[:, None], 0.0, h)
            c = jnp.where(dones[:, None], 0.0, c)

            return (new_states, (h, c)), (obs, actions, log_probs, values, rewards, dones)

        keys = jax.random.split(key, num_steps)
        (final_states, final_hidden), transitions = jax.lax.scan(
            step_fn, (states, hidden), keys
        )
        return final_states, final_hidden, Trajectory(*transitions)

    @jax.jit
    def compute_advantages(rewards, values, dones, importance):
        return compute_vtrace_advantages(
            rewards, values, dones, importance,
            gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip,
        )

    @eqx.filter_jit
    def train_step(model, opt_state, obs, actions, old_log_probs, old_values,
                   advantages, returns, prio_weights, init_hidden):
        (loss, aux), grads = eqx.filter_value_and_grad(ppo_loss_lstm, has_aux=True)(
            model, obs, actions, old_log_probs, old_values,
            advantages, returns, prio_weights, init_hidden,
            clip_coef, vf_clip_coef, vf_coef, ent_coef,
        )
        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, loss, aux

    # Training loop
    total_steps = 0
    best_ep_ret = -float("inf")
    episode_returns = []
    env_episode_returns = jnp.zeros(num_envs)

    for update in range(num_updates):
        t0 = time.perf_counter()

        init_hidden = hidden
        key, rollout_key = jax.random.split(key)
        states, hidden, trajectory = collect_rollout(model, states, hidden, rollout_key)

        # Track episode returns
        for t in range(num_steps):
            env_episode_returns = env_episode_returns + trajectory.rewards[t]
            done_mask = trajectory.dones[t]
            if bool(done_mask.any()):
                done_returns = env_episode_returns[done_mask]
                episode_returns.extend(done_returns.tolist())
            env_episode_returns = jnp.where(done_mask, 0.0, env_episode_returns)

        clipped_rewards = jnp.clip(trajectory.rewards, -1.0, 1.0)

        # V-trace + ratio retracking + prioritized sampling
        ratios = jnp.ones_like(clipped_rewards)
        values = trajectory.values

        anneal_beta = prio_beta0 + (1 - prio_beta0) * prio_alpha * update / max(
            num_updates - 1, 1
        )

        for _epoch in range(update_epochs):
            for _mb in range(num_minibatches):
                advantages = compute_advantages(
                    clipped_rewards, values, trajectory.dones, ratios
                )
                returns = advantages + values

                # Prioritized sampling
                adv_per_env = jnp.abs(advantages).sum(axis=0)
                weights = jnp.nan_to_num(adv_per_env**prio_alpha, 0.0, 0.0, 0.0)
                probs = (weights + 1e-6) / (weights.sum() + 1e-6)

                key, mb_key = jax.random.split(key)
                env_idx = jax.random.choice(
                    mb_key, num_envs, shape=(minibatch_envs,),
                    p=probs, replace=False,
                )

                mb_prio = (num_envs * probs[env_idx]) ** -anneal_beta

                mb_obs = trajectory.obs[:, env_idx]
                mb_actions = trajectory.actions[:, env_idx]
                mb_log_probs = trajectory.log_probs[:, env_idx]
                mb_old_values = values[:, env_idx]
                mb_adv = advantages[:, env_idx]
                mb_ret = returns[:, env_idx]
                mb_init_hidden = (
                    init_hidden[0][env_idx],
                    init_hidden[1][env_idx],
                )

                model, opt_state, loss, aux = train_step(
                    model, opt_state, mb_obs, mb_actions, mb_log_probs,
                    mb_old_values, mb_adv, mb_ret, mb_prio, mb_init_hidden,
                )

                # Ratio and value retracking
                pg_loss, v_loss, entropy, new_ratio, _, new_values = aux
                ratios = ratios.at[:, env_idx].set(
                    new_ratio.reshape(num_steps, minibatch_envs)
                )
                values = values.at[:, env_idx].set(
                    new_values.reshape(num_steps, minibatch_envs)
                )

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
