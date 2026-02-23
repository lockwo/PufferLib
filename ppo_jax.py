"""PPO in JAX/Equinox, closely following PufferLib's PuffeRL.

Implements PufferLib's "Big Three" unique techniques:
1. V-trace advantage computation (C/CUDA kernel reimplemented in JAX)
2. Importance ratio retracking across minibatches
3. Prioritized experience sampling by advantage magnitude

See: pufferlib/pufferl.py, pufferlib/extensions/pufferlib.cpp


  Three files created:                                                                                                              
                                                                                                                                    
  1. ppo_jax.py — JAX/Equinox PPO implementation                                                                                  
  - PPOPolicy — 2-layer MLP (no LSTM, compensates with extra depth)
  - compute_vtrace_advantages() — exact reimplementation of PufferLib's C/CUDA kernel using lax.scan
  - ppo_loss() — clipped surrogate + clipped value loss + entropy, with priority-weighted advantage normalization

  2. train_breakout_jax.py — Training script
  - Rollout via lax.scan + vmap (entire rollout compiles to one XLA program)
  - PufferLib's "Big Three": V-trace advantages, ratio retracking, prioritized sampling
  - Value retracking across minibatches
  - Hyperparams from PufferLib's Protein-swept breakout.ini (LR lowered for non-LSTM)
  - Saves model with eqx.tree_serialise_leaves

  3. eval_breakout_puffer.py — Evaluate on PufferLib C Breakout
  - Loads model, runs greedy or stochastic policy on PufferLib's C Breakout
  - Reports mean/median/max return, episode length, SPS
  - --render flag to watch it play, --num-envs for parallel eval

  Usage:

  # Train in JAX
  python train_breakout_jax.py

  # Evaluate on PufferLib C Breakout
  python eval_breakout_puffer.py
  python eval_breakout_puffer.py --render          # watch it play
  python eval_breakout_puffer.py --stochastic      # sample actions

  # PufferLib's own training (uses LSTM + full Protein-swept config):
  puffer train puffer_breakout

  Key differences from PufferLib:

  - No LSTM — PufferLib's Breakout config uses rnn_name = Recurrent. Our version uses a 2-layer feedforward MLP, so it has no memory
   of ball trajectory across frames. Performance will be lower.
  - Single vmap — PufferLib uses 8 worker processes × 1024 envs. Our version uses 1024 envs in a single jax.vmap, no
  multiprocessing.
  - LR lowered — PufferLib's sweep found lr=0.045 (very high, likely LSTM-specific). We use 1e-3.
"""

import math

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import NamedTuple


def orthogonal_init(layer, std, key):
    """Orthogonal weight initialization matching PufferLib/CleanRL.

    PufferLib uses: orthogonal_(weight, std), constant_(bias, 0).
    """
    w = layer.weight
    m, n = w.shape
    # Generate random matrix and compute QR decomposition
    random_mat = jax.random.normal(key, (max(m, n), max(m, n)))
    q, r = jnp.linalg.qr(random_mat)
    # Ensure uniform distribution (correct signs)
    q = q * jnp.sign(jnp.diag(r))
    q = q[:m, :n] * std
    new_bias = jnp.zeros_like(layer.bias) if layer.bias is not None else None
    return eqx.tree_at(
        lambda l: (l.weight, l.bias) if l.bias is not None else (l.weight,),
        layer,
        (q, new_bias) if new_bias is not None else (q,),
    )


class PPOPolicy(eqx.Module):
    """MLP policy for discrete action spaces.

    PufferLib default uses Linear+GELU (1 layer) + LSTM.
    We use 2 hidden layers + ReLU (no LSTM) for simplicity.
    """

    l1: eqx.nn.Linear
    l2: eqx.nn.Linear
    actor: eqx.nn.Linear
    critic: eqx.nn.Linear

    def __init__(self, obs_size, hidden_size, num_actions, key):
        k1, k2, k3, k4, ki1, ki2, ki3, ki4 = jax.random.split(key, 8)
        self.l1 = orthogonal_init(
            eqx.nn.Linear(obs_size, hidden_size, key=k1), math.sqrt(2), ki1
        )
        self.l2 = orthogonal_init(
            eqx.nn.Linear(hidden_size, hidden_size, key=k2), math.sqrt(2), ki2
        )
        self.actor = orthogonal_init(
            eqx.nn.Linear(hidden_size, num_actions, key=k3), 0.01, ki3
        )
        self.critic = orthogonal_init(
            eqx.nn.Linear(hidden_size, 1, key=k4), 1.0, ki4
        )

    def __call__(self, obs):
        x = jax.nn.relu(self.l1(obs))
        x = jax.nn.relu(self.l2(x))
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return logits, value


class Trajectory(NamedTuple):
    obs: jnp.ndarray  # (T, E, obs_size)
    actions: jnp.ndarray  # (T, E)
    log_probs: jnp.ndarray  # (T, E)
    values: jnp.ndarray  # (T, E)
    rewards: jnp.ndarray  # (T, E)
    dones: jnp.ndarray  # (T, E)


def compute_vtrace_advantages(
    rewards, values, dones, importance, gamma, lam, rho_clip, c_clip
):
    """V-trace advantage computation matching PufferLib's C/CUDA kernel.

    PufferLib kernel (pufferlib/extensions/pufferlib.cpp):
        for t = horizon-2 down to 0:
            rho_t = min(importance[t], rho_clip)
            c_t   = min(importance[t], c_clip)
            delta = rho_t * (r[t+1] + gamma*V[t+1]*(1-done[t+1]) - V[t])
            adv[t] = delta + gamma*lam*c_t * adv[t+1] * (1-done[t+1])

    Args:
        rewards, values, dones, importance: (T, E)
        gamma, lam, rho_clip, c_clip: scalars

    Returns:
        advantages: (T, E)
    """
    E = rewards.shape[1]

    def scan_fn(next_adv, t_data):
        reward_next, value_next, value_curr, done_next, imp_curr = t_data
        nonterminal = 1.0 - done_next
        rho = jnp.minimum(imp_curr, rho_clip)
        c = jnp.minimum(imp_curr, c_clip)
        delta = rho * (reward_next + gamma * value_next * nonterminal - value_curr)
        adv = delta + gamma * lam * c * next_adv * nonterminal
        return adv, adv

    # Build shifted arrays matching PufferLib's indexing:
    # reward_next[t] = rewards[t+1], value_next[t] = values[t+1], etc.
    # Iterate t from T-2 down to 0 (T-1 elements)
    scan_data = (
        rewards[1:],  # reward[t+1]
        values[1:],  # value[t+1]
        values[:-1],  # value[t]
        dones[1:],  # done[t+1]
        importance[:-1],  # importance[t]
    )

    _, advantages_inner = jax.lax.scan(scan_fn, jnp.zeros(E), scan_data, reverse=True)

    # advantages_inner: (T-1, E) for t=0..T-2; advantage[T-1] = 0
    return jnp.concatenate([advantages_inner, jnp.zeros((1, E))], axis=0)


def ppo_loss(
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
):
    """PPO clipped loss with priority-weighted advantages, matching PufferLib.

    Args:
        model: PPOPolicy
        obs: (M, obs_size)
        actions: (M,) int32
        old_log_probs: (M,) from collection
        old_values: (M,) from collection (or retracked)
        advantages: (M,) from V-trace
        returns: (M,) = advantages + values
        prio_weights: (M,) importance sampling correction
        clip_coef, vf_clip_coef, vf_coef, ent_coef: scalars

    Returns:
        loss, (pg_loss, v_loss, entropy, ratio, new_log_probs, new_values)
    """
    logits, new_values = jax.vmap(model)(obs)
    log_probs_all = jax.nn.log_softmax(logits)
    new_log_probs = jnp.take_along_axis(
        log_probs_all, actions[:, None], axis=1
    ).squeeze(-1)
    ratio = jnp.exp(new_log_probs - old_log_probs)

    # Priority-weighted advantage normalization
    # PufferLib: adv = mb_prio * (adv - adv.mean()) / (adv.std() + 1e-8)
    adv = prio_weights * (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Clipped surrogate loss
    pg_loss1 = -adv * ratio
    pg_loss2 = -adv * jnp.clip(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
    pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

    # Clipped value loss (PufferLib uses separate vf_clip_coef)
    v_clipped = old_values + jnp.clip(
        new_values - old_values, -vf_clip_coef, vf_clip_coef
    )
    v_loss = (
        0.5
        * jnp.maximum((new_values - returns) ** 2, (v_clipped - returns) ** 2).mean()
    )

    # Entropy bonus
    probs = jax.nn.softmax(logits)
    entropy = -(probs * log_probs_all).sum(axis=-1).mean()

    loss = pg_loss + vf_coef * v_loss - ent_coef * entropy
    return loss, (pg_loss, v_loss, entropy, ratio, new_log_probs, new_values)


class PPOPolicyLSTM(eqx.Module):
    """LSTM policy matching PufferLib's architecture.

    PufferLib: Linear(obs, hidden) + GELU -> LSTM -> actor/critic heads.
    """

    encoder: eqx.nn.Linear
    lstm: eqx.nn.LSTMCell
    actor: eqx.nn.Linear
    critic: eqx.nn.Linear
    hidden_size: int = eqx.field(static=True)

    def __init__(self, obs_size, hidden_size, num_actions, key):
        k1, k2, k3, k4, ki1, ki3, ki4, ki5 = jax.random.split(key, 8)
        self.encoder = orthogonal_init(
            eqx.nn.Linear(obs_size, hidden_size, key=k1), math.sqrt(2), ki1
        )
        self.lstm = eqx.nn.LSTMCell(hidden_size, hidden_size, key=k2)
        # Orthogonal init for LSTM weights (std=1.0, matching PufferLib)
        def init_lstm(lstm, key):
            leaves, treedef = jax.tree.flatten(lstm)
            keys = jax.random.split(key, len(leaves))
            new_leaves = []
            for leaf, k in zip(leaves, keys):
                if leaf.ndim >= 2:
                    m, n = leaf.shape
                    random_mat = jax.random.normal(k, (max(m, n), max(m, n)))
                    q, r = jnp.linalg.qr(random_mat)
                    q = q * jnp.sign(jnp.diag(r))
                    new_leaves.append(q[:m, :n])
                elif leaf.ndim == 1:
                    new_leaves.append(jnp.zeros_like(leaf))
                else:
                    new_leaves.append(leaf)
            return jax.tree.unflatten(treedef, new_leaves)
        self.lstm = init_lstm(self.lstm, ki5)
        self.actor = orthogonal_init(
            eqx.nn.Linear(hidden_size, num_actions, key=k3), 0.01, ki3
        )
        self.critic = orthogonal_init(
            eqx.nn.Linear(hidden_size, 1, key=k4), 1.0, ki4
        )
        self.hidden_size = hidden_size

    def __call__(self, obs, hidden):
        """Single step: obs (obs_size,) + hidden ((H,),(H,)) -> logits, value, new_hidden."""
        x = jax.nn.gelu(self.encoder(obs))
        new_hidden = self.lstm(x, hidden)
        h, _ = new_hidden
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value, new_hidden


def ppo_loss_lstm(
    model,
    obs,
    actions,
    old_log_probs,
    old_values,
    advantages,
    returns,
    prio_weights,
    init_hidden,
    clip_coef,
    vf_clip_coef,
    vf_coef,
    ent_coef,
):
    """PPO loss with LSTM — sequential forward pass via lax.scan.

    Args:
        model: PPOPolicyLSTM
        obs: (T, mb_envs, obs_size)
        actions, old_log_probs, old_values, advantages, returns: (T, mb_envs)
        prio_weights: (mb_envs,) IS correction per env
        init_hidden: ((mb_envs, H), (mb_envs, H))

    Returns:
        loss, (pg_loss, v_loss, entropy, ratio, new_log_probs, new_values)
        where ratio, new_log_probs, new_values are flat (T*mb_envs,)
    """
    T, mb_envs = obs.shape[:2]

    # LSTM forward through time (BPTT, no hidden reset at dones — matches PufferLib)
    def scan_fn(hidden, obs_t):
        logits, values, new_hidden = jax.vmap(model)(obs_t, hidden)
        return new_hidden, (logits, values)

    _, (all_logits, all_values) = jax.lax.scan(scan_fn, init_hidden, obs)

    # Flatten: (T, mb_envs, ...) -> (T*mb_envs, ...)
    logits = all_logits.reshape(T * mb_envs, -1)
    new_values = all_values.reshape(T * mb_envs)
    flat_actions = actions.reshape(T * mb_envs).astype(jnp.int32)
    flat_old_lp = old_log_probs.reshape(T * mb_envs)
    flat_old_v = old_values.reshape(T * mb_envs)
    flat_adv = advantages.reshape(T * mb_envs)
    flat_ret = returns.reshape(T * mb_envs)
    flat_prio = jnp.broadcast_to(prio_weights[None, :], (T, mb_envs)).reshape(
        T * mb_envs
    )

    log_probs_all = jax.nn.log_softmax(logits)
    new_log_probs = jnp.take_along_axis(
        log_probs_all, flat_actions[:, None], axis=1
    ).squeeze(-1)
    ratio = jnp.exp(new_log_probs - flat_old_lp)

    # Priority-weighted advantage normalization
    adv = flat_prio * (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

    # Clipped surrogate loss
    pg_loss1 = -adv * ratio
    pg_loss2 = -adv * jnp.clip(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
    pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

    # Clipped value loss
    v_clipped = flat_old_v + jnp.clip(
        new_values - flat_old_v, -vf_clip_coef, vf_clip_coef
    )
    v_loss = (
        0.5
        * jnp.maximum(
            (new_values - flat_ret) ** 2, (v_clipped - flat_ret) ** 2
        ).mean()
    )

    # Entropy
    probs = jax.nn.softmax(logits)
    entropy = -(probs * log_probs_all).sum(axis=-1).mean()

    loss = pg_loss + vf_coef * v_loss - ent_coef * entropy
    return loss, (pg_loss, v_loss, entropy, ratio, new_log_probs, new_values)
