"""Compare C and JAX breakout environments step by step."""
import numpy as np
import jax
import jax.numpy as jnp
from breakout_jax import BreakoutJAX

# JAX env
jax_env = BreakoutJAX()
key = jax.random.key(0)
key, reset_key = jax.random.split(key)
jax_state, jax_obs, _ = jax_env.reset(reset_key)

# C env
from pufferlib.ocean.breakout.breakout import Breakout
c_env = Breakout(num_envs=1)
c_env.reset()

# Use fixed action sequence: RIGHT for 10 steps then LEFT for 10 steps, repeat
actions = [2]*10 + [1]*10 + [2]*10 + [1]*10 + [0]*10 + [2]*20 + [1]*20

print(f"{'Step':>4} | {'Action':>6} | {'C ball_xy':>18} | {'JAX ball_xy':>18} | {'C vel':>18} | {'JAX vel':>18} | {'C rew':>5} | {'J rew':>5} | {'Match':>5}")
print("-" * 120)

max_diff = 0.0
for step, action in enumerate(actions):
    # C step
    c_env.actions[:] = action
    c_env.step(c_env.actions)
    c_obs = c_env.observations[0]
    c_rew = c_env.rewards[0]
    c_done = c_env.terminals[0]

    # JAX step
    key, step_key = jax.random.split(key)
    jax_state, jax_obs_new, jax_rew, jax_done, _, _ = jax_env.step(
        jax_state, jnp.int32(action), step_key
    )

    j_obs = np.array(jax_obs_new)

    c_bxy = f"({c_obs[2]:.4f},{c_obs[3]:.4f})"
    j_bxy = f"({j_obs[2]:.4f},{j_obs[3]:.4f})"
    c_vel = f"({c_obs[4]:.5f},{c_obs[5]:.5f})"
    j_vel = f"({j_obs[4]:.5f},{j_obs[5]:.5f})"

    diff = np.max(np.abs(c_obs[:10] - j_obs[:10]))
    max_diff = max(max_diff, diff)
    match = "OK" if diff < 0.01 else f"DIFF={diff:.4f}"

    if step < 50 or step % 10 == 0 or c_done or bool(jax_done) or c_rew > 0 or float(jax_rew) > 0:
        print(f"{step:4d} | {action:6d} | {c_bxy:>18s} | {j_bxy:>18s} | {c_vel:>18s} | {j_vel:>18s} | {c_rew:5.0f} | {float(jax_rew):5.0f} | {match:>5s}")

    if c_done or bool(jax_done):
        print(f"  ** C done={c_done}, JAX done={bool(jax_done)} **")
        break

print(f"\nMax obs difference: {max_diff:.6f}")
