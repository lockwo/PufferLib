"""Quick diagnostic: run a few random episodes in JAX Breakout, check rewards and observations."""
import jax
import jax.numpy as jnp
from breakout_jax import BreakoutJAX

env = BreakoutJAX()
key = jax.random.key(42)

# Run a single random episode
key, reset_key = jax.random.split(key)
state, obs, _ = env.reset(reset_key)
print(f"Initial obs shape: {obs.shape}")
print(f"Initial obs: {obs}")
print(f"Initial ball_vx={state.ball_vx}, ball_vy={state.ball_vy}")
print(f"Initial ball_x={state.ball_x}, ball_y={state.ball_y}")
print(f"Paddle_x={state.paddle_x}, paddle_y={state.paddle_y}")
print(f"balls_fired={state.balls_fired}, num_balls={state.num_balls}")
print()

total_reward = 0.0
for step in range(2000):
    key, action_key, step_key = jax.random.split(key, 3)
    action = jax.random.randint(action_key, (), 0, 3)
    state, obs, reward, terminated, truncated, _ = env.step(state, action, step_key)
    total_reward += float(reward)

    if step < 10:
        print(f"Step {step}: action={int(action)}, reward={float(reward):.1f}, "
              f"ball_xy=({float(state.ball_x):.1f},{float(state.ball_y):.1f}), "
              f"ball_v=({float(state.ball_vx):.3f},{float(state.ball_vy):.3f}), "
              f"score={int(state.score)}, balls_fired={int(state.balls_fired)}, "
              f"num_balls={int(state.num_balls)}, done={bool(terminated)}")

    if bool(terminated):
        print(f"\nEpisode ended at step {step}, total_reward={total_reward:.0f}, score={int(state.score)}")
        print(f"Post-terminal state: ball_xy=({float(state.ball_x):.1f},{float(state.ball_y):.1f})")
        print(f"Post-terminal obs[:10]: {obs[:10]}")
        break
else:
    print(f"\nNo termination after 2000 steps, total_reward={total_reward:.0f}")
    print(f"Score={int(state.score)}, num_balls={int(state.num_balls)}")

# Check: with NOOP policy, does the ball still work?
print("\n--- NOOP policy test ---")
key, reset_key = jax.random.split(key)
state, obs, _ = env.reset(reset_key)
total_reward = 0.0
for step in range(200):
    key, step_key = jax.random.split(key)
    action = jnp.int32(0)  # NOOP
    state, obs, reward, terminated, truncated, _ = env.step(state, action, step_key)
    total_reward += float(reward)
    if step < 20:
        print(f"Step {step}: ball_xy=({float(state.ball_x):.1f},{float(state.ball_y):.1f}), "
              f"ball_v=({float(state.ball_vx):.3f},{float(state.ball_vy):.3f}), "
              f"reward={float(reward):.1f}, balls_fired={int(state.balls_fired)}")
print(f"After 200 NOOP steps: reward={total_reward:.0f}, score={int(state.score)}")

# Check vmap works
print("\n--- Vmap test (1024 envs, 64 steps, random actions) ---")
num_envs = 1024
key, init_key = jax.random.split(key)
init_keys = jax.random.split(init_key, num_envs)
states, _, _ = jax.vmap(env.reset)(init_keys)

vmapped_step = jax.vmap(env.step)
total_rewards = jnp.zeros(num_envs)
total_episodes = 0
for step in range(64):
    key, action_key, step_key = jax.random.split(key, 3)
    actions = jax.random.randint(action_key, (num_envs,), 0, 3)
    step_keys = jax.random.split(step_key, num_envs)
    states, _, rewards, terminated, _, _ = vmapped_step(states, actions, step_keys)
    total_rewards += rewards
    total_episodes += terminated.sum()

print(f"Mean reward per env (64 steps): {float(total_rewards.mean()):.2f}")
print(f"Max reward: {float(total_rewards.max()):.1f}")
print(f"Total episodes: {int(total_episodes)}")
print(f"Mean score: {float(jax.vmap(lambda s: s.score)(states).mean()):.1f}")
