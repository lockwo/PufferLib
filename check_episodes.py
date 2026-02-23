"""Check episode lengths and returns with a trained model."""
import jax
import jax.numpy as jnp
import equinox as eqx
from breakout_jax import BreakoutJAX
from ppo_jax import PPOPolicyLSTM

env = BreakoutJAX()
key = jax.random.key(0)

# Load trained model
key, model_key = jax.random.split(key)
template = PPOPolicyLSTM(env.observation_size, 128, env.num_actions, model_key)
model = eqx.tree_deserialise_leaves("breakout_jax_model.eqx", template)

# Run single episodes
hidden = (jnp.zeros(128), jnp.zeros(128))

for ep in range(5):
    key, reset_key = jax.random.split(key)
    state, obs, _ = env.reset(reset_key)
    hidden = (jnp.zeros(128), jnp.zeros(128))
    total_reward = 0.0
    steps = 0

    for step in range(5000):
        obs = env._observe(state)
        logits, value, hidden = model(obs, hidden)
        action = jnp.argmax(logits)  # greedy

        key, step_key = jax.random.split(key)
        state, obs_new, reward, terminated, truncated, _ = env.step(state, action, step_key)
        total_reward += float(reward)
        steps += 1

        if step < 30 and ep == 0:
            action_name = ["NOOP", "LEFT", "RIGHT"][int(action)]
            print(f"  Step {step:3d}: {action_name:>5s} ball=({float(state.ball_x):6.1f},{float(state.ball_y):6.1f}) "
                  f"paddle_x={float(state.paddle_x):6.1f} rew={float(reward):.0f} score={int(state.score)}")

        if bool(terminated):
            break

    print(f"Episode {ep}: steps={steps}, total_reward={total_reward:.0f}, final_score={int(state.score)}, num_balls={int(state.num_balls)}")
