# PufferLib Unofficial Docs

A high-performance RL library centered around PPO, with 50+ built-in C environments (Ocean), fast vectorization via shared memory, and wrappers for 39+ third-party environments.

---

## Table of Contents

1. [Ocean Environments (Complete List)](#ocean-environments)
2. [Third-Party Environment Wrappers](#third-party-environment-wrappers)
3. [How Vectorization Works](#how-vectorization-works)
4. [The Buffer System](#the-buffer-system)
5. [Writing Your Own Environment](#writing-your-own-environment)
6. [How Rendering Works](#how-rendering-works)
7. [The Training Algorithm (PuffeRL)](#the-training-algorithm-pufferl)
8. [Configuration System](#configuration-system)
9. [Core Files Reference](#core-files-reference)
10. [Using PufferLib with JAX](#using-pufferlib-with-jax)
11. [Benchmarking: PufferLib vs PGX](#benchmarking-pufferlib-vs-pgx)
12. [Porting Environments to JAX](#porting-environments-to-jax)

---

## Ocean Environments

All Ocean environments are run via `puffer train puffer_<name>`. Each has a `.ini` config in `pufferlib/config/ocean/`. The C source is in `pufferlib/ocean/<name>/`.

### Arcade / Classic Games

| CLI Name | Agents | Obs Type | Action Type | Key Params | Description |
|---|---|---|---|---|---|
| `puffer_breakout` | 1 per env | float32 vector (10 + rows*cols) | Discrete(3) or continuous | `width=576, height=330, brick_rows=6, brick_cols=18, frameskip=4, continuous=False` | Brick breaker. Obs encodes paddle/ball position + brick grid |
| `puffer_pong` | 1 per env | float32 | Discrete | `frameskip=8` | Classic Pong |
| `puffer_enduro` | 1 per env | float32 | Discrete | — | Enduro racing |
| `puffer_freeway` | 1 per env | float32 | Discrete | `use_dense_rewards=True, difficulty=0, frameskip=4` | Cross the highway |
| `puffer_pacman` | 1 per env | uint8 grid | Discrete | `randomize_starting_position=1` | Pacman |
| `puffer_asteroids` | 1 per env | float32 | Discrete | — | Space shooter |
| `puffer_tetris` | 1 per env | float32 | Discrete | `n_rows=20, n_cols=10, use_deck_obs=True` | Tetris |
| `puffer_blastar` | 1 per env | float32 | Discrete | — | Blastar clone |
| `puffer_g2048` | 1 per env | float32 | Discrete | `reward_scaler=0.67, endgame_env_prob=0.05` | 2048 puzzle |

### Board / Card Games

| CLI Name | Agents | Key Params | Description |
|---|---|---|---|
| `puffer_connect4` | 2 per env | — | Connect 4 |
| `puffer_checkers` | 2 per env | `size=8` | Checkers |
| `puffer_go` | 2 per env | `grid_size=7`, many reward params | Go |
| `puffer_tripletriad` | 2 per env | — | Triple Triad card game |

### Navigation / Grid Worlds

| CLI Name | Agents | Key Params | Description |
|---|---|---|---|
| `puffer_squared` | 1 per env | `distance_to_target=3, num_targets=1` | Tutorial env: navigate grid to target |
| `puffer_pysquared` | 1 per env | same | Pure Python squared (slower, good sanity check) |
| `puffer_target` | 8 per env | `num_agents=8, num_goals=4` | Multi-agent reaching goals |
| `puffer_grid` | 1 per env | `max_size=47, num_maps=8192` | Grid navigation |
| `puffer_tmaze` | 1 per env | `size=16` | T-maze (requires memory/RNN) |
| `puffer_tower_climb` | 1 per env | `num_maps=200, reward_climb_row=0.27, reward_move_block=0.18` | Push blocks, climb tower |

### Multi-Agent Simulations

| CLI Name | Agents | Key Params | Description |
|---|---|---|---|
| `puffer_snake` | 256 per env | `width=640, height=360, num_snakes=256, num_food=4096, vision=5, reward_food=0.1, reward_death=-1.0` | Mass multi-agent snake. Obs is local (2*vision+1)^2 grid |
| `puffer_battle` | 128 per env | `num_agents=128, num_armies=2` | Army vs army combat |
| `puffer_convert` | 1024 per env | `num_agents=1024, num_factories=32, num_resources=8` | Resource conversion game |
| `puffer_convert_circle` | varies | `equidistant=1, radius=400` | Circular variant of convert |
| `puffer_drive` | 1024 per env | `num_agents=1024, reward_vehicle_collision=-0.5, spawn_immunity_timer=50, num_maps=80000` | Multi-agent driving |
| `puffer_boids` | 64 per env | `num_boids=64, margin_turn_factor, centering_factor, avoid_factor, matching_factor` | Flocking behavior sim |
| `puffer_drone` | 64 per env | `num_drones=64, max_rings=10` | Drone swarm coordination |
| `puffer_rware` | 8 per env | `map_choice=2, num_agents=8, num_requested_shelves=8` | Robotic warehouse |
| `puffer_trash_pickup` | 8 per env | `grid_size=20, num_agents=8, num_trash=40, num_bins=2, max_steps=500` | Multi-agent cleanup task |
| `puffer_shared_pool` | 8 per env | `vision=3, num_agents=8, reward_food=0.1, food_base_spawn_rate=2e-3` | Common pool resource game |
| `puffer_onlyfish` | 8 per env | `num_agents=8` | Fish behavior |
| `puffer_slimevolley` | 1-2 per env | `num_agents=1` (vs bot) or `2` (self-play) | Volleyball |
| `puffer_impulse_wars` | 2 per env | `num_drones=2, continuous=False` | RTS warfare (Box2D physics) |

### Complex / Large-Scale

| CLI Name | Agents | Key Params | Description |
|---|---|---|---|
| `puffer_nmmo3` | many | `reward_combat_level=1.0, reward_prof_level=1.0, reward_item_level=1.0, reward_death=-1.0` | Neural MMO 3. Massively multiagent RPG |
| `puffer_moba` | many | `reward_xp=0.0016926..., reward_tower=4.525...` | MOBA-style team game |

### Control / Physics

| CLI Name | Agents | Key Params | Description |
|---|---|---|---|
| `puffer_cartpole` | 1 per env | `cart_mass=1.0, pole_mass=0.1, pole_length=0.5, gravity=9.8, force_mag=10.0, dt=0.02` | Classic CartPole in C |
| `puffer_rocket_lander` | 1 per env | — | Lunar lander variant |
| `puffer_whisker_racer` | 1 per env | `width=1080, height=720, track_width=75, num_radial_sectors=180` | Racing with whisker sensors |

### Science / Misc

| CLI Name | Agents | Key Params | Description |
|---|---|---|---|
| `puffer_matsci` | 1 per env | `num_atoms=128` | Material science (LAMMPS) |
| `puffer_terraform` | 1 per env | `num_agents=1, reset_frequency=1024, reward_scale=0.11` | Terraforming |
| `puffer_robocode` | multi | — | Robot battle |
| `puffer_tactical` | multi | — | Tactical combat |

### Benchmarks / Sanity Checks

These are pure Python (no C), used for testing and debugging:

| CLI Name | Key Params | Description |
|---|---|---|
| `puffer_bandit` | `num_actions=10, reward_scale=1` | Multi-armed bandit |
| `puffer_memory` | `mem_length=2, mem_delay=2` | Sequence memory test |
| `puffer_password` | `password_length=5` | Password guessing |
| `puffer_stochastic` | `p=0.7` | Stochastic reward env |
| `puffer_performance` | `delay_mean=0, delay_std=0, bandwidth=1` | Benchmark env stepping speed |
| `puffer_continuous` | `discretize=False` | Continuous action test |
| `puffer_spaces` | — | Tests structured observation/action spaces |
| `puffer_multiagent` | — | Two-agent PettingZoo test |
| `puffer_chain_mdp` | `size=128` | Chain MDP (long-horizon credit assignment) |
| `puffer_onestateworld` | `mean_left=0.1, mean_right=0.5, var_right=10` | Single-state bandit variant |

### Template

| CLI Name | Description |
|---|---|
| `puffer_template` | Copy `pufferlib/ocean/template/` as starting point for new envs |

---

## Third-Party Environment Wrappers

These live in `pufferlib/environments/<name>/` and wrap existing RL libraries. Install with `pip install pufferlib[<name>]`.

| Wrapper | Package | Type | Notes |
|---|---|---|---|
| `atari` | ALE | Single-agent | Classic Atari games. Custom fast resize (not OpenCV) |
| `procgen` | procgen-mirror | Single-agent | Procedural arcade games |
| `nethack` | NLE | Single-agent | Extremely complex roguelike. 600+ line wrapper |
| `pokemon_red` | pokegym | Single-agent | Pokemon Red via Gameboy emulator |
| `box2d` | gymnasium[box2d] | Single-agent | LunarLander, BipedalWalker |
| `classic_control` | gymnasium | Single-agent | CartPole, MountainCar, Pendulum, Acrobot |
| `mujoco` | gymnasium[mujoco] | Single-agent | HalfCheetah, Ant, Humanoid, etc. |
| `dm_control` | DeepMind Control | Single-agent | Continuous control tasks |
| `dm_lab` | DeepMind Lab | Single-agent | 3D navigation environments |
| `crafter` | crafter | Single-agent | 2D Minecraft clone |
| `craftax` | craftax | Single-agent | JAX-based crafter variant |
| `minigrid` | minigrid | Single-agent | Grid navigation with partial obs |
| `minihack` | minihack | Single-agent | Simplified NetHack |
| `vizdoom` | vizdoom | Single-agent | Doom FPS environments |
| `stable_retro` | gym-retro | Single-agent | Classic console games |
| `minerl` | MineRL | Single-agent | Minecraft |
| `mani_skill` | ManiSkill | Single-agent | Robot manipulation |
| `bsuite` | bsuite | Single-agent | DeepMind behaviour suite |
| `griddly` | griddly | Single/Multi | Fast grid game framework |
| `kinetix` | kinetix | Single-agent | Motion environments |
| `butterfly` | PettingZoo | Multi-agent | Cooperative/competitive games |
| `magent` | magent2 | Multi-agent | Large-scale agent simulation |
| `open_spiel` | OpenSpiel | Multi-agent | DeepMind multi-game suite |
| `smac` | StarCraft II | Multi-agent | StarCraft multi-agent challenge |
| `microrts` | gym-microrts | Multi-agent | Real-time strategy (Java, finicky) |
| `nmmo` | neural-mmo | Multi-agent | Massively multiagent MMO |
| `cogames` | cogames | Multi-agent | Cooperative games |
| `gpudrive` | gpudrive | Multi-agent | GPU-accelerated driving (1M FPS) |
| `slimevolley` | slimevolley | 1-2 agent | Volleyball |
| `trade_sim` | trade_sim | — | Trading simulation |
| `tribal_village` | tribal_village | — | Social simulation |
| `links_awaken` | links_awaken | — | Zelda-like |
| `metta` | metta | Multi-agent | Multi-agent cooperation |

---

## How Vectorization Works

### Core file: `pufferlib/vector.py`

PufferLib has three vectorization backends that all expose the same API:

### Serial (single-threaded)
```python
pufferlib.vector.Serial
```
Runs all environments sequentially on the main process. Use for debugging.

### Multiprocessing (the fast one)
```python
pufferlib.vector.Multiprocessing
```
This is where the performance comes from. Here's what happens under the hood:

1. **Shared memory allocation** (`multiprocessing.RawArray`): One giant contiguous block is allocated for observations, rewards, terminals, truncations, masks, actions, and semaphore flags. All workers and the main process see the same memory.

2. **Worker processes**: Each worker gets a slice of the shared memory arrays. When a worker calls `env.step()`, observations and rewards are written *directly into shared memory* — no serialization, no pipes, no copies.

3. **Busy-wait semaphores**: Workers spin on a shared uint8 flag instead of using pipes/queues for synchronization. This virtually eliminates IPC overhead. Pipes are only used for info dicts (once per episode).

4. **Batch formation**: When enough workers finish, the main process reads directly from the shared memory buffer. With zero_copy=True, it returns a *view* (numpy slice) of the shared array — literally zero copies.

5. **Async batching**: You can run more environments than your batch size. Workers that finish early get new work immediately while slower ones catch up. The main process grabs the first N ready workers.

```
Main Process                    Workers (busy-wait on semaphore)
┌──────────────┐               ┌──────────┐
│ Shared Memory │◄─────────────│ Worker 0 │ writes obs directly
│ [obs|rew|done│               └──────────┘
│  |act|flags] │◄─────────────┌──────────┐
│              │               │ Worker 1 │ writes obs directly
│  recv() just │               └──────────┘
│  returns a   │◄─────────────┌──────────┐
│  slice ──────│               │ Worker 2 │ writes obs directly
└──────────────┘               └──────────┘
```

### Ray (distributed)
```python
pufferlib.vector.Ray
```
Uses Ray for cluster-scale distribution. Same API but copies data over the network.

### The API

```python
import pufferlib.vector

vecenv = pufferlib.vector.make(
    env_creator,                    # callable that returns a PufferEnv
    env_args=[], env_kwargs={},
    backend=pufferlib.vector.Multiprocessing,
    num_envs=16,                    # total environments
    num_workers=8,                  # processes (default: num_envs)
    batch_size=8,                   # envs returned per recv()
    zero_copy=True,                 # return views, not copies
    seed=0,
)

# Synchronous
obs, infos = vecenv.reset()
obs, rewards, terminals, truncations, infos = vecenv.step(actions)

# Asynchronous (faster — returns batches as soon as ready)
vecenv.async_reset()
obs, rewards, terminals, truncations, infos, env_ids, masks = vecenv.recv()
vecenv.send(actions)
obs, rewards, terminals, truncations, infos, env_ids, masks = vecenv.recv()
# ... loop ...

vecenv.close()
```

Key constraint: `num_envs` must be divisible by `num_workers`, and `batch_size` must be divisible by `(num_envs / num_workers)`.

---

## The Buffer System

### Do you need to care about buffers when writing an environment?

**Short answer: No.** Just accept `buf=None` in your constructor and pass it to `super().__init__(buf)`. PufferLib handles everything.

### What's actually happening

When you write:
```python
class MyEnv(pufferlib.PufferEnv):
    def __init__(self, buf=None):
        self.single_observation_space = gymnasium.spaces.Box(...)
        self.single_action_space = gymnasium.spaces.Discrete(5)
        self.num_agents = 4
        super().__init__(buf)  # <-- this is where the magic happens
```

`super().__init__(buf)` calls `set_buffers(self, buf)` in `pufferlib/pufferlib.py:22-43`:

- **If `buf=None`** (standalone use): Allocates fresh numpy arrays for `self.observations`, `self.rewards`, `self.terminals`, `self.truncations`, `self.masks`, `self.actions`.

- **If `buf` is a dict** (multiprocessing): Uses pre-allocated shared memory slices. Your env's `self.observations` *is* a view into the main process's batch array. When you write `self.observations[i] = something`, it appears in the main process's buffer with zero copies.

### The flow during multiprocessing

```
1. Main process allocates RawArray (shared memory) for ALL envs
2. Each worker gets a slice: buf = { observations: shm[start:end], ... }
3. Worker creates env: MyEnv(buf=buf)
4. env.observations IS shm[start:end] (same memory)
5. env.step() writes to self.observations in-place
6. Main process reads from shm — data is already there
```

### For C environments

The Python wrapper passes the numpy buffer pointers to C:
```python
# In your __init__:
self.c_envs = binding.vec_init(
    self.observations,   # numpy array backed by shared memory
    self.actions,
    self.rewards,
    self.terminals,
    self.truncations,
    num_envs, seed, **params
)
```
The C code stores these pointers and writes directly to them during `c_step()`. The C struct references the same memory:
```c
typedef struct {
    unsigned char* observations;  // points to Python's shared memory
    int* actions;
    float* rewards;
    unsigned char* terminals;
    // ...
} MyEnv;
```

### Important: in-place semantics

Because everything is shared memory, calling `step()` again overwrites the previous observations/rewards. If you need to keep old data, copy it before the next step. The training loop handles this by copying into its own experience buffer tensors.

---

## Writing Your Own Environment

### Pure Python (slower, simpler)

```python
import gymnasium
import numpy as np
import pufferlib

class MyEnv(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, buf=None, seed=0, **kwargs):
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(84,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(4)
        self.num_agents = num_envs  # 1 agent per env, num_envs envs
        super().__init__(buf)
        # your init here

    def reset(self, seed=None):
        # write to self.observations in-place
        self.observations[:] = 0
        return self.observations, [{}]  # info must be a list of dicts

    def step(self, actions):
        # read from actions, write to observations/rewards/terminals in-place
        self.actions[:] = actions
        self.rewards[:] = 0
        self.terminals[:] = False
        self.truncations[:] = False
        # your logic here, modifying self.observations, self.rewards, etc.
        return (self.observations, self.rewards, self.terminals,
                self.truncations, [])  # info list

    def render(self):
        pass  # optional

    def close(self):
        pass
```

### C Environment (fast, the recommended way)

The pattern used by all Ocean envs:

1. **Write C code** (`myenv.h`): Define your env struct with `Log log`, `observations`, `actions`, `rewards`, `terminals` pointers. Implement `c_reset()`, `c_step()`, `c_render()`, `c_close()`.

2. **Write binding** (`binding.c`): Uses `pufferlib/ocean/env_binding.h` macros to expose `vec_init`, `vec_step`, `vec_log` to Python.

3. **Write Python wrapper** (`myenv.py`): Subclass `PufferEnv`, pass buffers to C via `binding.vec_init()`, call `binding.vec_step()` each step.

4. **Register**: Add to `pufferlib/ocean/environment.py` MAKE_FUNCTIONS dict and create a `.ini` config.

5. **Build**: `python setup.py build_ext --inplace --force`

The key insight: your C code never allocates observation/reward memory. It receives pointers from Python (which point to shared memory) and writes to them in-place.

### How the C Log struct gets back to Python

Your C env accumulates metrics in a `Log` struct (all floats). When Python calls `binding.vec_log()`:

1. C sums Log structs across all environments
2. Divides by `log.n` (episode count) to get averages
3. Your `my_log()` function in `binding.c` exports fields to a Python dict via `assign_to_dict()`
4. Returns the dict to the training loop

```c
// In binding.c — choose which Log fields to export
static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    return 0;
}
```

---

## How Rendering Works

### Only one environment renders at a time

During training, nothing renders. During eval (`puffer eval`), the system creates a **single environment** (`num_envs=1`) and calls `driver.render()` each step.

```python
# From pufferl.py eval():
vecenv = pufferlib.vector.make(env_creator, backend=Serial, num_envs=1)
driver = vecenv.driver_env  # the single underlying env
# ...
render = driver.render()
```

### Ocean environments use Raylib

C environments implement `c_render()` which creates a Raylib window:
```c
void c_render(MyEnv* env) {
    if (!IsWindowReady()) {
        InitWindow(width, height, "My Env");
        SetTargetFPS(30);
    }
    BeginDrawing();
    // draw stuff
    EndDrawing();
}
```

### Display modes

- **Raylib window** (default for Ocean): Opens a native window. Press ESC to exit. F12 for screenshots, Ctrl+F12 for GIF recording.
- **ANSI** (text): Prints to terminal, overwriting previous frame.
- **RGB array**: Returns numpy arrays for programmatic capture. Used to export GIFs via imageio.

### You cannot render during training

The vectorized environments in training share memory across workers. There's no mechanism to render individual environments mid-training. If you want to watch training progress, use the terminal dashboard (`PuffeRL.print_dashboard()`), which shows stats like SPS, returns, and losses.

---

## The Training Algorithm (PuffeRL)

### Core file: `pufferlib/pufferl.py`

PuffeRL is PPO (Proximal Policy Optimization) with several enhancements. The training loop is:

```python
trainer = pufferl.PuffeRL(config, vecenv, policy)
while not trainer.done:
    trainer.evaluate()       # collect experience
    trainer.train()          # PPO update
    trainer.mean_and_log()   # aggregate stats
    trainer.print_dashboard()
trainer.close()
```

### Experience Collection (`evaluate()`)

Fills an experience buffer by stepping the vectorized environment:

```
for each step until buffer is full:
    obs, rewards, terminals, ..., env_ids = vecenv.recv()
    logits, value = policy.forward_eval(obs)
    action = sample(logits)
    store (obs, action, logprob, reward, terminal, value) in buffer
    vecenv.send(action)
```

Experience is organized into segments of length `bptt_horizon` (backprop-through-time horizon). Each segment is one row in the buffer. When a segment fills up, it's marked complete and a new row starts.

### The Big Three (Unique to PufferLib)

#### 1. Custom V-trace Advantage Kernel (C/CUDA)

Instead of standard GAE, PufferLib computes advantages with importance-weighted V-trace:

```
for t = horizon-2 down to 0:
    rho_t = min(importance[t], rho_clip)
    c_t   = min(importance[t], c_clip)
    delta = rho_t * (r[t+1] + gamma * V[t+1] * (1-done[t+1]) - V[t])
    advantage[t] = delta + gamma * lambda * c_t * advantage[t+1] * (1-done[t+1])
```

The importance ratio tracks how stale each trajectory is relative to the current policy. This is computed in a custom C kernel (CPU) or CUDA kernel (one thread per trajectory row). Allows reuse of off-policy data without blowing up.

- **CUDA version** (`pufferlib/extensions/cuda/pufferlib.cu`): Parallelizes across trajectories — one thread per row
- **CPU version** (`pufferlib/extensions/pufferlib.cpp`): Sequential fallback

#### 2. Importance Ratio Retracking

PufferLib stores per-trajectory log-probability ratios across minibatch updates:

```python
self.ratio = torch.ones(segments, horizon)  # persists across minibatches
# after each forward pass:
self.ratio[idx] = ratio.detach()  # update for next pass
```

When advantages are recomputed, these stored ratios feed into the V-trace kernel. This means trajectories collected 3 minibatches ago get their advantages retroactively corrected for policy drift. Vanilla PPO just recomputes advantages once and uses them for all epochs.

#### 3. Prioritized Experience Sampling

Instead of uniform minibatch sampling, PufferLib weights by advantage magnitude:

```python
adv = advantages.abs().sum(axis=1)  # per-trajectory priority
weights = adv ** alpha
probs = weights / weights.sum()
idx = torch.multinomial(probs, minibatch_size)
```

High-advantage trajectories (where the policy made surprising decisions) get sampled more. Annealing `beta` corrects the bias over training. This is like Prioritized Experience Replay but for PPO minibatches.

**The standout innovation is techniques 1-3 working together**: V-trace advantages + ratio retracking + prioritized sampling. This lets PufferLib squeeze more learning from each batch of experience, effectively making each environment step more sample-efficient than vanilla PPO.

### Performance Engineering

#### 4. torch.compile + CUDA Graphs

Compiles the policy forward pass, eval forward pass, and `sample_logits` separately. CUDA graph marking between evaluate/train phases.

#### 5. Gradient Accumulation

`max_minibatch_size` caps GPU memory. If your logical minibatch is larger, it accumulates gradients over multiple sub-batches.

#### 6. CPU Offload

Observations stored in pinned CPU memory, transferred to GPU only during training. Cuts peak VRAM significantly for large observation spaces.

#### 7. AMP (bfloat16)

`torch.amp.autocast` during both eval and training forward passes.

#### 8. LSTMCell vs LSTM

During rollouts, uses `LSTMCell` (single timestep, ~3x faster) instead of `LSTM`. During training, uses full `LSTM` over the BPTT horizon for backprop.

### Standard but Well-Implemented

- Cosine annealing LR with configurable min ratio
- Value function clipping (symmetric PPO-style)
- Reward clipping to [-1, 1]
- Muon optimizer option (from heavyball library) as alternative to Adam
- Segment-based buffer — experience stored as `[segments, bptt_horizon]` for cache-friendly GPU access

### PPO Loss (`train()`)

```python
# Policy loss (clipped surrogate objective)
ratio = exp(new_logprob - old_logprob)
pg_loss1 = -advantage * ratio
pg_loss2 = -advantage * clamp(ratio, 1-clip_coef, 1+clip_coef)
pg_loss = max(pg_loss1, pg_loss2).mean()

# Value loss (clipped)
v_clipped = old_value + clamp(new_value - old_value, -vf_clip, vf_clip)
v_loss = 0.5 * max((new_value - returns)^2, (v_clipped - returns)^2).mean()

# Entropy bonus
entropy_loss = entropy.mean()

# Total
loss = pg_loss + vf_coef * v_loss - ent_coef * entropy_loss
```

### Key Training Hyperparameters

| Param | Default | Description |
|---|---|---|
| `total_timesteps` | varies | Total training steps |
| `learning_rate` | varies | Learning rate (often 0.001) |
| `anneal_lr` | True/False | Cosine annealing of LR |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `clip_coef` | 0.1-0.5 | PPO clipping range |
| `vf_coef` | 1.0 | Value function loss weight |
| `ent_coef` | 0.01 | Entropy bonus weight |
| `max_grad_norm` | 0.5 | Gradient clipping |
| `update_epochs` | 1-4 | PPO epochs per batch |
| `batch_size` | varies | Total batch size |
| `minibatch_size` | varies | SGD minibatch size |
| `bptt_horizon` | 16-64 | LSTM unroll length / segment size |
| `vtrace_rho_clip` | 1.0 | V-trace rho clipping |
| `vtrace_c_clip` | 1.0 | V-trace c clipping |
| `prio_alpha` | 0.0 | Prioritized sampling alpha (0 = uniform) |
| `optimizer` | adam | `adam` or `muon` (from heavyball) |
| `precision` | float32 | `float32` or `bfloat16` |
| `compile` | False | Use `torch.compile` |
| `cpu_offload` | False | Keep obs on CPU (saves GPU memory) |

### Protein Sweep System

`pufferlib/sweep.py` implements Bayesian hyperparameter optimization:

1. Samples from Sobol sequences over parameter distributions (log_normal, uniform, logit_normal, int_uniform, uniform_pow2)
2. Fits a Gaussian Process (GPyTorch, Matern kernel) to observed (params → score) data
3. Tracks Pareto front of score vs. compute cost
4. Prunes runs that are below the efficiency frontier
5. Supports early stopping based on running score trends

Run with: `puffer sweep puffer_<env> --wandb --tag my_sweep`

---

## Configuration System

### INI file structure

Each environment has a `.ini` in `pufferlib/config/ocean/<name>.ini`:

```ini
[base]
package = ocean
env_name = puffer_squared
policy_name = Default        # or custom: Boids, NMMO3, Drive, Snake, etc.
rnn_name = None              # or Recurrent for LSTM

[env]
num_envs = 1024              # environments in the vectorized batch
# ... environment-specific params ...

[vec]
num_envs = 2                 # number of vectorized workers
num_workers = 2
batch_size = auto

[policy]
hidden_size = 128

[rnn]
input_size = 128
hidden_size = 128

[train]
total_timesteps = 10_000_000
learning_rate = 0.001
gamma = 0.99
gae_lambda = 0.95
clip_coef = 0.2
vf_coef = 1.0
ent_coef = 0.01
batch_size = auto
minibatch_size = 16384
bptt_horizon = 64
update_epochs = 1
device = cuda

[sweep]
method = Protein
metric = score
goal = maximize

[sweep.parameters.train.parameters.learning_rate]
distribution = log_normal
min = 0.0001
mean = 0.001
max = 0.1
```

### CLI overrides

Any config value can be overridden from the command line:
```bash
puffer train puffer_snake --train.learning-rate 0.0005 --env.num_snakes 512 --train.device cpu
```

Note: CLI uses hyphens (`learning-rate`), config uses underscores (`learning_rate`).

---

## Core Files Reference

| File | What It Does |
|---|---|
| `pufferlib/pufferlib.py` | `PufferEnv` base class, `set_buffers()`, postprocessing wrappers |
| `pufferlib/vector.py` | `Serial`, `Multiprocessing`, `Ray` vectorization backends, `make()` factory |
| `pufferlib/pufferl.py` | Main training script: `PuffeRL` class, `train()`, `eval()`, `sweep()`, CLI entry point |
| `pufferlib/emulation.py` | `GymnasiumPufferEnv`, `PettingZooPufferEnv` wrappers for third-party envs |
| `pufferlib/models.py` | Default policy network, `LSTMWrapper` |
| `pufferlib/spaces.py` | Space utilities, `joint_space()` |
| `pufferlib/sweep.py` | Protein/Random hyperparameter sweep, GP-based optimization |
| `pufferlib/pytorch.py` | `sample_logits()`, `layer_init()`, observation unflattening |
| `pufferlib/ocean/environment.py` | Environment registry (`MAKE_FUNCTIONS` dict), factory functions |
| `pufferlib/ocean/env_binding.h` | Shared C macros for binding envs to Python (`vec_init`, `vec_step`, `vec_log`) |
| `pufferlib/ocean/torch.py` | Custom policies for specific Ocean envs (Boids, NMMO3, Snake, etc.) |
| `pufferlib/extensions/pufferlib.cpp` | CPU advantage kernel |
| `pufferlib/extensions/cuda/pufferlib.cu` | CUDA advantage kernel |
| `pufferlib/config/ocean/*.ini` | Per-environment configuration files |
| `pufferlib/environments/*/environment.py` | Third-party env wrappers |
| `setup.py` | Build system: C extensions, Raylib, Box2D, CUDA compilation |

### Build environment variables

| Variable | Effect |
|---|---|
| `DEBUG=1` | Enable debug symbols and AddressSanitizer |
| `NO_OCEAN=1` | Skip compiling Ocean C environments |
| `NO_TRAIN=1` | Skip compiling training extensions (C/CUDA advantage kernels) |

---

## Using PufferLib with JAX

### Are PufferLib environments fast with JAX?

Yes. PufferLib's Ocean environments are written in C and return numpy arrays. Since JAX works with numpy arrays natively (via `jax.numpy.array()`), the environments are fully compatible. The speed comes from the C environment stepping, not from the framework — your JAX policy can consume observations and produce actions with `jax.jit`, while the env stepping happens at C speed regardless of framework.

The typical loop looks like:
```python
import jax
import pufferlib

env = MyOceanEnv(num_envs=1024)
obs, info = env.reset()

# Convert to JAX for policy inference
obs_jax = jax.numpy.array(obs)
action = jitted_policy(obs_jax)

# Convert back to numpy for env
obs, rewards, terminals, truncations, infos = env.step(np.array(action))
```

### JAX advantage kernel equivalent

PufferLib's CUDA advantage kernel (`pufferlib/extensions/cuda/pufferlib.cu`) can be replicated in JAX using `jax.lax.scan`:

```python
def compute_gae_vtrace(rewards, values, dones, importance, gamma, lam, rho_clip, c_clip):
    def scan_fn(advantage, t_data):
        reward, value, next_value, done, imp = t_data
        rho = jnp.minimum(imp, rho_clip)
        c = jnp.minimum(imp, c_clip)
        delta = rho * (reward + gamma * next_value * (1 - done) - value)
        advantage = delta + gamma * lam * c * advantage * (1 - done)
        return advantage, advantage

    # Scan in reverse
    _, advantages = jax.lax.scan(scan_fn, 0.0, (rewards, values, next_values, dones, importance), reverse=True)
    return advantages
```

### numpy version conflict

PufferLib requires `numpy<2.0`, while modern JAX requires `numpy>=2.0`. If you need to run both in the same workflow, use separate conda environments and persist results to JSON files.

---

## Benchmarking: PufferLib vs PGX

### What is PGX?

PGX is a JAX-native game library that implements board/puzzle games as pure JAX functions. Because everything compiles to a single XLA program, PGX can leverage `jax.lax.scan` to step thousands of environments in one fused kernel with no Python overhead between steps.

### Games with overlap

Three games exist in both PufferLib (C) and PGX (JAX):

| Game | PufferLib | PGX |
|---|---|---|
| 2048 | `pufferlib.ocean.g2048.g2048.G2048` | `pgx.make("2048")` |
| Connect Four | `pufferlib.ocean.connect4.connect4.Connect4` | `pgx.make("connect_four")` |
| Go | `pufferlib.ocean.go.go.Go` | `pgx.make("go_9x9")` |

### Benchmark script: `benchmark_2048.py`

A ready-to-use benchmark script is included at the repo root. It compares PufferLib (C, single-threaded) vs PGX (JAX, `jax.lax.scan`) across varying numbers of environments.

```bash
# Run in PufferLib conda env (numpy<2.0)
python benchmark_2048.py --pufferlib --game 2048

# Run in JAX conda env (numpy>=2.0)
python benchmark_2048.py --pgx --game 2048

# Plot combined results
python benchmark_2048.py --plot --game 2048
```

Supported games: `2048`, `connect4`, `go`.

### How the benchmark works

- **PufferLib**: Calls `binding.vec_step()` directly (bypasses Python logging for pure C speed). Steps a fixed number of times in a Python loop.
- **PGX**: Compiles a full rollout (init + N steps) into a single XLA program via `jax.lax.scan`. This means one host→device dispatch for the entire rollout, not one per step.
- Both use the same number of steps (1000) and environments (1 to 8192), taking the best of multiple trials.

### Key insight: `jax.lax.scan` vs Python loops

For JAX benchmarks, always use `jax.lax.scan` instead of a Python `for` loop:

```python
# WRONG — 1000 separate dispatches, measures Python+dispatch overhead
for step in range(1000):
    state = jitted_step(state, actions[step])

# RIGHT — single compiled XLA program, measures actual compute
def rollout(state, actions):
    def body(state, action):
        state = vmapped_step(state, action)
        return state, None
    final, _ = jax.lax.scan(body, state, actions)
    return final
jitted_rollout = jax.jit(rollout)
```

### Typical results (M3 MacBook, CPU only)

PufferLib (C) achieves ~2.6M-13.7M SPS on 2048, scaling well with num_envs since the C code is cache-friendly. PGX benefits from XLA fusion but has higher per-env overhead at small batch sizes. The crossover point depends on hardware — GPUs strongly favor PGX at high env counts.

---

## Porting Environments to JAX

### Why port a C environment to JAX?

- **GPU acceleration**: JAX environments run on GPU, scaling to millions of environments
- **End-to-end differentiability**: The environment can be part of the computation graph
- **Single program**: `jax.lax.scan` compiles the entire rollout (1000+ steps) into one XLA kernel — zero Python overhead between steps

### Example: Breakout (`breakout_jax.py`)

A complete JAX port of PufferLib's Breakout is included as `breakout_jax.py`. It uses a gym-style API with `NamedTuple` state:

```python
from breakout_jax import BreakoutJAX

env = BreakoutJAX()

# Single environment
state, obs, info = env.reset(jax.random.key(0))
state, obs, reward, terminated, truncated, info = env.step(state, action, key)

# Vectorized (1024 environments)
reset_fn = jax.jit(jax.vmap(env.reset))
step_fn = jax.jit(jax.vmap(env.step))
keys = jax.random.split(jax.random.key(0), 1024)
states, obs, infos = reset_fn(keys)
states, obs, rewards, terminated, truncated, infos = step_fn(states, actions, keys)
```

### Porting guide

1. **State as NamedTuple**: All game state goes into a `typing.NamedTuple`. This is automatically a JAX pytree — no extra dependencies needed.

2. **Pure functions**: `reset(key)` and `step(state, action, key)` must be pure (no side effects). All randomness comes from explicit `key` arguments.

3. **Branching**: Replace `if/else` with `jnp.where()`. JAX traces both branches, so both must produce the same shapes.

4. **Loops**: Replace `for` loops with `jax.lax.scan` (fixed iterations) or `jax.lax.while_loop` (dynamic).

5. **Auto-reset**: Handle terminal states inside `step()` using `jax.tree.map` with `jnp.where`:
   ```python
   reset_state, _, _ = self.reset(key)
   game_over = state.num_balls < 0
   state = jax.tree.map(lambda r, s: jnp.where(game_over, r, s), reset_state, state)
   ```

### GPU performance: indexing vs multiplication

When destroying bricks or modifying array elements by index, prefer multiplication masks over scatter-based indexing:

```python
# SLOW on GPU — .at[idx].set() compiles to scatter, requires atomic ops
bricks = bricks.at[hit_idx].set(1.0)

# FAST on GPU — elementwise multiply, trivially parallel
hit_mask = jnp.arange(num_bricks) == hit_idx
bricks = jnp.where(hit_mask, 1.0, bricks)
```

Scatter operations serialize on GPU because they need atomic memory access. A full-array multiply seems wasteful (touches all elements) but is actually faster because it maps perfectly to SIMD/GPU parallelism.

### What's different from the C version?

- **Collision detection**: C uses early-exit loops (check bricks sequentially, stop at first hit). JAX checks all bricks in parallel via `jax.vmap`, then picks the earliest. This is slower for single envs but scales better with batch size.
- **State management**: C uses mutable structs with pointer-based shared memory. JAX uses immutable NamedTuples returned from pure functions.
- **Frameskip**: C uses a `for` loop. JAX uses `jax.lax.scan` over the frameskip count.
