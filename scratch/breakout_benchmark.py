"""JAX port of PufferLib's Breakout environment.

Standalone, gym-style API. Pure functional — fully jit-able and vmap-able.

Port of pufferlib/ocean/breakout/breakout.h — same physics, same observations,
same game rules. Can be used standalone or benchmarked against the C version.

Usage:
    import jax
    from breakout_jax import BreakoutJAX

    env = BreakoutJAX()  # auto-detects CPU/GPU for collision strategy

    # Single env
    state, obs, info = env.reset(jax.random.key(0))
    state, obs, reward, terminated, truncated, info = env.step(state, action, key)

    # Vectorized
    reset_fn = jax.jit(jax.vmap(env.reset))
    step_fn = jax.jit(jax.vmap(env.step))
    keys = jax.random.split(jax.random.key(0), 1024)
    states, obs, infos = reset_fn(keys)
    states, obs, rewards, terminated, truncated, infos = step_fn(states, actions, keys)

    # Force collision mode: "direct" (default) or "vmap"
    env = BreakoutJAX(collision_mode="vmap")

Benchmark:
    python breakout_jax.py --pufferlib   # PufferLib (C) — run in numpy<2 env
    python breakout_jax.py --jax          # JAX — run in JAX env
    python breakout_jax.py --plot         # combine results
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple

# ── Constants (matching C) ──────────────────────────────────────────────────

NOOP = 0
LEFT = 1
RIGHT = 2

HALF_PADDLE_WIDTH = 31.0
Y_OFFSET = 50.0
TICK_RATE = 1.0 / 60.0


# ── State ───────────────────────────────────────────────────────────────────


class BreakoutState(NamedTuple):
    """Game state. NamedTuple is a JAX pytree automatically."""

    paddle_x: jnp.ndarray
    paddle_y: jnp.ndarray
    ball_x: jnp.ndarray
    ball_y: jnp.ndarray
    ball_vx: jnp.ndarray
    ball_vy: jnp.ndarray
    ball_speed: jnp.ndarray
    paddle_width: jnp.ndarray
    brick_states: jnp.ndarray  # (num_bricks,) float32 — 0=alive, 1=destroyed
    score: jnp.ndarray
    num_balls: jnp.ndarray
    balls_fired: jnp.ndarray
    hits: jnp.ndarray
    tick: jnp.ndarray


# ── Environment ─────────────────────────────────────────────────────────────


class BreakoutJAX:
    """JAX-native Breakout environment with gym-style API.

    reset(key) -> (state, obs, info)
    step(state, action, key) -> (state, obs, reward, terminated, truncated, info)
    """

    def __init__(
        self,
        frameskip=4,
        width=576,
        height=330,
        paddle_width=62,
        paddle_height=8,
        ball_width=32,
        ball_height=32,
        brick_width=32,
        brick_height=12,
        brick_rows=6,
        brick_cols=18,
        initial_ball_speed=256.0,
        max_ball_speed=448.0,
        paddle_speed=620.0,
        collision_mode="auto",
    ):
        self.frameskip = frameskip
        self.width = float(width)
        self.height = float(height)
        self.initial_paddle_width = float(paddle_width)
        self.paddle_height = float(paddle_height)
        self.ball_width = float(ball_width)
        self.ball_height = float(ball_height)
        self.brick_width = float(brick_width)
        self.brick_height = float(brick_height)
        self.brick_rows = brick_rows
        self.brick_cols = brick_cols
        self.num_bricks = brick_rows * brick_cols
        self.initial_ball_speed = initial_ball_speed
        self.max_ball_speed = max_ball_speed
        self.paddle_speed = paddle_speed
        self.num_actions = 3
        self.observation_size = 10 + self.num_bricks

        # collision_mode: "direct" (flat array ops, default — fast everywhere),
        # "vmap" (parallel via jax.vmap — alternative for GPU)
        if collision_mode == "auto":
            self.collision_mode = "direct"
        else:
            self.collision_mode = collision_mode

        # Pre-compute brick positions (static)
        bx = jnp.array(
            [c * brick_width for r in range(brick_rows) for c in range(brick_cols)],
            dtype=jnp.float32,
        )
        by = jnp.array(
            [
                r * brick_height + Y_OFFSET
                for r in range(brick_rows)
                for c in range(brick_cols)
            ],
            dtype=jnp.float32,
        )
        self.brick_x = bx
        self.brick_y = by

        # Score per brick: 7 - 3*(row//2)
        self.brick_scores = jnp.array(
            [
                7.0 - 3.0 * (r // 2)
                for r in range(brick_rows)
                for _ in range(brick_cols)
            ],
            dtype=jnp.float32,
        )
        self.half_max_score = int(self.brick_scores.sum())
        self.max_score = 2 * self.half_max_score

        # Which bricks are in the top 3 rows (hitting these maxes ball speed)
        self.top_brick_mask = jnp.array(
            [r < 3 for r in range(brick_rows) for _ in range(brick_cols)]
        )

    # ── Reset ────────────────────────────────────────────────────────────────

    def reset(self, key):
        """Initialize a single environment.

        Returns: (state, obs, info)
        """
        paddle_x = self.width / 2.0 - self.initial_paddle_width / 2.0
        paddle_y = self.height - self.paddle_height - 10.0
        ball_x = paddle_x + (self.initial_paddle_width / 2.0 - self.ball_width / 2.0)
        ball_y = self.height / 2.0 - 30.0

        state = BreakoutState(
            paddle_x=jnp.float32(paddle_x),
            paddle_y=jnp.float32(paddle_y),
            ball_x=jnp.float32(ball_x),
            ball_y=jnp.float32(ball_y),
            ball_vx=jnp.float32(0.0),
            ball_vy=jnp.float32(0.0),
            ball_speed=jnp.float32(self.initial_ball_speed),
            paddle_width=jnp.float32(self.initial_paddle_width),
            brick_states=jnp.zeros(self.num_bricks, dtype=jnp.float32),
            score=jnp.int32(0),
            num_balls=jnp.int32(5),
            balls_fired=jnp.int32(0),
            hits=jnp.int32(0),
            tick=jnp.int32(0),
        )
        obs = self._observe(state)
        return state, obs, {}

    # ── Step ─────────────────────────────────────────────────────────────────

    def step(self, state, action, key):
        """Step the environment.

        Args:
            state: BreakoutState
            action: int in {0, 1, 2} (noop, left, right)
            key: PRNGKey for ball launch direction

        Returns: (state, obs, reward, terminated, truncated, info)
        """
        # Auto-reset: if game was over, reset first
        reset_state, reset_obs, _ = self.reset(key)
        game_over = (state.num_balls < 0) | (state.score >= self.max_score)
        state = jax.tree.map(
            lambda r, s: jnp.where(game_over, r, s), reset_state, state
        )

        reward = jnp.float32(0.0)
        terminated = jnp.bool_(False)

        # Determine paddle movement
        act = jnp.where(action == LEFT, -1.0, jnp.where(action == RIGHT, 1.0, 0.0))

        # On first action, launch ball
        direction = jnp.float32(jnp.pi / 3.25)
        launch_vy = jnp.cos(direction) * state.ball_speed * TICK_RATE
        launch_vx = jnp.sin(direction) * state.ball_speed * TICK_RATE
        launch_vx = jnp.where(jax.random.bernoulli(key), launch_vx, -launch_vx)

        ball_vx = jnp.where(state.balls_fired == 0, launch_vx, state.ball_vx)
        ball_vy = jnp.where(state.balls_fired == 0, launch_vy, state.ball_vy)
        act = jnp.where(state.balls_fired == 0, 0.0, act)
        balls_fired = jnp.where(state.balls_fired == 0, 1, state.balls_fired)

        ball_x = state.ball_x
        ball_y = state.ball_y
        paddle_x = state.paddle_x
        paddle_width = state.paddle_width
        ball_speed = state.ball_speed
        brick_states = state.brick_states
        score = state.score
        hits = state.hits
        num_balls = state.num_balls
        tick = state.tick

        # Run frameskip frames
        def frame_body(carry, _):
            (
                ball_x,
                ball_y,
                ball_vx,
                ball_vy,
                paddle_x,
                paddle_width,
                ball_speed,
                brick_states,
                score,
                hits,
                num_balls,
                reward,
                terminated,
                tick,
                balls_fired,
            ) = carry

            tick = tick + 1

            # Move paddle
            paddle_x = paddle_x + act * self.paddle_speed * TICK_RATE
            paddle_x = jnp.clip(paddle_x, 0.0, self.width - paddle_width)

            # Wall bounds safety
            offset = self.max_ball_speed * 1.1 * TICK_RATE
            ball_x = jnp.where(ball_x < 0, ball_x + offset, ball_x)
            ball_x = jnp.where(ball_x > self.width, ball_x - offset, ball_x)
            ball_y = jnp.where(ball_y < 0, ball_y + offset, ball_y)

            # ── Collision detection ────────────────────────────────────
            #
            # GPU PERFORMANCE NOTE: For brick destruction, we use a
            # multiplication mask approach:
            #   new_bricks = bricks * (1 - hit_mask)
            # instead of scatter-based indexing:
            #   bricks = bricks.at[idx].set(1.0)
            # On GPUs, .at[idx].set() compiles to a scatter op which can be
            # significantly slower than a full-array multiply, even though
            # multiply touches all elements. Scatter requires atomic ops and
            # serialization; multiply is a trivially parallel elementwise op.

            (
                brick_col_t,
                brick_col_vx,
                brick_col_vy,
                brick_col_x,
                brick_col_y,
                hit_brick_idx,
            ) = self._brick_collisions(ball_x, ball_y, ball_vx, ball_vy, brick_states)

            # Wall collisions
            wall_t, wall_vx, wall_vy, wall_x, wall_y, wall_type = (
                self._check_wall_collisions(ball_x, ball_y, ball_vx, ball_vy)
            )

            # Paddle collision
            paddle_t, paddle_vx, paddle_vy, paddle_col_x, paddle_col_y, paddle_hit = (
                self._check_paddle_collision(
                    ball_x,
                    ball_y,
                    ball_vx,
                    ball_vy,
                    paddle_x,
                    state.paddle_y,
                    paddle_width,
                    ball_speed,
                )
            )

            # Find earliest collision
            best_t = jnp.float32(2.0)
            best_vx = ball_vx
            best_vy = ball_vy
            best_x = ball_x
            best_y = ball_y
            is_brick = jnp.bool_(False)
            is_backwall = jnp.bool_(False)
            is_paddle = jnp.bool_(False)

            # Check brick collision
            brick_hit = brick_col_t < best_t
            best_t = jnp.where(brick_hit, brick_col_t, best_t)
            best_vx = jnp.where(brick_hit, brick_col_vx, best_vx)
            best_vy = jnp.where(brick_hit, brick_col_vy, best_vy)
            best_x = jnp.where(brick_hit, brick_col_x, best_x)
            best_y = jnp.where(brick_hit, brick_col_y, best_y)
            is_brick = brick_hit

            # Check wall collision
            wall_hit = (wall_t < best_t) & (wall_t <= 1.0)
            best_t = jnp.where(wall_hit, wall_t, best_t)
            best_vx = jnp.where(wall_hit, wall_vx, best_vx)
            best_vy = jnp.where(wall_hit, wall_vy, best_vy)
            best_x = jnp.where(wall_hit, wall_x, best_x)
            best_y = jnp.where(wall_hit, wall_y, best_y)
            is_brick = jnp.where(wall_hit, False, is_brick)
            is_backwall = wall_hit & (wall_type == 1)

            # Check paddle collision
            paddle_valid = paddle_hit & (paddle_t <= best_t)
            is_paddle = paddle_valid

            # Apply collision results
            any_collision = best_t <= 1.0
            no_paddle = ~is_paddle

            final_vx = jnp.where(
                is_paddle,
                paddle_vx,
                jnp.where(any_collision & no_paddle, best_vx, ball_vx),
            )
            final_vy = jnp.where(
                is_paddle,
                paddle_vy,
                jnp.where(any_collision & no_paddle, best_vy, ball_vy),
            )

            new_ball_x = jnp.where(
                is_paddle,
                paddle_col_x,
                jnp.where(any_collision & no_paddle, best_x, ball_x + ball_vx),
            )
            new_ball_y = jnp.where(
                is_paddle,
                paddle_col_y,
                jnp.where(any_collision & no_paddle, best_y, ball_y + ball_vy),
            )

            new_ball_x = jnp.where(
                ~any_collision & ~is_paddle, ball_x + ball_vx, new_ball_x
            )
            new_ball_y = jnp.where(
                ~any_collision & ~is_paddle, ball_y + ball_vy, new_ball_y
            )

            # Destroy brick — multiply mask (GPU-friendly, avoids scatter)
            brick_hit_mask = (jnp.arange(self.num_bricks) == hit_brick_idx) & is_brick
            brick_reward = jnp.where(is_brick, self.brick_scores[hit_brick_idx], 0.0)
            new_brick_states = jnp.where(brick_hit_mask, 1.0, brick_states)
            new_score = score + jnp.where(is_brick, brick_reward.astype(jnp.int32), 0)
            reward = reward + brick_reward

            # Ball speed increase on top brick hit
            hit_top = is_brick & self.top_brick_mask[hit_brick_idx]
            new_ball_speed = jnp.where(hit_top, self.max_ball_speed, ball_speed)

            # Paddle hit: increment hits, speed up every 4 hits
            new_hits = jnp.where(is_paddle, hits + 1, hits)
            speed_bump = (
                is_paddle & (new_hits % 4 == 0) & (new_ball_speed < self.max_ball_speed)
            )
            new_ball_speed = jnp.where(
                speed_bump, new_ball_speed + 64.0, new_ball_speed
            )

            # Backwall hit: shrink paddle
            new_paddle_width = jnp.where(is_backwall, HALF_PADDLE_WIDTH, paddle_width)

            # Half-score brick reset
            at_half = new_score == self.half_max_score
            new_brick_states = jnp.where(
                at_half, jnp.zeros_like(new_brick_states), new_brick_states
            )

            # Ball fell below paddle
            ball_lost = new_ball_y >= (state.paddle_y + self.paddle_height)
            new_num_balls = jnp.where(ball_lost, num_balls - 1, num_balls)

            # Reset round on ball loss
            r_paddle_x = self.width / 2.0 - self.initial_paddle_width / 2.0
            r_ball_x = r_paddle_x + (
                self.initial_paddle_width / 2.0 - self.ball_width / 2.0
            )
            r_ball_y = self.height / 2.0 - 30.0

            new_ball_x = jnp.where(ball_lost, r_ball_x, new_ball_x)
            new_ball_y = jnp.where(ball_lost, r_ball_y, new_ball_y)
            final_vx = jnp.where(ball_lost, 0.0, final_vx)
            final_vy = jnp.where(ball_lost, 0.0, final_vy)
            new_ball_speed = jnp.where(
                ball_lost, jnp.float32(self.initial_ball_speed), new_ball_speed
            )
            new_paddle_width = jnp.where(
                ball_lost, jnp.float32(self.initial_paddle_width), new_paddle_width
            )
            new_hits = jnp.where(ball_lost, 0, new_hits)
            new_balls_fired = jnp.where(ball_lost, jnp.int32(0), balls_fired)
            r_paddle_x_reset = self.width / 2.0 - new_paddle_width / 2.0
            new_paddle_x = jnp.where(ball_lost, r_paddle_x_reset, paddle_x)

            # Terminal: out of balls or max score
            game_over = (new_num_balls < 0) | (new_score >= self.max_score)
            new_terminated = terminated | game_over

            carry = (
                new_ball_x,
                new_ball_y,
                final_vx,
                final_vy,
                new_paddle_x,
                new_paddle_width,
                new_ball_speed,
                new_brick_states,
                new_score,
                new_hits,
                new_num_balls,
                reward,
                new_terminated,
                tick,
                new_balls_fired,
            )
            return carry, None

        init_carry = (
            ball_x,
            ball_y,
            ball_vx,
            ball_vy,
            paddle_x,
            paddle_width,
            ball_speed,
            brick_states,
            score,
            hits,
            num_balls,
            reward,
            terminated,
            tick,
            balls_fired,
        )

        final_carry, _ = jax.lax.scan(
            frame_body, init_carry, None, length=self.frameskip
        )

        (
            ball_x,
            ball_y,
            ball_vx,
            ball_vy,
            paddle_x,
            paddle_width,
            ball_speed,
            brick_states,
            score,
            hits,
            num_balls,
            reward,
            terminated,
            tick,
            balls_fired,
        ) = final_carry

        new_state = BreakoutState(
            ball_x=ball_x,
            ball_y=ball_y,
            ball_vx=ball_vx,
            ball_vy=ball_vy,
            paddle_x=paddle_x,
            paddle_y=state.paddle_y,
            paddle_width=paddle_width,
            ball_speed=ball_speed,
            brick_states=brick_states,
            score=score,
            hits=hits,
            num_balls=num_balls,
            balls_fired=balls_fired,
            tick=tick,
        )
        obs = self._observe(new_state)
        # Breakout has no truncation (game ends on terminal)
        return new_state, obs, reward, terminated, jnp.bool_(False), {}

    # ── Observation ─────────────────────────────────────────────────────────

    def _observe(self, state):
        return jnp.concatenate(
            [
                jnp.array(
                    [
                        state.paddle_x / self.width,
                        state.paddle_y / self.height,
                        state.ball_x / self.width,
                        state.ball_y / self.height,
                        state.ball_vx / 512.0,
                        state.ball_vy / 512.0,
                        state.balls_fired / 5.0,
                        state.score / 864.0,
                        state.num_balls / 5.0,
                        state.paddle_width / (2.0 * HALF_PADDLE_WIDTH),
                    ]
                ),
                state.brick_states,
            ]
        )

    # ── Collision helpers ───────────────────────────────────────────────────

    def _brick_collisions(self, ball_x, ball_y, ball_vx, ball_vy, brick_states):
        """Find earliest brick collision. Returns (t, vx, vy, x, y, brick_idx).

        All 108 bricks × 4 faces = 432 candidates are evaluated as flat array
        ops — no vmap, no scan, no per-brick function calls. The XLA graph is
        just elementwise ops on (108,) arrays + one argmin on (432,).
        """
        INF = jnp.float32(1e9)
        bw, bh = self.brick_width, self.brick_height
        baw, bah = self.ball_width, self.ball_height
        bx, by = self.brick_x, self.brick_y  # (108,) each
        alive = brick_states == 0.0  # (108,)

        # Face 1: ball right side vs brick left wall (ball moving right)
        t1 = (bx - (ball_x + baw)) / ball_vx
        ov1 = jnp.minimum(by + bh, ball_y + bah + ball_vy * t1) - jnp.maximum(
            by, ball_y + ball_vy * t1
        )
        valid1 = (ov1 > 0) & (t1 > 0) & (t1 <= 1.0) & (ball_vx > 0) & alive
        t1 = jnp.where(valid1, t1, INF)
        ov1 = jnp.where(valid1, ov1, -1.0)
        cx1 = bx - baw
        cy1 = ball_y + ball_vy * t1

        # Face 2: ball left side vs brick right wall (ball moving left)
        t2 = ((bx + bw) - ball_x) / ball_vx
        ov2 = jnp.minimum(by + bh, ball_y + bah + ball_vy * t2) - jnp.maximum(
            by, ball_y + ball_vy * t2
        )
        valid2 = (ov2 > 0) & (t2 > 0) & (t2 <= 1.0) & (ball_vx < 0) & alive
        t2 = jnp.where(valid2, t2, INF)
        ov2 = jnp.where(valid2, ov2, -1.0)
        cx2 = bx + bw
        cy2 = ball_y + ball_vy * t2

        # Face 3: ball bottom vs brick top (ball moving down)
        t3 = (by - (ball_y + bah)) / ball_vy
        ov3 = jnp.minimum(bx + bw, ball_x + baw + ball_vx * t3) - jnp.maximum(
            bx, ball_x + ball_vx * t3
        )
        valid3 = (ov3 > 0) & (t3 > 0) & (t3 <= 1.0) & (ball_vy > 0) & alive
        t3 = jnp.where(valid3, t3, INF)
        ov3 = jnp.where(valid3, ov3, -1.0)
        cx3 = ball_x + ball_vx * t3
        cy3 = by - bah

        # Face 4: ball top vs brick bottom (ball moving up)
        t4 = ((by + bh) - ball_y) / ball_vy
        ov4 = jnp.minimum(bx + bw, ball_x + baw + ball_vx * t4) - jnp.maximum(
            bx, ball_x + ball_vx * t4
        )
        valid4 = (ov4 > 0) & (t4 > 0) & (t4 <= 1.0) & (ball_vy < 0) & alive
        t4 = jnp.where(valid4, t4, INF)
        ov4 = jnp.where(valid4, ov4, -1.0)
        cx4 = ball_x + ball_vx * t4
        cy4 = by + bh

        # Flatten all 4*108 = 432 candidates and find global best
        all_t = jnp.concatenate([t1, t2, t3, t4])
        all_ov = jnp.concatenate([ov1, ov2, ov3, ov4])
        all_cx = jnp.concatenate([cx1, cx2, cx3, cx4])
        all_cy = jnp.concatenate([cy1, cy2, cy3, cy4])

        sort_key = all_t - all_ov * 1e-6
        winner = jnp.argmin(sort_key)

        # Determine face type from winner index to get reflected velocity
        n = self.num_bricks
        face = winner // n  # 0,1 = vline (reflect vx), 2,3 = hline (reflect vy)
        brick_idx = winner % n
        is_vline = face < 2
        out_vx = jnp.where(is_vline, -ball_vx, ball_vx)
        out_vy = jnp.where(is_vline, ball_vy, -ball_vy)

        return all_t[winner], out_vx, out_vy, all_cx[winner], all_cy[winner], brick_idx

    def _check_wall_collisions(self, ball_x, ball_y, ball_vx, ball_vy):
        """Check left wall, right wall, and back (top) wall."""
        INF = jnp.float32(1e9)
        bw = self.ball_width
        bh = self.ball_height

        best_t = INF
        best_vx = ball_vx
        best_vy = ball_vy
        best_x = ball_x
        best_y = ball_y
        wall_type = jnp.int32(0)  # 0=side, 1=back

        # Left wall (ball moving left): vline at x=0
        t = (0.0 - ball_x) / ball_vx
        top = jnp.minimum(self.height, ball_y + bh + ball_vy * t)
        bot = jnp.maximum(0.0, ball_y + ball_vy * t)
        ov = top - bot
        hit = (ov > 0) & (t > 0) & (t <= 1.0) & (ball_vx < 0) & (t < best_t)
        best_t = jnp.where(hit, t, best_t)
        best_vx = jnp.where(hit, -ball_vx, best_vx)
        best_x = jnp.where(hit, 0.0, best_x)
        best_y = jnp.where(hit, ball_y + ball_vy * t, best_y)

        # Right wall (ball moving right): vline at x=width
        t = (self.width - (ball_x + bw)) / ball_vx
        top = jnp.minimum(self.height, ball_y + bh + ball_vy * t)
        bot = jnp.maximum(0.0, ball_y + ball_vy * t)
        ov = top - bot
        hit = (ov > 0) & (t > 0) & (t <= 1.0) & (ball_vx > 0) & (t < best_t)
        best_t = jnp.where(hit, t, best_t)
        best_vx = jnp.where(hit, -ball_vx, best_vx)
        best_x = jnp.where(hit, self.width - bw, best_x)
        best_y = jnp.where(hit, ball_y + ball_vy * t, best_y)

        # Back wall / top (ball moving up): hline at y=0
        t = (0.0 - ball_y) / ball_vy
        right = jnp.minimum(self.width, ball_x + bw + ball_vx * t)
        left = jnp.maximum(0.0, ball_x + ball_vx * t)
        ov = right - left
        hit = (ov > 0) & (t > 0) & (t <= 1.0) & (ball_vy < 0) & (t < best_t)
        best_t = jnp.where(hit, t, best_t)
        best_vy = jnp.where(hit, -ball_vy, best_vy)
        best_x = jnp.where(hit, ball_x + ball_vx * t, best_x)
        best_y = jnp.where(hit, 0.0, best_y)
        wall_type = jnp.where(hit, 1, wall_type)

        return best_t, best_vx, best_vy, best_x, best_y, wall_type

    def _check_paddle_collision(
        self,
        ball_x,
        ball_y,
        ball_vx,
        ball_vy,
        paddle_x,
        paddle_y,
        paddle_width,
        ball_speed,
    ):
        """Check paddle collision. Returns angle-based reflected velocity."""
        bw = self.ball_width
        bh = self.ball_height

        above = (ball_y + bh + ball_vy) < paddle_y

        # hline collision: ball bottom vs paddle top
        t = (paddle_y - (ball_y + bh)) / ball_vy
        right = jnp.minimum(paddle_x + paddle_width, ball_x + bw + ball_vx * t)
        left = jnp.maximum(paddle_x, ball_x + ball_vx * t)
        ov = right - left
        hit = (ov > 0) & (t > 0) & (t <= 1.0) & (~above)

        cx = ball_x + ball_vx * t

        # Angle-based reflection
        relative = ((ball_x + bw / 2.0) - paddle_x) / paddle_width
        base_angle = jnp.float32(jnp.pi / 4.0)
        angle = -base_angle + relative * 2.0 * base_angle
        new_vx = jnp.sin(angle) * ball_speed * TICK_RATE
        new_vy = -jnp.cos(angle) * ball_speed * TICK_RATE

        col_y = paddle_y - bh
        return t, new_vx, new_vy, cx, col_y, hit


# ── Benchmark ───────────────────────────────────────────────────────────────

NUM_ENVS_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
NUM_STEPS = 1000
NUM_TRIALS = 2
WARMUP_TRIALS = 1


def result_paths():
    d = os.path.dirname(os.path.abspath(__file__))
    return (
        os.path.join(d, "bench_pufferlib_breakout.json"),
        os.path.join(d, "bench_jax_breakout.json"),
    )


def bench_pufferlib():
    import numpy as np
    from pufferlib.ocean.breakout.breakout import Breakout
    from pufferlib.ocean.breakout import binding

    results = []
    for n in NUM_ENVS_LIST:
        env = Breakout(num_envs=n)
        env.reset()
        actions = np.random.randint(0, 3, (NUM_STEPS, n), dtype=np.int32).astype(
            np.float32
        )

        best_sps = 0
        for trial in range(WARMUP_TRIALS + NUM_TRIALS):
            env.reset()
            start = time.perf_counter()
            for s in range(NUM_STEPS):
                env.actions[:] = actions[s]
                binding.vec_step(env.c_envs)
            elapsed = time.perf_counter() - start

            if trial >= WARMUP_TRIALS:
                sps = (NUM_STEPS * n) / elapsed
                best_sps = max(best_sps, sps)

        env.close()
        print(f"  PufferLib  num_envs={n:>6}  SPS={best_sps:>12,.0f}")
        results.append(best_sps)

    puffer_json, _ = result_paths()
    with open(puffer_json, "w") as f:
        json.dump({"num_envs": NUM_ENVS_LIST, "sps": results}, f)
    print(f"Saved to {puffer_json}")
    return results


def bench_jax():
    env = BreakoutJAX()

    reset_fn = jax.jit(jax.vmap(env.reset))
    vmapped_step = jax.vmap(env.step)

    def rollout(states, actions, keys):
        def body(states, inputs):
            action, key = inputs
            states, obs, reward, terminated, truncated, info = vmapped_step(
                states, action, key
            )
            return states, None

        final_states, _ = jax.lax.scan(body, states, (actions, keys))
        return final_states

    jitted_rollout = jax.jit(rollout)

    results = []
    for n in NUM_ENVS_LIST:
        rng = jax.random.key(0)
        rng, rng_init, rng_actions, rng_keys = jax.random.split(rng, 4)
        init_keys = jax.random.split(rng_init, n)
        actions = jax.random.randint(rng_actions, (NUM_STEPS, n), 0, 3)
        step_keys = jax.random.split(rng_keys, NUM_STEPS * n).reshape(
            NUM_STEPS, n, *jax.random.key(0).shape
        )

        states, obs, infos = reset_fn(init_keys)

        # Warmup (includes JIT compilation)
        final = jitted_rollout(states, actions, step_keys)
        _ = final.tick.block_until_ready()

        best_sps = 0
        for trial in range(NUM_TRIALS):
            states, obs, infos = reset_fn(init_keys)
            _ = states.tick.block_until_ready()
            start = time.perf_counter()
            final = jitted_rollout(states, actions, step_keys)
            _ = final.tick.block_until_ready()
            elapsed = time.perf_counter() - start
            sps = (NUM_STEPS * n) / elapsed
            best_sps = max(best_sps, sps)

        print(f"  JAX        num_envs={n:>6}  SPS={best_sps:>12,.0f}")
        results.append(best_sps)

    _, jax_json = result_paths()
    with open(jax_json, "w") as f:
        json.dump({"num_envs": NUM_ENVS_LIST, "sps": results}, f)
    print(f"Saved to {jax_json}")
    return results


def plot():
    import matplotlib.pyplot as plt

    puffer_json, jax_json = result_paths()
    fig, ax = plt.subplots(figsize=(10, 6))

    if os.path.exists(puffer_json):
        with open(puffer_json) as f:
            d = json.load(f)
        ax.plot(
            d["num_envs"],
            d["sps"],
            "o-",
            label="PufferLib (C)",
            linewidth=2,
            markersize=6,
        )
        print(f"Loaded PufferLib results from {puffer_json}")
    else:
        print(f"Warning: {puffer_json} not found. Run with --pufferlib first.")

    if os.path.exists(jax_json):
        with open(jax_json) as f:
            d = json.load(f)
        ax.plot(
            d["num_envs"],
            d["sps"],
            "s-",
            label="JAX (lax.scan)",
            linewidth=2,
            markersize=6,
        )
        print(f"Loaded JAX results from {jax_json}")
    else:
        print(f"Warning: {jax_json} not found. Run with --jax first.")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Environments", fontsize=13)
    ax.set_ylabel("Steps / Second", fontsize=13)
    ax.set_title("Breakout Environment Throughput: PufferLib (C) vs JAX", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.xaxis.set_major_locator(plt.LogLocator(base=10))
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    fig.tight_layout()
    outfile = "benchmark_breakout.png"
    fig.savefig(outfile, dpi=150)
    print(f"Plot saved to {outfile}")
    plt.show()


if __name__ == "__main__":
    import argparse
    import json
    import os
    import time

    parser = argparse.ArgumentParser(
        description="Benchmark Breakout: PufferLib (C) vs JAX"
    )
    parser.add_argument(
        "--pufferlib", action="store_true", help="Run PufferLib (C) benchmark"
    )
    parser.add_argument("--jax", action="store_true", help="Run JAX benchmark")
    parser.add_argument(
        "--plot", action="store_true", help="Plot results from JSON files"
    )
    args = parser.parse_args()

    if not (args.pufferlib or args.jax or args.plot):
        parser.print_help()
        print("\nRun --pufferlib and --jax in separate conda envs, then --plot.")
        exit(1)

    if args.pufferlib:
        print("=== Breakout — PufferLib (C) ===")
        bench_pufferlib()

    if args.jax:
        print("=== Breakout — JAX ===")
        bench_jax()

    if args.plot or args.pufferlib or args.jax:
        plot()
