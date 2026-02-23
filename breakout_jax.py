import jax
import jax.numpy as jnp
from typing import NamedTuple

NOOP = 0
LEFT = 1
RIGHT = 2

HALF_PADDLE_WIDTH = 31.0
Y_OFFSET = 50.0
TICK_RATE = 1.0 / 60.0


class BreakoutState(NamedTuple):
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




class BreakoutJAX:

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

        self.top_brick_mask = jnp.array(
            [r < 3 for r in range(brick_rows) for _ in range(brick_cols)]
        )

    def reset(self, key):
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

    def step(self, state, action, key):
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

            (
                brick_col_t,
                brick_col_vx,
                brick_col_vy,
                brick_col_x,
                brick_col_y,
                hit_brick_idx,
            ) = self._brick_collisions(ball_x, ball_y, ball_vx, ball_vy, brick_states)

            wall_t, wall_vx, wall_vy, wall_x, wall_y, wall_type = (
                self._check_wall_collisions(ball_x, ball_y, ball_vx, ball_vy)
            )

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

            # Destroy brick
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

        # Auto-reset at END (like PufferLib C): if game is over, return reset state
        # but keep the terminal reward and terminated flag from this step
        game_over = (num_balls < 0) | (score >= self.max_score)
        key, reset_key = jax.random.split(key)
        reset_state = self.reset(reset_key)[0]
        new_state = jax.tree.map(
            lambda r, s: jnp.where(game_over, r, s), reset_state, new_state
        )

        obs = self._observe(new_state)
        # Breakout has no truncation (game ends on terminal)
        return new_state, obs, reward, terminated, jnp.bool_(False), {}

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

    def _brick_collisions(self, ball_x, ball_y, ball_vx, ball_vy, brick_states):
        """Find earliest brick collision. Returns (t, vx, vy, x, y, brick_idx)."""
        INF = jnp.float32(1e9)
        bw, bh = self.brick_width, self.brick_height
        baw, bah = self.ball_width, self.ball_height
        bx, by = self.brick_x, self.brick_y  # (108,)
        alive = brick_states == 0.0  # (108,)

        # ball right side vs brick left wall
        t1 = (bx - (ball_x + baw)) / ball_vx
        ov1 = jnp.minimum(by + bh, ball_y + bah + ball_vy * t1) - jnp.maximum(
            by, ball_y + ball_vy * t1
        )
        valid1 = (ov1 > 0) & (t1 > 0) & (t1 <= 1.0) & (ball_vx > 0) & alive
        t1 = jnp.where(valid1, t1, INF)
        ov1 = jnp.where(valid1, ov1, -1.0)
        cx1 = bx - baw
        cy1 = ball_y + ball_vy * t1

        # ball left side vs brick right wall
        t2 = ((bx + bw) - ball_x) / ball_vx
        ov2 = jnp.minimum(by + bh, ball_y + bah + ball_vy * t2) - jnp.maximum(
            by, ball_y + ball_vy * t2
        )
        valid2 = (ov2 > 0) & (t2 > 0) & (t2 <= 1.0) & (ball_vx < 0) & alive
        t2 = jnp.where(valid2, t2, INF)
        ov2 = jnp.where(valid2, ov2, -1.0)
        cx2 = bx + bw
        cy2 = ball_y + ball_vy * t2

        # ball bottom vs brick top
        t3 = (by - (ball_y + bah)) / ball_vy
        ov3 = jnp.minimum(bx + bw, ball_x + baw + ball_vx * t3) - jnp.maximum(
            bx, ball_x + ball_vx * t3
        )
        valid3 = (ov3 > 0) & (t3 > 0) & (t3 <= 1.0) & (ball_vy > 0) & alive
        t3 = jnp.where(valid3, t3, INF)
        ov3 = jnp.where(valid3, ov3, -1.0)
        cx3 = ball_x + ball_vx * t3
        cy3 = by - bah

        # ball top vs brick bottom
        t4 = ((by + bh) - ball_y) / ball_vy
        ov4 = jnp.minimum(bx + bw, ball_x + baw + ball_vx * t4) - jnp.maximum(
            bx, ball_x + ball_vx * t4
        )
        valid4 = (ov4 > 0) & (t4 > 0) & (t4 <= 1.0) & (ball_vy < 0) & alive
        t4 = jnp.where(valid4, t4, INF)
        ov4 = jnp.where(valid4, ov4, -1.0)
        cx4 = ball_x + ball_vx * t4
        cy4 = by + bh

        all_t = jnp.concatenate([t1, t2, t3, t4])
        all_ov = jnp.concatenate([ov1, ov2, ov3, ov4])
        all_cx = jnp.concatenate([cx1, cx2, cx3, cx4])
        all_cy = jnp.concatenate([cy1, cy2, cy3, cy4])

        sort_key = all_t - all_ov * 1e-6
        winner = jnp.argmin(sort_key)

        n = self.num_bricks
        face = winner // n  # 0,1 = vline (reflect vx), 2,3 = hline (reflect vy)
        brick_idx = winner % n
        is_vline = face < 2
        out_vx = jnp.where(is_vline, -ball_vx, ball_vx)
        out_vy = jnp.where(is_vline, ball_vy, -ball_vy)

        return all_t[winner], out_vx, out_vy, all_cx[winner], all_cy[winner], brick_idx

    def _check_wall_collisions(self, ball_x, ball_y, ball_vx, ball_vy):
        INF = jnp.float32(1e9)
        bw = self.ball_width
        bh = self.ball_height

        best_t = INF
        best_vx = ball_vx
        best_vy = ball_vy
        best_x = ball_x
        best_y = ball_y
        wall_type = jnp.int32(0)  # 0=side, 1=back

        # Left wall: vline at x=0
        t = (0.0 - ball_x) / ball_vx
        top = jnp.minimum(self.height, ball_y + bh + ball_vy * t)
        bot = jnp.maximum(0.0, ball_y + ball_vy * t)
        ov = top - bot
        hit = (ov > 0) & (t > 0) & (t <= 1.0) & (ball_vx < 0) & (t < best_t)
        best_t = jnp.where(hit, t, best_t)
        best_vx = jnp.where(hit, -ball_vx, best_vx)
        best_x = jnp.where(hit, 0.0, best_x)
        best_y = jnp.where(hit, ball_y + ball_vy * t, best_y)

        # Right wall: vline at x=width
        t = (self.width - (ball_x + bw)) / ball_vx
        top = jnp.minimum(self.height, ball_y + bh + ball_vy * t)
        bot = jnp.maximum(0.0, ball_y + ball_vy * t)
        ov = top - bot
        hit = (ov > 0) & (t > 0) & (t <= 1.0) & (ball_vx > 0) & (t < best_t)
        best_t = jnp.where(hit, t, best_t)
        best_vx = jnp.where(hit, -ball_vx, best_vx)
        best_x = jnp.where(hit, self.width - bw, best_x)
        best_y = jnp.where(hit, ball_y + ball_vy * t, best_y)

        # Back wall / top: hline at y=0
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
        bw = self.ball_width
        bh = self.ball_height

        above = (ball_y + bh + ball_vy) < paddle_y

        # ball bottom vs paddle top
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


