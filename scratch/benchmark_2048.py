"""Benchmark: PufferLib (C) vs PGX (JAX) environment throughput.

PufferLib requires numpy<2.0, PGX/JAX requires numpy>=2.0.
Run each half in its own conda env, results are saved to JSON and merged for plotting.

Usage:
    python benchmark_2048.py --pufferlib --game 2048
    python benchmark_2048.py --pgx --game connect4
    python benchmark_2048.py --plot --game 2048

Supported games: 2048, connect4, go
"""

import argparse
import json
import time
import os

import numpy as np

# --- Config ---
NUM_ENVS_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
NUM_STEPS = 1000
NUM_TRIALS = 2
WARMUP_TRIALS = 1

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Game definitions ---
GAMES = {
    "2048": dict(
        pufferlib_module="pufferlib.ocean.g2048.g2048",
        pufferlib_class="G2048",
        pufferlib_kwargs={},
        num_actions=4,
        pgx_id="2048",
        pgx_stochastic=True,  # tile spawns need a key
    ),
    "connect4": dict(
        pufferlib_module="pufferlib.ocean.connect4.connect4",
        pufferlib_class="Connect4",
        pufferlib_kwargs={},
        num_actions=7,
        pgx_id="connect_four",
        pgx_stochastic=False,
    ),
    "go": dict(
        pufferlib_module="pufferlib.ocean.go.go",
        pufferlib_class="Go",
        pufferlib_kwargs={"grid_size": 9},
        num_actions=82,  # 9*9 + 1 pass
        pgx_id="go_9x9",
        pgx_stochastic=False,
    ),
}


def result_paths(game):
    return (
        os.path.join(RESULTS_DIR, f"bench_pufferlib_{game}.json"),
        os.path.join(RESULTS_DIR, f"bench_pgx_{game}.json"),
    )


def bench_pufferlib(game):
    import importlib

    cfg = GAMES[game]
    mod = importlib.import_module(cfg["pufferlib_module"])
    EnvClass = getattr(mod, cfg["pufferlib_class"])

    results = []
    for n in NUM_ENVS_LIST:
        env = EnvClass(num_envs=n, **cfg["pufferlib_kwargs"])
        env.reset()
        actions = np.random.randint(
            0, cfg["num_actions"], (NUM_STEPS, n), dtype=np.int32
        )

        # Raw step: call C binding directly, skip Python logging logic
        import importlib

        binding = importlib.import_module(
            cfg["pufferlib_module"].rsplit(".", 1)[0] + ".binding"
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

    puffer_json, _ = result_paths(game)
    with open(puffer_json, "w") as f:
        json.dump({"num_envs": NUM_ENVS_LIST, "sps": results}, f)
    print(f"Saved to {puffer_json}")
    return results


def bench_pgx(game):
    import jax
    import pgx

    cfg = GAMES[game]
    env = pgx.make(cfg["pgx_id"])
    stochastic = cfg["pgx_stochastic"]

    results = []
    for n in NUM_ENVS_LIST:
        init_fn = jax.jit(jax.vmap(env.init))
        vmapped_step = jax.vmap(env.step)

        if stochastic:

            def rollout(state, actions, keys):
                def body(state, inputs):
                    action, key = inputs
                    state = vmapped_step(state, action, key)
                    return state, None

                final_state, _ = jax.lax.scan(body, state, (actions, keys))
                return final_state
        else:

            def rollout(state, actions):
                def body(state, action):
                    state = vmapped_step(state, action)
                    return state, None

                final_state, _ = jax.lax.scan(body, state, actions)
                return final_state

        jitted_rollout = jax.jit(rollout)

        rng = jax.random.key(0)
        rng, rng_init, rng_actions = jax.random.split(rng, 3)
        init_keys = jax.random.split(rng_init, n)
        actions = jax.random.randint(rng_actions, (NUM_STEPS, n), 0, cfg["num_actions"])

        if stochastic:
            rng, rng_keys = jax.random.split(rng)
            step_keys = jax.random.split(rng_keys, NUM_STEPS * n).reshape(
                NUM_STEPS, n, *jax.random.key(0).shape
            )

        state = init_fn(init_keys)

        best_sps = 0
        for trial in range(WARMUP_TRIALS + NUM_TRIALS):
            state = init_fn(init_keys)
            _ = state.terminated.block_until_ready()

            start = time.perf_counter()
            if stochastic:
                final_state = jitted_rollout(state, actions, step_keys)
            else:
                final_state = jitted_rollout(state, actions)
            _ = final_state.terminated.block_until_ready()
            elapsed = time.perf_counter() - start

            if trial >= WARMUP_TRIALS:
                sps = (NUM_STEPS * n) / elapsed
                best_sps = max(best_sps, sps)

        print(f"  PGX        num_envs={n:>6}  SPS={best_sps:>12,.0f}")
        results.append(best_sps)

    _, pgx_json = result_paths(game)
    with open(pgx_json, "w") as f:
        json.dump({"num_envs": NUM_ENVS_LIST, "sps": results}, f)
    print(f"Saved to {pgx_json}")
    return results


def plot(game):
    import matplotlib.pyplot as plt

    puffer_json, pgx_json = result_paths(game)
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

    if os.path.exists(pgx_json):
        with open(pgx_json) as f:
            d = json.load(f)
        ax.plot(
            d["num_envs"],
            d["sps"],
            "s-",
            label="PGX (JAX, lax.scan)",
            linewidth=2,
            markersize=6,
        )
        print(f"Loaded PGX results from {pgx_json}")
    else:
        print(f"Warning: {pgx_json} not found. Run with --pgx first.")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Environments", fontsize=13)
    ax.set_ylabel("Steps / Second", fontsize=13)
    ax.set_title(f"{game} Environment Throughput: PufferLib vs PGX", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.xaxis.set_major_locator(plt.LogLocator(base=10))
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    fig.tight_layout()
    outfile = f"benchmark_{game}.png"
    fig.savefig(outfile, dpi=150)
    print(f"Plot saved to {outfile}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pufferlib", action="store_true", help="Run PufferLib benchmark"
    )
    parser.add_argument("--pgx", action="store_true", help="Run PGX benchmark")
    parser.add_argument(
        "--plot", action="store_true", help="Plot results from JSON files"
    )
    parser.add_argument(
        "--game",
        default="2048",
        choices=list(GAMES.keys()),
        help="Game to benchmark (default: 2048)",
    )
    args = parser.parse_args()

    if not (args.pufferlib or args.pgx or args.plot):
        parser.print_help()
        print(f"\nAvailable games: {', '.join(GAMES.keys())}")
        print(
            "Run --pufferlib and --pgx in separate conda envs, then --plot to combine."
        )
        exit(1)

    if args.pufferlib:
        print(f"=== PufferLib (C) — {args.game} ===")
        bench_pufferlib(args.game)

    if args.pgx:
        print(f"=== PGX (JAX) — {args.game} ===")
        bench_pgx(args.game)

    if args.plot or args.pufferlib or args.pgx:
        plot(args.game)
