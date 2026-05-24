# MAGLAC: Multi-Agent Graph-Based Lyapunov Actor-Critic

This repository contains the official implementation of

**"Safe and Scalable Multi-Agent Control with Stability Guarantees via Graph-Based Lyapunov Reinforcement Learning"**

MAGLAC learns a **distributed, safe, and scalable control policy** for multi-agent navigation in dense, dynamic, obstacle-cluttered environments. Each agent has only a limited sensing radius, the number of neighbors and detected obstacles changes over time, and safety is enforced as a **hard Lyapunov constraint** rather than a soft reward penalty.

<p align="center">
  <img src="assets/overview.png" width="780" alt="MAGLAC overview"/>
</p>

<p align="center"><i>Figure 1. Overview of MAGLAC: graph-based observation encoder, distributed actor, centralized Q / Lyapunov critics with multi-head self-attention, and adaptive Lagrangian optimization.</i></p>

---

## Repository Structure

```
MAGLAC/
├── train.py                # Training entry point
├── evaluate.py             # Evaluation / rollout / video rendering
├── requirements.txt        # Python dependencies
├── maglac/
│   ├── custom_envs/        # Multi-agent navigation environments + plotting
│   │   ├── plot.py         # Trajectory / video / static rendering
│   │   └── ...
│   ├── rl_agent/
│   │   ├── maglac.py       # MAGLACAgent, rollout & evaluation routines
│   │   ├── replay_buffer.py
│   │   └── data.py
│   └── utils/              # JAX utilities, typing, helpers
├── pretrain/MAGLAC/        # Pretrained checkpoints
├── logs/                   # (auto-created) training logs, configs, models
└── README.md
```

---

## Installation

The code is built on **JAX / Flax** with  `wandb`. A CUDA-enabled NVIDIA GPU is recommended.

```bash
# 1. clone
git clone https://github.com/lazysmoon/MAGLAC.git
cd MAGLAC

# 2. (recommended) create a conda env
conda create -n maglac python=3.10 -y
conda activate maglac

# 3. install dependencies
pip install -r requirements.txt
```

> JAX needs to match your local CUDA toolkit. If the default install in `requirements.txt` does not match your GPU, please refer to https://github.com/jax-ml/jax#installation for the correct wheel.

---

## Quick Start

### 1. Train a policy

The default configuration trains with `N = 8` agents, `O = 8` obstacles in a 4 m × 4 m workspace, 256 steps per episode:

```bash
python train.py \
    --env Second_Order \
    --rl_algo MAGLAC \
    --num-agents 8 \
    --obs 8 \
    --area-size 4 \
    --seed 0
```

Logs, the merged config (`config.yaml`), and checkpoints are saved under

```
logs/<env>/<algo>/num_agents<N>_obs<O>/seed<seed>_<timestamp>/
```

Add `--debug` to disable `wandb` and JAX JIT for quick local debugging. See `train.py` for the full list of arguments.

### 2. Evaluate a trained policy

```bash
python evaluate.py \
    --model_dir ./pretrain/MAGLAC/models \
    --prefix checkpoint_ \
    --num-agents 8 \
    --obs 8 \
    --area-size 4 \
    --epi 100
```

The script loads the checkpoint at the requested step. Omit `--checkpoint_step` to automatically scan all available steps and pick the best, or pass a specific integer step. It then runs `--epi` rollouts and writes:

- `output_seed<seed>.txt` — mean return, success rate, safe rate, failed episode indices,
- one MP4 video and one trajectory PNG per selected episode.

A pretrained checkpoint is provided under `pretrain/MAGLAC/` for direct evaluation.

### 3. Render a single graph / trajectory

`maglac/custom_envs/plot.py` exposes three rendering helpers used by `evaluate.py`:

- `render_single_graph(graph, save_path, side_length, n_agent, n_rays, r)` — static figure of one frame (agents as circles, goals as squares, obstacles, observation edges).
- `render_trajectory(rollout, save_path, side_length, dim, n_agent, r)` — continuous time-graded trajectory plot with a colorbar.
- `render_video(rollout, video_path, side_length, dim, n_agent, n_rays, r)` — full MP4 animation including the Lyapunov-informative observation graph.

---

## Method Overview

The multi-agent safe navigation problem is formulated as a **Constrained Markov Decision Process** (CMDP), in which each agent sees only a local observation whose dimension varies in time (the number of neighbors and detected obstacle points changes as agents move).

**Graph-based observation.** Each agent's local observation is turned into a directed graph with four node types — ego, neighboring agents, obstacle-ray endpoints, and goal — each tagged with a one-hot semantic feature. Edge features encode relative position and velocity to the ego-agent. A graph attention encoder aggregates this variable-dimensional graph into a fixed-length embedding, which a distributed actor maps to bounded control commands.

**Lyapunov critic and UUB stability.** A centralized Lyapunov critic is trained to approximate the cumulative discounted safety cost. A data-driven drift condition is derived that guarantees the closed-loop multi-agent system is **uniformly ultimately bounded** with respect to the global safety cost. The global condition is decoupled into per-agent local constraints, so the stability guarantee is enforced through purely local supervision.

**Adaptive Lagrangian optimization.** The Lyapunov stability condition is combined with a maximum-entropy objective via Lagrangian duality. Both the Lagrangian multiplier and the entropy temperature are auto-tuned by dual ascent, removing the need for manual safety/task weight tuning.

---

## Experimental Results

### Training curves

<p align="center">
  <img src="assets/training_curves.png" width="820" alt="Training curves"/>
</p>
<p align="center"><i>Figure 3. Training curves of MAGLAC under three configurations (3 / 8 / 10 agents).</i></p>

MAGLAC stabilizes near a mean reward of **140** across all three settings, and reaches this level within the first 20% of training.

### Zero-shot scaling and generalization

<p align="center">
  <img src="assets/scalability.png" width="820" alt="Scalability"/>
</p>
<p align="center"><i>Figure 4. Zero-shot scaling. Top: collision rate (left) and success rate (right) vs. number of agents (N_obs = 0). Bottom: same metrics vs. number of obstacles (N = 48).</i></p>

A policy trained at `N = 8, N_obs = 8` is evaluated zero-shot on different workspaces ($L = 8$ m).

- **Agent-crowding family** (`N_obs = 0`, `N ∈ {32, 48, 64, 80, 96}`): MAGLAC keeps collisions below 6% across all populations and sustains a success rate above 87% even at `N = 96`.
- **Obstacle-density family** (`N = 48`, `N_obs ∈ {8, 12, 16, 20, 24}`): MAGLAC maintains collision rate below 4% and success rate above 76% across all obstacle densities.

These results indicate that the dynamic graph representation enables the policy to generalize across diverse interaction topologies, while the Lyapunov-based stability condition maintains safety under conditions outside the training distribution.

### Software-in-the-loop

The learned policy is further deployed on a **ROS 2 Humble + Crazyswarm2** SITL platform running on Ubuntu 22.04 LTS, with the official Crazyflie firmware handling low-level control. Eight quadrotors fly at a fixed altitude of 1.0 m through workspaces containing ten obstacles, drawn from random seeds unseen during training. The actor outputs are sent to the firmware through the `cmd_full_state` interface of Crazyswarm2.

<p align="center">
  <img src="assets/sitl_trajectories.png" width="720" alt="SITL trajectories"/>
</p>
<p align="center"><i>Figure 5. Trajectories of eight Crazyflie quadrotors across six SITL scenarios.</i></p>

In every scenario, all eight quadrotors reach their assigned goals without collision, and the trajectories remain smooth even in regions where multiple agents and obstacles fall within a single sensing radius — confirming that the policy trained in numerical simulation transfers directly to the firmware-driven SITL environment.

---

## Contact

Issues and pull requests are very welcome.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
