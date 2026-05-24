import os
import numpy as np
import jax
import argparse
import matplotlib
import matplotlib.pyplot as plt
from flax.training import checkpoints
import jax.tree_util as jtu
from maglac.utils.utils import jax_jit_np, jax_vmap
from maglac.custom_envs.base import RolloutResult
from maglac.rl_agent.maglac import MAGLACAgent, rollout_single_episode, run_parallel_evaluation
from maglac.custom_envs import make_env
from tqdm import tqdm
import sys, yaml

plt.rcParams['font.cursive'] = ['Comic Sans MS']
# Prevent matplotlib from failing to find the 'cursive' font family
matplotlib.rcParams['font.cursive'] = ['DejaVu Sans']
from flax.training.checkpoints import available_steps


def is_debug_mode():
    """Check whether the script is running in debug mode."""
    return sys.gettrace() is not None


def load_config(config_path: str) -> tuple[dict, dict, dict]:
    parent_dir = os.path.dirname(os.path.normpath(config_path))
    config_path = os.path.join(parent_dir, f"config.yaml")
    try:
        with open(config_path, 'r') as f:
            full_config = yaml.full_load(f)
        if not isinstance(full_config, dict):
            raise TypeError(f"The top level of config file {config_path} is not a dict.")

        # Use .get() to safely extract sub-dicts, returning empty dict if the key is missing
        env_params = full_config.get('env_params', {})

        print(f"Config successfully loaded from {config_path}.")
        return env_params

    except FileNotFoundError:
        print(f"Error: config file not found at '{config_path}'")
        return {}
    except yaml.YAMLError as e:
        print(f"Error: failed to parse YAML file: {e}")
        return {}


def get_checkpoint_path_by_step(ckpt_dir, prefix, step):
    """
    Find a checkpoint path by a specific step number.

    Args:
        ckpt_dir: checkpoint directory.
        prefix: checkpoint prefix (e.g. "my_model").
        step: target step number (e.g. 3000).

    Returns:
        The checkpoint path for the given step, or None if it does not exist.
    """
    ckpt_path = os.path.join(ckpt_dir, f"{prefix}{step}")
    if os.path.exists(ckpt_path):
        return ckpt_path
    else:
        return None


def test_model(args):
    print(f"> Running evaluation with args: {args}")
    if args.debug:
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["JAX_DISABLE_JIT"] = "True"
    env_params = load_config(f'{args.model_dir}')

    # --- 1. Set up environment and agent ---
    env = make_env(
        env_id=args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        n_rays=args.n_rays,
        area_size=args.area_size,
        max_step=args.max_step,
        r_c_params=env_params
    )

    # Initialize the agent (its structure must match the one used during training)
    agent = MAGLACAgent(
        env=env,
        n_agents=env.num_agents,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        seed=args.seed,
        # Other SAC hyperparameters can be left at any value, they are not used at eval time
    )
    if args.nojit_rollout:
        print("Only jit step, no jit rollout!")
        is_unsafe_fn = None
        is_finish_fn = None
    else:
        print("jit rollout!")
        is_unsafe_fn = jax_jit_np(jax_vmap(env.collision_mask))
        is_finish_fn = jax_jit_np(jax_vmap(env.finish_mask))

    # --- 2. Load the trained model ---
    prefix = args.prefix

    best_success_rate = -0.1
    best_safe_rate = -0.1
    selected_step = args.checkpoint_step
    if selected_step is None:
        steps = available_steps(args.model_dir, prefix=prefix)
        if not steps:
            raise FileNotFoundError(f"No checkpoint found in directory: {args.model_dir}")
        best_success_step = steps[0]
        best_safe_step = steps[0]
        for step in steps:
            load_path = get_checkpoint_path_by_step(args.model_dir, prefix, step)
            if not load_path:
                raise FileNotFoundError(f"No checkpoint found in directory: {args.model_dir}")
            load_path = os.path.abspath(load_path)
            print(f"Loading agent states from: {load_path}")
            agent.load_agent_states(load_path)
            all_successful_flag, all_safe_flag, _ = run_parallel_evaluation(
                agent=agent,
                eval_env=env,
                max_steps=env.max_step,
                eval_episodes=args.epi,
                actor_params=agent.actor_state.params,  # use the latest parameters
                seed=args.seed,
            )
            # Compute averages
            success_rate = all_successful_flag.mean().item()
            safe_rate = all_safe_flag.mean().item()
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_success_step = step
            if safe_rate > best_safe_rate:
                best_safe_rate = safe_rate
                best_safe_step = step
        selected_step = best_success_step
        text = (
            f"best_success_rate:{best_success_rate*100}% with step {best_success_step}\n,"
            f"best_safe_rate:{best_safe_rate*100}% with step {best_safe_step}"
        )
        print(text)
        txt_path = os.path.join(
            args.model_dir,
            f"agents{args.num_agents}obs{args.obs}_seed{args.seed}_all_{prefix}_steps.txt",
        )
        with open(txt_path, "w", encoding="utf-8") as file:
            file.writelines([text])
        print(f"Results successfully written to {prefix}of_allstep.txt")

    load_path = get_checkpoint_path_by_step(args.model_dir, prefix, selected_step)
    if not load_path:
        # Fall back to the latest checkpoint if the specific one is not found
        load_path = checkpoints.latest_checkpoint(ckpt_dir=args.model_dir, prefix=prefix)
    if not load_path:
        raise FileNotFoundError(f"No checkpoint found in directory: {args.model_dir}")
    load_path = os.path.abspath(load_path)
    print(f"Loading agent states from: {load_path}")
    agent.load_agent_states(load_path)
    episodes = []
    episodes_returns = []
    episode_dist2tgt = []

    # --- 4. Run the evaluation loop ---
    # Mirrors the structure of SACTrainer for data collection
    pbar = tqdm(total=args.epi, desc="Evaluating Episodes")
    success_time = 0
    safe_time = 0
    false_idxs = []
    success_idxs = []

    for i in range(args.epi):
        keys = jax.random.PRNGKey(args.seed + i)

        episodes_return, _, all_transitions, infos, _ = rollout_single_episode(
            agent, env, env.max_step, agent.actor_state.params, keys
        )
        episodes_returns.append(episodes_return)
        (graph, action, reward, cost, done, next_graph) = all_transitions
        done_indices = np.where(done)[0]
        done_index = done_indices[0]
        infos_np = jtu.tree_map(np.asarray, infos)
        dist2tgt = infos_np['dist2tgt'][done_index]
        episode_dist2tgt.append(dist2tgt)

        # T_... data has length T
        episode_transitions = jtu.tree_map(
            lambda x: x[0:done_index + 1],
            (action, reward, cost, done)
        )

        # Tp1_graph data has length T+1
        episode_graph = jtu.tree_map(
            lambda x: x[0:done_index + 1],
            graph
        )
        episodes.append(RolloutResult(
            Tp1_graph=episode_graph,
            T_action=episode_transitions[0],
            T_reward=episode_transitions[1].mean(axis=-1),
            T_cost=episode_transitions[2].mean(axis=-1),
            T_done=episode_transitions[3],
            T_info=None
        ))
        episode_verbose = (
            f"Episode {i+1}: episodes_return={episodes_return.mean():.2f}, "
            f"Episode_Length={done_index}, dist2tgt = {dist2tgt}"
        )
        dist2tgt = np.array(dist2tgt)
        if not np.any(dist2tgt > 0.2):
            success_time += 1
            success_idxs.append(i)
        else:
            false_idxs.append(i)
        if done_index + 1 >= args.max_step:
            safe_time += 1
        tqdm.write(episode_verbose)
        pbar.update(1)

    # --- 5. Print summary ---
    mean_return = np.mean(episodes_returns)
    std_return = np.std(episodes_returns)
    mean_return_text = f"Mean_return: {mean_return:.2f}\n"
    print("\n----------------------------------------------------")
    print(f"Evaluation over {args.epi} episodes:")
    print(f"Mean Return: {mean_return:.2f} +/- {std_return:.2f}")
    success_rate = success_time / args.epi * 100
    safe_rate = safe_time / args.epi * 100
    success_text = f"Success times: {success_time}. Success Rate : {success_rate:.2f} %\n"
    safe_text = f"Safe times: {safe_time}. Safe Rate : {safe_rate:.2f} %\n"
    print(success_text, safe_text)
    print("----------------------------------------------------")
    save_dir = os.path.join(
        args.model_dir,
        f"eval_agent{args.num_agents}_obs{args.obs}_seed{args.seed}_{prefix}{selected_step}",
    )
    os.makedirs(save_dir, exist_ok=True)
    # Open the file and write contents (create it if it does not exist)
    txt_path = os.path.join(save_dir, f"output_seed{args.seed}.txt")
    with open(txt_path, "w", encoding="utf-8") as file:
        file.writelines([mean_return_text, success_text, safe_text, str(false_idxs)])
    print("Results successfully written to output.txt")

    idx_for_vd = np.random.randint(0, len(episodes), size=10)
    idx_set = set(idx_for_vd)
    idx_list = list(idx_set)
    
    for i, episode_rollout in enumerate(episodes):
        if i in idx_list:
            video_path = os.path.join(
                save_dir, f"seed{args.seed}_ep{i}_return{episodes_returns[i].mean():.2f}.mp4"
            )
            png_path = os.path.join(
                save_dir, f"seed{args.seed}_ep{i}_return{episodes_returns[i].mean():.2f}.png"
            )
            print(f"Rendering video for episode {i} to {video_path} ...")
            # Compute is_unsafe for this episode
            Ta_is_unsafe = is_unsafe_fn(episode_rollout.Tp1_graph)
            env.render_trajectory(rollout=episode_rollout, save_path=png_path, dpi=300)
            # Call the rendering function
            print(f"Rendering video for episode {i} to {video_path} ...")
            env.render_video(
                rollout=episode_rollout,
                video_path=video_path,
                Ta_is_unsafe=Ta_is_unsafe,
                dpi=300,
            )


def main():
    parser = argparse.ArgumentParser()

    # --- Core arguments ---
    parser.add_argument("--model_dir", type=str, default='./pretrain/MAGLAC/models', help="Directory where the trained models are saved.")
    parser.add_argument("--env", type=str, default='Second_Order', help="Name of the environment.")
    parser.add_argument("--prefix", type=str, default="checkpoint_", help="Name of the model.")
    parser.add_argument("--checkpoint_step", type=int, default=None, help="Checkpoint step to load. Omit it to scan all available steps and pick the best.")

    # --- Evaluation arguments ---
    parser.add_argument("--seed", type=int, default=299, help="Random seed for evaluation.")
    parser.add_argument("--epi", type=int, default=100, help="Number of episodes to run for evaluation.")
    parser.add_argument("--max_step", type=int, default=320, help="Maximum steps per episode.")

    # --- Environment-specific arguments (must match training, or be set as needed) ---
    parser.add_argument("--num-agents", type=int, default=8, help="Number of agents.")
    parser.add_argument("--obs", type=int, default=8)
    parser.add_argument("--area-size", type=float, default=4, help="Size of the environment area.")
    parser.add_argument("--n-rays", type=int, default=32)

    # --- Optional features ---
    parser.add_argument("--nojit-rollout", action="store_true", default=False)
    parser.add_argument("--no-video", action="store_true", help="Do not generate and save videos.")
    parser.add_argument("--dpi", type=int, default=100, help="DPI for saved videos.")
    parser.add_argument("--debug", action="store_true", default=is_debug_mode())
    args = parser.parse_args()
    test_model(args)


if __name__ == "__main__":
    main()
