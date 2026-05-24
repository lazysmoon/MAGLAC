import argparse
import datetime
import os
import numpy as np
import wandb
import yaml
import sys
from maglac.custom_envs import make_env
from maglac.utils.utils import is_connected
from maglac.rl_agent import make_agent
from maglac.rl_agent.maglac import MAGLAC_Trainer

def load_config(args) -> tuple[dict]:
    env_params={
            'success_reward': args.success_reward,
            'reach_reward': args.reach_reward,
            'correction_cost_dist': args.correction_cost_dist,
            'w_delta1': args.w_delta1,
            'w_delta2': args.w_delta2,
            'danger_penalty_coeff': args.danger_penalty_coeff,
            'potential_obs_reward_coeff': args.potential_obs_reward_coeff,
            'tgt_reward_coeff': args.tgt_reward_coeff,
            'tgt_reward_low': args.tgt_reward_low,
            'tgt_reward_high': args.tgt_reward_high,
            'reward_scale': args.reward_scale,
            'cost': args.cost,
            'cost_coeff': args.cost_coeff,
            'cost_obs_dist': args.cost_obs_dist,
            'cost_agent_dist': args.cost_agent_dist,
            'warning_dist2obs': args.warning_dist2obs,
            'warning_dist2agent': args.warning_dist2agent,
            }
    return env_params

def train(args):
    print(f"> Running train.py {args}")
    env_params = load_config(args)
    
    # set up environment variables and seed
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if not is_connected():
        os.environ["WANDB_MODE"] = "offline"
    np.random.seed(args.seed)
    
    if args.debug:
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["JAX_DISABLE_JIT"] = "True"

    # create environments
    env = make_env(
        env_id=args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        n_rays=args.n_rays,
        area_size=args.area_size,
        max_step=args.max_step,
        r_c_params=env_params
    )
    env_test = make_env(
        env_id=args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        n_rays=args.n_rays,
        area_size=args.area_size,
        max_step=args.max_step,
        r_c_params=env_params
    )
    
    agent = make_agent(
        algo=args.rl_algo,
        env=env,
        n_agents=env.num_agents,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        seed=args.seed,
        actor_lr=args.actor_lr, 
        critic_lr=args.critic_lr, 
        alpha_lr=args.alpha_lr,
        gamma=args.gamma, 
        tau=args.tau,
        hidden_dims=args.hidden_dims,
        lyapunov_loss_coeff=args.lyapunov_loss_coeff,
        alpha3=args.alpha3,
    ) 
    
    # set up logger
    start_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
    
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(f"{args.log_dir}/{args.env}"):
        os.makedirs(f"{args.log_dir}/{args.env}")
    if not os.path.exists(f"{args.log_dir}/{args.env}/{args.rl_algo}"):
        os.makedirs(f"{args.log_dir}/{args.env}/{args.rl_algo}")
        
    log_dir = f"{args.log_dir}/{args.env}/{args.rl_algo}/num_agents{args.num_agents}_obs{args.obs}/seed{args.seed}_{start_time}"
    run_name = f"{args.rl_algo}_{args.env}_{start_time}" if args.name is None else args.name

    # get training parameters
    train_params = {
        "run_name": run_name,
        "training_steps": args.steps,
        "eval_interval": args.eval_interval,
        "eval_epi": args.eval_epi,
        "save_interval": args.save_interval,
        "batch_size": args.batch_size,
        "total_timesteps": args.total_timesteps,
        "total_episodes": args.total_episodes,
        "start_timesteps": args.start_timesteps,
        "world_model_cfg": {
            'ensemble_size': 5,
            'batch_size': args.batch_size,
            'learning_rate': 1e-4,
            'n_agents': 1,         # Number of agents in the scene
            'action_dim': 2,       # Action dimension for each agent
            'state_dim': 4,        # State dimension for each agent
            'node_dim': 3,         # Node feature dimension in the graph
            
            # Hyperparameters for GNN and MLP heads (can be kept default or customized)
            'gnn_layers': 1,
            'gnn_out_dim': 128,    # Feature dimension of GNN output
            'hidden_dim': 256,
            'head_hidden_layers': 2,
            'log_var_bound_weight': 0.01
        }
     }

    # Create and run Trainer
    trainer = MAGLAC_Trainer(
        env=env,
        env_test=env_test,
        agent=agent,
        log_dir=log_dir,
        seed=args.seed,
        params=train_params,
        save_log=not args.debug,
    )
    
    full_config = {
        'command_line_args': vars(args),
        'env_params': env_params
    }

    if not args.debug:
        os.makedirs(log_dir, exist_ok=True) 
        config_path = os.path.join(log_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(full_config, f, indent=4)
        print(f"Hyperparameters saved to {config_path}")

    wandb.config.update(args)
    trainer.train()

def is_debug_mode():
    """Check if running in debug mode"""
    return sys.gettrace() is not None

def main():
    parser = argparse.ArgumentParser()

    # custom arguments
    parser.add_argument("-n", "--num-agents", type=int, default=8)
    parser.add_argument("--obs", type=int, default=8)
    parser.add_argument("--rl_algo", type=str, default="MAGLAC")
    parser.add_argument("--env", type=str, default="Second_Order")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--debug", action="store_true", default=is_debug_mode())
   
    parser.add_argument("--n-rays", type=int, default=32)
    parser.add_argument("--area-size", type=float, default=4)
    parser.add_argument("--max_step", type=int, default=256, help="Maximum steps per episode.")

    # custom_r_c arguments
    parser.add_argument("--success_reward", type=float, default=100)
    parser.add_argument("--reach_reward", type=float, default=5.0)
    parser.add_argument("--correction_cost_dist", type=float, default=0.05)
    parser.add_argument("--w_delta1", type=float, default=5)
    parser.add_argument("--w_delta2", type=float, default=2)
    parser.add_argument("--danger_penalty_coeff", type=float, default=40)
    parser.add_argument("--potential_obs_reward_coeff", type=float, default=20)
    parser.add_argument("--tgt_reward_coeff", type=float, default=100)
    parser.add_argument("--tgt_reward_low", type=float, default=-0.1)
    parser.add_argument("--tgt_reward_high", type=float, default=4)
    parser.add_argument("--reward_scale", type=float, default=5)
    parser.add_argument("--cost", type=float, default=10)
    parser.add_argument("--cost_obs_dist", type=float, default=4)
    parser.add_argument("--cost_agent_dist", type=float, default=6)
    parser.add_argument("--cost_coeff", type=float, default=10)
    parser.add_argument("--warning_dist2obs", type=float, default=4)
    parser.add_argument("--warning_dist2agent", type=float, default=5)

    # RL arguments
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--alpha_lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--hidden_dims", type=str, default=(256, 256),
                    help="Hidden layer dimensions, format is '(dim1, dim2, ...)'")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lyapunov_loss_coeff", type=float, default=0.4)
    parser.add_argument("--alpha3", type=float, default=0.8)

    # default arguments
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--eval-interval", type=int, default=40)
    parser.add_argument("--eval-epi", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--total_timesteps", type=int, default=256*2000)
    parser.add_argument("--total_episodes", type=int, default=2300)
    parser.add_argument("--start_timesteps", type=int, default=3000)
    args = parser.parse_args()
    
    train(args)

if __name__ == "__main__":
    main()
