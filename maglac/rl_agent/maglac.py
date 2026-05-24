from .replay_buffer import PyTreeReplayBuffer
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from flax.training.train_state import TrainState
from flax.training import checkpoints
import optax
import functools as ft
from maglac.utils.typing import Array
from maglac.custom_envs.base import MultiAgentEnv
from .networks import ActorWithGNN, DoubleCriticWithGNN, LyapunovCritic
from .data import Rollout
from maglac.utils.graph import GraphsTuple
import os
import time
import jax.random as jr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

plt.rcParams['font.cursive'] = ['Comic Sans MS']
matplotlib.rcParams['font.cursive'] = ['DejaVu Sans']


def apply_tanh_correction(dist, action):
    log_prob = dist.log_prob(action).sum(axis=-1)
    log_prob -= (2 * (jnp.log(2) - action - jax.nn.softplus(-2 * action))).sum(axis=-1)
    return log_prob


class MAGLACAgent:
    def __init__(self,
                 env: MultiAgentEnv,
                 n_agents: int,
                 node_dim: int,
                 edge_dim: int,
                 state_dim: int,
                 action_dim: int,
                 seed: int,
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
                 gamma=0.99, tau=0.005, hidden_dims=(256, 256),
                 use_lyapunov: bool = True,
                 lyapunov_loss_coeff: float = 0.2,
                 alpha3: float = 0.8, ):

        self.gamma = gamma
        self.tau = tau
        self.n_agents = n_agents
        self.use_lyapunov = use_lyapunov

        self.key = jax.random.PRNGKey(seed)
        actor_key, critic_key, lyapunov_key, alpha_key = jax.random.split(self.key, 4)

        # Create dummy graph for network initialization
        nominal_graph = GraphsTuple(
            nodes=jnp.zeros((n_agents, node_dim)),
            edges=jnp.zeros((n_agents, edge_dim)),
            states=jnp.zeros((n_agents, state_dim)),
            n_node=jnp.array(n_agents),
            n_edge=jnp.array(n_agents),
            senders=jnp.arange(n_agents),
            receivers=jnp.arange(n_agents),
            node_type=jnp.zeros((n_agents,)),
            env_states=jnp.zeros((n_agents,)),
        )
        self.nominal_graph = nominal_graph

        # Initialize Actor network
        actor_model = ActorWithGNN(action_dim=action_dim,
                                   n_agents=self.n_agents,
                                   hidden_dims=hidden_dims)

        actor_params = actor_model.init(actor_key, nominal_graph)['params']
        self.actor_state = TrainState.create(
            apply_fn=actor_model.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr)
        )

        # Initialize Double Critic network
        dummy_actions = jnp.zeros((self.n_agents, action_dim))

        critic_model = DoubleCriticWithGNN(n_agents=self.n_agents,
                                           hidden_dims=hidden_dims)

        critic_params = critic_model.init(critic_key, nominal_graph, dummy_actions)['params']
        self.critic_state = TrainState.create(
            apply_fn=critic_model.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=critic_lr)
        )
        self.target_critic_params = critic_params

        # Initialize Lyapunov Critic
        if self.use_lyapunov:
            lyapunov_model = LyapunovCritic(n_agents=self.n_agents, hidden_dims=hidden_dims)
            lyapunov_params = lyapunov_model.init(lyapunov_key, nominal_graph, dummy_actions)['params']
            self.lyapunov_state = TrainState.create(
                apply_fn=lyapunov_model.apply,
                params=lyapunov_params,
                tx=optax.adam(learning_rate=critic_lr)
            )
            self.target_lyapunov_params = lyapunov_params

            # Initialize Lagrange multiplier Lambda
            self.log_lambda = jnp.array(0.0)
            self.lambda_state = TrainState.create(
                apply_fn=None,
                params={'log_lambda': self.log_lambda},
                tx=optax.adam(learning_rate=actor_lr)
            )

        self.lyapunov_loss_coeff = lyapunov_loss_coeff
        self.alpha3 = alpha3

        # Initialize entropy temperature Alpha
        self.target_entropy = -action_dim
        self.log_alpha = jnp.array(0.0)
        self.alpha_state = TrainState.create(
            apply_fn=None,
            params={'log_alpha': self.log_alpha},
            tx=optax.adam(learning_rate=alpha_lr)
        )

        # JIT compile the update function
        self._update_step = jax.jit(self._update)

    @ft.partial(jax.jit, static_argnames=('self', 'deterministic'))
    def select_action(self, key, params, obs: GraphsTuple, deterministic: bool = False):
        dist = self.actor_state.apply_fn({'params': params}, obs)

        if deterministic:
            raw_action = dist.mean()
        else:
            raw_action = dist.sample(seed=key)

        action = jnp.tanh(raw_action)
        return action

    def update(self, main_batch: Rollout, edge_batch: Rollout):
        use_edge_batch_flag = 1.0 if edge_batch is not None else 0.0
        if edge_batch is None:
            edge_batch = jtu.tree_map(jnp.zeros_like, main_batch)
        key, self.key = jax.random.split(self.key, 2)
        self.actor_state, self.critic_state, self.alpha_state, self.target_critic_params, \
            self.lyapunov_state, self.target_lyapunov_params, self.lambda_state, metrics = self._update_step(
            key,
            self.actor_state, self.critic_state, self.alpha_state, self.target_critic_params,
            self.lyapunov_state, self.target_lyapunov_params, self.lambda_state, main_batch, edge_batch,
            jnp.array(use_edge_batch_flag))
        return metrics

    def _update(self, key, actor_state, critic_state, alpha_state, target_critic_params,
                lyapunov_state, target_lyapunov_params, lambda_state, main_batch: Rollout, edge_batch: Rollout,
                use_edge_batch_flag):
        actor_key, critic_key, lyapunov_key, alpha_key = jax.random.split(key, 4)

        # Unpack main batch data
        obs, actions, rewards, costs, dones, next_obs = main_batch

        # Unpack edge batch data
        edge_obs, edge_actions, edge_rewards, edge_costs, edge_dones, edge_next_obs = edge_batch

        def single_actor_forward(params, single_obs):
            return actor_state.apply_fn({'params': params}, single_obs)

        def single_critic_forward(params, single_obs, single_action):
            q1, q2 = critic_state.apply_fn({'params': params}, single_obs, single_action)
            return jnp.squeeze(q1), jnp.squeeze(q2)

        next_dist_fn = jax.vmap(single_actor_forward, in_axes=(None, 0))
        next_q_fn = jax.vmap(single_critic_forward, in_axes=(None, 0, 0))

        # Critic Update
        def critic_loss_fn(critic_params):
            next_dist = next_dist_fn(actor_state.params, next_obs)
            next_raw_actions = next_dist.sample(seed=critic_key)

            next_log_probs = apply_tanh_correction(next_dist, next_raw_actions)
            next_actions = jnp.tanh(next_raw_actions)

            next_q1, next_q2 = next_q_fn(target_critic_params, next_obs, next_actions)
            next_q = jnp.minimum(next_q1, next_q2)
            alpha = jnp.exp(alpha_state.params['log_alpha'])
            next_log_probs = jnp.squeeze(next_log_probs)

            target_q = rewards + self.gamma * (1 - dones[:, None]) * (next_q - alpha * next_log_probs)

            current_q1, current_q2 = next_q_fn(critic_params, obs, actions)

            loss = ((current_q1 - target_q) ** 2 + (current_q2 - target_q) ** 2).mean()
            return loss, {'critic_loss': loss, 'q1': current_q1.mean(), 'q2': current_q2.mean()}

        (critic_loss_val, critic_metrics), critic_grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(
            critic_state.params)
        new_critic_state = critic_state.apply_gradients(grads=critic_grads)

        def single_lyapunov_forward(params, single_obs, single_action):
            lyapunov_value = lyapunov_state.apply_fn({'params': params}, single_obs, single_action)
            return jnp.squeeze(lyapunov_value)

        next_lyapunov_fn = jax.vmap(single_lyapunov_forward, in_axes=(None, 0, 0))

        # Lyapunov Critic Update
        def lyapunov_loss_fn(lyapunov_params):
            next_dist = next_dist_fn(actor_state.params, next_obs)
            next_raw_actions = next_dist.sample(seed=lyapunov_key)
            next_actions = jnp.tanh(next_raw_actions)

            l_next = next_lyapunov_fn(target_lyapunov_params, next_obs, next_actions)
            l_target = costs + self.gamma * (1 - dones[:, None]) * l_next

            l_current = next_lyapunov_fn(lyapunov_params, obs, actions)

            loss = ((l_current - l_target) ** 2).mean() * self.use_lyapunov
            return loss, {'lyapunov_loss': loss}

        (l_loss_val, l_metrics), l_grads = jax.value_and_grad(lyapunov_loss_fn, has_aux=True)(lyapunov_state.params)
        new_lyapunov_state = lyapunov_state.apply_gradients(grads=l_grads)

        # Actor and Alpha Update
        def actor_alpha_loss_fn(actor_params, alpha_params, lambda_params):
            dist_new = next_dist_fn(actor_params, obs)
            raw_actions_new = dist_new.sample(seed=actor_key)

            log_probs_new = apply_tanh_correction(dist_new, raw_actions_new)
            actions_new = jnp.tanh(raw_actions_new)

            q1, q2 = next_q_fn(new_critic_state.params, obs, actions_new)
            q = jnp.minimum(q1, q2)

            alpha_detached = jnp.exp(jax.lax.stop_gradient(alpha_params['log_alpha']))
            actor_loss_sac = (alpha_detached * jnp.squeeze(log_probs_new) - q).mean()

            alpha = jnp.exp(alpha_params['log_alpha'])
            log_probs_detached = jax.lax.stop_gradient(log_probs_new)
            alpha_loss = alpha * (-log_probs_detached.mean() - self.target_entropy)

            # Lyapunov loss components
            edge_dist_next = next_dist_fn(actor_params, edge_next_obs)
            edge_raw_actions_next = edge_dist_next.sample(seed=actor_key)
            edge_actions_next = jnp.tanh(edge_raw_actions_next)

            l_current_for_actor = jax.lax.stop_gradient(
                next_lyapunov_fn(new_lyapunov_state.params, edge_obs, edge_actions))
            l_next_for_actor = next_lyapunov_fn(new_lyapunov_state.params, edge_next_obs, edge_actions_next)

            l_delta = (l_next_for_actor * (edge_next_obs.env_states.edge_mask) -
                      (l_current_for_actor - self.alpha3 * edge_costs) * (
                                  edge_obs.env_states.edge_mask)).mean() * use_edge_batch_flag

            lambda_val = jnp.clip(jnp.exp(lambda_params['log_lambda']), 0, 50)
            actor_loss_lyapunov = (jax.lax.stop_gradient(
                lambda_val) * l_delta) * self.use_lyapunov * use_edge_batch_flag

            lambda_loss = - lambda_params['log_lambda'] * jax.lax.stop_gradient(
                l_delta) * self.use_lyapunov * use_edge_batch_flag

            actor_loss = actor_loss_sac + self.lyapunov_loss_coeff * actor_loss_lyapunov
            total_loss = actor_loss + alpha_loss + lambda_loss
            return total_loss, (actor_loss, alpha_loss, lambda_loss, {
                'actor_loss': actor_loss,
                'alpha_loss': alpha_loss,
                'alpha': alpha,
                'entropy': -log_probs_detached.mean(),
                'lambda_loss': lambda_loss,
                'lambda': lambda_val,
                'l_delta': l_delta
            })

        grad_fn = jax.value_and_grad(actor_alpha_loss_fn, argnums=(0, 1, 2), has_aux=True)
        ((_, (actor_loss_val, _, _, actor_alpha_metrics)),
         (actor_grads, alpha_grads, lambda_grads)) = grad_fn(actor_state.params, alpha_state.params, lambda_state.params)

        new_actor_state = actor_state.apply_gradients(grads=actor_grads)
        new_alpha_state = alpha_state.apply_gradients(grads=alpha_grads)
        new_lambda_state = lambda_state.apply_gradients(grads=lambda_grads)

        # Soft update target critic
        new_target_critic_params = jtu.tree_map(
            lambda target, online: target * (1 - self.tau) + online * self.tau,
            target_critic_params, new_critic_state.params
        )

        new_target_lyapunov_params = jtu.tree_map(
            lambda target, online: target * (1 - self.tau) + online * self.tau,
            target_lyapunov_params, new_lyapunov_state.params
        )

        metrics = {**critic_metrics, **l_metrics, **actor_alpha_metrics}

        return new_actor_state, new_critic_state, new_alpha_state, new_target_critic_params, \
            new_lyapunov_state, new_target_lyapunov_params, new_lambda_state, metrics

    def save_agent_states(self, save_path, step, prefix="best_"):
        save_data = {
            'actor': self.actor_state,
            'critic': self.critic_state,
            'alpha': self.alpha_state
        }
        checkpoints.save_checkpoint(ckpt_dir=save_path, target=save_data, step=step, prefix=prefix, keep=30,
                                    overwrite=False)
        print(f"Agent states saved to directory: {save_path}")

    def load_agent_states(self, load_path):
        template_states = {
            'actor': self.actor_state,
            'critic': self.critic_state,
            'alpha': self.alpha_state
        }
        loaded_states = checkpoints.restore_checkpoint(ckpt_dir=load_path, target=template_states)
        print(f"Loading model from: {load_path}")
        self.actor_state = loaded_states['actor']
        self.critic_state = loaded_states['critic']
        self.alpha_state = loaded_states['alpha']
        self.target_critic_params = self.critic_state.params
        print(f"Agent states loaded from directory: {load_path}")


@ft.partial(jax.jit, static_argnames=('agent', 'env', 'max_steps'))
def rollout_single_episode(
        agent,
        env,
        max_steps: int,
        actor_params,
        key):
    # Reset environment
    reset_key, rollout_key = jr.split(key)
    initial_graph = env.reset(reset_key)

    initial_N = -1

    # Step function for scan loop
    def step_fn(carry, _):
        prev_graph, cumulative_reward, cumulative_cost, key, done_flag, prev_N = carry

        def do_step():
            a_next_key, next_key = jax.random.split(key)
            action = agent.select_action(a_next_key, actor_params, prev_graph, deterministic=False)
            next_graph, reward, cost, done, info = env.step(prev_graph, action)
            edge_mask = jnp.any(next_graph.env_states.edge_mask)
            current_N = jnp.where(
                edge_mask,
                next_graph.env_states.timestep,
                prev_N
            )
            transition = (prev_graph, action, reward, cost, done, next_graph)
            new_cumulative_reward = cumulative_reward + reward
            new_cumulative_cost = cumulative_cost + cost
            return next_graph, action, new_cumulative_reward, new_cumulative_cost, next_key, done, transition, current_N, info

        def skip_step():
            action = jnp.zeros((env.num_agents, env.action_dim))
            next_graph, reward, cost, done, info = env.step(prev_graph, action)
            transition = (prev_graph, action, jnp.zeros((env.num_agents,)), jnp.zeros((env.num_agents,)), done_flag,
                          prev_graph)
            return prev_graph, action, cumulative_reward, cumulative_cost, key, done_flag, transition, prev_N, info

        next_graph, action, new_cumulative_reward, new_cumulative_cost, new_key, current_done_signal, current_transition, current_N, info = jax.lax.cond(
            done_flag,
            skip_step,
            do_step
        )

        new_done_flag = current_done_signal

        return (next_graph, new_cumulative_reward, new_cumulative_cost, new_key, new_done_flag, current_N), (
        current_transition, info)

    initial_carry = (initial_graph, jnp.zeros((env.num_agents,)), jnp.zeros((env.num_agents,)), rollout_key,
                     jnp.array(False), initial_N)

    final_carry, (all_transition, infos) = jax.lax.scan(
        step_fn,
        initial_carry,
        None,
        length=max_steps
    )

    final_graph, final_reward, final_cost, key, _, edge_N = final_carry
    return final_reward, final_cost, all_transition, infos, edge_N


@ft.partial(jax.jit, static_argnames=('agent', 'eval_env', 'max_steps', 'seed', 'eval_episodes'))
def run_parallel_evaluation(
        agent,
        eval_env,
        max_steps: int,
        eval_episodes: int,
        actor_params,
        seed: int,
):
    def rollout_single_episode(key):
        reset_key, rollout_key = jr.split(key)
        initial_graph = eval_env.reset(reset_key)

        def step_fn(carry, _):
            prev_graph, cumulative_reward, key, done_flag = carry

            def do_step():
                a_key, next_key = jax.random.split(key)
                action = agent.select_action(a_key, actor_params, prev_graph, deterministic=False)
                next_graph, reward, cost, done, info = eval_env.step(prev_graph, action)
                transition = (prev_graph, action, reward, cost, done, next_graph)
                new_cumulative_reward = cumulative_reward + reward
                return next_graph, new_cumulative_reward, next_key, done, transition

            def skip_step():
                action = agent.select_action(key, actor_params, prev_graph, deterministic=False)
                transition = (prev_graph, action, jnp.zeros((eval_env.num_agents,)), jnp.zeros((eval_env.num_agents,)),
                              done_flag, prev_graph)
                return prev_graph, cumulative_reward, key, done_flag, transition

            next_graph, new_cumulative_reward, new_key, current_done_signal, current_transition = jax.lax.cond(
                done_flag,
                skip_step,
                do_step
            )

            new_done_flag = jnp.logical_or(done_flag, current_done_signal)

            return (next_graph, new_cumulative_reward, new_key, new_done_flag), current_transition

        initial_carry = (initial_graph, jnp.zeros((eval_env.num_agents,)), rollout_key, jnp.array(False))

        final_carry, all_transition = jax.lax.scan(
            step_fn,
            initial_carry,
            None,
            length=max_steps
        )

        final_graph, final_reward, _, _ = final_carry
        dist2tgt = final_graph.env_states.dist2tgt
        successful_flag = jnp.all(dist2tgt <= 0.2)
        safe_flag = jnp.where(final_graph.env_states.timestep >= 254, 1, 0)
        return successful_flag, safe_flag, final_reward

    keys = jnp.array([jr.PRNGKey(eval_seed) for eval_seed in range(seed, seed + eval_episodes)])
    all_successful_flag, all_safe_flag, all_rewards = jax.vmap(rollout_single_episode)(keys)

    return all_successful_flag, all_safe_flag, all_rewards


class MAGLAC_Trainer:

    def __init__(
            self,
            env: MultiAgentEnv,
            env_test: MultiAgentEnv,
            agent: MAGLACAgent,
            log_dir: str,
            seed: int,
            params: dict,
            save_log: bool = True
    ):
        self.env = env
        self.env_test = env_test
        self.agent = agent
        graph = env.reset(jax.random.PRNGKey(0))
        dummy_actions = jnp.ones((env.num_agents, env.action_dim))
        next_graph, reward, cost, done, info = env.step(graph, dummy_actions)
        dummy_transition = (graph, dummy_actions, reward, cost, done, next_graph)
        self.PyTreereplay_buffer = PyTreeReplayBuffer(capacity=int(1e6), dummy_input=dummy_transition)
        self.log_dir = os.path.abspath(log_dir)
        self.seed = seed
        self.action_low, self.action_high = env.action_lim()

        if MAGLAC_Trainer._check_params(params):
            self.params = params

        self.total_steps = params['total_timesteps']
        self.total_episodes = params['total_episodes']
        self.start_steps = params['start_timesteps']
        self.batch_size = params['batch_size']
        self.eval_interval = params['eval_interval']
        self.eval_epi = params['eval_epi']
        self.save_interval = params['save_interval']
        self.save_log = save_log
        self.max_episode_steps = env.max_step
        self.horizon = 32

        if self.save_log:
            self.model_dir = os.path.join(self.log_dir, 'models')
            os.makedirs(self.model_dir, exist_ok=True)

        wandb.login()
        wandb.init(
            name=params['run_name'],
            project='MAGA2C_SG_DoubleIntergtor',
            dir=self.log_dir,
            config=params
        )
        self.key = jax.random.PRNGKey(seed)
        self.env_model_error = 0
        self.model_steps = 100
        self.update_steps = 0
        self.best_eval_reward = -np.inf

    @staticmethod
    def _check_params(params: dict) -> bool:
        assert 'run_name' in params
        assert 'total_timesteps' in params
        assert 'start_timesteps' in params
        assert 'batch_size' in params
        assert 'eval_interval' in params and params['eval_interval'] > 0
        assert 'eval_epi' in params and params['eval_epi'] >= 1
        assert 'save_interval' in params and params['save_interval'] > 0
        return True

    @ft.partial(jax.jit, static_argnums=(0,))
    def safe_mask(self, graph: GraphsTuple) -> jnp.ndarray:
        def safe_rollout(single_rollout_mask: Array) -> Array:
            safe_rollout_mask = jnp.ones_like(single_rollout_mask).astype(jnp.bool_)
            for i in range(single_rollout_mask.shape[0]):
                start = 0 if i < self.horizon else i - self.horizon
                safe_mask = ((1 - single_rollout_mask[i]) * safe_rollout_mask[start: i + 1]).astype(jnp.bool_)
                safe_rollout_mask = safe_rollout_mask.at[start: i + 1].set(safe_mask)
                safe_rollout_mask = safe_rollout_mask.at[0].set(jnp.array(1).astype(jnp.bool_))
            return safe_rollout_mask

        safe = safe_rollout(graph.env_states.unsafe_mask)
        state_with_safe = graph.env_states._replace(safe_mask=safe)
        graph = graph._replace(env_states=state_with_safe)
        return graph

    def train(self):
        start_time = time.time()
        key_x0, self.key = jax.random.split(self.key)
        current_step = 1
        collect_time = 0
        pbar = tqdm(total=int(self.total_episodes), ncols=80)

        while collect_time <= self.total_episodes:
            key_x0, self.key = jax.random.split(self.key)

            episodes_return, episodes_cost, all_transitions, summaries, edge_N = rollout_single_episode(
                self.agent, self.env,
                self.env.max_step, self.agent.actor_state.params, key_x0)

            (graph, action, reward, cost, done, next_graph) = all_transitions
            done_indices = np.where(done)[0]
            done_index = done_indices[0]
            episode_reward = episodes_return
            episode_length = done_index + 1

            infos_np = jtu.tree_map(np.asarray, summaries)
            dist2tgt = infos_np['dist2tgt'][done_index]

            episode_transitions = jtu.tree_map(
                lambda x: x[0:done_index + 1],
                all_transitions
            )

            episode_verbose = (
                f"Episode_Length={episode_length}, "
                f"dist2tgt={dist2tgt}")
            tqdm.write(episode_verbose)

            wandb.log({
                "rollout/episode_reward": float(episode_reward.mean()),
                "rollout/episode_length": episode_length
            }, step=collect_time)

            self.PyTreereplay_buffer.add_batch(episode_transitions)
            self.PyTreereplay_buffer.add_edge(edge_N, episode_transitions)
            current_step += episode_length

            # Policy update
            if current_step >= self.start_steps and self.PyTreereplay_buffer.size > self.batch_size:
                train_per_cycle = 40
                for _ in range(train_per_cycle):
                    main_batch, edge_batch = self.PyTreereplay_buffer.sample(self.batch_size)
                    update_info = self.agent.update(main_batch, edge_batch)
                    if self.update_steps % 100 == 0:
                        wandb.log({f"train/{k}": v.item() for k, v in update_info.items()}, step=collect_time)
                    self.update_steps += 1

            # Evaluation
            if collect_time % self.eval_interval == 0:
                all_successful_flag, all_safe_flag, all_episode_rewards = run_parallel_evaluation(
                    agent=self.agent,
                    eval_env=self.env_test,
                    max_steps=self.max_episode_steps,
                    eval_episodes=self.eval_epi,
                    actor_params=self.agent.actor_state.params,
                    seed=self.seed)

                all_episode_rewards_np = np.array(all_episode_rewards)
                eval_reward = all_episode_rewards_np.mean()
                eval_successful_rate = all_successful_flag.mean().item()
                eval_safe_rate = all_safe_flag.mean().item()

                wandb.log({"eval/mean_reward": eval_reward}, step=collect_time)
                wandb.log({"eval/eval_successful_rate": eval_successful_rate}, step=collect_time)
                wandb.log({"eval/eval_safe_rate": eval_safe_rate}, step=collect_time)

                time_since_start = time.time() - start_time
                eval_verbose = (f'Episode: {collect_time}, Time: {time_since_start:.0f}s, Eval Reward: {eval_reward:.2f}')
                tqdm.write(eval_verbose)

                if self.save_log:
                    if eval_reward > self.best_eval_reward:
                        self.best_eval_reward = eval_reward
                        texts = f"Safe Rate : {eval_safe_rate * 100} %\n Success Rate : {eval_successful_rate * 100} %\n"
                        eval_dir = os.path.join(self.model_dir, f"eval_train_best")
                        os.makedirs(eval_dir, exist_ok=True)
                        txt_path = os.path.join(eval_dir, f"output.txt")
                        with open(txt_path, "w", encoding="utf-8") as file:
                            file.writelines([texts])
                        tqdm.write(f"New best model found! Saving...")
                        self.agent.save_agent_states(self.model_dir, collect_time, prefix="best_")

            # Periodic checkpoint save
            if collect_time % self.save_interval == 0 and self.save_log:
                tqdm.write(f"Saving interval checkpoint...")
                self.agent.save_agent_states(self.model_dir, collect_time, prefix="checkpoint_")

            collect_time += 1
            pbar.update(1)

        # Save final model
        if self.save_log:
            print("Training finished. Saving final model.")
            self.agent.save_agent_states(self.model_dir, collect_time, prefix="final_")

        wandb.finish()
