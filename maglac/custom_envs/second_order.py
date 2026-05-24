import functools as ft
import pathlib
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from typing import NamedTuple, Tuple, Optional
from ..utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ..utils.typing import Action, AgentState, Array, Cost, Done, Info, Reward, State
from ..utils.utils import merge01, jax_vmap
from .base import MultiAgentEnv, RolloutResult
from .obstacle import Obstacle, Rectangle
from .plot import render_video, render_trajectory
from .utils import get_lidar, inside_obstacles, lqr, get_node_goal_rng

class Second_Order(MultiAgentEnv):
    AGENT = 0
    GOAL = 1
    OBS = 2

    class EnvState(NamedTuple):
        agent: AgentState
        goal: State
        obstacle: Obstacle
        timestep: jnp.ndarray # <--- newly added field
        dist2tgt: jnp.ndarray
        min_dist2obs: Optional[jnp.ndarray] = jnp.array(0.0)
        min_dist2agent: Optional[jnp.ndarray] = jnp.array(0.0)
        edge_mask: jnp.ndarray = jnp.array(0)
        unsafe_mask: Optional[jnp.ndarray] = jnp.array(0).astype(jnp.bool_)
        safe_mask: Optional[jnp.ndarray] = jnp.array(0).astype(jnp.bool_)
        pre_action: Optional[jnp.ndarray] = None
        
        @property
        def n_agent(self) -> int:
            return self.agent.shape[0]

    EnvGraphsTuple = GraphsTuple[State, EnvState]

    PARAMS = {
        "car_radius": 0.05,
        "comm_radius": 0.5,
        "n_rays": 32,
        "obs_len_range": [0.1, 0.5],
        "n_obs": 8,
        "m": 0.1,  # mass
    }

    def __init__(
            self,
            num_agents: int,
            area_size: float,
            max_step: int = 256,
            max_travel: float = None,
            dt: float = 0.03,
            params: dict = None,
            r_c_params: dict = None,
    ):
        super(Second_Order, self).__init__(num_agents, area_size, max_step, max_travel, dt, params)
        A = np.zeros((self.state_dim, self.state_dim), dtype=np.float32)
        A[0, 2] = 1.0
        A[1, 3] = 1.0
        self._A = A * self._dt + np.eye(self.state_dim)
        self._B = (
            np.array([[0.0, 0.0], [0.0, 0.0], [1.0 / self._params["m"], 0.0], [0.0, 1.0 / self._params["m"]]])
            * self._dt
        )
        self.goal_threshold = 0.05
        self.edge_threshold = 0.25
        self.edge_threshold2 = 0.15 # Collision between drones, detection radius is doubled
        self.action_low, self.action_high = self.action_lim()
        self.max_step = max_step
        self.success_reward = r_c_params['success_reward']
        self.reach_reward = r_c_params['reach_reward']
        self.w_delta1 = r_c_params['w_delta1']
        self.w_delta2 = r_c_params['w_delta2']
        self.correction_cost_dist = r_c_params['correction_cost_dist']
        self.danger_penalty_coeff = r_c_params['danger_penalty_coeff']
        self.warning_dist2obs = r_c_params['warning_dist2obs']
        self.warning_dist2agent = r_c_params['warning_dist2agent']
        self.potential_obs_reward_coeff = r_c_params['potential_obs_reward_coeff']
        self.tgt_reward_coeff = r_c_params['tgt_reward_coeff']
        self.tgt_reward_low = r_c_params['tgt_reward_low']
        self.tgt_reward_high = r_c_params['tgt_reward_high']
        self.reward_scale = r_c_params['reward_scale']
        self.cost_coeff = r_c_params['cost_coeff']
        self.cost = r_c_params['cost']
        self.cost_obs_dist = r_c_params['cost_obs_dist']
        self.cost_agent_dist = r_c_params['cost_agent_dist']

        self._Q = np.eye(self.state_dim) * 5
        self._R = np.eye(self.action_dim)
        self._K = jnp.array(lqr(self._A, self._B, self._Q, self._R))
        self.create_obstacles = jax_vmap(Rectangle.create)

    @property
    def state_dim(self) -> int:
        return 4  # x, y, vx, vy

    @property
    def node_dim(self) -> int:
        return 3  # indicator: agent: 001, goal: 010, obstacle: 100

    @property
    def edge_dim(self) -> int:
        return 4  # x_rel, y_rel, vx_rel, vy_rel

    @property
    def action_dim(self) -> int:
        return 2  # fx, fy

    def sample_non_overlapping_obstacles_numpy(self, key, n_obs, min_gap=0.05):
        """Perform rejection sampling in numpy, then convert back to jax array"""
        area = self.area_size
        lo, hi = self._params["obs_len_range"]
        
        # Define a pure numpy internal function (does not accept traced values)
        def _sample_numpy(seed):
            # seed is a specific int (when executed on the host side)
            rng = np.random.default_rng(int(seed))
            centers, half_diags, ws, hs, thetas = [], [], [], [], []
            for _ in range(n_obs):
                for _ in range(500):
                    pos   = rng.uniform(0, area, size=2)
                    w     = rng.uniform(lo, hi)
                    h     = rng.uniform(lo, hi)
                    theta = rng.uniform(0, 2 * np.pi)
                    hd    = np.sqrt(w**2 + h**2) / 2
                    if len(centers) == 0:
                        break
                    dists    = np.linalg.norm(np.array(centers) - pos, axis=-1)
                    min_dist = np.array(half_diags) + hd + min_gap
                    if np.all(dists >= min_dist):
                        break
                centers.append(pos)
                half_diags.append(hd)
                ws.append(w); hs.append(h); thetas.append(theta)
            
            # Return numpy array (must have fixed shape and dtype)
            return (
                np.array(centers, dtype=np.float32),
                np.array(ws, dtype=np.float32),
                np.array(hs, dtype=np.float32),
                np.array(thetas, dtype=np.float32),
            )
        
        # Generate an int seed from key
        numpy_seed = jax.random.randint(key, shape=(), minval=0, maxval=2147483647)
        
        # Declare output shape and dtype (must be fixed!)
        result_shape_dtype = (
            jax.ShapeDtypeStruct((n_obs, 2), jnp.float32),  # centers
            jax.ShapeDtypeStruct((n_obs,), jnp.float32),    # ws
            jax.ShapeDtypeStruct((n_obs,), jnp.float32),    # hs
            jax.ShapeDtypeStruct((n_obs,), jnp.float32),    # thetas
        )
        
        # Use pure_callback to break out of JAX trace
        all_pos, all_w, all_h, all_theta = jax.pure_callback(
            _sample_numpy,
            result_shape_dtype,
            numpy_seed,
            vmap_method='sequential',  # Key: call batch by batch during vmap
        )
        
        return self.create_obstacles(all_pos, all_w, all_h, all_theta)

    def reset(self, key: Array) -> GraphsTuple:
        self._t = 0

        # randomly generate obstacles
        n_rng_obs = self._params["n_obs"]
        assert n_rng_obs >= 0
        obstacle_key, key = jr.split(key, 2)
        obs_pos = jr.uniform(obstacle_key, (n_rng_obs, 2), minval=0, maxval=self.area_size)
        length_key, key = jr.split(key, 2)
        obs_len = jr.uniform(
            length_key,
            (n_rng_obs, 2),
            minval=self._params["obs_len_range"][0],
            maxval=self._params["obs_len_range"][1],
        )
        theta_key, key = jr.split(key, 2)
        obs_theta = jr.uniform(theta_key, (n_rng_obs,), minval=0, maxval=2 * np.pi)
        obstacles = self.create_obstacles(obs_pos, obs_len[:, 0], obs_len[:, 1], obs_theta)
        obstacles = self.sample_non_overlapping_obstacles_numpy(key, n_rng_obs)
        # randomly generate agent and goal
        states, goals = get_node_goal_rng(
            key, self.area_size, 2, obstacles, self.num_agents, 4 * self.params["car_radius"], self.max_travel)

        # add zero velocity
        states = jnp.concatenate([states, jnp.zeros((self.num_agents, 2))], axis=1)
        goals = jnp.concatenate([goals, jnp.zeros((self.num_agents, 2))], axis=1)
        
        agent_pos = states[:, :2]
        goal_pos = goals[:, :2]
        dist2tgt = jnp.linalg.norm(agent_pos - goal_pos, axis=-1) 

        dummy_actions = jnp.zeros((self.num_agents, self.action_dim)) # (n_agents, action_dim)
        env_states = self.EnvState(states, goals, obstacles, jnp.array(0), jnp.array(dist2tgt), 
                                   jnp.array(0.0), jnp.array(0.0), jnp.array(0),
                                   jnp.array(0).astype(jnp.bool_), jnp.array(0).astype(jnp.bool_), dummy_actions)
        graph, _ = self.get_graph(env_states)

        return graph

    def agent_step_exact(self, agent_states: AgentState, action: Action) -> AgentState:
        assert action.shape == (self.num_agents, self.action_dim)
        # [x, y, vx, vy]
        assert agent_states.shape == (self.num_agents, self.state_dim)
        n_accel = self.agent_accel(action)
        n_pos_new = agent_states[:, :2] + agent_states[:, 2:] * self.dt + n_accel * self.dt**2 / 2
        n_vel_new = agent_states[:, 2:] + n_accel * self.dt
        n_state_agent_new = jnp.concatenate([n_pos_new, n_vel_new], axis=1)
        assert n_state_agent_new.shape == (self.num_agents, self.state_dim)
        return n_state_agent_new

    def agent_accel(self, action: Action) -> Action:
        return action / self._params["m"]

    def agent_step_euler(self, agent_states: AgentState, action: Action) -> AgentState:
        assert action.shape == (self.num_agents, self.action_dim)
        # [x, y, vx, vy]
        assert agent_states.shape == (self.num_agents, self.state_dim)
        x_dot = self.agent_xdot(agent_states, action)
        n_state_agent_new = x_dot * self.dt + agent_states
        assert n_state_agent_new.shape == (self.num_agents, self.state_dim)
        return self.clip_state(n_state_agent_new)

    def agent_xdot(self, agent_states: AgentState, action: Action) -> AgentState:
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)
        n_accel = self.agent_accel(action)
        x_dot = jnp.concatenate([agent_states[:, 2:], n_accel], axis=1)
        assert x_dot.shape == (self.num_agents, self.state_dim)
        return x_dot

    def get_min_dist_to_obstacles(self, agent_pos: jnp.ndarray, obstacles: Obstacle) -> jnp.ndarray:
        """
        Use get_lidar function to calculate the distance from each agent to the nearest obstacle.
        Args:
            agent_pos: positions of all agents, shape (n_agents, 2) or (n_agents, 3)
            obstacles: a Pytree containing all obstacle information
        Returns:
            An array containing the distance from each agent to the nearest obstacle, shape (n_agents,)
        """
        # 1. Create a vmap version of the get_lidar function
        #    This function can now receive a batch of agent_pos as input
        get_lidar_vmap = jax.vmap(
            ft.partial(
                get_lidar,  
                obstacles=obstacles,
                num_beams=self._params["n_rays"],
                sense_range=self._params["comm_radius"],
                max_returns=self._params["n_rays"] # Ensure returning results of all beams
            )
        )
        
        # 2. Compute LiDAR data for all agents in parallel
        #    agent_pos shape: (n_agents, 2)
        #    all_lidar_data shape: (n_agents, n_rays, 2)
        all_lidar_data = get_lidar_vmap(agent_pos)

        # 3. Calculate the distance between each agent's LiDAR hit point and its own position
        #    a. agent_pos needs to expand dimensions to match all_lidar_data
        #       agent_pos_expanded shape: (n_agents, 1, 2)
        agent_pos_expanded = jnp.expand_dims(agent_pos, axis=1)
        
        #    b. Calculate the distance from each hit point to the corresponding agent
        #       distances_all_beams shape: (n_agents, n_rays)
        distances_all_beams = jnp.linalg.norm(all_lidar_data - agent_pos_expanded, axis=-1)
        
        # 4. Find the minimum value for each agent from the distances of all beams
        #    min_dists shape: (n_agents,)
        min_dists = jnp.min(distances_all_beams, axis=1)

        return min_dists

    def step(
            self, graph: EnvGraphsTuple, delta_action: Action, get_eval_info: bool = False
        ) -> Tuple[EnvGraphsTuple, Reward, Cost, Done, Info]:
            self._t += 1
            current_t = graph.env_states.timestep

            agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
            goal_states = graph.type_states(type_idx=1, n_type=self.num_agents)
            agent_pos = agent_states[:, :2]
            obstacles = graph.env_states.obstacle
            pre_dist2tgt = graph.env_states.dist2tgt
            pre_min_dist2obs = graph.env_states.min_dist2obs
            pre_min_dist2agent = graph.env_states.min_dist2agent
            pre_action = graph.env_states.pre_action
            
            delta_action = delta_action
            u_ref = self.u_ref(graph)
            action = delta_action + 1 * u_ref
            action = self.clip_action(action)

            assert action.shape == (self.num_agents, self.action_dim)
            assert agent_states.shape == (self.num_agents, self.state_dim)

            next_agent_states = self.agent_step_euler(agent_states, action)
            next_t = current_t + 1
            done = jnp.array(False)

            reward = jnp.zeros(()).astype(jnp.float32)
            
            # --- 4. Check termination conditions ---
            next_agent_pos = next_agent_states[:, :2]
            goal_pos = goal_states[:, :2]
            dist2tgt = jnp.linalg.norm(next_agent_pos - goal_pos, axis=-1) 
            is_reach = dist2tgt < self.goal_threshold
            is_timeout = next_t >= self.max_step
            all_reach = jnp.all(is_reach)
 
            is_success = jnp.logical_and(dist2tgt < self.goal_threshold, is_timeout)
            success_reward = jnp.where(is_success, self.success_reward, 0.0)

            next_state = self.EnvState(next_agent_states, goal_states, obstacles, next_t, jnp.array(dist2tgt), 
                                       jnp.array(0.0), jnp.array(0.0), jnp.array(0), jnp.array(0).astype(jnp.bool_),
                                       jnp.array(0).astype(jnp.bool_), delta_action)
            
            next_graph, extra_info = self.get_graph(next_state)
            is_collision = self.collision_mask(next_graph)
            any_collision = jnp.any(is_collision)
            done = jnp.logical_or(any_collision, is_timeout)
            done = jnp.array(done)
            cost = self.get_cost(next_graph, is_collision)
            min_dist_to_agent = next_graph.env_states.min_dist2agent
            
            # --- 5. Calculate reward function ---
            min_dist_to_obs = extra_info['min_dist_to_obs']

            reach_reward = jnp.where(is_reach, self.reach_reward, 0.0) 

            assert done.shape == tuple()

            warning_dist = self._params["car_radius"] * self.warning_dist2obs
            is_in_danger_zone1 = pre_min_dist2obs < warning_dist
            danger_penalty = - (jnp.maximum(0, warning_dist - min_dist_to_obs) * 40)
            warning_dist2agent = self._params["car_radius"] * self.warning_dist2agent 
            is_in_danger_zone2 = min_dist_to_agent < warning_dist2agent 
            danger_penalty -= (jnp.maximum(0, warning_dist2agent - min_dist_to_agent) * self.danger_penalty_coeff)

            is_in_danger_zone = jnp.logical_or(is_in_danger_zone1, is_in_danger_zone2)
            correction_cost = jnp.where(
                is_in_danger_zone,
                -self.w_delta2 * (jnp.linalg.norm(delta_action - pre_action, axis=-1)**2),
                -self.w_delta1 * (jnp.linalg.norm(delta_action, axis=-1)**2) 
            )
            lat_v = self.lateral_velocity(next_agent_pos, agent_states[:, 2:4], goal_pos)
            lateral_reward = jnp.where(is_in_danger_zone, 2 * lat_v, 0.0)

            tgt_reward = (pre_dist2tgt - dist2tgt) * self.tgt_reward_coeff 
            tgt_reward = jnp.clip(tgt_reward, self.tgt_reward_low, self.tgt_reward_high)

            reward_per_agent = (
                reach_reward +
                success_reward +
                correction_cost +
                lateral_reward +
                tgt_reward
            )
            
            reward = reward_per_agent / self.reward_scale
            reward = jnp.array(reward)
            info = {}
            if get_eval_info:
                # collision between agents and obstacles
                info["inside_obstacles"] = is_collision
            info["success"] = is_success.astype(jnp.int32) 
            info["dist2tgt"] = dist2tgt
            return next_graph, reward, cost, done, info
    
    
    def get_cost(self, graph: EnvGraphsTuple, is_collision) -> Cost:
        min_dist_to_obs = graph.env_states.min_dist2obs
        min_dist_to_obs = jnp.minimum(0.5, min_dist_to_obs)
        min_dist_to_agent = graph.env_states.min_dist2agent
        
        cost = jnp.maximum(0, self._params["car_radius"] * self.cost_obs_dist - min_dist_to_obs) * self.cost_coeff
        cost += jnp.maximum(0, self._params["car_radius"] * self.cost_agent_dist - min_dist_to_agent) * self.cost_coeff

        return cost
    
    def lateral_velocity(self, agent_pos, agent_vel, goal_pos, eps=1e-6):
        to_goal = goal_pos - agent_pos
        to_goal_norm = jnp.linalg.norm(to_goal, axis=-1, keepdims=True)
        e_parallel = to_goal / (to_goal_norm + eps)

        v_parallel = jnp.sum(agent_vel * e_parallel, axis=-1, keepdims=True) * e_parallel
        v_lateral = agent_vel - v_parallel
        return jnp.linalg.norm(v_lateral, axis=-1)

   
    def render_video(
            self,
            rollout: RolloutResult,
            video_path: pathlib.Path,
            Ta_is_unsafe=None,
            viz_opts: dict = None,
            dpi: int = 100,
            **kwargs
    ) -> None:
        render_video(
            rollout=rollout,
            video_path=video_path,
            side_length=self.area_size,
            dim=2,
            n_agent=self.num_agents,
            n_rays=self.params["n_rays"],
            r=self.params["car_radius"],
            Ta_is_unsafe=Ta_is_unsafe,
            viz_opts=viz_opts,
            dpi=dpi,
            **kwargs
        )

    def render_trajectory(
            self,
            rollout: RolloutResult,
            save_path: pathlib.Path,
            dpi: int = 100,
            **kwargs
    ) -> None:
        render_trajectory(
            rollout=rollout,
            save_path=save_path,
            side_length=self.area_size,
            dim=2,
            n_agent=self.num_agents,
            r=self.params["car_radius"],
            dt=self._dt,
            dpi=dpi,
            **kwargs
        )

    def edge_blocks(self, state: EnvState, lidar_data: State) -> list[EdgeBlock]:
        n_hits = self._params["n_rays"] * self.num_agents

        # agent - agent connection
        agent_pos = state.agent[:, :2]
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(dist.shape[1]) * (self._params["comm_radius"] + 1)
        min_dist2agent = jnp.min(dist, axis=1)
        edge_mask = jnp.where(min_dist2agent < self.edge_threshold, 1, 0)
        state_diff = state.agent[:, None, :] - state.agent[None, :, :]
        agent_agent_mask = jnp.less(dist, self._params["comm_radius"])
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(state_diff, agent_agent_mask, id_agent, id_agent)

        # agent - goal connection, clipped to avoid too long edges
        id_goal = jnp.arange(self.num_agents, self.num_agents * 2)
        agent_goal_mask = jnp.eye(self.num_agents)
        agent_goal_feats = state.agent[:, None, :] - state.goal[None, :, :]
        feats_norm = jnp.sqrt(1e-6 + jnp.sum(agent_goal_feats[:, :2] ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(feats_norm, comm_radius)
        coef = jnp.where(feats_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        agent_goal_feats = agent_goal_feats.at[:, :2].set(agent_goal_feats[:, :2] * coef)
        agent_goal_edges = EdgeBlock(
            agent_goal_feats, agent_goal_mask, id_agent, id_goal
        )

        # agent - obs connection
        id_obs = jnp.arange(self.num_agents * 2, self.num_agents * 2 + n_hits)
        agent_obs_edges = []
        for i in range(self.num_agents):
            id_hits = jnp.arange(i * self._params["n_rays"], (i + 1) * self._params["n_rays"])
            lidar_pos = agent_pos[i, :] - lidar_data[id_hits, :2]
            lidar_feats = state.agent[i, :] - lidar_data[id_hits, :]
            lidar_dist = jnp.linalg.norm(lidar_pos, axis=-1)
            active_lidar = jnp.less(lidar_dist, self._params["comm_radius"] - 1e-1)
            agent_obs_mask = jnp.ones((1, self._params["n_rays"]))
            agent_obs_mask = jnp.logical_and(agent_obs_mask, active_lidar)
            agent_obs_edges.append(
                EdgeBlock(lidar_feats[None, :, :], agent_obs_mask, id_agent[i][None], id_obs[id_hits])
            )

        return [agent_agent_edges, agent_goal_edges] + agent_obs_edges, min_dist2agent, edge_mask

    def control_affine_dyn(self, state: State) -> [Array, Array]:
        assert state.ndim == 2
        f = jnp.concatenate([state[:, 2:], jnp.zeros((state.shape[0], 2))], axis=1)
        g = jnp.concatenate([jnp.zeros((2, 2)), jnp.eye(2) / self._params['m']], axis=0)
        g = jnp.expand_dims(g, axis=0).repeat(f.shape[0], axis=0)
        assert f.shape == state.shape
        assert g.shape == (state.shape[0], self.state_dim, self.action_dim)
        return f, g

    def add_edge_feats(self, graph: GraphsTuple, state: State) -> GraphsTuple:
        assert graph.is_single
        assert state.ndim == 2

        edge_feats = state[graph.receivers] - state[graph.senders]
        feats_norm = jnp.sqrt(1e-6 + jnp.sum(edge_feats[:, :2] ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(feats_norm, comm_radius)
        coef = jnp.where(feats_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        edge_feats = edge_feats.at[:, :2].set(edge_feats[:, :2] * coef)

        return graph._replace(edges=edge_feats, states=state)

    def get_graph(self, state: EnvState, adjacency: Array = None) -> GraphsTuple:
        # node features
        n_hits = self._params["n_rays"] * self.num_agents
        n_nodes = 2 * self.num_agents + n_hits
        node_feats = jnp.zeros((self.num_agents * 2 + n_hits, 3))
        node_feats = node_feats.at[: self.num_agents, 2].set(1)  # agent feats
        node_feats = node_feats.at[self.num_agents: self.num_agents * 2, 1].set(1)  # goal feats
        node_feats = node_feats.at[-n_hits:, 0].set(1)  # obs feats

        node_type = jnp.zeros(n_nodes, dtype=jnp.int32)
        node_type = node_type.at[self.num_agents: self.num_agents * 2].set(Second_Order.GOAL)
        node_type = node_type.at[-n_hits:].set(Second_Order.OBS)

        get_lidar_vmap = jax_vmap(
            ft.partial(
                get_lidar,
                obstacles=state.obstacle,
                num_beams=self._params["n_rays"],
                sense_range=self._params["comm_radius"],
            )
        )
        lidar_data = merge01(get_lidar_vmap(state.agent[:, :2]))
        lidar_data = jnp.concatenate([lidar_data, jnp.zeros_like(lidar_data)], axis=-1)
        edge_blocks, min_dist2agent, edge_mask1 = self.edge_blocks(state, lidar_data)
        
        # --- b. Calculate minimum distance from lidar_data ---
        agent_pos = state.agent[:, :2]
        agent_pos_expanded = jnp.expand_dims(agent_pos, axis=1)
        
        # Shape of all_lidar_data is (n_agents, n_rays, 2)
        # lidar_data is (n_agents * n_rays, 2), needs reshape
        all_lidar_data = lidar_data[:, :2].reshape(self.num_agents, self._params["n_rays"], 2)
        
        distances_all_beams = jnp.linalg.norm(all_lidar_data - agent_pos_expanded, axis=-1)
        min_dist2obs = jnp.min(distances_all_beams, axis=1)

        edge_mask2 = jnp.where(min_dist2obs < self.edge_threshold2, 1, 0)
        edge_mask = jnp.logical_or(edge_mask1, edge_mask2)
        state_with_dist = state._replace(min_dist2obs=min_dist2obs, min_dist2agent=min_dist2agent, edge_mask=edge_mask)
        
        # create graph
        graph = GetGraph(
            nodes=node_feats,
            node_type=node_type,
            edge_blocks=edge_blocks,
            env_states=state_with_dist,
            states=jnp.concatenate([state.agent, state.goal, lidar_data], axis=0),
        ).to_padded()
    
        # --- c. Pack and return extra information ---
        extra_info = {
            'min_dist_to_obs': min_dist2obs
        }
        return graph, extra_info

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        lower_lim = jnp.array([-jnp.inf, -jnp.inf, -0.5, -0.5])
        upper_lim = jnp.array([jnp.inf, jnp.inf, 0.5, 0.5])
        return lower_lim, upper_lim

    def action_lim(self) -> Tuple[Action, Action]:
        lower_lim = jnp.ones(2) * -1.0
        upper_lim = jnp.ones(2)
        return lower_lim, upper_lim

    def u_ref(self, graph: GraphsTuple) -> Action:
        agent = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal = graph.type_states(type_idx=1, n_type=self.num_agents)
        error = goal - agent
        error_max = jnp.abs(error / jnp.linalg.norm(error, axis=-1, keepdims=True) * self._params["comm_radius"])
        error = jnp.clip(error, -error_max, error_max)
        return self.clip_action(error @ self._K.T)

    def forward_graph(self, graph: GraphsTuple, action: Action) -> GraphsTuple:
        # calculate next graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal_states = graph.type_states(type_idx=1, n_type=self.num_agents)
        obs_states = graph.type_states(type_idx=2, n_type=self._params["n_rays"] * self.num_agents)
        action = self.clip_action(action)

        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        next_agent_states = self.agent_step_euler(agent_states, action)
        next_states = jnp.concatenate([next_agent_states, goal_states, obs_states], axis=0)

        next_graph = self.add_edge_feats(graph, next_states)
        return next_graph

    @ft.partial(jax.jit, static_argnums=(0,))
    def safe_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]

        # agents are not colliding
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist = dist + jnp.eye(dist.shape[1]) * (self._params["car_radius"] * 2 + 1)  # remove self connection
        safe_agent = jnp.greater(dist, self._params["car_radius"] * 4)

        safe_agent = jnp.min(safe_agent, axis=1)

        safe_obs = jnp.logical_not(
            inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"] * 2)
        )

        safe_mask = jnp.logical_and(safe_agent, safe_obs)

        return safe_mask
        
    @ft.partial(jax.jit, static_argnums=(0,))
    def unsafe_mask(self, graph: GraphsTuple) -> Array:
        agent_state = graph.type_states(type_idx=0, n_type=self.num_agents)
        agent_pos = agent_state[:, :2]

        # agents are colliding
        agent_pos_diff = agent_pos[None, :, :] - agent_pos[:, None, :]
        agent_dist = jnp.linalg.norm(agent_pos_diff, axis=-1)
        agent_dist = agent_dist + jnp.eye(agent_dist.shape[1]) * (self._params["car_radius"] * 2 + 1)
        unsafe_agent = jnp.less(agent_dist, self._params["car_radius"] * 2)
        unsafe_agent = jnp.max(unsafe_agent, axis=1)

        # agents are colliding with obstacles
        unsafe_obs = inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"])

        collision_mask = jnp.logical_or(unsafe_agent, unsafe_obs)

        # unsafe direction
        agent_warn_dist = 3 * self._params["car_radius"]
        obs_warn_dist = 2 * self._params["car_radius"]
        obs_pos = graph.type_states(type_idx=2, n_type=self._params["n_rays"] * self.num_agents)[:, :2]
        obs_pos_diff = obs_pos[None, :, :] - agent_pos[:, None, :]
        obs_dist = jnp.linalg.norm(obs_pos_diff, axis=-1)
        pos_diff = jnp.concatenate([agent_pos_diff, obs_pos_diff], axis=1)
        warn_zone = jnp.concatenate([jnp.less(agent_dist, agent_warn_dist), jnp.less(obs_dist, obs_warn_dist)], axis=1)
        pos_vec = (pos_diff / (jnp.linalg.norm(pos_diff, axis=2, keepdims=True) + 0.0001))
        speed_agent = jnp.linalg.norm(agent_state[:, 2:], axis=1, keepdims=True)
        heading_vec0 = (agent_state[:, 2:] / (speed_agent + 0.0001))[:, None, :]
        heading_vec = heading_vec0.repeat(pos_vec.shape[1], axis=1)
        inner_prod = jnp.sum(pos_vec * heading_vec, axis=2)
        unsafe_theta_agent = jnp.arctan2(self._params['car_radius'] * 2,
                                         jnp.sqrt(agent_dist**2 - 4 * self._params['car_radius']**2))
        unsafe_theta_obs = jnp.arctan2(self._params['car_radius'],
                                       jnp.sqrt(obs_dist**2 - self._params['car_radius']**2))
        unsafe_theta = jnp.concatenate([unsafe_theta_agent, unsafe_theta_obs], axis=1)
        lidar_mask = jnp.ones((self._params["n_rays"],))
        lidar_mask = jax.scipy.linalg.block_diag(*[lidar_mask] * self.num_agents)
        valid_mask = jnp.concatenate([jnp.ones((self.num_agents, self.num_agents)), lidar_mask], axis=-1)
        warn_zone = jnp.logical_and(warn_zone, valid_mask)
        unsafe_dir = jnp.max(jnp.logical_and(warn_zone, jnp.greater(inner_prod, jnp.cos(unsafe_theta))), axis=1)
        unsafe_mask = jnp.logical_or(collision_mask, unsafe_dir)
        final_unsafe_mask = jnp.any(unsafe_mask, axis=-1)
        state_with_safe_mask = graph.env_states._replace(unsafe_mask=final_unsafe_mask)
        graph = graph._replace(env_states=state_with_safe_mask)
        return graph 

    def collision_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]

        # agents are colliding
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist = dist + jnp.eye(dist.shape[1]) * (self._params["car_radius"] * 2 + 1)  # remove self connection
        unsafe_agent = jnp.less(dist, self._params["car_radius"] * 2)
        unsafe_agent = jnp.max(unsafe_agent, axis=1)

        # agents are colliding with obstacles
        unsafe_obs = inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"])

        collision_mask = jnp.logical_or(unsafe_agent, unsafe_obs)

        return collision_mask

    def finish_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]
        goal_pos = graph.env_states.goal[:, :2]
        reach = jnp.linalg.norm(agent_pos - goal_pos, axis=1) < self._params["car_radius"] * 2
        return reach