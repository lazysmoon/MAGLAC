import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from colour import hsl2hex
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.pyplot import Axes
from matplotlib.patches import Polygon, Circle
from mpl_toolkits.mplot3d import proj3d, Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from ..utils.utils import merge01, tree_index, MutablePatchCollection, save_anim
from .obstacle import Cuboid, Sphere, Obstacle, Rectangle
from .base import RolloutResult
from matplotlib.colors import Normalize

def get_obs_collection(
        obstacles: Obstacle, color: str, alpha: float
):
    n_obs = len(obstacles.center)
    patches = []

    # Get the type array. Older Rectangle classes may not have a `type` field;
    # in that case default to rectangle.
    obs_types = getattr(obstacles, 'type', None)

    for i in range(n_obs):
        # Detect circles (type == 1).
        # obs_types[i] may be a JAX array; comparison still works, but we only
        # check the value when obs_types exists.
        is_circle = False
        if obs_types is not None:
            # Assume 1 stands for CIRCLE
            if obs_types[i] == 1:
                is_circle = True

        if is_circle:
            # --- Draw a circle ---
            # center: (x, y), radius: float.
            # Matplotlib expects Python floats / tuples; JAX arrays usually work,
            # but wrap with np.array() if needed.
            xy = (obstacles.center[i][0], obstacles.center[i][1])
            r = obstacles.radius[i]
            patches.append(Circle(xy, radius=r))
        else:
            # --- Draw a rectangle / polygon ---
            # points shape: (4, 2)
            poly_points = obstacles.points[i]
            patches.append(Polygon(poly_points, closed=True))

    # Build the collection. match_original=False lets us override the color uniformly;
    # match_original=True would keep each patch's original color.
    obs_col = PatchCollection(patches, facecolor=color, alpha=alpha, zorder=99)
    return obs_col


def render_video(
        rollout: RolloutResult,
        video_path: pathlib.Path,
        side_length: float,
        dim: int,
        n_agent: int,
        n_rays: int,
        r: float,
        Ta_is_unsafe=None,
        viz_opts: dict = None,
        dpi: int = 100,
        **kwargs
):
    assert dim == 2 or dim == 3

    # set up visualization option
    if dim == 2:
        ax: Axes
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=dpi)
    else:
        fig = plt.figure(figsize=(10, 10), dpi=dpi)
        ax: Axes3D = fig.add_subplot(projection='3d')
    ax.set_xlim(0., side_length)
    ax.set_ylim(0., side_length)
    if dim == 3:
        ax.set_zlim(0., side_length)
    ax.set(aspect="equal")
    if dim == 2:
        plt.axis("off")

    if viz_opts is None:
        viz_opts = {}

    # plot the first frame
    T_graph = rollout.Tp1_graph
    graph0 = tree_index(T_graph, 0)

    agent_color = "#0068ff"
    goal_color = "#2fdd00"
    obs_color = "#404040"
    edge_goal_color = goal_color

    # plot obstacles
    obs = graph0.env_states.obstacle
    ax.add_collection(get_obs_collection(obs, obs_color, alpha=0.8))

    # plot agents
    n_hits = n_agent * n_rays
    n_color = [agent_color] * n_agent + [goal_color] * n_agent
    n_pos = graph0.states[:n_agent * 2, :dim]
    n_radius = np.array([r] * n_agent * 2)
    if dim == 2:
        agent_list = []  # holds every patch artist (circles + squares)
        for ii in range(n_agent * 2):
            if ii < n_agent:
                # --- First half are agents: draw as circles ---
                patch = plt.Circle(n_pos[ii], n_radius[ii], color=n_color[ii], linewidth=0.0)
            else:
                # --- Second half are goals: draw as squares ---
                # Rectangle (x, y) refers to the bottom-left corner, so subtract r to center the shape.
                bottom_left_x = n_pos[ii, 0] - r
                bottom_left_y = n_pos[ii, 1] - r
                side = r * 2  # side length == diameter
                patch = plt.Rectangle((bottom_left_x, bottom_left_y), side, side,
                                      color=n_color[ii], linewidth=0.0)
            agent_list.append(patch)

        agent_col = MutablePatchCollection([i for i in reversed(agent_list)], match_original=True, zorder=6)
        ax.add_collection(agent_col)
    else:
        plot_r = ax.transData.transform([r, 0])[0] - ax.transData.transform([0, 0])[0]
        agent_col = ax.scatter(n_pos[:, 0], n_pos[:, 1], n_pos[:, 2],
                               s=plot_r, c=n_color, zorder=5)  # todo: the size of the agent might not be correct

    # plot edges
    all_pos = graph0.states[:n_agent * 2 + n_hits, :dim]
    edge_index = np.stack([graph0.senders, graph0.receivers], axis=0)
    is_pad = np.any(edge_index == n_agent * 2 + n_hits, axis=0)
    e_edge_index = edge_index[:, ~is_pad]
    e_start, e_end = all_pos[e_edge_index[0, :]], all_pos[e_edge_index[1, :]]
    e_lines = np.stack([e_start, e_end], axis=1)  # (e, n_pts, dim)
    e_is_goal = (n_agent <= graph0.senders) & (graph0.senders < n_agent * 2)
    e_is_goal = e_is_goal[~is_pad]
    e_colors = [edge_goal_color if e_is_goal[ii] else "0.2" for ii in range(len(e_start))]
    if dim == 2:
        edge_col = LineCollection(e_lines, colors=e_colors, linewidths=2, alpha=0.5, zorder=3)
    else:
        edge_col = Line3DCollection(e_lines, colors=e_colors, linewidths=2, alpha=0.5, zorder=3)
    ax.add_collection(edge_col)

    # text for cost and reward
    text_font_opts = dict(
        size=16,
        color="k",
        family="cursive",
        weight="normal",
        transform=ax.transAxes,
    )
    if dim == 2:
        cost_text = ax.text(0.02, 1.04, "dist2obs: 1.0, dist2tgt: 1.0, Reward: 1.0", va="bottom", **text_font_opts)
    else:
        cost_text = ax.text2D(0.02, 1.04, "dist2obs: 1.0, dist2tgt: 1.0, Reward: 1.0", va="bottom", **text_font_opts)

    # text for time step
    if dim == 2:
        kk_text = ax.text(0.99, 0.99, "kk=0", va="top", ha="right", **text_font_opts)
    else:
        kk_text = ax.text2D(0.99, 0.99, "kk=0", va="top", ha="right", **text_font_opts)

    # init function for animation
    def init_fn() -> list[plt.Artist]:
        return [agent_col, edge_col, cost_text, kk_text]

    # update function for animation
    def update(kk: int) -> list[plt.Artist]:
        graph = tree_index(T_graph, kk)
        n_pos_t = graph.states[:-1, :dim]

        # update agent positions
        if dim == 2:
            for ii in range(n_agent * 2):
                if ii < n_agent:
                    # --- Agent (circle): update the center ---
                    agent_list[ii].set_center(tuple(n_pos_t[ii]))
                else:
                    # --- Goal (square): update the bottom-left corner ---
                    # Subtract r to keep it centered on the goal position.
                    new_x = n_pos_t[ii, 0] - r
                    new_y = n_pos_t[ii, 1] - r
                    agent_list[ii].set_xy((new_x, new_y))
        else:
            agent_col.set_offsets(n_pos_t[:n_agent * 2, :2])
            agent_col.set_3d_properties(n_pos_t[:n_agent * 2, 2], zdir='z')

        # update edges
        e_edge_index_t = np.stack([graph.senders, graph.receivers], axis=0)
        is_pad_t = np.any(e_edge_index_t == n_agent * 2 + n_hits, axis=0)
        e_edge_index_t = e_edge_index_t[:, ~is_pad_t]
        e_start_t, e_end_t = n_pos_t[e_edge_index_t[0, :]], n_pos_t[e_edge_index_t[1, :]]
        e_is_goal_t = (n_agent <= graph.senders) & (graph.senders < n_agent * 2)
        e_is_goal_t = e_is_goal_t[~is_pad_t]
        e_colors_t = [edge_goal_color if e_is_goal_t[ii] else "0.2" for ii in range(len(e_start_t))]
        e_lines_t = np.stack([e_start_t, e_end_t], axis=1)
        edge_col.set_segments(e_lines_t)
        edge_col.set_colors(e_colors_t)

        # update cost and safe labels
        if kk < len(rollout.T_cost):
            dist2tgt = rollout.Tp1_graph.env_states.dist2tgt[kk][0]
            min_dist2obs = min(rollout.Tp1_graph.env_states.min_dist2obs[kk])
            cost_text.set_text("dist2obs: {:5.4f}, dist2tgt: {:5.4f}, Reward: {:5.4f}".format(
                min_dist2obs, dist2tgt, rollout.T_reward[kk]))
        else:
            cost_text.set_text("")

        kk_text.set_text("kk={:04}".format(kk))

        return [agent_col, edge_col, cost_text, kk_text]

    fps = 30.0
    spf = 1 / fps
    mspf = 1_000 * spf
    anim_T = len(T_graph.n_node)
    ani = FuncAnimation(fig, update, frames=anim_T, init_func=init_fn, interval=mspf, blit=True)
    save_anim(ani, video_path)


def render_trajectory(
        rollout: RolloutResult,
        save_path: pathlib.Path,
        side_length: float,
        dim: int,
        n_agent: int,
        r: float,
        dt: float = 0.03,
        dpi: int = 150,
        **kwargs
):
    """
    Render a continuous static trajectory plot: time-graded color lines,
    circular start markers, and square goal markers.
    """
    if dim != 2:
        raise NotImplementedError("The static trajectory plot only supports 2D environments")

    # 1. Initialize the canvas
    fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=dpi)
    ax.set_xlim(0., side_length)
    ax.set_ylim(0., side_length)
    ax.set_aspect("equal")

    # 2. Collect positions for every time step
    T_graph = rollout.Tp1_graph
    total_steps = len(T_graph.n_node)

    all_positions = []
    for kk in range(total_steps):
        graph = tree_index(T_graph, kk)
        pos = graph.states[:n_agent, :dim]
        all_positions.append(pos)

    # shape: (Time, n_agent, 2)
    all_positions = np.array(all_positions)

    # 3. Draw obstacles
    graph0 = tree_index(T_graph, 0)
    obs = graph0.env_states.obstacle
    try:
        obs_col = get_obs_collection(obs, color="k", alpha=1.0)
        obs_col.set_facecolor("#666666")
        obs_col.set_edgecolor('#202020')
        obs_col.set_linewidth(1)
        obs_col.set_alpha(0.8)
        obs_col.set_zorder(10)  # obstacle z-order
        ax.add_collection(obs_col)
    except:
        pass

    # 4. Draw continuous trajectories (the main part)

    # Define the color map (e.g. 'viridis', 'plasma', 'coolwarm')
    cmap = plt.get_cmap('viridis')
    # Normalizer used to map time to color
    norm = Normalize(vmin=0, vmax=total_steps * dt)

    lc = None  # handle kept around for the colorbar

    special_blue = '#007bff'

    for i in range(n_agent):
        # Full trajectory for this agent, shape (T, 2).
        # Use the full data (no subsampling) to keep the line smooth.
        full_traj = all_positions[:, i, :]

        points = full_traj.reshape(-1, 1, 2)
        # segments shape: (T-1, 2, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # --- Create the LineCollection ---
        lc = LineCollection(segments, cmap=cmap, norm=norm)

        # Set the scalar value of each segment (we use time here).
        # The time array length should match the number of segments (T-1).
        segment_times = np.arange(len(segments)) * dt
        lc.set_array(segment_times)

        # Line styling
        lc.set_linewidth(2.0)
        lc.set_alpha(0.9)
        lc.set_zorder(5)
        ax.add_collection(lc)

        # --- Start and end markers ---
        # Start (circle)
        ax.scatter(full_traj[0, 0], full_traj[0, 1], c=special_blue, s=60, marker='o',
                   edgecolors='white', linewidth=1.5, zorder=20, label='Start' if i == 0 else "")

        # End (square)
        ax.scatter(full_traj[-1, 0], full_traj[-1, 1], c=special_blue, s=80, marker='s',
                   edgecolors='white', linewidth=1.5, zorder=20, label='Goal' if i == 0 else "")

    # 5. Add the colorbar
    # Use the last LineCollection as the mappable
    if lc is not None:
        cbar = plt.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('time (s)', fontsize=12)
        # Configure ticks
        max_time = total_steps * dt
        cbar_ticks = np.linspace(0, max_time, 5)
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels([f"{t:.1f}" for t in cbar_ticks])
        # Tidy up the colorbar (drop internal lines, keep the outer frame)
        cbar.solids.set_edgecolor("face")
        cbar.outline.set_linewidth(1)

    # 6. Save the figure
    plt.tight_layout()
    print(f"Saving continuous trajectory plot to {save_path}")
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', dpi=300)
    plt.close(fig)    