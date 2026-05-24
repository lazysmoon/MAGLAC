import pathlib
import jax.lax as lax
import einops as ei
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.collections as mcollections
import numpy as np
from datetime import timedelta
from typing import Any, Callable, Iterable, ParamSpec, Sequence, TypeVar, List, NamedTuple
from jax import numpy as jnp, tree_util as jtu
from jax._src.lib import xla_client as xc
from matplotlib.animation import FuncAnimation
from rich.progress import Progress, ProgressColumn
from rich.text import Text
import socket
import optax
from typing import Callable
from matplotlib.colors import CenteredNorm



def has_nan(x):
    return jtu.tree_map(lambda y: jnp.isnan(y).any(), x)


def has_any_nan(x):
    return jnp.array(jtu.tree_flatten(has_nan(x))[0]).any()


def compute_norm(grad):
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jtu.tree_leaves(grad)))


def compute_norm_and_clip(grad, max_norm: float):
    g_norm = compute_norm(grad)
    clipped_g_norm = jnp.maximum(max_norm, g_norm)
    clipped_grad = jtu.tree_map(lambda t: (t / clipped_g_norm) * max_norm, grad)

    return clipped_grad, g_norm


def tree_copy(tree):
    return jtu.tree_map(lambda x: x.copy(), tree)


def empty_grad_tx() -> optax.GradientTransformation:
    def init_fn(params):
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        return None, None

    return optax.GradientTransformation(init_fn, update_fn)


def jax2np(x):
    return jtu.tree_map(lambda y: np.array(y), x)


def np2jax(x):
    return jtu.tree_map(lambda y: jnp.array(y), x)


def centered_norm(vmin: float | list[float], vmax: float | list[float]):
    if isinstance(vmin, list):
        vmin = min(vmin)
    if isinstance(vmax, list):
        vmin = max(vmax)
    halfrange = max(abs(vmin), abs(vmax))
    return CenteredNorm(0, halfrange)

def is_connected():
    try:
        sock = socket.create_connection(("www.google.com", 80), timeout=3)
        if sock is not None:
            sock.close()
        return True
    except OSError:
        pass
    print('No internet connection')
    return False

def merge01(x):
    return ei.rearrange(x, "n1 n2 ... -> (n1 n2) ...")


_P = ParamSpec("_P")
_R = TypeVar("_R")
_Fn = Callable[_P, _R]

_PyTree = TypeVar("_PyTree")


def jax_vmap(fn: _Fn, in_axes: int | Sequence[Any] = 0, out_axes: Any = 0) -> _Fn:
    return jax.vmap(fn, in_axes, out_axes)


def concat_at_front(arr1: jnp.ndarray, arr2: jnp.ndarray, axis: int) -> jnp.ndarray:
    """
    :param arr1: (nx, )
    :param arr2: (T, nx)
    :param axis: Which axis for arr2 to concat under.
    :return: (T + 1, nx) with [arr1 arr2]
    """
    # The shapes of arr1 and arr2 should be the same without the dim at axis for arr1.
    arr2_shape = list(arr2.shape)
    del arr2_shape[axis]
    assert np.all(np.array(arr2_shape) == np.array(arr1.shape))

    if isinstance(arr1, np.ndarray):
        return np.concatenate([np.expand_dims(arr1, axis=axis), arr2], axis=axis)
    else:
        return jnp.concatenate([jnp.expand_dims(arr1, axis=axis), arr2], axis=axis)


def tree_concat_at_front(tree1: _PyTree, tree2: _PyTree, axis: int) -> _PyTree:
    def tree_concat_at_front_inner(arr1: jnp.ndarray, arr2: jnp.ndarray):
        return concat_at_front(arr1, arr2, axis=axis)

    return jtu.tree_map(tree_concat_at_front_inner, tree1, tree2)


def tree_index(tree: _PyTree, idx: int) -> _PyTree:
    return jtu.tree_map(lambda x: x[idx], tree)


def jax2np(pytree: _PyTree) -> _PyTree:
    return jtu.tree_map(np.array, pytree)


def np2jax(pytree: _PyTree) -> _PyTree:
    return jtu.tree_map(jnp.array, pytree)


def mask2index(mask: jnp.ndarray, n_true: int) -> jnp.ndarray:
    idx = lax.top_k(mask, n_true)[1]
    return idx


def jax_jit_np(
        fn: _Fn,
        static_argnums: int | Sequence[int] | None = None,
        static_argnames: str | Iterable[str] | None = None,
        donate_argnums: int | Sequence[int] = (),
        device: xc.Device = None,
        *args,
        **kwargs,
) -> _Fn:
    jit_fn = jax.jit(fn, static_argnums, static_argnames, donate_argnums, device, *args, **kwargs)

    def wrapper(*args, **kwargs) -> _R:
        return jax2np(jit_fn(*args, **kwargs))

    return wrapper


def chunk_vmap(fn: _Fn, chunks: int) -> _Fn:
    fn_jit_vmap = jax_jit_np(jax.vmap(fn))

    def wrapper(*args) -> _R:
        args = list(args)
        # 1: Get the batch size.
        batch_size = len(jtu.tree_leaves(args[0])[0])
        chunk_idxs = np.array_split(np.arange(batch_size), chunks)

        out = []
        for idxs in chunk_idxs:
            chunk_input = jtu.tree_map(lambda x: x[idxs], args)
            out.append(fn_jit_vmap(*chunk_input))
        
        # 2: Concatenate the output.
        out = tree_merge(out)
        return out

    return wrapper

class MutablePatchCollection(mcollections.PatchCollection):
    def __init__(self, patches, *args, **kwargs):
        self._paths = None
        self.patches = patches
        mcollections.PatchCollection.__init__(self, patches, *args, **kwargs)

    def get_paths(self):
        self.set_paths(self.patches)
        return self._paths


class CustomTimeElapsedColumn(ProgressColumn):
    """Renders time elapsed."""

    def render(self, task: "Task") -> Text:
        """Show time elapsed."""
        elapsed = task.finished_time if task.finished else task.elapsed
        if elapsed is None:
            return Text("-:--:--", style="progress.elapsed")
        delta = timedelta(seconds=elapsed)
        delta = timedelta(seconds=delta.seconds, milliseconds=round(delta.microseconds // 1000))
        delta_str = str(delta)
        return Text(delta_str, style="progress.elapsed")


def save_anim(ani: FuncAnimation, path: pathlib.Path):
    pbar = Progress(*Progress.get_default_columns(), CustomTimeElapsedColumn())
    pbar.start()
    task = pbar.add_task("Animating", total=ani.save_count)

    def progress_callback(curr_frame: int, total_frames: int):
        pbar.update(task, advance=1)

    ani.save(path, progress_callback=progress_callback)
    pbar.stop()


def tree_merge(data: List[NamedTuple]):
    def body(*x):
        x = list(x)
        if isinstance(x[0], np.ndarray):
            return np.concatenate(x, axis=0)
        else:
            return jnp.concatenate(x, axis=0)
    out = jtu.tree_map(body, *data)
    return out


def tree_stack(trees: list):
    def tree_stack_inner(*arrs):
        arrs = list(arrs)
        if isinstance(arrs[0], np.ndarray):
            return np.stack(arrs, axis=0)
        return np.stack(arrs, axis=0)

    return jtu.tree_map(tree_stack_inner, *trees)
