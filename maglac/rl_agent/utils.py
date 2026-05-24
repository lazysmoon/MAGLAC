import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

def jax2np(x):
    return jtu.tree_map(lambda y: np.array(y), x)


def np2jax(x):
    return jtu.tree_map(lambda y: jnp.array(y), x)

