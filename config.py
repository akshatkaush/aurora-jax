import jax.numpy as jnp

surf_weights = {
    "2t": 3.5,
    "10u": 0.77,
    "10v": 0.66,
    "msl": 1.6,
}

num_levels = 13
atmos_weights = {
    "z": jnp.full(num_levels, 3.5),
    "q": jnp.full(num_levels, 0.8),
    "t": jnp.full(num_levels, 1.7),
    "u": jnp.full(num_levels, 0.87),
    "v": jnp.full(num_levels, 0.6),
}

gamma = 1.0
alpha = 1.0 / 4
beta = 1.0

weight_decay = 5e-6
