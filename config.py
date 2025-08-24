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

REPLAY_BUFFER_CAPACITY = 200
DATASET_SAMPLING_PERIOD = 10
LEAD_TIME_SCHEDULE_STEP_THRESHOLD = 5000

HOURS_PER_STEP = 6
INITIAL_LEAD_TIME_LIMIT_DAYS = 4
FULL_LEAD_TIME_LIMIT_DAYS = 10

INITIAL_LEAD_TIME_LIMIT_STEPS = INITIAL_LEAD_TIME_LIMIT_DAYS * (24 // HOURS_PER_STEP)
FULL_LEAD_TIME_LIMIT_STEPS = FULL_LEAD_TIME_LIMIT_DAYS * (24 // HOURS_PER_STEP)

ZARR_PATH = "/home1/a/akaush/aurora/hresDataset/hres_t0_2021-2022mid.zarr"
STATIC_DATA_PATH = "/home1/a/akaush/aurora/datasetEnviousScratch/static.nc"

# --- Time Constants ---
SECONDS_PER_HOUR = 3600
TIMESTEP_DURATION_SECONDS = HOURS_PER_STEP * SECONDS_PER_HOUR

# --- Model Specific (Optional placeholder) ---
# MODEL_PATCH_SIZE = 64
