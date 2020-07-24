import body

# Input parameters (and their dependencies)
resolution = []
dt = []
diffusivity = []
grid_size = []
dx2 = []
dy2 = []
nsteps = []
diff_scheme = []

# Type of run
verbose = 0

# Main processes I/O
cratering_on = 1
secondaries_on = 1
diffusion_on = 1

# Tracers I/O
tracers_on = 1
periodic_particles = 1

# Output
save_grid = 1
save_movie = 0
save_trajectories = 1
save_craters = 0

# Grid parameters
grid_width = 2000.0

# Time
model_time = 3.48e9	# Total model time in years

# Miscellany

continuous_ejecta_blanket_factor = 5.0	# Ejecta blanket extends to 3 crater radii
max_secondary_factor = 0.04	# Largest secondary is 4% of primary
v_min_sec = 250.0
min_primary_for_secondaries = (2.0/3.0)*(v_min_sec**2)/body.g
#min_primary_for_secondaries = (min_crater)/(max_secondary_factor)	# Smallest primary that can produce a resolvable secondary

# Tracer particles
n_particles_per_layer = 25

# Sampling
sampling_depth = 0.1
surface_depth = 0.001
