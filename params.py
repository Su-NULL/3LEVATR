# Model parameters
#import numpy as np

# Type of run
full_run = 1
verbose = 1

# Main processes I/O
cratering_on = 1
secondaries_on = 1
diffusion_on = 1
implicit_diffusion = 0
explicit_diffusion = 0
crank_nicolson_diffusion = 1

save_grid = 0
save_movie = 0

# Tracers I/O
tracers_on = 1
pixel_noise_on = 0
periodic_particles = 1
save_trajectories = 0

if full_run:
    resolution = 4.0
    grid_width = 2000.0
    grid_size = int(grid_width/resolution)

# Initial state of the grid.  Flat or load in a cratered highlands surface (coming soon)
#grid_initial = np.zeros((grid_size, grid_size))

dx2 = resolution**2
dy2 = resolution**2

# Time
model_time = 3.48e9	# Total model time in years
dt = 1.e6		# Model timestep in years
nsteps = int(model_time/dt)

# Miscellany
min_crater = 2.0*resolution
continuous_ejecta_blanket_factor = 3.0	# Ejecta blanket extends to 3 crater radii
max_secondary_factor = 0.04	# Largest secondary is 4% of primary

min_primary_for_secondaries = (min_crater)/(max_secondary_factor)	# Smallest primary that can produce a resolvable secondary

# Diffusion
diffusivity = []

# Tracer particles
n_particles_per_layer = 25

# Sampling
sampling_depth = 0.1
surface_depth = 0.001

#nsteps = 10

# Output directory
save_dir = './output/'
