# Model parameters
#import numpy as np

# Type of run
full_run = 1
noise_run = 0
verbose = 1

# Main processes I/O
cratering_on = 1
secondaries_on = 1
diffusion_on = 1
implicit_diffusion = 1
explicit_diffusion = 0

save_grid = 0

# Tracers I/O
tracers_on = 1
pixel_noise_on = 1
periodic_particles = 1
save_trajectories = 0

if full_run:
    resolution = 17.0
    grid_size = 250
    grid_width = resolution*float(grid_size)

elif noise_run:
    resolution = 1.7
    grid_size = 100
    grid_width = resolution*float(grid_size)

# Initial state of the grid.  Flat or load in a cratered highlands surface (coming soon)
#grid_initial = np.zeros((grid_size, grid_size))

dx2 = resolution**2
dy2 = resolution**2

# Time
model_time = 3.5e9	# Total model time in years
if full_run:
    nsteps = 100
    diff_itr_max = 10
elif noise_run:
    nsteps = 500
    diff_itr_max = 50
dt = float(model_time/nsteps)		# Model timestep in years

# Miscellany
min_crater = 2.0*resolution
continuous_ejecta_blanket_factor = 3.0	# Ejecta blanket extends to 3 crater radii
max_secondary_factor = 0.05	# Largest secondary is 5% of primary

min_primary_for_secondaries = (min_crater)/(max_secondary_factor)	# Smallest primary that can produce a resolvable secondary

# Diffusion
diffusivity = []

# Tracer particles
n_particles_per_layer = 25

# Sampling
sampling_depth = 0.1
surface_depth = 0.001

#steps = 1

# Output directory
save_dir = '/extra/pob/Model_17m/'
