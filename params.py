# Model parameters
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import sys

# Type of run
full_run = 1
noise_run = 0
verbose = 1

# Main processes on/off
cratering_on = 1
secondaries_on = 1
diffusion_on = 1
implicit_diffusion = 1
explicit_diffusion = 0
pixel_noise_on = 0

# Tracers on/off
tracers_on = 0

if full_run:
	resolution = 17.0
	grid_size = 250
	grid_width = resolution*float(grid_size)

elif noise_run:
	grid_width = 17.0
	grid_size = 100
	resolution = grid_width/grid_size

# Initial state of the grid.  Flat or load in a cratered highlands surface (coming soon)
#grid_initial = np.zeros((grid_size, grid_size))

dx2 = resolution**2
dy2 = resolution**2

# Time
model_time = 3.5e9	# Total model time in years
nsteps = 100
dt = float(model_time/nsteps)		# Model timestep in years

# If this is a noise run, set the total model time to the timestep you will be using in the full-scale model
if noise_run:
	model_time = dt
	dt = model_time/nsteps	# New timestep, keeping the total number of timesteps as the full-scale model

# Miscellany
min_crater = 2.0*resolution
continuous_ejecta_blanket_factor = 3.0	# Ejecta blanket extends to 3 crater radii
max_secondary_factor = 0.05	# Largest secondary is 5% of primary

min_primary_for_secondaries = min_crater/(max_secondary_factor)	# Smallest primary that can produce a resolvable secondary

# Diffusion
diffusivity = []

# Tracer particles
n_particles_per_layer = 25

# Sampling
sampling_depth = 0.1
surface_depth = 0.001

#nsteps = 5