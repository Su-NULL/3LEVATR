# Model parameters
import numpy as np
import sys

# Type of run
full_run = 0
noise_run = 1
verbose = 1

# Main processes on/off
cratering_on = 1
secondaries_on = 0
diffusion_on = 1
implicit_diffusion = 1
explicit_diffusion = 0
pixel_noise_on = 0

# Tracers on/off
tracers_on = 0

grid_size = 100
resolution = (17.0/10)
grid_width = grid_size*resolution

dx2 = resolution**2
dy2 = resolution**2

# Miscellany
min_crater = 2.0*resolution
continuous_ejecta_blanket_factor = 3.0	# Ejecta blanket extends to 3 crater radii
max_secondary_factor = 0.05	# Largest secondary is 5% of primary

min_primary_for_secondaries = 2.0*resolution/(max_secondary_factor)	# Smallest primary that can produce a resolvable secondary

diffusivity = []
model_time = []
dt = []
nsteps = []
