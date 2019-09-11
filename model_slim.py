######################### ---------------------------------------------------------------------------------------------------------------------------------------- #########################
# Import modules and parameter files (body, params)
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import sys
import body
import params_slim as params
from scipy.interpolate import interp1d
from scipy.linalg import solve
from misc_functions import progress
import uuid
import gc
from time import time
import matplotlib.pyplot as plt
import tracemalloc
tracemalloc.start()

#np.random.seed(np.random.randint(10000) + int(sys.argv[2]))
np.random.seed(int(sys.argv[2]))
gc.collect()
######################### ---------------------------------------------------------------------------------------------------------------------------------------- #########################


######################### ---------------------------------------------------------------------------------------------------------------------------------------- #########################
# Grid class
class Grid:
	def __init__(self, grid_size, resolution, diffusivity, dt):
		# Initialize grid parameters
		self.grid_size = grid_size
		self.resolution = resolution
		self.grid_width = grid_size*resolution

		if params.diffusion_on:
			# Initialize parameters for grid diffusion
			alpha_diff = diffusivity*dt/(resolution**2)

			# Set up matrices to solve --> A*u_n+1 = b_n
			N = self.grid_size
			A_interior = np.zeros((N,N))
			A_edge = np.zeros((N,N))

			for i in range(N):
				if i==0:
					A_interior[i,:] = [1.0+3.0*alpha_diff if j==0 else (-alpha_diff) if j==1 else 0 for j in range(N)]
					A_edge[i,:] = [1.0+2.0*alpha_diff if j==0 else (-alpha_diff) if j==1 else 0 for j in range(N)]

				elif i==N-1:
					A_interior[i,:] = [1.0+3.0*alpha_diff if j==N-1 else (-alpha_diff) if j==N-2 else 0 for j in range(N)]
					A_edge[i,:] = [1.0+2.0*alpha_diff if j==N-1 else (-alpha_diff) if j==N-2 else 0 for j in range(N)]

				else:
					A_interior[i,:] = [(-alpha_diff) if j==i-1 or j==i+1 else 1.0+4.0*alpha_diff if j==i else 0 for j in range(N)]
					A_edge[i,:] = [(-alpha_diff) if j==i-1 or j==i+1 else 1.0+3.0*alpha_diff if j==i else 0 for j in range(N)]

			self.alpha_diff = alpha_diff
			self.A_interior = A_interior
			self.A_edge = A_edge

	def setUpGrid(self):
		return np.zeros((self.grid_size, self.grid_size))

	def implicit_diffusion2D(self, grid_old):
		alpha_diff = self.alpha_diff
		A_interior = self.A_interior
		A_edge = self.A_edge
		N = self.grid_size

		itr_max = 5
		grid_inter = np.copy(grid_old)

		itr = 0
		while itr < itr_max:
			#print(grid_inter[N/2, N/2])
			# Do all rows
			for j in range(N):
				if j == 0:    # Top row
					u_row = grid_old[:,j]

					u_row_below = grid_inter[:,j+1]

					grid_inter[:,j] = solve(A_edge, u_row + alpha_diff*u_row_below)

				elif j == N-1:    # Bottom rows
					u_row = grid_old[:,j]

					u_row_above = grid_inter[:,j-1]

					grid_inter[:,j] = solve(A_edge, u_row + alpha_diff*u_row_above)

				else:    # All other rows
					u_row = grid_old[:,j]

					u_row_above = grid_inter[:, j-1]
					u_row_below = grid_inter[:, j+1]

					grid_inter[:,j] = solve(A_interior, u_row + alpha_diff*u_row_above + alpha_diff*u_row_below)

			# Do all columns
			for i in range(N):
				if i == 0:    # Left column
					u_col = grid_old[i,:]

					u_col_right = grid_inter[i+1,:]

					grid_inter[i,:] = solve(A_edge, u_col + alpha_diff*u_col_right)

				elif i == N-1:    # Right column
					u_col = grid_old[i,:]

					u_col_left = grid_inter[i-1,:]

					grid_inter[i,:] = solve(A_edge, u_col + alpha_diff*u_col_left)

				else:    # All other columns
					u_col = grid_old[i,:]

					u_col_left = grid_inter[i-1,:]
					u_col_right = grid_inter[i+1,:]

					grid_inter[i,:] = solve(A_interior, u_col + alpha_diff*u_col_left + alpha_diff*u_col_right)

			itr += 1

		return np.copy(grid_inter)

	def calc_scale_factor(self, diameter, res, primary_index, continuous_ejecta_blanket_factor, X_grid, Y_grid, ones_grid, grid_size):

		grid_size_scale = int(3.0*diameter/res)
		if grid_size_scale > grid_size:
			grid_size = grid_size_scale
			X_grid, Y_grid = np.ogrid[:grid_size, :grid_size]

		x_crater = int(grid_size/2.0)
		y_crater = int(grid_size/2.0)

		radius = diameter/2.0

		depth = (269.0/81.0)*(0.04*diameter)*primary_index
		rim_height = 0.04*diameter*primary_index

		r_ejecta = continuous_ejecta_blanket_factor*radius

		dist_from_center = np.hypot(abs(X_grid - x_crater)*res, abs(Y_grid - y_crater)*res)

		# Grid pixels covered by the crater
		crater_mask = dist_from_center <= radius

		# Grid pixels covered by the ejecta blanket
		ejecta_mask = (dist_from_center > radius) & (dist_from_center <= r_ejecta)

		# Crater elevation profile
		delta_H_crater = (((dist_from_center/radius)**2)*(rim_height + depth)) - depth

		# Ejecta elevation profile
		with np.errstate(divide='ignore'):    # Divide by zero at r=0 but we don't care about that point since it's interior to the ejecta blanket
			delta_H_ejecta = rim_height*((dist_from_center/radius)**-3) - (rim_height/54.0)*((dist_from_center/radius) - 1.0)

		crater_grid = np.zeros((grid_size, grid_size))
		crater_grid[crater_mask] = delta_H_crater[crater_mask]
		crater_grid[ejecta_mask] = delta_H_ejecta[ejecta_mask]

		scaling_factor = np.sum(crater_grid[np.where(crater_grid >= 0.0)])/(np.abs(np.sum(crater_grid[np.where(crater_grid < 0.0)])))

		return scaling_factor

	def add_crater(self, grid, x_center, y_center, diameter, res, primary_index, continuous_ejecta_blanket_factor, X_grid, Y_grid, ones_grid):

		current_grid = np.copy(grid)
		grid_size = int(current_grid.shape[0])

		radius = diameter/2.0

		depth = (269.0/81.0)*(0.04*diameter)*primary_index
		rim_height = 0.04*diameter*primary_index

		r_ejecta = continuous_ejecta_blanket_factor*radius

		dist_from_center = np.hypot(abs(X_grid - x_center)*res, abs(Y_grid - y_center)*res)

		# Grid pixels covered by the crater
		crater_mask = dist_from_center <= radius

		# Grid pixels covered by the ejecta blanket
		ejecta_mask = (dist_from_center > radius) & (dist_from_center <= r_ejecta)

		full_crater_mask = dist_from_center <= r_ejecta

		if np.sum(crater_mask) == 0 and np.sum(ejecta_mask) == 0:
			# Crater does not actually overlap the grid. These cases are included for completeness, simply return the pre-crater grid
			ret_grid = np.copy(current_grid)

		else:
			# Crater somewhat overlaps the grid
			# Inheritance parameter, set as a constant, see Howard 2007, I=0 --> crater rim horizontal, I=1 --> crater rim parallel to pre-existing slope
			I_i = 0.9

			# Reference elevation is the average of the pre-crater grid within the crater area
			# Creter interior weighted by 1, ejecta blanket weighted by (distance/radius)**-n, n=3
			weights = np.copy(ones_grid)
			outside_mask = dist_from_center > radius

			weights[outside_mask] = (dist_from_center[outside_mask]/radius)**(-3)
			weighted_grid = current_grid*weights

			E_r = np.average(weighted_grid[full_crater_mask])

			# Crater elevation profile
			delta_H_crater = (((dist_from_center/radius)**2)*(rim_height + depth)) - depth

			# Ejecta elevation profile
			with np.errstate(divide='ignore'):    # Divide by zero at r=0 but we don't care about that point since it's interior to the ejecta blanket
				delta_H_ejecta = rim_height*((dist_from_center/radius)**-3) - (rim_height/54.0)*((dist_from_center/radius) - 1.0)

			# Inheritance matrices - determines how the crater is integrated into the existing grid
			G_grid = (1.0 - I_i)*ones_grid
			min_mask = G_grid > delta_H_ejecta/rim_height
			G_grid[min_mask] = delta_H_ejecta[min_mask]/rim_height

			crater_inh_profile = (E_r - current_grid)*(1.0 - (I_i*(dist_from_center/radius)**2))
			ejecta_inh_profile = G_grid*(E_r - current_grid)

			delta_E_crater = delta_H_crater + crater_inh_profile
			delta_E_ejecta = delta_H_ejecta + ejecta_inh_profile

			# Add calculated elevations to the grid at the corresponding pixels
			crater_grid = np.zeros((grid_size, grid_size))
			crater_grid[crater_mask] = delta_E_crater[crater_mask]
			crater_grid[ejecta_mask] = delta_E_ejecta[ejecta_mask]

			if diameter <= (params.grid_width/3.0):
				scaling_factor = self.calc_scale_factor(diameter, res, primary_index, continuous_ejecta_blanket_factor, X_grid, Y_grid, ones_grid, grid_size)
			else:
				scaling_factor = 1.0

			crater_grid[np.where(crater_grid < 0.0)] *= scaling_factor

			current_grid += crater_grid

			ret_grid = np.copy(current_grid)

		return ret_grid

######################### ---------------------------------------------------------------------------------------------------------------------------------------- #########################


######################### ---------------------------------------------------------------------------------------------------------------------------------------- #########################
# Impactor population class
class ImpactorPopulation:
	def __init__(self):
		pass

	def calc_min_impactor(self, min_crater, v_i, theta_i):
		# Calculates the minimum impactor diameter needed to make a resolvable crater under the most
		# favorable impact conditions, e.g. head-on, maximum velocity impact into cohesive regolith
		# To solve the H&H (2007) equation simply for d_i, must use the pure strength regime

		# Returns the minimum impactor diameter in km*****

		# d_i = impactor diameter in m
		# v_i = impactor velocity in m/s
		# target = target body (moon, ceres, mercury)
		# delta_i = impactor density, 2.7 g/cm^3 from Marchi et al. 2009
		# theta_i = impact angle
		g = body.g
		delta_i = body.impactor_density
		rho_t = body.regolith_density
		R_min = min_crater/2.0

		nu = 0.4

		# Values for cohesive soils
		k = 1.03
		mu = 0.41

		v_perp = v_i*np.cos(theta_i)

		Y_bar = body.regolith_strength
		a_min = (R_min/k)*((Y_bar/(rho_t*(v_perp**2)))**((2.0 + mu)/2.0)*(rho_t/delta_i)**(nu*(2.0+mu)/mu))**(mu/(2.0+mu))

		return (2.0*a_min)/1000.0

	def pi_group_scale_small(self, d_i, v_i, theta_i, frac_depth):
		# d_i = impactor diameter in m
		# v_i = impactor velocity in m/s
		# target = target body (moon, ceres, mercury)
		# delta_i = impactor density, 2.7 g/cm^3 from Marchi et al. 2009
		# theta_i = impact angle

		# Returns final crater diameter in meters*****

		g = body.g
		delta_i = body.impactor_density
		rho_surf = body.regolith_density
		rho_depth = body.bedrock_density
		Y_surf = body.regolith_strength
		Y_depth = body.bedrock_strength

		a_i = d_i/2.0

		if frac_depth == 0.0:
			Y = Y_depth
			rho_t = rho_depth
		elif 10.0*a_i <= frac_depth:
			Y = ((Y_depth - Y_surf)/frac_depth)*(5.0*a_i) + 0.0
			rho_t = ((rho_depth - rho_surf)/frac_depth)*(5.0*a_i) + rho_surf
		elif 10.0*a_i > frac_depth:
			Y = ((Y_depth + Y_surf)/2.0)*(frac_depth/(10.0*a_i)) + Y_depth*(10.0*a_i - frac_depth)/(10.0*a_i)
			rho_t = ((rho_depth + rho_surf)/2.0)*(frac_depth/(10.0*a_i)) + rho_depth*(10.0*a_i - frac_depth)/(10.0*a_i)

		# Values for cohesive soils
		k = 1.03
		mu = 0.41

		nu = 0.4

		v_perp = v_i*np.cos(theta_i)

		term1 = g*a_i/(v_perp**2)
		term2 = (rho_t/delta_i)**(2.0*nu/mu)

		term3 = (Y/(rho_t*v_perp**2))**((2.0+mu)/2.0)
		term4 = (rho_t/delta_i)**(nu*(2.0+mu)/mu)

		crater_radius = k*a_i*(( term1*term2 + term3*term4)**(-mu/(2.0+mu)))

		return 2.0*crater_radius

	def pi_group_scale_large(self, d_i, v_i, theta_i, frac_depth):
		# d_i = impactor diameter in m
		# v_i = impactor velocity in m/s
		# target = target body (moon, ceres, mercury)
		# delta_i = impactor density, 2.7 g/cm^3 from Marchi et al. 2009
		# theta_i = impact angle

		# Returns final crater diameter in meters*****

		g = body.g
		delta_i = body.impactor_density
		rho_surf = body.regolith_density
		rho_depth = body.bedrock_density
		Y_surf = body.regolith_strength
		Y_depth = body.bedrock_strength

		a_i = d_i/2.0

		if frac_depth == 0.0:
			Y = Y_depth
			rho_t = rho_depth
		elif 10.0*a_i <= frac_depth:
			Y = ((Y_depth - Y_surf)/frac_depth)*(5.0*a_i) + 0.0
			rho_t = ((rho_depth - rho_surf)/frac_depth)*(5.0*a_i) + rho_surf
		elif 10.0*a_i > frac_depth:
			Y = ((Y_depth + Y_surf)/2.0)*(frac_depth/(10.0*a_i)) + Y_depth*(10.0*a_i - frac_depth)/(10.0*a_i)
			rho_t = ((rho_depth + rho_surf)/2.0)*(frac_depth/(10.0*a_i)) + rho_depth*(10.0*a_i - frac_depth)/(10.0*a_i)

		# Values for rock
		k = 0.93
		mu = 0.55

		nu = 0.4

		v_perp = v_i*np.cos(theta_i)

		term1 = g*a_i/(v_perp**2)
		term2 = (rho_t/delta_i)**(2.0*nu/mu)

		term3 = (Y/(rho_t*v_perp**2))**((2.0+mu)/2.0)
		term4 = (rho_t/delta_i)**(nu*(2.0+mu)/mu)

		crater_radius = k*a_i*(( term1*term2 + term3*term4)**(-mu/(2.0+mu)))

		return 2.0*crater_radius

	def sample_impact_velocities(self, n_samples=1):
		# Inverse transform sampling for impact velocity distribution
		vels = body.velocities
		cum_prob_vels = body.velocity_cumulative_probability
		d_vel = body.velocity_delta

		R_prob = np.random.uniform(0.0, 1.0, n_samples)

		gen_vels = [vels[np.argwhere(cum_prob_vels == np.min(cum_prob_vels[(cum_prob_vels - r) > 0]))] + np.random.uniform(-d_vel, d_vel, size=1) for r in R_prob]
		gen_vels = [val[0][0] for val in gen_vels]

		return np.array(gen_vels)

	def sample_impact_angles(self, n_samples=1):
		# Inverse transform sampling for impact angle distribution
		angs = body.impact_angles
		cum_prob_angs = body.angle_cumulative_probability
		d_ang = body.angle_delta

		R_prob = np.random.uniform(0.0, 1.0, n_samples)

		gen_angs = [angs[np.argwhere(cum_prob_angs == np.min(cum_prob_angs[(cum_prob_angs - r) > 0]))] + np.random.uniform(-d_ang, d_ang, size=1) for r in R_prob]
		gen_angs = [val[0][0] for val in gen_angs]

		return np.array(gen_angs)

	def D_max(self, r, D_c, r_body):
		# Beta is the power law exponent of the ejected fragment size-velocity distribution
		#f_beta = np.poly1d([0.92701526, 0.28685168])
		f_beta = np.poly1d([0.75392577, 0.77708292])

		beta = f_beta(np.log10(D_c/1000.0))

		if beta < 0.555:
			beta = 0.555

		gamma = 0.05		# Very largest primary is 10% of primary diameter (Bierhaus et al 2018)

		R_c = D_c/2.0
		R_N = 3.0*R_c

		numerator = (np.tan(R_N/(2.0*r_body)) + 1.0)*np.tan(r/(2.0*r_body))
		denominator = (np.tan(r/(2.0*r_body)) + 1.0)*np.tan(R_N/(2.0*r_body))

		return (gamma*D_c)*( (numerator/denominator)**((0.554 - beta)/(2.554)))

	def secondaries(self, r, D_p, D_min, D_max, r_body, grid_area, max_dist_for_sec, num_annuli_imps):
		# r is distance between primary and center of grid, in meters
		# D_p is primary crater diameter in meters
		# R is radius of the Moon in meters
		# resolution is pixel resolution in meters/pixels
		# grid_area is total area of the grid in meters^2

		R_p = D_p/2.0
		b_cum = 4.0

		annulus_area = (np.pi*((r + max_dist_for_sec)**2 - (r - max_dist_for_sec)**2))/(1.e6)		# Area of annulus containing the grid in km^2
		c_sfd = (D_max**b_cum)/annulus_area

		# Define diameter bins with sqrt(2) spacing between D_min and D_max
		diam_bins = [D_min]
		while diam_bins[-1] < D_max:
			diam_bins.append(np.sqrt(2.0)*diam_bins[-1])
		diam_bins = np.array(diam_bins)

		# Calculate the incremental number of craters in each diameter bin using the cumulative SFD
		inc_num = np.array([c_sfd*(diam_bins[i]**(-b_cum) - diam_bins[i+1]**(-b_cum)) for i in range(len(diam_bins)-1)])

		# Poisson lambdas for each bin by multiplying by the grid area in km^2
		secondary_lams = inc_num*(grid_area/(1.e6))*num_annuli_imps
		secondary_lams[secondary_lams < 0] = 0

		# Make diameter bins the geometric average of each bin
		diam_bins = diam_bins[0:-1]*(2.0**(1.0/4.0))

		# Poisson sampling of the number of secondaries in each secondary crater diameter bin
		num_secs_on_grid_arr = np.random.poisson(lam=secondary_lams, size=len(secondary_lams))

		# Add a random amount to each bin so diameters are uniformly spaced within the bin (instead of all at the geometric average)
		#secondary_diameters = np.concatenate([diam_bins[i]*np.ones(num_secs_on_grid_arr[i]) + np.random.rand(num_secs_on_grid_arr[i])*(geom_factor*diam_bins[i] - diam_bins[i]) for i in range(len(diam_bins))])
		# For speedup, just use the geometric average
		#secondary_diameters =  np.concatenate([diam_bins[i]*np.ones(num_secs_on_grid_arr[i], dtype=np.float) for i in range(len(diam_bins))])

		return np.concatenate([diam_bins[i]*np.ones(num_secs_on_grid_arr[i], dtype=np.float) for i in range(len(diam_bins))])

	def grid_secondaries(self, r, D_p, D_min, D_max, r_body, grid_area, max_dist_for_sec, continuous_ejecta_blanket_factor):
		# r is distance between primary and center of grid, in meters
		# D_p is primary crater diameter in meters
		# R is radius of the Moon in meters
		# resolution is pixel resolution in meters/pixels
		# grid_area is total area of the grid in meters^2

		R_p = D_p/2.0
		b_cum = 4.0

		annulus_area = (np.pi*((continuous_ejecta_blanket_factor*R_p + max_dist_for_sec)**2 - (continuous_ejecta_blanket_factor*R_p)**2))/(1.e6)		# Area of annulus containing the grid in km^2
		c_sfd = (D_max**b_cum)/annulus_area

		# Define diameter bins with sqrt(2) spacing between D_min and D_max
		diam_bins = [D_min]
		while diam_bins[-1] < D_max:
			diam_bins.append(np.sqrt(2.0)*diam_bins[-1])
		diam_bins = np.array(diam_bins)

		# Calculate the incremental number of craters in each diameter bin using the cumulative SFD
		inc_num = np.array([c_sfd*diam_bins[i]**(-b_cum) - c_sfd*diam_bins[i+1]**(-b_cum) for i in range(len(diam_bins)-1)])

		# Poisson lambdas for each bin by multiplying by the grid area in km^2
		secondary_lams = inc_num*(grid_area/(1.e6))
		secondary_lams[secondary_lams < 0] = 0.0

		# Make diameter bins the geometric average of each bin
		diam_bins = diam_bins[0:-1]*(2.0**(1.0/4.0))

		# Poisson sampling of the number of secondaries in each secondary crater diameter bin
		num_secs_on_grid_arr = np.random.poisson(lam=secondary_lams, size=len(secondary_lams))

		# Add a random amount to each bin so diameters are uniformly spaced within the bin (instead of all at one of the edges or the geometric average)
		#secondary_diameters = np.concatenate([diam_bins[i]*np.ones(num_secs_on_grid_arr[i]) + np.random.rand(num_secs_on_grid_arr[i])*(geom_factor*diam_bins[i] - diam_bins[i]) for i in range(len(diam_bins))])
		# For speedup, just use the geometric average
		#secondary_diameters = np.concatenate([diam_bins[i]*np.ones(num_secs_on_grid_arr[i], dtype=np.float) for i in range(len(diam_bins))])

		return  np.concatenate([diam_bins[i]*np.ones(num_secs_on_grid_arr[i], dtype=np.float) for i in range(len(diam_bins))])

	def sample_timestep_craters(self, t, avg_imp_diam, primary_lams, max_grid_dist, avg_crater_diam, num_inc, dt, min_crater, resolution, grid_size, grid_width, min_primary_for_secondaries, secondaries_on, r_body, continuous_ejecta_blanket_factor, X_grid, Y_grid):

		#np.random.seed(int((time()+t*1000 + int(sys.argv[2]))))

		timestep_diams = []
		timestep_x = []
		timestep_y = []
		timestep_primary_index = []
		timestep_time = []

		current_time = t*dt
		regolith_thickness = (body.model_avg_regolith_thickness/(body.model_avg_age**(0.5)))*(current_time**(0.5))
		frac_depth = (body.known_avg_fractured_depth/(body.known_avg_age**(0.5)))*(current_time**(0.5))

		for i in range(len(avg_imp_diam)):
			cur_imp_diam = avg_imp_diam[i]

			##### PRIMARY IMPACTS #####
			num_cur_grid_imps = np.random.poisson(lam=primary_lams[i])

			# Sample velocities and compute crater diameters for all primaries that hit within area of influence
			primary_imp_vels = self.sample_impact_velocities(num_cur_grid_imps) # km/s
			primary_imp_angs = self.sample_impact_angles(num_cur_grid_imps)
			if cur_imp_diam*1000.0 <= (1.0/20.0)*frac_depth:
				primary_crater_diams = self.pi_group_scale_small(cur_imp_diam*1000.0, primary_imp_vels*1000.0, primary_imp_angs, frac_depth)    # m
			else:
				primary_crater_diams = self.pi_group_scale_large(cur_imp_diam*1000.0, primary_imp_vels*1000.0, primary_imp_angs, frac_depth)    # m

			# Downsample to just the craters that end up being larger than two pixels
			min_crater_mask = np.where(primary_crater_diams >= min_crater)
			primary_crater_diams = primary_crater_diams[min_crater_mask]

			if len(primary_crater_diams) > 0:
				# Randomly sample distances and azimuthal angles from the center of the grid
				primary_crater_dists = max_grid_dist[i]*np.sqrt(np.random.uniform(size=len(primary_crater_diams)))
				primary_crater_phis = np.random.uniform(size=len(primary_crater_diams))*2.0*np.pi

				# Calculate x,y position of craters
				primary_crater_x_pix = np.array(((primary_crater_dists*np.cos(primary_crater_phis)/resolution) + grid_size/2.0)).astype('int')
				primary_crater_y_pix = np.array((primary_crater_dists*np.sin(primary_crater_phis)/resolution) + grid_size/2.0).astype('int')

				timestep_diams.append(primary_crater_diams)
				timestep_x.append(primary_crater_x_pix)
				timestep_y.append(primary_crater_y_pix)
				timestep_primary_index.append(np.ones(len(primary_crater_diams)))


				for j in range(len(primary_crater_diams)):
					D_p = primary_crater_diams[j]

					if D_p >= min_primary_for_secondaries and (0.1*D_p/1.17) >= regolith_thickness and secondaries_on:
						# On-grid crater large enough to produce secondaries
						x_crater = primary_crater_x_pix[j]
						y_crater = primary_crater_y_pix[j]

						dx = abs(X_grid - x_crater)*resolution
						dy = abs(Y_grid - y_crater)*resolution

						dist_from_crater_center = np.hypot(dx, dy)
						dist_from_crater_center[dist_from_crater_center <= continuous_ejecta_blanket_factor*(D_p/2.0)] = np.nan


						if np.any(~np.isnan(dist_from_crater_center)):
							avg_dist_from_crater_center = np.nanmean(dist_from_crater_center)
							avg_D_max = self.D_max(avg_dist_from_crater_center, D_p, r_body)


							if avg_D_max >= min_crater:
								max_dist_for_sec = np.sqrt(2.0*grid_width**2) + continuous_ejecta_blanket_factor*(avg_D_max/2.0)

								# Annulus where secondaries can occur.  Inner radius is the edge of the primary's continuous ejecta blanket. Outer edge is the distance at which the largest possible secondary at this distance from the grid center could influence the grid elevation.
								secondary_area = np.pi*(max_dist_for_sec**2 - (continuous_ejecta_blanket_factor*(D_p/2.0))**2)

								# Diameters of secondaries produced by this on-grid primary, sampled from the appropriate cumulative SFD
								secondary_crater_diams = self.grid_secondaries(avg_dist_from_crater_center, D_p, min_crater, avg_D_max, r_body, secondary_area, max_dist_for_sec, continuous_ejecta_blanket_factor)

								if len(secondary_crater_diams) > 0:
									# Randomly sample distances from the primary for these diameters.  Must occur outside the continuous ejecta blanket and inside the maximum secondary distance
									secondary_crater_dists = (continuous_ejecta_blanket_factor*(D_p/2.0)) + (max_dist_for_sec - (continuous_ejecta_blanket_factor*(D_p/2.0)))*np.random.rand(len(secondary_crater_diams))

									# Randomly sample azimuthal angles for secondaries between 0 and 2pi
									secondary_crater_phis = (2.0*np.pi)*np.random.rand(len(secondary_crater_diams))

									# Calculate x,y position of secondaries
									secondary_crater_x_pix = np.array((secondary_crater_dists*np.cos(secondary_crater_phis) + x_crater*resolution)/resolution).astype('int')
									secondary_crater_y_pix = np.array((secondary_crater_dists*np.sin(secondary_crater_phis) + y_crater*resolution)/resolution).astype('int')


									# Remove craters who land too far off the grid to influence elevations
									dist_from_grid_center = np.hypot((secondary_crater_x_pix - grid_size/2), (secondary_crater_y_pix - grid_size/2))*resolution
									max_dist_for_sec = np.sqrt(grid_width**2/2.0) + continuous_ejecta_blanket_factor*(secondary_crater_diams/2.0)

									on_grid_secondaries = (dist_from_grid_center <= max_dist_for_sec)

									secondary_crater_diameters = secondary_crater_diams[on_grid_secondaries]
									secondary_crater_x_pix = secondary_crater_x_pix[on_grid_secondaries]
									secondary_crater_y_pix = secondary_crater_y_pix[on_grid_secondaries]

									secondary_shallowing_factor = 0.5*(avg_dist_from_crater_center/(np.pi*body.radius_body) + 1.0)

									if len(secondary_crater_diameters) > 0:
										timestep_diams.append(secondary_crater_diameters)
										timestep_x.append(secondary_crater_x_pix)
										timestep_y.append(secondary_crater_y_pix)
										timestep_primary_index.append(secondary_shallowing_factor*np.ones(len(secondary_crater_diameters)))


			##### SECONDARY IMPACTS FROM OFF-GRID CRATERS #####
			# Global craters in this diameter bin are large enough to potentially produce resolvable secondaries
			if avg_crater_diam[i] >= min_primary_for_secondaries and (0.1*avg_crater_diam[i]/1.17) >= regolith_thickness and secondaries_on:
				D_p = avg_crater_diam[i]

				# Create annuli and compute surface area and distance from the grid of each annulus
				theta0 = max_grid_dist[i]/r_body		# radians
				theta_max = np.pi

				theta_arr = np.linspace(theta0, theta_max, 101,  endpoint=True, dtype=np.float)		# radians
				d_theta = theta_arr[1] - theta_arr[0]

				# Array of surface area in each annulus
				surface_area_arr = np.array([2.0*np.pi*r_body**2*(np.cos(theta_arr[j-1]) - np.cos(theta_arr[j])) for j in range(1, len(theta_arr))])
				theta_arr = theta_arr[1:]

				avg_dist_arr = r_body*(theta_arr - d_theta)		# m

				# Calculate the maximum secondary diameter that could be produced on the grid from each annulus
				max_sec_diam_arr = self.D_max(avg_dist_arr, D_p, r_body)

				# Sample primary craters of this size in all annuli at the current timestep
				annuli_lams = num_inc[i]*(surface_area_arr/(1.e6))*dt
				num_annuli_imps_arr = np.random.poisson(lam=annuli_lams)


				# Loop through each annulus
				for j in range(len(avg_dist_arr)):
					max_sec_diam = max_sec_diam_arr[j]

					# If maximum secondary diameter at this distance is resolvable, proceed
					if max_sec_diam >= min_crater:
						# Initialize array of secondary diameters from primary impacts of the current size in this annulus
						annulus_secondary_diameters = []

						# Average distance for the current annulus
						avg_dist = avg_dist_arr[j]

						# Surface area of the current annulus in m^2
						surface_area_annulus = surface_area_arr[j]

						# Distance at which the maximum secondary crater at this range could influence the grid
						max_dist_for_sec = np.sqrt(grid_width**2/2.0) + continuous_ejecta_blanket_factor*(max_sec_diam/2.0)

						# Circle defining the largest area for which secondaries at this distance could influence the grid
						max_sec_area = np.pi*max_dist_for_sec**2

						# Number of primary impacts of the current diameter in this annulus at the current timestep
						num_annuli_imps = num_annuli_imps_arr[j]

						if num_annuli_imps > 0:
							annulus_secondary_diameters.append(self.secondaries(avg_dist, D_p, min_crater, max_sec_diam, r_body, max_sec_area, max_dist_for_sec, num_annuli_imps))

						if len(annulus_secondary_diameters) > 0:
							annulus_secondary_diameters = np.concatenate(annulus_secondary_diameters)

							if len(annulus_secondary_diameters) > 0:
								# Randomly sample distances and azimuthal angles from the center of the grid
								sec_crater_dists = max_dist_for_sec*np.sqrt(np.random.uniform(size=len(annulus_secondary_diameters)))

								# Downsample to secondaries which are close enough that they COULD be on grid
								sec_on_grid = np.where(sec_crater_dists <= (np.sqrt(2.0)*grid_width + continuous_ejecta_blanket_factor*annulus_secondary_diameters))

								annulus_secondary_diameters = annulus_secondary_diameters[sec_on_grid]
								sec_crater_dists = sec_crater_dists[sec_on_grid]

								sec_crater_phis = np.random.uniform(size=len(annulus_secondary_diameters))*2.0*np.pi

								sec_dists_from_grid = avg_dist*np.ones(len(annulus_secondary_diameters))

								# Calculate x,y position of craters
								secondary_crater_x_pix = np.array((sec_crater_dists*np.cos(sec_crater_phis)/resolution) + grid_size/2.0).astype('int')
								secondary_crater_y_pix = np.array((-sec_crater_dists*np.sin(sec_crater_phis)/resolution) + grid_size/2.0).astype('int')

								# Remove craters who land too far off the grid to influence elevations
								dist_from_grid_center = np.hypot((secondary_crater_x_pix - grid_size/2), (secondary_crater_y_pix - grid_size/2))*resolution
								max_dist_for_sec = np.sqrt(grid_width**2/2.0) + continuous_ejecta_blanket_factor*(annulus_secondary_diameters/2.0)

								on_grid_secondaries = (dist_from_grid_center <= max_dist_for_sec)

								annulus_secondary_diameters = annulus_secondary_diameters[on_grid_secondaries]
								secondary_crater_x_pix = secondary_crater_x_pix[on_grid_secondaries]
								secondary_crater_y_pix = secondary_crater_y_pix[on_grid_secondaries]
								sec_dists_from_grid = sec_dists_from_grid[on_grid_secondaries]

								secondary_shallowing_factor = 0.5*(avg_dist/(np.pi*body.radius_body) + 1.0)

								if len(annulus_secondary_diameters) > 0:
									timestep_diams.append(annulus_secondary_diameters)
									timestep_x.append(secondary_crater_x_pix)
									timestep_y.append(secondary_crater_y_pix)
									timestep_primary_index.append(secondary_shallowing_factor*np.ones(len(annulus_secondary_diameters)))
			
		if len(timestep_diams) > 0:
			timestep_diams = np.concatenate(np.array(timestep_diams))
			timestep_x = np.concatenate(np.array(timestep_x))
			timestep_y = np.concatenate(np.array(timestep_y))
			timestep_primary_index = np.concatenate(np.array(timestep_primary_index))
			timestep_time = t*np.ones(len(timestep_diams))

		return timestep_diams, timestep_x, timestep_y, timestep_primary_index, timestep_time


	def sample_all_craters(self):

		nsteps = params.nsteps
		grid_width = params.grid_width
		X_grid, Y_grid = np.ogrid[:params.grid_size, :params.grid_size]

		d_min = self.calc_min_impactor(params.min_crater, body.velocity_max*1000.0, 0.0)

		diam_bins = [d_min]
		while diam_bins[-1] < body.diameter_bins_raw[-1]:
			diam_bins.append(np.sqrt(2.0)*diam_bins[-1])

		diam_bins = np.array(diam_bins)
		avg_imp_diam = diam_bins*(2.0**(1.0/4.0))

		# Interpolate diameter bins to get cumulative number in each bin from your SFD
		f = interp1d(np.log10(body.diameter_bins_raw), np.log10(body.cumulative_number_raw), fill_value='extrapolate')
		cum_num = 10**f(np.log10(diam_bins))

		max_crater_diam = np.zeros(len(diam_bins))
		avg_crater_diam = np.zeros(len(diam_bins))
		for i in range(len(diam_bins)):
			if avg_imp_diam[i]*1000.0 <= (1.0/20.0)*body.avg_fractured_depth:
				max_crater_diam[i] = self.pi_group_scale_small(avg_imp_diam[i]*1000.0, body.velocity_max*1000.0, 0.0, body.avg_fractured_depth)
				avg_crater_diam[i] = self.pi_group_scale_small(avg_imp_diam[i]*1000.0, body.velocity_average*1000.0, np.deg2rad(45.0), body.avg_fractured_depth)
			else:
				max_crater_diam[i] = self.pi_group_scale_large(avg_imp_diam[i]*1000.0, body.velocity_max*1000.0, 0.0, body.avg_fractured_depth)
				avg_crater_diam[i] = self.pi_group_scale_large(avg_imp_diam[i]*1000.0, body.velocity_average*1000.0, np.deg2rad(45.0), body.avg_fractured_depth)

		max_crater_radius = max_crater_diam/2.0
		avg_crater_radius = avg_crater_diam/2.0

		max_grid_dist = (np.sqrt(grid_width**2/2.0) + params.continuous_ejecta_blanket_factor*max_crater_radius)#*1.25
		max_grid_area = np.pi*max_grid_dist**2

		num_inc = [cum_num[i] - cum_num[i+1] for i in range(len(diam_bins)-1)]
		diam_bins = diam_bins[0:-1]
		avg_imp_diam = avg_imp_diam[0:-1]

		avg_crater_diam = avg_crater_diam[0:-1]
		avg_crater_radius = avg_crater_radius[0:-1]

		max_grid_dist = max_grid_dist[0:-1]
		max_grid_area = max_grid_area[0:-1]

		primary_lams = num_inc*(max_grid_area/(1.0e6))*params.dt

		d_craters = []
		x_craters = []
		y_craters = []
		index_craters = []
		t_craters = []

		for t in range(nsteps):
			timestep_diams, timestep_x, timestep_y, timestep_primary_index, timestep_time = self.sample_timestep_craters(t, avg_imp_diam, primary_lams, max_grid_dist, avg_crater_diam, num_inc, params.dt, params.min_crater, params.resolution, params.grid_size, params.grid_width, params.min_primary_for_secondaries, params.secondaries_on, body.radius_body, params.continuous_ejecta_blanket_factor, X_grid, Y_grid)

			current, peak =  tracemalloc.get_traced_memory()
			print('Current: {}, Peak: {}'.format(current, peak))

			d_craters.append(timestep_diams)
			x_craters.append(timestep_x)
			y_craters.append(timestep_y)
			index_craters.append(timestep_primary_index)
			t_craters.append(timestep_time)

			gc.collect()
			if params.verbose:
				progress(t,nsteps)

		current, peak =  tracemalloc.get_traced_memory()
		print('Current: {}, Peak: {}'.format(current, peak))
		sys.exit()
		if params.verbose:
			progress(nsteps, nsteps)
			print('')

		d_craters = np.array(np.concatenate(d_craters)).astype('float')
		x_craters = np.array(np.concatenate(x_craters)).astype('int')
		y_craters = np.array(np.concatenate(y_craters)).astype('int')
		index_craters = np.array(np.concatenate(index_craters)).astype('float')
		t_craters = np.array(np.concatenate(t_craters)).astype('int')

		return d_craters, x_craters, y_craters, index_craters, t_craters

######################### ---------------------------------------------------------------------------------------------------------------------------------------- #########################


######################### ---------------------------------------------------------------------------------------------------------------------------------------- #########################
class Model:
	def __init__(self):
		if params.verbose:
			os.system('clear')
			print('')
			print('########## ----------------------------------------------------------------------- ##########')
			print('Running 3d Landscape-EVolution and TRansport model (3LEVaTR)...')
			print('Components:')
			if params.cratering_on and params.secondaries_on:
				print('Impact cratering (primary and secondary craters)')
			elif params.cratering_on:
				print('Impact cratering (primary craters only)')
			if params.diffusion_on:
				print('Topographic diffusion (from micrometeorites, seismic shaking, creep, etc)')
			if params.pixel_noise_on:
				print('Pixel noise from sub-resolution craters')
			if params.tracers_on:
				print('Tracer particles tracked under the effects of the processes listed above')
			print('')
		else:
			pass

	def run(self):
		starttime = time()
		gc.collect()
		# Set up the grid

		params.diffusivity = float(sys.argv[1])

		grid = Grid(params.grid_size, params.resolution, params.diffusivity, params.dt)
		grid_old = grid.setUpGrid()
		grid_new = np.copy(grid_old)

		X_grid, Y_grid = np.ogrid[:params.grid_size, :params.grid_size]
		ones_grid = np.ones((params.grid_size, params.grid_size))

		continuous_ejecta_blanket_factor = params.continuous_ejecta_blanket_factor
		resolution = params.resolution
		grid_size = params.grid_size

		if params.cratering_on:
			if params.verbose:
				print('########## ----------------------------------------------------------------------- ##########')
				print('Sampling impactor population...')

			impPop = ImpactorPopulation()

			d_craters, x_craters, y_craters, index_craters, t_craters = impPop.sample_all_craters()
			current, peak =  tracemalloc.get_traced_memory()
			print('Current: {}, Peak: {}'.format(current, peak))
			sys.exit()
			gc.collect()

		for t in range(params.nsteps):
			#np.random.seed(int((time()+t*1000 + int(sys.argv[2]))))

			if params.cratering_on:
				##### ------------------------------------------------------------------------- #####
				# IMPACT CRATERING
				# Primary and secondary resolved craters at the current timestep
				timestep_index = np.where(t_craters == t)

				current_diameters = d_craters[timestep_index]
				current_x = x_craters[timestep_index]
				current_y = y_craters[timestep_index]
				current_index = index_craters[timestep_index]

				index_shuf = list(range(len(current_diameters)))
				np.random.shuffle(index_shuf)
				current_diameters = np.array([current_diameters[j] for j in index_shuf])
				current_x = np.array([current_x[j] for j in index_shuf])
				current_y = np.array([current_y[j] for j in index_shuf])
				current_index = np.array([current_index[j] for j in index_shuf])

				# Thickness of the regolith at the current timestep
				regolith_thickness = (body.model_avg_regolith_thickness/(body.model_avg_age**(0.5)))*(t**(0.5))

				##### ------------------------------------------------------------------------- #####
				# Add primary and background secondary craters
				for i in range(len(current_diameters)):
					crater_diam = current_diameters[i]
					crater_radius = crater_diam/2.0
					crater_index = current_index[i]

					x_crater_pix = int(current_x[i])
					y_crater_pix = int(current_y[i])

					grid_new = grid.add_crater(np.copy(grid_old), x_crater_pix, y_crater_pix, crater_diam, resolution, crater_index, continuous_ejecta_blanket_factor, X_grid, Y_grid, ones_grid)

					##### --------------------------------------------------------------------- #####
					# Update grid after adding crater
					grid_old = np.copy(grid_new)

				##### --------------------------------------------------------------------- #####
				# Update grid after adding all craters
				grid_old = np.copy(grid_new)

			if params.diffusion_on:
				##### --------------------------------------------------------------------- #####
				# DIFFUSION
				# Compute topographic diffusion.  If you are using explicit diffusion make sure that the timestep meets the Courant stability criterion
				if params.implicit_diffusion:
					grid_new = grid.implicit_diffusion2D(grid_old)
				elif params.explicit_diffusion:
					grid_new = grid.explicit_diffusion2D(grid_old)

				# Update grid after diffusion
				grid_old = np.copy(grid_new)

			progress(t, params.nsteps)
			gc.collect()

		gc.collect()

		'''
		# Size of variables in MB
		var_arr = []
		size_arr = []
		for var, obj in locals().items():
			var_arr.append(var)
			size_arr.append(sys.getsizeof(obj)*(1.e-6))

		var_arr = np.array(var_arr)
		size_arr = np.array(size_arr)
		ind_sort = size_arr.argsort()
		var_arr = var_arr[ind_sort]
		size_arr = size_arr[ind_sort]

		for l in range(len(var_arr)):
			print(var_arr[l], size_arr[l])
		'''

		current, peak =  tracemalloc.get_traced_memory()
		print('Current: {}, Peak: {}'.format(current, peak))
		tracemalloc.stop()

		'''
		plt.figure()
		plt.imshow(grid_old.T)
		plt.show()


        fname = '/extra/pob/Calibration_17m/' + str(sys.argv[1]) + '/' + str(uuid.uuid4()) + '.txt'

        np.savetxt(fname, grid_old)
		'''

######################### ---------------------------------------------------------------------------------------------------------------------------------------- #########################


######################### ---------------------------------------------------------------------------------------------------------------------------------------- #########################
def main():
	model = Model()
	model.run()

######################### ---------------------------------------------------------------------------------------------------------------------------------------- #########################


######################### ---------------------------------------------------------------------------------------------------------------------------------------- #########################
if __name__ == "__main__":
	main()

######################### ---------------------------------------------------------------------------------------------------------------------------------------- #########################
