######################### ---------------------------------------------------------------------------------------------------------------------------------------- #########################
# Import modules and parameter files (body, params)
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import sys
import body
import params as params
from scipy.interpolate import interp1d
from time import time, process_time
import matplotlib.pyplot as plt
from misc_functions import *
from matplotlib.colors import LightSource
from scipy.optimize import minimize
from scipy import interpolate
import scipy
import uuid

#np.random.seed(np.random.randint(10000) + int(sys.argv[2]))
np.random.seed(int(sys.argv[2]))
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

					grid_inter[:,j] = scipy.linalg.solve(A_edge, u_row + alpha_diff*u_row_below)

				elif j == N-1:    # Bottom rows
					u_row = grid_old[:,j]

					u_row_above = grid_inter[:,j-1]

					grid_inter[:,j] = scipy.linalg.solve(A_edge, u_row + alpha_diff*u_row_above)

				else:    # All other rows
					u_row = grid_old[:,j]

					u_row_above = grid_inter[:, j-1]
					u_row_below = grid_inter[:, j+1]

					grid_inter[:,j] = scipy.linalg.solve(A_interior, u_row + alpha_diff*u_row_above + alpha_diff*u_row_below)

			# Do all columns
			for i in range(N):
				if i == 0:    # Left column
					u_col = grid_old[i,:]

					u_col_right = grid_inter[i+1,:]

					grid_inter[i,:] = scipy.linalg.solve(A_edge, u_col + alpha_diff*u_col_right)

				elif i == N-1:    # Right column
					u_col = grid_old[i,:]

					u_col_left = grid_inter[i-1,:]

					grid_inter[i,:] = scipy.linalg.solve(A_edge, u_col + alpha_diff*u_col_left)

				else:    # All other columns
					u_col = grid_old[i,:]

					u_col_left = grid_inter[i-1,:]
					u_col_right = grid_inter[i+1,:]

					grid_inter[i,:] = scipy.linalg.solve(A_interior, u_col + alpha_diff*u_col_left + alpha_diff*u_col_right)

			itr += 1

		return np.copy(grid_inter)

	def explicit_diffusion2D(self, grid_old):
		 # Compute diffusion for grid using input parameters
		grid_size = self.grid_size
		grid_new = np.zeros((grid_size, grid_size))
		D = params.diffusivity
		dt = params.dt
		dx2 = params.dx2
		dy2 = params.dy2

		ind = range(0, grid_size)
		ind_up = np.roll(ind, -1)
		ind_down = np.roll(ind, 1)

		grid_new[0:,0:] = grid_old[0:,0:] + D * dt * ( (grid_old[ind_up, 0:] - 2.0*grid_old[0:, 0:] + grid_old[ind_down, 0:])/dx2 + (grid_old[0:, ind_up] - 2.0*grid_old[0:, 0:] + grid_old[0:, ind_down])/dy2 )

		return grid_new

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

		# Inheritance parameter, set as a constant, see Howard 2007, I=0 --> crater rim horizontal, I=1 --> crater rim parallel to pre-existing slope
		I_i = 0.9

		weighted_grid = np.copy(grid)
		outside_mask = dist_from_center > radius

		weighted_grid[outside_mask] = current_grid[outside_mask]*(1.0/dist_from_center[outside_mask])

		if np.max(crater_mask) > 0.0 and np.max(ejecta_mask) > 0.0:
			E_r = np.average(weighted_grid[crater_mask]) + np.average(weighted_grid[outside_mask])
		elif np.max(crater_mask) > 0.0:
			E_r = np.average(weighted_grid[crater_mask])
		elif np.max(ejecta_mask > 0.0):
			E_r = np.average(weighted_grid[outside_mask])
		else:
			E_r = np.zeros((grid_size, grid_size))

		# Crater elevation profile
		delta_H_crater = (((dist_from_center/radius)**2)*(rim_height + depth)) - depth

		# Ejecta elevation profile
		with np.errstate(divide='ignore'):    # Divide by zero at r=0 but we don't care about that point since it's interior to the ejecta blanket
			delta_H_ejecta = rim_height*((dist_from_center/radius)**-3) - (rim_height/54.0)*((dist_from_center/radius) - 1.0)

		# Inheritance matrices - determines how the crater is integrated into the existing grid
		G_grid = (1.0 - I_i)*ones_grid
		min_mask = G_grid > delta_H_ejecta/rim_height
		G_grid[min_mask] = delta_H_ejecta[min_mask]/rim_height

		crater_inh_profile = (1.0 - I_i*((dist_from_center/radius)**2.0))*(E_r - current_grid)
		ejecta_inh_profile = G_grid*(E_r - current_grid)

		delta_E_crater = delta_H_crater + crater_inh_profile
		delta_E_ejecta = delta_H_ejecta + ejecta_inh_profile

		# Add calculated elevations to the grid at the corresponding pixels
		grid[crater_mask] += delta_E_crater[crater_mask]
		grid[ejecta_mask] += delta_E_ejecta[ejecta_mask]

		return grid

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

		# Define diameter bins with geometric spacing between D_min and D_max
		diam_bins = np.geomspace(D_min, D_max, 10, endpoint=True, dtype=np.float)
		geom_factor = diam_bins[1]/diam_bins[0]

		# Calculate the incremental number of craters in each diameter bin using the cumulative SFD
		inc_num = np.array([c_sfd*(diam_bins[i]**(-b_cum) - diam_bins[i+1]**(-b_cum)) for i in range(len(diam_bins)-1)])

		# Poisson lambdas for each bin by multiplying by the grid area in km^2
		secondary_lams = inc_num*(grid_area/(1.e6))
		secondary_lams[secondary_lams < 0] = 0

		# Make diameter bins the geometric average of each bin
		diam_bins = diam_bins[0:-1]*np.sqrt(geom_factor)

		# Poisson sampling of the number of secondaries in each secondary crater diameter bin
		num_secs_on_grid_arr = np.random.poisson(lam=secondary_lams, size=len(secondary_lams))*num_annuli_imps

		# Add a random amount to each bin so diameters are uniformly spaced within the bin (instead of all at the geometric average)
		#secondary_diameters = np.concatenate([diam_bins[i]*np.ones(num_secs_on_grid_arr[i]) + np.random.rand(num_secs_on_grid_arr[i])*(geom_factor*diam_bins[i] - diam_bins[i]) for i in range(len(diam_bins))])
		# For speedup, just use the geometric average
		secondary_diameters =  np.concatenate([diam_bins[i]*np.ones(num_secs_on_grid_arr[i], dtype=np.float) for i in range(len(diam_bins))])

		return secondary_diameters

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
		diam_bins = np.geomspace(D_min, D_max, 10, endpoint=True, dtype=np.float)
		geom_factor = diam_bins[1]/diam_bins[0]

		# Calculate the incremental number of craters in each diameter bin using the cumulative SFD
		inc_num = np.array([c_sfd*diam_bins[i]**(-b_cum) - c_sfd*diam_bins[i+1]**(-b_cum) for i in range(len(diam_bins)-1)])

		# Poisson lambdas for each bin by multiplying by the grid area in km^2
		secondary_lams = inc_num*(grid_area/(1.e6))
		secondary_lams[secondary_lams < 0] = 0.0

		# Make diameter bins the geometri average of each bin
		diam_bins = diam_bins[0:-1]*np.sqrt(geom_factor)

		# Poisson sampling of the number of secondaries in each secondary crater diameter bin
		num_secs_on_grid_arr = np.random.poisson(lam=secondary_lams, size=len(secondary_lams))

		# Add a random amount to each bin so diameters are uniformly spaced within the bin (instead of all at one of the edges or the geometric average)
		#secondary_diameters = [diam_bins[i]*np.ones(num_secs_on_grid_arr[i]) + np.random.rand(num_secs_on_grid_arr[i])*(geom_factor*diam_bins[i] - diam_bins[i]) for i in range(len(diam_bins))]
		secondary_diameters = np.concatenate([diam_bins[i]*np.ones(num_secs_on_grid_arr[i], dtype=np.float) for i in range(len(diam_bins))])

		return secondary_diameters

	def sample_timestep_craters(self, t, avg_imp_diam, primary_lams, max_grid_dist, avg_crater_diam, num_inc, dt, min_crater, resolution, grid_size, grid_width, min_primary_for_secondaries, secondaries_on, r_body, continuous_ejecta_blanket_factor, X_grid, Y_grid):

		#np.random.seed(int((time()+t*1000 + int(sys.argv[2]))))

		timestep_diams = []
		timestep_x = []
		timestep_y = []
		timestep_primary_index = []
		timestep_time = []
		sec_dist_arr = []

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
				primary_crater_x = primary_crater_dists*np.cos(primary_crater_phis)
				primary_crater_y = primary_crater_dists*np.sin(primary_crater_phis)

				primary_crater_x_pix = np.array(primary_crater_x/resolution + grid_size/2.0).astype('int')
				primary_crater_y_pix = np.array(-primary_crater_y/resolution + grid_size/2.0).astype('int')

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

								# Randomly sample distances from the primary for these diameters.  Must occur outside the continuous ejecta blanket and inside the maximum secondary distance
								secondary_crater_dists = (continuous_ejecta_blanket_factor*(D_p/2.0)) + (max_dist_for_sec - (continuous_ejecta_blanket_factor*(D_p/2.0)))*np.random.rand(len(secondary_crater_diams))

								# Randomly sample azimuthal angles for secondaries between 0 and 2pi
								secondary_crater_phis = (2.0*np.pi)*np.random.rand(len(secondary_crater_diams))

								# Calculate x,y position of secondaries
								secondary_crater_x_primary = secondary_crater_dists*np.cos(secondary_crater_phis) + x_crater*resolution
								secondary_crater_y_primary = secondary_crater_dists*np.sin(secondary_crater_phis) + y_crater*resolution

								secondary_crater_x_pix = np.array(secondary_crater_x_primary/resolution).astype('int')
								secondary_crater_y_pix = np.array(secondary_crater_y_primary/resolution).astype('int')

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
						'''
						# Loop through each primary impact and compute all secondaries from that impact that overlap the grid
						for k in range(num_annuli_imps):
							annulus_secondary_diameters.append(self.secondaries(avg_dist, D_p, min_crater, max_sec_diam, r_body, max_sec_area, max_dist_for_sec, num_annuli_imps))
						'''
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
								secondary_crater_x = sec_crater_dists*np.cos(sec_crater_phis)
								secondary_crater_y = sec_crater_dists*np.sin(sec_crater_phis)

								secondary_crater_x_pix = np.array(secondary_crater_x/resolution + grid_size/2.0).astype('int')
								secondary_crater_y_pix = np.array(-secondary_crater_y/resolution + grid_size/2.0).astype('int')

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
									sec_dist_arr.append(sec_dists_from_grid)

		if len(timestep_diams) > 0:
			#print('some impacts')
			timestep_diams = np.concatenate(np.array(timestep_diams))
			timestep_x = np.concatenate(np.array(timestep_x))
			timestep_y = np.concatenate(np.array(timestep_y))
			timestep_primary_index = np.concatenate(np.array(timestep_primary_index))
			timestep_time = t*np.ones(len(timestep_diams))
			if len(sec_dist_arr) > 0:
				sec_dist_arr = np.concatenate(sec_dist_arr)

		return timestep_diams, timestep_x, timestep_y, timestep_primary_index, timestep_time, sec_dist_arr



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
		dist_secs = []

		for t in range(nsteps):
			timestep_diams, timestep_x, timestep_y, timestep_primary_index, timestep_time, sec_dist_arr = self.sample_timestep_craters(t, avg_imp_diam, primary_lams, max_grid_dist, avg_crater_diam, num_inc, params.dt, params.min_crater, params.resolution, params.grid_size, params.grid_width, params.min_primary_for_secondaries, params.secondaries_on, body.radius_body, params.continuous_ejecta_blanket_factor, X_grid, Y_grid)

			d_craters.append(timestep_diams)
			x_craters.append(timestep_x)
			y_craters.append(timestep_y)
			index_craters.append(timestep_primary_index)
			t_craters.append(timestep_time)
			dist_secs.append(sec_dist_arr)

			if params.verbose:
				progress(t,nsteps)

		if params.verbose:
			progress(nsteps, nsteps)
			print('')

		d_craters = np.array(np.concatenate(d_craters)).astype('float')
		x_craters = np.array(np.concatenate(x_craters)).astype('float')
		y_craters = np.array(np.concatenate(y_craters)).astype('float')
		index_craters = np.array(np.concatenate(index_craters)).astype('float')
		t_craters = np.array(np.concatenate(t_craters)).astype('int')
		dist_secs = np.array(np.concatenate(dist_secs)).astype('float')

		return d_craters, x_craters, y_craters, index_craters, t_craters, dist_secs

######################### ---------------------------------------------------------------------------------------------------------------------------------------- #########################


######################### ---------------------------------------------------------------------------------------------------------------------------------------- #########################
class Tracer:
		def __init__(self, x_p0, y_p0, z_p0):
			# Initialize particle position with x,y,z position
			# Create arrays to store the particle's trajectory
			self.x_p = x_p0
			self.y_p = y_p0
			self.z_p = z_p0
			self.x_arr = []
			self.y_arr = []
			self.z_arr = []
			self.d_arr = []
			self.slope_arr = []

		def current_position(self):
			# Return the particle's current position
			# Should be used to get inputs for transportation methods
			return [self.x_p, self.y_p, self.z_p]

		def update_position(self, new_position):
			# Update particle's current position
			self.x_p = new_position[0]
			self.y_p = new_position[1]
			self.z_p = new_position[2]

		def update_trajectory(self, x_p, y_p, z_p, d_p, slope_p):
			# Update particle history at the end of each timestep
			# Store x,y,z position as well as depth and the slope of the grid at the particle's position
			self.x_arr.append(x_p)
			self.y_arr.append(y_p)
			self.z_arr.append(z_p)
			self.d_arr.append(d_p)
			self.slope_arr.append(slope_p)

		def tracer_particle_diffusion(self, z0, z, particle_pos):
			diffusivity = params.diffusivity
			dt = params.dt
			dx2 = params.dx2
			dy2 = params.dy2
			grid_size = params.grid_size
			resolution = params.resolution

			if np.isnan(particle_pos[0]):
			# CLASS 0 - NaN particle passed to function
			# Should not happen, but if it does just return NaNs again
				x_p = np.nan
				y_p = np.nan
				z_p = np.nan

			elif particle_pos[0] == 0 or particle_pos[0] == (grid_size-1) or particle_pos[1] == 0 or particle_pos[1] == (grid_size-1):
			# CLASS 1 - Particle starts on the edge of the grid
			# Cannot compute elevation difference in at least one direction so particle is lost
				x_p = np.nan
				y_p = np.nan
				z_p = np.nan

			else:
			# CLASS 2 - Particle possibly moved by diffusion
				x_p = int(round(particle_pos[0]))
				y_p = int(round(particle_pos[1]))
				z_p = particle_pos[2]

				z_new = z[x_p, y_p]
				z_old = z0[x_p, y_p]
				d_p0 = z_old - z_p

				if z_p < z_new:
				# SCENARIO 1 - Particle still at or below the surface after the elevation change from diffusion.  Particle remains where it is
					x_p = int(x_p)
					y_p = int(y_p)
					z_p = z_p
					# FINAL PARTICLE POSITION DETERMINED - C2:S1

				elif z_p >= z_new:
				# SCENARIO 2 - Diffusion elevation change at the particle's position would put it at negative depth.
				# Particle moves one pixel in the steepest downhill direction
					delta_h = z_old - z_new		# Change in elevation at the current pixel

					# Possible directions to move, I will be going from left to right then down to the next row
					# (i-1, j+1)    (i, j+1)    (i+1, j+1)
					# (i-1, j)      (i, j)      (i+1, j)
					# (i-1, j-1)    (i, j-1)    (i+1, j-1)
					delta_z_11 = float(z0[x_p, y_p] - z0[x_p-1, y_p+1])
					delta_z_12 = float(z0[x_p, y_p] - z0[x_p, y_p+1])
					delta_z_13 = float(z0[x_p, y_p] - z0[x_p+1, y_p+1])
					delta_z_21 = float(z0[x_p, y_p] - z0[x_p-1, y_p])
					delta_z_22 = float(z0[x_p, y_p] - z0[x_p, y_p])	# trivial, particle must move somewhere other than the current pixel
					delta_z_23 = float(z0[x_p, y_p] - z0[x_p+1, y_p])
					delta_z_31 = float(z0[x_p, y_p] - z0[x_p-1, y_p-1])
					delta_z_32 = float(z0[x_p, y_p] - z0[x_p, y_p-1])
					delta_z_33 = float(z0[x_p, y_p] - z0[x_p+1, y_p-1])

					delta_z = np.zeros((3,3))
					delta_z[0,0] = delta_z_11
					delta_z[0,1] = delta_z_12
					delta_z[0,2] = delta_z_13
					delta_z[1,0] = delta_z_21
					delta_z[1,1] = delta_z_22
					delta_z[1,2] = delta_z_23
					delta_z[2,0] = delta_z_31
					delta_z[2,1] = delta_z_32
					delta_z[2,2] = delta_z_33

					rad_ext_11 = np.sqrt(2.0*resolution**2)
					rad_ext_12 = resolution
					rad_ext_13 = np.sqrt(2.0*resolution**2)
					rad_ext_21 = resolution
					rad_ext_22 = resolution
					rad_ext_23 = resolution
					rad_ext_31 = np.sqrt(2.0*resolution**2)
					rad_ext_32 = resolution
					rad_ext_33 = np.sqrt(2.0*resolution**2)

					slope_11 = np.arctan2(delta_z_11, rad_ext_11)
					slope_12 = np.arctan2(delta_z_12, rad_ext_12)
					slope_13 = np.arctan2(delta_z_13, rad_ext_13)
					slope_21 = np.arctan2(delta_z_21, rad_ext_21)
					slope_22 = 0
					slope_23 = np.arctan2(delta_z_23, rad_ext_23)
					slope_31 = np.arctan2(delta_z_31, rad_ext_31)
					slope_32 = np.arctan2(delta_z_32, rad_ext_32)
					slope_33 = np.arctan2(delta_z_33, rad_ext_33)

					x_change = [-1, 0, 1, -1, 0, 1, -1,  0,  1]
					y_change = [ 1, 1, 1,  0, 0, 0, -1, -1, -1]

					max_slope_dir = np.argmax([slope_11, slope_12, slope_13, slope_21, slope_22, slope_23, slope_31, slope_32, slope_33], axis=0)

					x_p += x_change[max_slope_dir]
					y_p += y_change[max_slope_dir]

					if 0 <= x_p <= (grid_size-1) and 0 <= y_p <= (grid_size):
					# Particle moves somewhere on the grid.  Placed at a depth correspondingly inversely to its initial depth (layers are flipped as material near the surface moves downslope first, followed by material buried deeper)
						x_p = int(round(x_p))
						y_p = int(round(y_p))
						z_p = z[x_p, y_p] - delta_h + d_p0
					else:
					# Particle moves off the grid
						x_p = np.nan
						y_p = np.nan
						z_p = np.nan
					# FINAL PARTICLE POSITION DETERMINED - C2:S2

			return [x_p, y_p, z_p]

		def solve_for_landing_point(self, t, *args):
			resolution = params.resolution
			g = body.g

			R0 = args[0]
			ejection_velocity_vertical = args[1]
			x_crater_pix = args[2]
			y_crater_pix = args[3]
			phi0 = args[4]
			f_surf_new = args[5]
			z_p0 = args[6]

			r_t_flight = R0 + ejection_velocity_vertical*t
			z_t_flight = z_p0 + ejection_velocity_vertical*t - 0.5*g*(t**2)

			R_t = np.hypot( r_t_flight, z_t_flight )
			theta_t = np.arccos(z_t_flight/R_t)

			x_t_flight = R_t*np.sin(theta_t)*np.cos(phi0)/resolution + x_crater_pix
			y_t_flight = R_t*np.sin(theta_t)*np.sin(phi0)/resolution + y_crater_pix

			z_t_surf = f_surf_new(y_t_flight, x_t_flight)[0]

			return abs(z_t_surf - z_t_flight)

		def solve_for_ejection_point(self, t,*args):
			resolution = params.resolution

			R0 = args[0]
			x_crater_pix = args[1]
			y_crater_pix = args[2]
			phi0 = args[3]
			theta0 = args[4]
			alpha = args[5]
			f_surf_old = args[6]

			R_t_flow = (R0**4 + 4*alpha*t)**(1.0/4.0)
			theta_t_flow = np.arccos(1.0 - ( (1.0-np.cos(theta0))*(R_t_flow/R0)   ))

			x_t_flow = (R_t_flow*np.sin(theta_t_flow)*np.cos(phi0))/resolution + x_crater_pix
			y_t_flow = (R_t_flow*np.sin(theta_t_flow)*np.sin(phi0))/resolution + y_crater_pix
			z_t_flow = -1.0*R_t_flow*np.cos(theta_t_flow)

			z_t_surf = f_surf_old(y_t_flow, x_t_flow)[0]

			return abs(z_t_surf - z_t_flow)

		def tracer_particle_crater(self, x_p0, y_p0, z_p0, d_p0, dx, dy, dz, x_crater_pix, y_crater_pix, R0, crater_radius, grid_old, grid_new):
			plot_on = 0
			print_on = 0

			grid_size = params.grid_size
			g = body.g
			continuous_ejecta_blanket_factor = params.continuous_ejecta_blanket_factor
			resolution = params.resolution

			if np.isnan(x_p0) or np.isnan(y_p0) or np.isnan(z_p0):
				##### ------------------------------------------------------------------ #####
				# CLASS 0: NaN Particle passed.  Should not make it into the function but just return NaNs again if it does
				print('CLASS 0')
				x_p = np.nan
				y_p = np.nan
				z_p = np.nan
				# FINAL PARTICLE POSITION DETERMINED - C0

			else:
				# Particle passed with real coordinates
				# Define variables that will be used by all or most paths
				transient_crater_radius = crater_radius/1.18
				alpha = np.sqrt( (transient_crater_radius**7)*g/12.0 )
				t_flow = (1.0/(4.0*alpha))*(12.0*(alpha**2)/g)**(4.0/7.0)

				if d_p0 < 0.0:
					##### ------------------------------------------------------------------ #####
					# CLASS 1: Particle is above the surface (un-physical)
					print('CLASS 0: PARTICLE ABOVE THE SURFACE')
					# FINAL PARTICLE POSITION DETERMINED - C1
					sys.exit()

				elif d_p0 == 0.0:
					##### ------------------------------------------------------------------ #####
					# CLASS 2: Particle on the surface within the sphere of influence

					if R0 == 0.0:
						# SCENARIO 1: Obliteration
						if print_on:
							print('CLASS 2 - SCENARIO 1')
						# Particle on the surface at the impact site
						x_p = np.nan
						y_p = np.nan
						z_p = np.nan
						# FINAL PARTICLE POSITION DETERMINED - C2:S1

						if plot_on:
							plt.plot(x_p0, z_p0, 'kX')

					elif 0.0 < R0 <= transient_crater_radius:
						# SCENARIO 2: Surface ejection
						if print_on:
							print('CLASS 2 - SCENARIO 2')

						##### Interpolate on post-crater grid for computing landing position
						xi = range(grid_size)
						yi = range(grid_size)
						XX, YY = np.meshgrid(xi, yi)
						f_surf_new = interpolate.interp2d(xi,yi,grid_new, kind='linear')

						phi0 = float(np.arctan2(dy, dx))

						ejection_velocity_vertical = alpha/(R0**3)

						# Initial guess for the time it will take the particle to land back on the surface
						flight_time = 2.0*ejection_velocity_vertical/g		# Would be exact if no pre-existing topography

						# Solve for when and where the particle lands
						res_solve = minimize(self.solve_for_landing_point, flight_time, args=(R0, ejection_velocity_vertical, x_crater_pix, y_crater_pix, phi0, f_surf_new, z_p0), bounds=[(0.0, np.inf)], method = 'L-BFGS-B')
						t_land = res_solve.x[0]		# Flight time that minimizes the distance between the particle and the surface (aka finds when it lands on the landscape)

						r_flight = R0 + ejection_velocity_vertical*t_land
						z_flight = z_p0 + ejection_velocity_vertical*t_land - 0.5*g*(t_land**2)

						R_land = np.hypot( r_flight, z_flight )
						theta_land = np.arccos(z_flight/R_land)

						if np.isnan(theta_land):
							print('NAN LANDING THETA- SURFACE EJECTION')
							sys.exit()

						x_land = int(round(R_land*np.sin(theta_land)*np.cos(phi0)/resolution + x_crater_pix))
						y_land = int(round(R_land*np.sin(theta_land)*np.sin(phi0)/resolution + y_crater_pix))

						if (0 <= x_land <= (grid_size-1)) and (0 <= y_land <= (grid_size-1)):
							# Particle lands on the grid
							x_p = int(x_land)
							y_p = int(y_land)

							dx_final = (x_p - x_crater_pix)*resolution
							dy_final = (y_p - y_crater_pix)*resolution
							dist_from_crater = np.hypot(dx_final, dy_final)

							# If particle lands within the continuous ejecta blanket, place it randomly within the ejecta blanket thickness at its landing point
							if dist_from_crater <= continuous_ejecta_blanket_factor*crater_radius:
								z_p = grid_new[x_p, y_p] - np.random.rand()*abs(grid_new[x_p, y_p] - grid_old[x_p, y_p])
							else:
								z_p = grid_new[x_p, y_p]

							if plot_on:
								##### For plotting purposes only
								plt.plot(x_p0, z_p0, 'yX')

								t_arr = np.linspace(0.0, t_land, 100, dtype=np.float)
								r_t_flight = R0 + ejection_velocity_vertical*t_arr
								z_t_flight = z_p0 + ejection_velocity_vertical*t_arr - 0.5*g*(t_arr**2)

								R_t_flight = np.hypot( r_t_flight, z_t_flight )
								theta_t_flight = np.arccos(z_t_flight/R_t_flight)

								x_t_flight = R_t_flight*np.sin(theta_t_flight)*np.cos(phi0)/resolution + x_crater_pix
								y_t_flight = R_t_flight*np.sin(theta_t_flight)*np.sin(phi0)/resolution + y_crater_pix

								plt.plot(x_t_flight, z_t_flight, 'y--')
								plt.plot(x_p, z_p, 'ro', markersize=3)
						else:
							# Particle lands off the grid
							x_p = np.nan
							y_p = np.nan
							z_p = np.nan

							if plot_on:
								plt.plot(x_p0, z_p0, 'cX')
						# FINAL PARTICLE POSITION DETERMINED - C2:S2

					elif transient_crater_radius < R0 <= crater_radius:
						# SCENARIO 3: Infill
						if print_on:
							print('CLASS 2 - SCENARIO 3')
						# Particle starts between the transient crater radius and final crater radius, ends up being part of the infilling material

						if dz <= 0.0:
							theta0 = np.arccos(abs(dz)/R0)
						elif dz > 0.0:
							theta0 = np.pi/2.0 + np.arcsin(dz/R0)
						phi0 = float(np.arctan2(dy, dx))

						R_final = np.random.uniform()*R0

						x_land = int(np.round(R_final*np.sin(theta0)*np.cos(phi0)/resolution + x_crater_pix))
						y_land = int(np.round(R_final*np.sin(theta0)*np.sin(phi0)/resolution + y_crater_pix))

						if (0 <= x_land <= (grid_size-1)) and (0 <= y_land <= (grid_size-1)):
							# Particle lands on the grid
							x_p = int(x_land)
							y_p = int(y_land)
							z_p = grid_new[x_p, y_p]

							if plot_on:
								##### For plotting purposes only
								plt.plot(x_p0, z_p0, 'go', markersize=3)
								plt.plot(x_p, z_p, 'ro', markersize=3)
								plt.plot([x_p0, x_p], [z_p0, z_p], 'b--')
						else:
							# Particle lands off the grid
							x_p = np.nan
							y_p = np.nan
							z_p = np.nan

							if plot_on:
								##### For plotting purposes only
								plt.plot(x_p0, z_p0, 'bX', markersize=3)
						# FINAL PARTICLE POSITION DETERMINED - C2:S3

					elif R0 > crater_radius:
						# SCENARIO 4: Surface burial
						if print_on:
							print('CLASS 2 - SCENARIO 4')
						# Particle starts outside the final crater radius, gets buried by continuous ejecta blanket

						x_p = int(x_p0)
						y_p = int(y_p0)
						z_p = z_p0

						if z_p > grid_new[x_p, y_p]:
							# Possible that interactions where craters land on top of other craters that could cause a net decrease in elevation even exterior the the new crater (see: Inheritance parameter)
							z_p = grid_new[x_p, y_p]

						if plot_on:
							##### For plotting purposes only
							plt.plot(x_p0, z_p0, 'gX')
						# FINAL PARTICLE POSITION DETERMINED - C2:S4


				elif d_p0 > 0.0:
					##### ------------------------------------------------------------------ #####
					# CLASS 3: Particle buried in the subsurface within the sphere of influence

					if R0 >= continuous_ejecta_blanket_factor*crater_radius:
						# SCENARIO 0: Particle outside the sphere of influence. Leave it where it is
						# I have extended the initial threshold radius to be passed to the function to catch edge cases so these might get thrown in
						x_p = x_p0
						y_p = y_p0
						z_p = z_p0

						# Possible that interactions where craters land on top of other craters that could cause a net decrease in elevation even exterior the the new crater (see: Inheritance parameter)
						if z_p > grid_new[x_p, y_p]:
							z_p = grid_new[x_p, y_p]
						# FINAL PARTICLE POSITION DETERMINED - C3:S0

					elif dx == 0.0 and dy == 0.0:
						# SCENARIO 1: Drilling
						if print_on:
							print('CLASS 3 - SCENARIO 1')
						# Only need to compute R(t) because the particle will not change x or y position. Theta0 = 0, phi0 = 0
						theta0 = 0.0
						phi0 = 0.0

						R_flow = -1.0*((R0**4 + 4.0*alpha*t_flow)**(0.25))

						x_p = int(x_p0)
						y_p = int(y_p0)
						z_p = R_flow

						if z_p > grid_new[x_p, y_p]:
							z_p = grid_new[x_p, y_p]
						# FINAL PARTICLE POSITION DETERMINED - C3:S1

						if plot_on:
							t_arr = np.linspace(0.0, t_flow, 100, dtype=np.float)
							R_flow = -1.0*((R0**4 + 4.0*alpha*t_arr)**(0.25))

							x_flow = x_p0*np.ones(len(R_flow))
							y_flow = y_p0*np.ones(len(R_flow))
							z_flow = R_flow

							plt.plot(x_p0, z_p0, 'go', markersize=3)
							plt.plot(x_flow, z_flow, 'r--')
							plt.plot(x_p, z_p, 'ro', markersize=3)


					else:
						# Particle will move along a streamline that changes its [x,y,z] position and possibly eject it from the subsurface

						if dz <= 0.0:
							theta0 = np.arccos(abs(dz)/R0)
						elif dz > 0.0:
							theta0 = np.pi/2.0 + np.arcsin(dz/R0)
						phi0 = float(np.arctan2(dy, dx))

						##### Interpolate on pre- and post-crater grid for computing ejection/landing positions
						xi = range(grid_size)
						yi = range(grid_size)
						XX, YY = np.meshgrid(xi, yi)
						f_surf_old = interpolate.interp2d(xi,yi,grid_old, kind='linear')

						# Initial guess for ejection time.  Guess it will be ejected at an angle of ~90 degrees from the vertical
						theta_eject_guess = np.pi/2.0
						t_eject_guess = (R0**4)/(4.0*alpha)*( ((1.0 - np.cos(theta0))**(-4)) - 1.0)
						t_max =  (R0**4)/(4.0*alpha)*( (16.0*(1.0 - np.cos(theta0))**(-4)) - 1.0)

						R_eject_guess = (R0**4 + 4.0*alpha*t_eject_guess)**(0.25)
						theta_eject_guess = np.arccos(1.0 - ( (1.0-np.cos(theta0))*(R_eject_guess/R0)   ))

						# Solve for ejection time
						res_solve = minimize(self.solve_for_ejection_point, t_eject_guess, args=(R0, x_crater_pix, y_crater_pix, phi0, theta0, alpha, f_surf_old), bounds=[(0.0, 0.99*t_max)], method = 'L-BFGS-B')
						t_eject = res_solve.x[0]		# Ejection time that minimizes the distance between the particle and the surface (i.e. finds when the streamline intersects the surface)
						R_eject = (R0**4 + 4.0*alpha*t_eject)**(0.25)

						if t_eject <= t_flow and R_eject <= transient_crater_radius:
							# SCENARIO 2: Subsurface ejection
							if print_on:
								print('CLASS 3 - SCENARIO 2')
							# Particle streamline reaches the surface before the flow freezes and at a radial distance of less than the transient crater radius
							f_surf_new = interpolate.interp2d(xi,yi,grid_new, kind='linear')

							theta_eject = np.arccos(1.0 - ( (1.0-np.cos(theta0))*(R_eject/R0)   ))

							x_eject = (R_eject*np.sin(theta_eject)*np.cos(phi0))/resolution + x_crater_pix
							y_eject = (R_eject*np.sin(theta_eject)*np.sin(phi0))/resolution + y_crater_pix
							z_eject = -1.0*R_eject*np.cos(theta_eject)

							ejection_velocity_vertical = alpha/(R_eject**3)

							# Initial guess for the time it will take the particle to land back on the surface
							flight_time = 2.0*ejection_velocity_vertical/g		# Would be exact if no pre-existing topography

							# Solve for when and where the particle lands
							res_solve = minimize(self.solve_for_landing_point, flight_time, args=(R_eject, ejection_velocity_vertical, x_crater_pix, y_crater_pix, phi0, f_surf_new, z_eject), bounds=[(0.0, np.inf)], method = 'L-BFGS-B')
							t_land = res_solve.x[0]		# Flight time that minimizes the distance between the particle and the surface (aka finds when it lands on the landscape)

							r_flight = R_eject + ejection_velocity_vertical*t_land
							z_flight = z_eject + ejection_velocity_vertical*t_land - 0.5*g*(t_land**2)

							R_land = np.hypot( r_flight, z_flight )
							theta_land = np.arccos(z_flight/R_land)

							if np.isnan(theta_land):
								print('NAN LANDING THETA- SUBSURFACE EJECTION')
								sys.exit()

							x_land = int(round(R_land*np.sin(theta_land)*np.cos(phi0)/resolution + x_crater_pix))
							y_land = int(round(R_land*np.sin(theta_land)*np.sin(phi0)/resolution + y_crater_pix))

							if (0 <= x_land <= (grid_size-1)) and (0 <= y_land <= (grid_size-1)):
								# Particle lands on the grid
								x_p = int(x_land)
								y_p = int(y_land)

								dx_final = (x_p - x_crater_pix)*resolution
								dy_final = (y_p - y_crater_pix)*resolution

								dist_from_crater = np.hypot(dx_final, dy_final)

								# If particle lands within the continuous ejecta blanket, place it randomly within the ejecta blanket thickness at its landing point
								if dist_from_crater < continuous_ejecta_blanket_factor*crater_radius:
									z_p = grid_new[x_p, y_p] - np.random.rand()*abs(grid_new[x_p, y_p] - grid_old[x_p, y_p])
								else:
									z_p = grid_new[x_p, y_p]

								if plot_on:
									##### For plotting purposes only
									t_arr = np.linspace(0.0, t_eject, 100, dtype=np.float)
									R_flow = (R0**4 + 4.0*alpha*t_arr)**(0.25)

									theta_flow = np.arccos(1.0 - ( (1.0-np.cos(theta0))*(R_flow/R0)   ))

									x_flow = (R_flow*np.sin(theta_flow)*np.cos(phi0))/resolution + x_crater_pix
									y_flow = (R_flow*np.sin(theta_flow)*np.sin(phi0))/resolution + y_crater_pix
									z_flow = -1.0*R_flow*np.cos(theta_flow)

									plt.plot(x_p0, z_p0, 'go', markersize=3)
									plt.plot(x_flow, z_flow, 'r--')
									plt.plot(x_flow[-1], z_flow[-1], 'yX')

									t_arr = np.linspace(0.0, t_land, 100, dtype=np.float)

									r_flight = R_eject + ejection_velocity_vertical*t_arr
									z_flight = z_eject + ejection_velocity_vertical*t_arr - 0.5*g*(t_arr**2)

									R_flight = np.hypot( r_flight, z_flight )
									theta_flight = np.arccos(z_flight/R_flight)

									x_flight = R_flight*np.sin(theta_flight)*np.cos(phi0)/resolution + x_crater_pix
									y_flight = R_flight*np.sin(theta_flight)*np.sin(phi0)/resolution + y_crater_pix

									plt.plot(x_flight, z_flight, 'y--')
									plt.plot(x_p, z_p, 'ro', markersize=3)

							else:
								# Particle lands off the grid
								x_p = np.nan
								y_p = np.nan
								z_p = np.nan

								if plot_on:

									t_arr = np.linspace(0.0, t_eject, 100, dtype=np.float)
									R_flow = (R0**4 + 4.0*alpha*t_arr)**(0.25)

									theta_flow = np.arccos(1.0 - ( (1.0-np.cos(theta0))*(R_flow/R0)   ))

									x_flow = (R_flow*np.sin(theta_flow)*np.cos(phi0))/resolution + x_crater_pix
									y_flow = (R_flow*np.sin(theta_flow)*np.sin(phi0))/resolution + y_crater_pix
									z_flow = -1.0*R_flow*np.cos(theta_flow)

									plt.plot(x_p0, z_p0, 'go', markersize=3)
									plt.plot(x_flow, z_flow, 'r--')
									plt.plot(x_flow[-1], z_flow[-1], 'cX')
							# FINAL PARTICLE POSITION DETERMINED - C3:S2


						else:
							# SCENARIO 3: Subsurface transport
							if print_on:
								print('CLASS 3 - SCENARIO 3')
							# Particle moves along a streamline that does not reach the surface before the flow freezes. Or it does reach the surface but
							# does so at a radial distance greater than the transient crater radius.  Particles at these distances are not observed to be ejected.
							# This artifact is likely due to our use of a constant alpha, meaning that the velocity field does not decay with time.

							R_flow = (R0**4 + 4.0*alpha*t_flow)**(0.25)
							theta_flow = np.arccos(1.0 - ( (1.0-np.cos(theta0))*(R_flow/R0)   ))

							try:
								x_flow = int(round((R_flow*np.sin(theta_flow)*np.cos(phi0))/resolution + x_crater_pix))
								y_flow = int(round((R_flow*np.sin(theta_flow)*np.sin(phi0))/resolution + y_crater_pix))
								z_flow = -1.0*R_flow*np.cos(theta_flow)
							except:
								print('NAN SUBSURFACE FLOW THETA')
								print(x_p0, y_p0, z_p0)
								print(dx, dy, dz)
								print(R0, crater_radius)
								sys.exit()

							if (0 <= x_flow <= (grid_size-1)) and (0 <= y_flow <= (grid_size-1)):
							# Particle flows on the grid
								x_p = int(x_flow)
								y_p = int(y_flow)
								z_p = z_flow
								# Super-surface flows - NEED TO FIX.  For now just place on the surface at the current position
								if z_p > grid_new[x_p, y_p]:
									z_p = grid_new[x_p, y_p]

								if plot_on:
									t_arr = np.linspace(0.0, t_flow, 100, dtype=np.float)
									R_flow = (R0**4 + 4.0*alpha*t_arr)**(0.25)

									theta_flow = np.arccos(1.0 - ( (1.0-np.cos(theta0))*(R_flow/R0)   ))

									x_flow = (R_flow*np.sin(theta_flow)*np.cos(phi0))/resolution + x_crater_pix
									y_flow = (R_flow*np.sin(theta_flow)*np.sin(phi0))/resolution + y_crater_pix
									z_flow = -1.0*R_flow*np.cos(theta_flow)

									plt.plot(x_p0, z_p0, 'go', markersize=3)
									plt.plot(x_flow, z_flow, 'r--')
									plt.plot(x_p, z_p, 'ro', markersize=3)
							else:
								# Particle flows off the grid
								x_p = np.nan
								y_p = np.nan
								z_p = np.nan

								if plot_on:
									plt.plot(x_p0, z_p0, 'rX')
							# FINAL PARTICLE POSITION DETERMINED - C3:S3

			if ~np.isnan(x_p):
				grid_old = np.copy(grid_new)
				d_p = grid_old[x_p, y_p] - z_p
				if d_p < 0.0:
					print('PARTICLE ABOVE THE SURFACE - END OF FUNCTION')
					sys.exit()

			return [x_p, y_p, z_p]


		def tracer_particle_noise(self, x_p0, y_p0, z_p0, grid_old, grid_new):
			# Movement of tracer particles from the addition of sub-pixel cratering
			# Particles are not moved horizontally.  If cratering removes material from the particle's pixel and excavates the particle to be above the surface,
			# Place the particle on the new surface at the same pixel

			if z_p0 > grid_new[x_p0, y_p0]:
				x_p = x_p0
				y_p = y_p0
				z_p = grid_new[x_p, y_p]
			else:
				x_p = x_p0
				y_p = y_p0
				z_p = z_p0

			return [x_p, y_p, z_p]


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
		# Set up the grid
		if params.verbose:
			print('########## ----------------------------------------------------------------------- ##########')
			print('Setting up the grid...')

		params.diffusivity = float(sys.argv[1])

		grid = Grid(params.grid_size, params.resolution, params.diffusivity, params.dt)
		grid_old = grid.setUpGrid()
		grid_new = np.copy(grid_old)

		X_grid, Y_grid = np.ogrid[:params.grid_size, :params.grid_size]
		ones_grid = np.ones((params.grid_size, params.grid_size))

		continuous_ejecta_blanket_factor = params.continuous_ejecta_blanket_factor
		resolution = params.resolution

		if params.verbose:
			print('Grid size (meters): {}'.format(grid.grid_width))
			print('Grid size (pixels): {}'.format(grid_old.shape))
			print('Grid resolution (m/px): {}'.format(grid.resolution))
			print('Total model time (Myr): {}'.format(params.model_time/(1.e6)))
			print('Timestep (Myr): {}'.format(params.dt/(1.e6)))
			print('Number of timesteps: {}'.format(params.nsteps))
			if params.diffusion_on:
				print('Grid diffusivity (m^2/yr): {}'.format(params.diffusivity))
			print('')

		if params.cratering_on:
			if params.verbose:
				print('########## ----------------------------------------------------------------------- ##########')
				print('Sampling impactor population...')

			impPop = ImpactorPopulation()

			d_craters, x_craters, y_craters, index_craters, t_craters, dist_secs = impPop.sample_all_craters()
			'''
			plt.figure()
			plt.scatter(x_craters, y_craters, s=1)
			plt.axvline(0.0, c='r')
			plt.axvline(params.grid_size, c='r')
			plt.axhline(0.0, c='r')
			plt.axhline(params.grid_size, c='r')

			x_craters = x_craters[np.where(x_craters < 200)]
			y_craters = y_craters[np.where(x_craters < 200)]

			plt.figure()
			plt.subplot(121)
			plt.hist(x_craters, bins=100)
			plt.axvline(0.0, c='r')
			plt.axvline(params.grid_size, c='r')
			plt.subplot(122)
			plt.hist(y_craters, bins=100)
			plt.axvline(0.0, c='r')
			plt.axvline(params.grid_size, c='r')
			plt.show()
			sys.exit()
			'''
			if params.verbose:
				if params.secondaries_on:

					d_sec_craters = d_craters[np.where(index_craters < 1.0)]
					index_sec_craters = index_craters[np.where(index_craters < 1.0)]
					print('Primary craters')
					print('Total number of craters: {}'.format(len(d_craters)))
					print('Number of primary craters: {}'.format(len(d_craters) - len(d_sec_craters)))
					print('Number of secondary craters: {}'.format(len(d_sec_craters)))

				else:
					print('Primary craters only')
					print('Total number of craters: {}'.format(len(d_craters)))
				print('Smallest crater: {}'.format(np.min(d_craters)))
				print('Largest crater: {}'.format(np.max(d_craters)))
				print('')

				if params.secondaries_on:
					plt.figure()
					plt.subplot(221)
					plt.hist(d_craters)
					plt.xlabel('Crater diameter (m)')
					plt.subplot(223)
					plt.hist(index_craters)
					plt.xlabel('Crater index (m)')
					plt.subplot(222)
					plt.hist(t_craters)
					plt.xlabel('Timestep')
					plt.subplot(224)
					plt.hist(dist_secs/1000.0)
					plt.xlabel('Secondary origination distance (km)')
					plt.show()

				else:
					plt.figure()
					plt.subplot(121)
					plt.hist(d_craters)
					plt.xlabel('Crater diameter (m)')
					plt.subplot(122)
					plt.hist(t_craters)
					plt.xlabel('Timestep')
					plt.show()

		if params.tracers_on:
			if params.verbose:
				print('########## ----------------------------------------------------------------------- ##########')
				print('Initializing tracer particles...')

			n_pd = params.n_particles_per_layer
			x_points = np.linspace(5, params.grid_size - 5, n_pd, dtype=np.int)
			y_points = np.linspace(5, params.grid_size - 5, n_pd, dtype=np.int)
			z_points = np.geomspace(1.0, (params.grid_width/4.0/1.17/10.0 + 1.0), n_pd, dtype=np.float)    # Particles end at excavation depth of crater 1/4 the grid width
			z_points = -1.0*(z_points - 1.0)
			z_points[0] = 0.0

			if params.verbose:
				print('Total number of particles: {}'.format(int(n_pd**3)))
				print('Particles per layer: {}'.format(n_pd))
				print('Min/Max depth of particles (m): {}/{}'.format(z_points[0], z_points[-1]))
				print('')
			XX, YY, ZZ = np.meshgrid(x_points, y_points, z_points)

			tracers = []
			for i in range(XX.shape[0]):
				for j in range(XX.shape[1]):
					for k in range(XX.shape[2]):
						x_p0 = XX[i,j,k]
						y_p0 = YY[i,j,k]
						z_p0 = ZZ[i,j,k]

						tracer = Tracer(x_p0, y_p0, z_p0)
						tracers.append(tracer)

		if params.verbose:
			print('########## ----------------------------------------------------------------------- ##########')
			print('Evolving the landscape...')

		median_slope_arr = np.zeros(params.nsteps)
		median_elev_arr = np.zeros(params.nsteps)

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

					if params.tracers_on:
						##### -------------------- #####
						# TRACERS
						# Effect of crater formation on tracer particles
						for j in range(len(tracers)):
							particle_position = tracers[j].current_position()

							x_p0 = particle_position[0]
							y_p0 = particle_position[1]
							z_p0 = particle_position[2]

							if ~np.isnan(x_p0) and ~np.isnan(y_p0) and ~np.isnan(z_p0):
								x_p0 = int(x_p0)
								y_p0 = int(y_p0)
								z_p0 = z_p0
								d_p0 = grid_old[x_p0, y_p0] - z_p0

								if d_p0 < 0.0:
									print('PARTICLE ABOVE SURFACE - BEGINNING OF TIMESTEP')
									sys.exit()
								dx = (x_p0 - x_crater_pix)*params.resolution
								dy = (y_p0 - y_crater_pix)*params.resolution

								if (0 <= x_crater_pix <= (params.grid_size - 1)) and (0 <= y_crater_pix <= (params.grid_size - 1)):
									dz = z_p0 - grid_old[x_crater_pix, y_crater_pix]
								else:
									dz = z_p0

								R0 = np.sqrt(dx**2 + dy**2 + dz**2)

								# If particle sufficiently close to sphere of influence, pass to tracer particle cratering method
								if R0 <= 1.75*params.continuous_ejecta_blanket_factor*crater_radius:
									particle_position_new = tracers[j].tracer_particle_crater(x_p0, y_p0, z_p0, d_p0, dx, dy, dz, x_crater_pix, y_crater_pix, R0, crater_radius, grid_old, grid_new)

									tracers[j].update_position(particle_position_new)

						##### -------------------- #####
					##### --------------------------------------------------------------------- #####
					# Update grid after adding crater
					grid_old = np.copy(grid_new)

				##### --------------------------------------------------------------------- #####
				# Update grid after adding all craters
				grid_old = np.copy(grid_new)

			if params.pixel_noise_on:
				##### ------------------------------------------------------------------------- #####
				# SUB-PIXEL NOISE

				#noise_grid = np.random.choice(params.noise_arr, (params.grid_size, params.grid_size))*np.random.choice([-1.0, 1.0], (params.grid_size, params.grid_size))

				noise_grid = -1.0*np.abs(np.random.choice(params.noise_arr, (params.grid_size, params.grid_size)))
				#noise_grid = -0.05*np.ones((params.grid_size, params.grid_size))

				'''
				plt.figure()
				plt.hist(noise_grid.flatten(), bins=50)
				plt.show()
				sys.exit()
				'''
				grid_new = grid_old + noise_grid

				if params.tracers_on:
					##### -------------------- #####
					# TRACERS
					# Effect of addition of sub-pixel noise on tracer particles
					for j in range(len(tracers)):
						particle_position = tracers[j].current_position()

						x_p0 = particle_position[0]
						y_p0 = particle_position[1]
						z_p0 = particle_position[2]

						if ~np.isnan(x_p0) and ~np.isnan(y_p0) and ~np.isnan(z_p0):
							x_p0 = int(x_p0)
							y_p0 = int(y_p0)
							z_p0 = z_p0

							particle_position_new = tracers[j].tracer_particle_noise(x_p0, y_p0, z_p0, grid_old, grid_new)

							tracers[j].update_position(particle_position_new)
					##### -------------------- #####

				# Update grid after noise
				grid_old = grid_new.copy()

			if params.diffusion_on:
				##### --------------------------------------------------------------------- #####
				# DIFFUSION
				# Compute topographic diffusion.  If you are using explicit diffusion make sure that the timestep meets the Courant stability criterion
				if params.implicit_diffusion:
					grid_new = grid.implicit_diffusion2D(grid_old)
				elif params.explicit_diffusion:
					grid_new = grid.explicit_diffusion2D(grid_old)

				if params.tracers_on:
					##### -------------------- #####
					# TRACERS
					# Effect of topographic diffusion on tracer particles
					for j in range(len(tracers)):
						particle_position = tracers[j].current_position()

						if np.isnan(particle_position[0]) or np.isnan(particle_position[1]) or np.isnan(particle_position[2]):
							pass
						else:
							particle_position_new = tracers[j].tracer_particle_diffusion(grid_old, grid_new, tracers[j].current_position())

							tracers[j].update_position(particle_position_new)
					##### -------------------- #####

				# Update grid after diffusion
				grid_old = grid_new.copy()

			if params.tracers_on:
				##### -------------------- #####
				# TRACERS
				# Store final tracer particle position, depth, and surface slope
				# Compute surface slopes at all points on the grid
				x_slope, y_slope = np.gradient(grid_old, params.resolution)
				slope_grid = np.rad2deg(np.arctan(np.sqrt( x_slope**2 + y_slope**2)))

				for j in range(len(tracers)):
					particle_position_final = tracers[j].current_position()

					x_final = particle_position_final[0]
					y_final = particle_position_final[1]
					z_final = particle_position_final[2]

					if np.isnan(x_final) or np.isnan(y_final) or np.isnan(z_final):
						depth_final = np.nan
						slope_final = np.nan
					else:
						x_final = int(x_final)
						y_final = int(y_final)

						depth_final = grid_old[x_final, y_final] - z_final
						slope_final = slope_grid[x_final, y_final]

					tracers[j].update_trajectory(x_final, y_final, z_final, depth_final, slope_final)

			if params.verbose:
				progress(t, params.nsteps)

			x_slope, y_slope = np.gradient(grid_old, params.resolution)
			slope_grid = np.rad2deg(np.arctan(np.sqrt( x_slope**2 + y_slope**2)))
			slope_grid = slope_grid[1:-1, 1:-1]

			median_slope = np.median(abs(slope_grid.flatten()))
			median_slope_arr[t] = (median_slope)
			median_elev_arr[t] = np.average(grid_old)

		if params.verbose:
			progress(params.nsteps, params.nsteps)
			print('')


		grid = np.copy(grid_old)

		grid = grid[int(grid.shape[0]/2 - 50):int(grid.shape[0]/2 + 50), int(grid.shape[0]/2 - 50):int(grid.shape[0]/2 + 50)]
		print(grid.shape)
		'''
        fname = '/extra/pob/Calibration_17m/' + str(sys.argv[1]) + '/' + str(uuid.uuid4()) + '.txt'

        np.savetxt(fname, grid)
		'''

		'''
		grid = grid_old.copy()

		x_slope, y_slope = np.gradient(grid, params.resolution)
		slope_grid = np.rad2deg(np.arctan(np.sqrt( x_slope**2 + y_slope**2)))
		slope_grid = slope_grid[1:-1, 1:-1]

		median_slope = np.median(abs(slope_grid.flatten()))
		'''

		if params.verbose:

			if params.tracers_on:
				ls = LightSource(azdeg=270, altdeg=30.0)

				plt.figure()
				plt.subplot(221)
				plt.imshow(ls.hillshade(grid.T, dx=params.resolution, dy=params.resolution), extent=(0, params.grid_width, 0, params.grid_width), cmap='gray')
				plt.subplot(222)
				plt.imshow(grid.T)

				plt.figure()
				plt.subplot(121)
				plt.plot(np.arange(len(median_slope_arr))*params.dt/(1.e6), median_slope_arr)
				plt.xlabel('Time (Myr)')
				plt.ylabel('Median slope (deg)')
				plt.subplot(122)
				plt.plot(np.arange(len(median_slope_arr))*params.dt/(1.e6), median_elev_arr)
				plt.axhline(0.0, color='k', linestyle='--')
				plt.xlabel('Time (Myr)')
				plt.ylabel('Mean grid elevation (m)')

				for j in range(len(tracers)):
					pos = tracers[j].current_position()
					plt.scatter(pos[0], pos[1], c='r', s=2)
				plt.xlabel('Distance (m)')
				plt.ylabel('Distance (m)')
				plt.title('Final landscape')


				surf_res = []
				lost_count = 0
				sample_count = 0
				sample_depth = 0.1

				plt.subplot(223)
				for j in range(len(tracers)):
					depth_arr = tracers[j].d_arr

					if ~np.isnan(depth_arr[-1]):
						if depth_arr[-1] <= params.sampling_depth:
							sample_count += 1
							surf_steps = sum(1 for d in depth_arr if d <= params.surface_depth)
							surf_time = surf_steps*params.dt
							surf_res.append(surf_time)

							plt.plot(range(params.nsteps), depth_arr)
					else:
						lost_count += 1
				surf_res = np.array(surf_res)/(1.e9)
				plt.subplot(224)
				plt.hist(surf_res, density=True)
				plt.xlabel('Surface residence time (Gyr)')
				plt.ylabel('Normalized Count')
				plt.title('Residence times of tracer particles - sample depth:{} (m)'.format(params.sampling_depth))

				print('Total model time: {}'.format((params.nsteps)*params.dt/(1.e9)))
				print('Particles lost: {}'.format(lost_count))
				print('Sampled particles: {}'.format(sample_count))
				print('Surface residence: ', np.nanmin(surf_res), np.nanmax(surf_res))
				print('Median slope: {} degrees'.format(median_slope))

				print('')
				runtime = time()- starttime
				print('Runtime (min): {}'.format(runtime/3600.0))
				print('')

			else:
				ls = LightSource(azdeg=270, altdeg=30.0)

				plt.figure()
				plt.subplot(121)
				plt.imshow(ls.hillshade(grid.T, dx=params.resolution, dy=params.resolution), extent=(0, params.grid_width, 0, params.grid_width), cmap='gray')
				plt.subplot(122)
				plt.imshow(grid.T)

				plt.figure()
				plt.subplot(121)
				plt.plot(np.arange(len(median_slope_arr))*params.dt/(1.e6), median_slope_arr)
				plt.xlabel('Time (Myr)')
				plt.ylabel('Median slope (deg)')
				plt.subplot(122)
				plt.plot(np.arange(len(median_slope_arr))*params.dt/(1.e6), median_elev_arr)
				plt.axhline(0.0, color='k', linestyle='--')
				plt.xlabel('Time (Myr)')
				plt.ylabel('Mean grid elevation (m)')

				print('')
				runtime = time()- starttime
				print('Runtime (min): {}'.format(runtime/60.0))
				print('')

			plt.show()

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
