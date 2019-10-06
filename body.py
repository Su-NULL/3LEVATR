# Planetary body parameters
import numpy as np
import matplotlib.pyplot as plt

radius_body = 1.7374e6  # m
g = 1.62

# Impactor size-frequency distribution
sfd = np.loadtxt('files/moon_impact_sfd.txt')
diameter_bins_raw = sfd[:, 0]
cumulative_number_raw = sfd[:, 1]			# (N>D) km^-2*yr^-1

# Impact velocity distribution
velocity_distribution = np.loadtxt('files/moon_velocity_distribution_v2.txt')
velocities = velocity_distribution[:, 0]
velocity_probability = velocity_distribution[:, 1]
velocity_cumulative_probability = velocity_distribution[:, 2]
velocity_delta = velocities[1] - velocities[0]
velocity_max = np.max(velocities)  # km/s
velocity_max_likelihood = [float(velocities[np.argwhere(velocity_cumulative_probability == min(
    velocity_cumulative_probability[(velocity_cumulative_probability - r) > 0]))]) for r in [0.5]][0]		# km/s
velocity_average = np.average(velocities, weights=velocity_probability)

# Impact angle distribution (most likely always going to be sin(2i))
impact_angles = np.linspace(0.0, np.pi/2.0, 1000, endpoint=True)

angle_delta = impact_angles[1] - impact_angles[0]
prob = 0.5*np.sin(2.0*impact_angles)
prob[-1] = 0.0
cdf = np.cumsum(prob)
angle_cumulative_probability = cdf/np.max(cdf)

# Impact process parameters for the body
impactor_density = 2700.0  # Average impactor density at the body, kg/m^3
regolith_density = 1500.0  # kg/m^3
bedrock_density = 2550.0  # kg/m^3, from Wieczorek et al. 2013, GRAIL results
regolith_strength = 1.0e3  # Pa, from Mitchell et al. 1972
bedrock_strength = 2.0e7  # Pa, from Marchi et al. 2009 via Asphaug et al. 1996

# Average fractured depth and regolith thickness
# Use value known over some other region (e.g. highlands)
known_avg_fractured_depth = 10000.0
known_avg_age = 4.5e9

model_avg_regolith_thickness = 5.0 # Weber, Encyclopedia of the Solar System, 3rd edition, 2014
model_avg_age = 3.5e9

# t^1/2 dependence to solve for the average thickness corresponding to your model region and age
avg_fractured_depth = (known_avg_fractured_depth /
                       (known_avg_age**(0.5)))*(model_avg_age**(0.5))
