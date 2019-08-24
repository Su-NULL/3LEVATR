# Function to sample from a cosine distribution approximated by a normal distribution for speed
def lat_cos_approx(n_samples):
	import numpy as np
	from scipy import stats
	lower, upper = -np.pi/2.0, np.pi/2.0
	mu, sigma = 0, 0.75*np.sqrt( (np.pi**2/3.0) - 2.0)
	X = stats.truncnorm(
	    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
	return X.rvs(n_samples)

def haversine(lon1, lat1, lon2, lat2, r_body):
	import numpy as np
	# Great-circle formula (needs to handle large arrays)
	"""
    	Calculate the great circle distance between two points 
    	on the earth (specified in decimal degrees)
	"""
    	# convert decimal degrees to radians 
	lon1 = np.deg2rad(lon1)
	lat1 = np.deg2rad(lat1)
	lon2 = np.deg2rad(lon2)
	lat2 = np.deg2rad(lat2)
    	# haversine formula 
	dlon = lon2 - lon1 
	dlat = lat2 - lat1 
	a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
	c = 2 * np.arcsin(np.sqrt(a)) 
	# Radius of earth in kilometers is 6371
	km = r_body* c
	bearing = np.arctan2(np.sin(lon2-lon1)*np.cos(lat2), np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(lon2-lon1))
	
	return km, bearing

# Progress bar function for display purposes
def progress(count, total, status=''):
	import sys
	bar_len = 60
	filled_len = int(round(bar_len * count / float(total)))
	
	percents = round(100.0 * count / float(total), 1)
	bar = '=' * filled_len + '-' * (bar_len - filled_len)
	
	sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
	sys.stdout.flush()

def grid_shade(z, res):
    import numpy as np
    inc = np.deg2rad(60.0)
    out_grid = np.zeros(z.shape)
    for i in range(z.shape[0]-1):
        for j in range(z.shape[1]-1):
            
            if i == z.shape[0]-1:
                out_grid[i,j] = 0.0
            else:
                slope = np.arctan2((z[i,j] - z[i+1,j]), (res))
                
                ang = np.cos(inc - slope)
                if ang < 0.0:
                    ang = 0.0
                else:
                    ang = ang
                    
                out_grid[i,j] = ang
                
    return out_grid
