import numpy as np
import scipy


class ParticleFilter:
	''' Implementation of particle filter for robot localization.'''

	def __init__(self, x_range, y_range, z_range, N):
		'''
		Initialize ParticleFilter object.
		Args:
			x_range: allowed range of x-values (implied in pixels)
			y_range: allowed range of y-values (implied in pixels)
			z_range: allowed range of z-values (height, assumed in meters)
				  N: number of particles to use
		'''
		self.x_range = x_range
		self.y_range = y_range
		self.z_range = z_range
		self.angle_range = (-180, 180)
		self.N = N
		
		self.particles = self.initialize_particles()
		self.weights = np.divide(np.ones(self.N), self.N)


	def reset_particles(self):
		self.particles = self.initialize_particles()
		self.weights = np.divide(np.ones(self.N), self.N)


	def initialize_particles(self):
		'''
		Initialize a set of 4-DOF particles with random values.
		'''
		particles = np.empty((self.N, 4))
		particles[:, 0] = np.random.uniform(self.x_range[0], self.x_range[1], size=self.N)
		particles[:, 1] = np.random.uniform(self.y_range[0], self.y_range[1], size=self.N)
		particles[:, 2] = np.random.uniform(self.z_range[0], self.z_range[1], size=self.N)
		particles[:, 3] = np.random.uniform(self.angle_range[0], self.angle_range[1], size=self.N)
		return particles


	def motion_update(self, dx, dy, noise_factor=0.1):
		'''
		Apply motion update to particles (modified with random noise).
		Args:
					  dx: change in x-coordinate from VisualOdometry
					  dy: change in y-coordinate form VisualOdometry
			noise_factor: multiplier of applied noise
		'''

		# calculate updates
		update_x = dx * np.cos(self.particles[:, 3]) + (np.random.randn(self.N)*dx * noise_factor)
		update_y = dy * np.sin(self.particles[:, 3]) + (np.random.randn(self.N)*dy * noise_factor)
		update_z = np.random.randn(self.N) * noise_factor
		update_angle = np.rad2deg(np.arctan2(dx, dy)) + (np.random.randn(self.N) * noise_factor)
		
		# apply updates
		self.particles[:, 0] += update_x
		self.particles[:, 1] += update_y
		self.particles[:, 2] += update_z

		# update heading
		self.particles[:, 3] += update_angle
		self.particles[:, 3] = ((self.particles[:, 3] + 180) % 360) - 180


	def sensor_update(self, new_weights):
		'''
		Perform sensor data update by using new particle weights.
		Args:
			new_weights: set of weights to modify old weights with
		'''
		new_weights = np.divide(new_weights, np.sum(new_weights))
		self.weights = np.multiply(self.weights, new_weights)
		self.weights = self.weights + 1.e-18  # avoid round-off to zero
		self.weights = np.divide(self.weights, np.sum(self.weights)) # normalize


	def estimate(self):
		'''
		Calculate weighted average and variance of particles.
		'''
		pos = self.particles[:, 0:3]
		mean = np.average(pos, weights=self.weights, axis=0)
		var  = np.average((pos - mean)**2, weights=self.weights, axis=0)
		return mean, var


	def resample(self):
		'''
		Perform resampling with replacement based on particle weights.
		'''
		retained_indices = np.random.choice(self.N, self.N, replace=True, p=self.weights)
		self.particles = self.particles[retained_indices]
		self.weights = self.weights[retained_indices]



if __name__ == "__main__":
	# checking module functionality with random values

	x_range = (0, 100)
	y_range = (0, 100)
	z_range = (0, 10)
	angle_range = (-180, 180)
	N = 10

	p_filter = ParticleFilter(x_range, y_range, z_range, N)
	p_filter.initialize_particles()

	print("Initiatlized {} particles.".format(N))
	print("Range of x values:", p_filter.x_range)
	print("Range of y values:", p_filter.y_range)
	print("Range of z values:", p_filter.z_range)
	print("Range of heading values:", p_filter.angle_range)
	print("--------------------------------------\n")

	for i in range(10):
		print("Iteration {}:".format(i+1))

		dx, dy = np.round(np.random.rand(2) * 10, 3)
		print("Applying motion_update:", dx, dy)
		p_filter.motion_update(dx, dy)

		new_weights = np.random.rand(N) * 100
		print("Applying sensor update.")	
		p_filter.sensor_update(new_weights)

		print("Resampling particles.\n")
		p_filter.resample()

		print("Position estimate:")
		pos_mean, pos_var = p_filter.estimate()
		print("Mean:", pos_mean)
		print("Variance:", pos_var)
		print("--------------------------------------\n")



		