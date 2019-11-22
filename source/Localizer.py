####################### EXTERNAL IMPORTS ###################################
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from keras.models import Sequential, load_model
import cv2
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
############################################################################

####################### INTERNAL IMPORTS ###################################
from Utility import rotate_image, pair_pixels, brief_encode
from Utility import calc_weight_brief, calc_weight_sse, calc_weight_cossim
from DataProcessor import *
from VisualOdometry import *
from ParticleFilter import *
############################################################################


class Localizer:
	'''Full localization solution, with various performance assessment functions.
	   Mostly centered around terrain classification. Does not work properly yet.'''

	def __init__(self, map_file, camera_path, weights_file, frames, num_particles=1000):
		'''
		Initialize Localizer object.
		Args:
			    map_file: path to file of the global map to use
			 camera_path: path to camera images taken during the flight
		    weights_file: path to weights file to use learning model with
				  frames: total count of images taken during the flight
		'''
		self._set_processor(map_file=map_file,
							camera_path=camera_path,
							weights_file=weights_file)

		self._set_filter(x_range=(600, 4200), y_range=(600, 4200), z_range=(30,150), N=num_particles)

		self._set_odometry(camera_path=camera_path, last_frame=frames)


	# ------------------------ OBJECT SETTERS/GETTERS ------------------------#
	def _set_processor(self, map_file, camera_path, weights_file):
		self.processor = DataProcessor(map_file, camera_path, weights_file)

	def get_processor(self):
		return self.processor

	def _set_filter(self, x_range, y_range, z_range, N):
		self.filter = ParticleFilter(x_range, y_range, z_range, N)

	def get_filter(self):
		return self.filter

	def _set_odometry(self, camera_path, last_frame):
		self.odometry = VisualOdometry(camera_path, last_frame)

	def get_odometry(self):
		return self.odometry
	# ------------------------------------------------------------------------#


	# -------------- PERFORMANCE ASSESSMENT AND RESULT PLOTTING --------------#
	
	def plot_trajectory(self, estimated_traj, true_traj, plot_title="Flight Trajectory"):
		'''
		Plot estimated and true trajectories (only considers x and y values).
		Args:
			estimated_traj: x,y-value pairs of trajectory estimated with particle filter
				 true_traj: x,y-value pairs of trajectory declared in ground truth
				plot_title: title for the plot
		'''

		# initialize plot
		fig, ax = plt.subplots()
		ax.title.set_text(plot_title)

		# show map
		img = plt.imread(self.processor.map_file)
		ax.imshow(img)

		# plot trajectories
		ax.plot(estimated_traj[:,0], estimated_traj[:,1], 'r', label="Estimated Trajectory")
		ax.plot(true_traj[:,0], true_traj[:,1], label="True Trajectory")
		ax.legend()
		ax.set_xlabel("X-values (pixels)")
		ax.set_ylabel("Y-values (pixels)")
		plt.show()


	def collect_mean_data(self, true_traj, runs=5, measurement="sse"):
		'''
		Calculate mean errors and variances in trajectory between the runs.
		Args:
			true_traj: array with x,y-value pairs of true trajectory
				 runs: number of times to run the particle filter
		Returns mean errors and variances.
		'''
		errors = []
		variances = []
		for i in range(runs):
			print("ITERATION:", i+1)
			estimated_traj, variance_traj = self.process_flight(measurement=measurement)
			self.odometry.reset_frames()
			self.filter.reset_particles()

			error_xy = np.abs(estimated_traj - true_traj)
			error_traj = np.hypot(error_xy[:,0], error_xy[:,1])

			errors.append(error_traj)
			variances.append(variance_traj)

		mean_err = np.mean(np.array(errors), axis=0)
		mean_var = np.mean(np.array(variances), axis=0)
		return mean_err, mean_var


	def plot_errors(self, mean_err, plot_title="Mean Trajectory Errors"):
		'''
		Plot mean errors in trajectory.
		Args:
			  mean_err: mean errors on each camera frame
			plot_title: title of the plot
		'''
		plt.title(plot_title)
		plt.plot(np.arange(1, mean_err.shape[0]+1), mean_err)
		plt.xlabel("Iteration (Camera Frame Number)")
		plt.ylabel("Mean Trajectory Error (pixels, 1px ~ 0.25m)")
		plt.show()


	def plot_variances(self, mean_var, plot_title="Mean Position Variances"):
		'''
		Plot mean variances of position.
		Args:
			mean_var: mean variances in x and y coordinates on each camera frame
		  plot_title: title of the plot
		'''
		plt.title(plot_title)
		plt.plot(np.arange(1, mean_var.shape[0]+1), mean_var[:,0], label="along x-axis")
		plt.plot(np.arange(1, mean_var.shape[0]+1), mean_var[:,1], label="along y-axis")
		plt.xlabel("Iteration (Camera Frame Number)")
		plt.ylabel("Mean Variance (pixels, 1px ~ 0.25m)")
		plt.legend()
		plt.show()
	# ------------------------------------------------------------------------#


	# ----------------------------- LOCALIZATION -----------------------------#
	def process_flight(self, measurement="brief"):
		print("Begin processing flight.")

		# initialize objects
		processor = self.get_processor()
		p_filter = self.get_filter()
		odometry = self.get_odometry()
		print("All objects initialized.")

		# initialize MCL particles
		p_filter.initialize_particles()
		particles = p_filter.particles
		print("MCL particles initialized.")

		mean_positions = []
		var_positions = []
		timer_start = time.time()
		# process all flight images
		for i in range(2, odometry.last_frame+1):
			# get camera image
			frame = processor.get_camera_image(i)
			print("Frame {}:".format(i))

			# extract mean optical flow, update particle positions
			dx, dy = odometry.process_next_frame()
			p_filter.motion_update(dx, dy)

			print("Received motion update from odometry: {:.3f}, {:.3f}".format(dx, dy))
			print("Begin measuring particles.")

			# keep frame resize constant to conserve resources (downscale by ~half on each side)
			frame = processor.adjust_for_height(frame, drone_height=90)
			frame_prediction = processor.construct_prediction(frame)
			pred_shape = [frame_prediction.shape[0], frame_prediction.shape[1]]

			# weigh particles
			weights = []
			for particle in particles:
				x, y, z, angle = particle[0], particle[1], particle[2], particle[3]
				particle_prediction = processor.get_particle_area(pX=x, pY=y, angle=angle,
																w=pred_shape[1], h=pred_shape[0])
				if measurement == "sse":
					weight = calc_weight_sse(frame_prediction, particle_prediction)
				else:
					weight = calc_weight_brief(frame_prediction, particle_prediction)
				weights.append(weight)

			# update particle weights, resample
			p_filter.sensor_update(np.array(weights))
			p_filter.resample()

			pos_mean, pos_var = p_filter.estimate()
			mean_positions.append(pos_mean)
			var_positions.append(pos_var)
			print("Mean:", np.round(pos_mean,3))
			print("Variance: {}\n".format(np.round(pos_var,3)))

		print("\nTotal time: {:.3f} seconds.".format(time.time()-timer_start))
		trajectory = np.array(mean_positions)[:, :2]
		var_positions = np.array(var_positions)[:, :2]
		return trajectory, var_positions


if __name__ == "__main__":

	# declare paths to camera images for each flight
	camera_flight1 = "../data/camera_feed/Flight1/"
	camera_flight2 = "../data/camera_feed/Flight2/"

	# global maps files
	map1_file = "../data/maps/AdM-01-2014.jpg"
	map2_file = "../data/maps/AdM-03-2014.jpg"

	# ground truth files
	truth_flight1 = "../data/ground_truth/Flight1/traj_truth.txt"
	truth_flight2 = "../data/ground_truth/Flight2/traj_truth.txt"

	# learning model weights
	weights_file = "../data/learning_model/sat6_weights.hdf5"


	# simulate flight 1 on map 1 and plot results
	localizer_fl1_map1 = Localizer(map1_file, camera_flight1, weights_file, frames=358, num_particles=1000)
	estimated_traj = localizer_fl1_map1.process_flight()[0]
	true_traj = localizer_fl1_map1.get_processor().read_truth(truth_flight1)
	localizer_fl1_map1 = plot_trajectory(estimated_traj, true_traj, plot_title="Flight Trajectory")

	





