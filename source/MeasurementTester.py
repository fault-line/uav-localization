import matplotlib.pyplot as plt
import numpy as np
import cv2

from Utility import calc_weight_brief, calc_weight_obj
from Localizer import *
from DataProcessor import *


''' Collection of functions for assessing measurement models.'''


def calc_truth_weights_terrain(map_file, camera_path, weights_file, truth_file, frames=10):
	'''
	Test observation and measurement model by calculating particle weights of ground truth
	observations.

	Args:
		map_file: path to file of the global map to use
		camera_path: path to camera images from the flight
		weights_file: file with weights for the learning model
		truth_file: file with ground truth trajectory
		frames: plot this many observations
	'''

	localizer = Localizer(map_file, camera_path, weights_file, frames, num_particles=1)
	processor = localizer.get_processor()
	truth = processor.read_truth(truth_file, all_dims=True)

	brief_weights = []
	for i in range(frames):
		x, y, z, angle = truth[i,0], truth[i,1], truth[i,2], truth[i,3]
		frame = processor.get_camera_image(i+2)
		frame = processor.resize_to_scale(frame, z)
		frame_pred = processor.construct_prediction(frame)
		particle_pred = processor.get_particle_area(x, y, angle, w=frame_pred.shape[1],
													h=frame_pred.shape[0])
		brief_weight = calc_weight_brief(frame_pred, particle_pred)
		brief_weights.append(brief_weight)

	brief_weights = np.array(brief_weights)
	return brief_weights


def read_objects(object_pred_file):
	'''
	Read YOLOv3 output file and store information about detected objects.
	'''
	f = open(object_pred_file)
	object_lines = f.readlines()

	# convert to array of floats
	object_truth = []
	for line in object_lines[2:]:
		line = line.strip()
		line = np.array(line.split(" "), dtype="float")

		object_truth.append(line)
	object_truth = np.array(object_truth)

	# filter for "building"-type object classes
	object_truth = object_truth[object_truth[:,3] > 45]
	return object_truth


def create_image_prediction(object_truth, img_x, img_y, img_c=60, patch_dim=28):
	'''
	Create a prediction tensor based on the results of object prediction with YOLOv3.
	'''
	img_prediction = np.zeros((img_y//patch_dim, img_x//patch_dim, img_c))

	for obj in object_truth:
		x1, y1, x2, y2, obj_class = obj[0], obj[1], obj[2], obj[3], int(obj[4])
		x1, y1, x2, y2 = int(x1/patch_dim), int(y1/patch_dim), int(x2/patch_dim), int(y2/patch_dim)
		img_prediction[y1:y2, x1:x2, obj_class] += 1
	return img_prediction


def calc_truth_weights_obj_detection(map_objects_file):
	'''
	Calculate the particle weights of observations at ground truth locations.
	CURRENTLY HARD-CODED.
	'''
	map_objects = read_objects(map_objects_file)
	map_prediction = create_image_prediction(map_objects, 4800, 4800)

	camera_prediction_path = "../data/object_detection/Flight2/"
	truth_file = "../data/ground_truth/Flight2/traj_truth.txt"

	processor = DataProcessor()
	true_traj = processor.read_truth(truth_file, all_dims=True)

	weights = []
	for i in range(266, 276):
		camera_prediction_file = "{}{:03d}.txt".format(camera_prediction_path, i)
		detected_objects = read_objects(camera_prediction_file)
		camera_prediction = create_image_prediction(detected_objects, 1280, 720)

		pX, pY, pZ, angle = true_traj[i+2][0], true_traj[i+2][1], true_traj[i+2][2], true_traj[i+2][3]
		#camera_prediction = processor.resize_to_scale(camera_prediction, pZ)
		particle_prediction = processor.get_particle_area(pX=pX, pY=pY, angle=angle,
			w=1280*pZ, h=720*pZ, another_map=map_prediction, padded_map=False)

		weight = calc_weight_obj(camera_prediction, particle_prediction)
		weights.append(weight)
	return weights


def plot_truth_weights_obj_detection():
	'''
	Plot the particle weights of observations (OBJECT DETECTION) at ground truth locations
	and compare to expected weights (=1.0).  CURRENTLY HARD-CODED.
	'''

	# declare paths to YOLOv3 results for global maps
	map1_objects_file = "../data/object_detection/maps/AdM-01-2014.txt"
	map2_objects_file = "../data/object_detection/maps/AdM-03-2014.txt"

	map1_truth_weights = calc_truth_weights_obj_detection(map1_objects_file)
	map2_truth_weights = calc_truth_weights_obj_detection(map2_objects_file)
	frames = np.arange(1,11)

	plot_title = "Particle Weights of Ground Truth Observations with Select Camera Frames,\n" +\
				 "Measurement: Object Detection/SSE-based Weighting."
	plt.title(plot_title)
	plt.plot(frames, map1_truth_weights, label="Actual Weight (Map 1)")
	plt.plot(frames, map2_truth_weights, label="Actual Weight (Map 2)")
	plt.plot(frames, [1]*len(frames), "--", label="Expected Weight")
	plt.xlabel("Camera Frame Number")
	plt.ylabel("Calculated Particle Weight")
	plt.legend()
	plt.show()


def plot_truth_weights_terrain_class():
	'''
	Plot the particle weights of observations (TERRAIN CLASSIFICATION) at ground truth locations
	and compare to expected weights (=1.0).  CURRENTLY HARD-CODED.
	'''
	map1_file = "../data/maps/AdM-01-2014.jpg"
	map2_file = "../data/maps/AdM-03-2014.jpg"
	camera_path = "../data/camera_feed/Flight2/"
	weights_file = "../data/learning_model/sat6_weights.hdf5"
	truth_file = "../data/ground_truth/Flight2/traj_truth.txt"

	map1_truth_weights = calc_truth_weights_terrain(map1_file, camera_path, weights_file,
		truth_file)
	map2_truth_weights = calc_truth_weights_terrain(map2_file, camera_path, weights_file,
		truth_file)

	plot_title = "Particle Weights of Ground Truth Observations with Select Camera Frames,\n" +\
				 "Measurement: Terrain Classification/BRIEF-based Weighting."
	frames = np.arange(1, 11)
	plt.title(plot_title)
	plt.plot(frames, map1_truth_weights, label="Actual Weight (Map 1)")
	plt.plot(frames, map2_truth_weights, label="Actual Weight (Map 2)")
	plt.plot(frames, [1]*len(frames), "--", label="Expected Weight")
	plt.plot(frames, [0.5]*len(frames), "--", label="Weight of Random Observation")
	plt.xlabel("Camera Frame Number")
	plt.ylabel("Calculated Particle Weight")
	plt.legend()
	plt.show()

if __name__ == "__main__":
	#plot_truth_weights_terrain_class()
	plot_truth_weights_obj_detection()
