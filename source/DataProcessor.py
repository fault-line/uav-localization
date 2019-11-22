from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from keras.models import Sequential, load_model
import cv2
import numpy as np
import pandas as pd
import time

from Utility import rotate_image

# CLASS CODES:
# 1 0 0 0 0 0 - BUILDING    - 0
# 0 1 0 0 0 0 - BARREN LAND - 1
# 0 0 1 0 0 0 - TREES       - 2
# 0 0 0 1 0 0 - GRASSLAND   - 3
# 0 0 0 0 1 0 - ROAD        - 4
# 0 0 0 0 0 1 - WATER       - 5


class DataProcessor:
	''' Module with various data and image processing functions.
		Mostly centered around terrain classification.'''

	def __init__(self, map_file=None, camera_path=None, weights_file=None):
		self.classes = 6 # number of terrain classes
		self.patch_dim = 28 # side dimensions of a patch (pixels)
		#self.max_x = 4787 # maximum x-coordinate (pixels)
		#self.max_y = 4787 # maximum y-coordinate (pixels)
		#self.min_z = 40   # minimum z-coordinate (drone height, meters)
		#self.max_z = 80  # maximum z-coordinate (drone height, meters)
		self.map_res = 0.243 # spatial resolution of the map (meters/pixel)
		self.cam_x = 1280 # drone camera image width  (pixels)
		self.cam_y = 720  # drone camera image height (pixels)

		# ratios estimated from declared diagonal FOV of drone camera (94 degrees):
		self.cam_xz_ratio = 1.87 # ratio of x-dimension covered by image (in meters) to drone height
		self.cam_yz_ratio = 1.05 # ratio of y-dimension covered by image (in meters) to drone height

		self.map_prediction = None
		if (map_file is not None) and (camera_path is not None) and (weights_file) is not None:
			self.make_model_CNN()
			self.model.load_weights(weights_file)

			self.camera_path = camera_path # path to drone camera images
			self.map_file = map_file
			self.map = cv2.imread(map_file)
			self.map_prediction = self.construct_prediction(self.map, pad=True)


	def load_learning_data(self, X_train_file, X_test_file, y_train_file, y_test_file):
		'''
		Load SAT6 dataset. Only used for training the neural network.
		Args:
			X_train_file: path to X_train csv file of SAT6
			 X_test_file: path to X_test csv file of SAT6
			y_train_file: path to y_train csv file of SAT6
			 y_test_file: path to y_test csv file of SAT6
		Returns training and validation SAT6 data as numpy arrays.
		'''
		X_train_df = pd.read_csv(X_train_file, header=None)
		X_test_df  = pd.read_csv(X_test_file, header=None)
		y_train_df = pd.read_csv(y_train_file, header=None)
		y_test_df  = pd.read_csv(y_test_file, header=None)

		X_train = X_train_df.values.reshape((-1,28,28,4)).clip(0,255).astype("uint8")[:,:,:,:3]
		X_test  = X_test_df.values.reshape((-1,28,28,4)).clip(0,255).astype("uint8")[:,:,:,:3]
		y_train = y_train_df.values.getfield(dtype="int8")
		y_test  = y_test_df.values.getfield(dtype="int8")
		return X_train, X_test, y_train, y_test


	def train_model(self):
		'''
		Train the learning model on SAT6 dataset.
		'''
		X_train, X_test, y_train, y_test = self.load_data(X_train_file="X_train_sat6.csv",
	                							X_test_file="X_test_sat6.csv",
	                							y_train_file="y_train_sat6.csv",
	                							y_test_file="y_test_sat6.csv")

		self.model.fit(X_train, y_train, batch_size=50, validation_data=(X_test, y_test), epochs=5)
		#model.save_weights('deepsat6-6epochs-weights.h5')


	def read_truth(self, truth_file, all_dims=False):
		'''
		Read ground truth file.
		Args:
			truth_file: file containing ground truth poses for each flight image
		Returns array of true x,y-value pairs.
		'''

		# open ground truth file and read lines
		f = open(truth_file)
		truth_lines = f.readlines()
		f.close()

		# convert to array of floats
		ground_truth = []
		for line in truth_lines[2:]:
			line = line.strip()
			line = np.array(line.split(" "), dtype="float")
			ground_truth.append(line)
		ground_truth = np.array(ground_truth)

		if not all_dims:
			# only x and y values
			ground_truth = ground_truth[:, :2]
		return ground_truth


	def get_camera_image(self, frame_number, format=".jpg"):
		'''
		Get camera image by number.
		Args:
			frame_number: number of image in the set
				  format: file format (".jpg" or ".png", for example)
		Returns specified image.
		'''
		image_file = "{}{:03d}{}".format(self.camera_path, frame_number, format)
		return cv2.imread(image_file)


	def make_model_CNN(self):
		'''
		Make convolutional neural network model for terrain classification.
		'''

		self.model = Sequential()

		self.model.add(Conv2D(16, (3,3), activation='relu', input_shape=(28,28,3)))
		self.model.add(Conv2D(32, (3,3), activation='relu'))
		self.model.add(MaxPool2D(pool_size=(2,2)))
		self.model.add(Dropout(0.5))

		self.model.add(Conv2D(32, (3,3), activation='relu'))
		self.model.add(Conv2D(64, (3,3), activation='relu'))
		self.model.add(MaxPool2D(pool_size=(2,2)))
		self.model.add(Dropout(0.5))

		self.model.add(Flatten())
		self.model.add(Dense(128, activation='relu'))

		self.model.add(Dropout(0.5))
		self.model.add(Dense(6, activation='softmax'))

		self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		print(self.model.summary())
		#model.save("sat6model.h5")


	def construct_prediction(self, img_array, pad=False):
		'''
		Split image into patches, classify them, then construct prediction tensor.
		Args:
			img_array: image to process
				  pad: whether to pad the prediction tensor (set to True if predicting global map)
		Returns prediction tensor for the image.
		'''

		image_patches = self.split_image(img_array)
		image_prediction = self.predict_patches(image_patches)

		#image_prediction = self.broadcast_predictions(patches_prediction)
		#image_prediction = self.build_image(image_prediction)
		if pad:
			pad_x = int((self.cam_x/self.patch_dim)//2)
			pad_y = int((self.cam_y/self.patch_dim)//2)
			image_prediction = np.pad(image_prediction,
									((pad_y, pad_y), (pad_x, pad_x), (0,0)), "constant")
		return image_prediction


	def adjust_for_height(self, img_array, drone_height):
		'''
		Resize image based on the specified drone height.
		Args:
			   img_array: image to resize
			drone_height: height of the drone in meters
		'''
		width  = int(drone_height * self.cam_xz_ratio // self.map_res)
		height = int(drone_height * self.cam_yz_ratio // self.map_res)
		img_array = cv2.resize(img_array, (width, height))
		return img_array


	def resize_to_scale(self, img, scale_factor):
		width, height = img.shape[1], img.shape[0]
		resized = cv2.resize(img, (int(width*scale_factor), int(height*scale_factor)))
		return resized


	def split_image(self, img_array):
		'''
		Split image into patches.
		Args:
			img_array: image to split
		Returns resulting patches.
		'''
		img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

		# calculate max dimensions divisible by patch size
		max_height = img_array.shape[0] // self.patch_dim * self.patch_dim
		max_width  = img_array.shape[1] // self.patch_dim * self.patch_dim

		# cut into 28x28 patches
		img_array = img_array[:max_height, :max_width]
		img_array = np.array(np.split(img_array, max_height//self.patch_dim, axis=0))
		img_array = np.array(np.split(img_array, max_width//self.patch_dim, axis=2))
		return img_array


	def predict_patches(self, patches_array):
		'''
		Apply learning model on a set of image patches to classify them.
		Args:
			patches_array: image patches to classify
		Returns predictions for every patch.
		'''
		predictions = []
		for i in range(len(patches_array)):
			predictions.append(np.round(self.model.predict(patches_array[i]), 3))
		predictions = np.array(predictions)
		return predictions


	def build_image(self, patches_array, write_file=False):
		'''
		Reconstruct image (or any tensor) from patches.
		Args:
			patches_array: reconstruct from these patches
			   write_file: save the result as a .jpg file on True
		Returns reconstructed image/tensor.
		'''
		vertical_strips = []
		for i in range(patches_array.shape[0]):
			vertical_strips.append(np.vstack(patches_array[i,:]))
		img = np.hstack(vertical_strips[:])

		if write_file:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			cv2.imwrite("rebuilt.jpg", img)
		return img


	def broadcast_predictions(self, predictions):
		'''
		Clone predictions for every patch to construct a pseudo-prediction for every pixel in image.
		Broadcasts from 1x1xCLASSES to PATCH_DIMxPATCH_DIMxCLASSES.

		Args:
			predictions: broadcast these predictions (assumes predictions for every image patch)

		Returns pseudo-prediction patches.
		'''
		image_prediction = []
		for i in range(predictions.shape[0]):
			column_prediction = []
			for j in range(predictions.shape[1]):
				padded_prediction = np.broadcast_to(predictions[i,j],
										(self.patch_dim, self.patch_dim, self.classes))
				column_prediction.append(padded_prediction)
			image_prediction.append(column_prediction)
		return np.array(image_prediction)


	def get_particle_area(self, pX, pY, angle, w, h, another_map=None,
						  padded_map=True, patch_size=28):
		'''
		Extract area around a particle (from ParticleFilter) on the global map.
		Extracts from the map prediction tensor, but can be applied directly to the map.

		Args:
					pX: x-coordinate of the particle
					pY: y-coordinate of the particle
				 angle: orientation of the particle
					 w: width of area to extract
					 h: height of area to extract
		   another_map: if specified, extract particle area from this tensor
			padded_map: set to True if the map is pre-padded

		Returns the extracted area.
		'''

		pX = int(pX//patch_size)
		pY = int(pY//patch_size)

		if padded_map:
			# account for padding
			pX += int(self.cam_x/patch_size//2)
			pY += int(self.cam_y/patch_size//2)

		# extract bounding box of particle zone
		(box_X, box_Y) = rotate_image(angle=angle, w=int(w), h=int(h))

		map_prediction = self.map_prediction
		if another_map is not None:
			map_prediction = another_map
		bounding_box = map_prediction[int(pY-box_Y//2) : int(pY+box_Y//2),
									  int(pX-box_X//2) : int(pX+box_X//2)]
		
		# extract particle area at the specified angle
		rotated_box = rotate_image(bounding_box, -angle)
		(rY, rX) = rotated_box.shape[:2]
		particle_area = rotated_box[int(rY//2-h//2) : int(rY//2+h//2),
									int(rX//2-w//2) : int(rX//2+w//2)]
		return particle_area


if __name__=="__main__":
	# set paths to data
	map_file = "../data/maps/UFRGS-01-2017.png"
	camera_path = "../data/camera_feed/"
	weights_file = "../data/learning_model/sat6_weights.hdf5"

	processor = DataProcessor(map_file, camera_path, weights_file)



