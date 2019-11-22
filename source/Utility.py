import cv2
import numpy as np

'''Collection of various utility functions.'''

def rotate_image(image=None, angle=0, w=-1, h=-1):
	'''
	Rotate image by angle without cutting.
	Also used to calculate bounding box dimensions for rotation.

	Args:
		image: image to transform (otherwise, only calculates bounding box)
		angle: counter-clockwise angle of rotation in degrees
			w: virtual width, if 'image' is not specified
			h: virutal height, if 'image' is not specified

	Returns:
		if 'image' is None: bounding box dimensions
				 otherwise: rotated image
	'''

	if image is not None:
		# determine dimensions, get center
		(h, w) = image.shape[:2]
	(cW, cH) = (w // 2, h // 2)

	# calculate sin and cos of rotation
	R = cv2.getRotationMatrix2D((cW, cH), angle, 1.0)
	cos_R = np.abs(R[0, 0])
	sin_R = np.abs(R[0, 1])

	# compute bounding box dimensions
	nW = int((h * sin_R) + (w * cos_R))
	nH = int((h * cos_R) + (w * sin_R))
	res = (nW, nH)

	if image is not None:
		# adjust rotation matrix to take into account translation
		R[0, 2] += (nW / 2) - cW
		R[1, 2] += (nH / 2) - cH

		# perform actual rotation and return image
		res = cv2.warpAffine(image, R, (nW, nH))
	return res


def pair_pixels(img_shape, num_pairs):
	'''
	Construct a set of random pixel pairs (from gaussian distribution).
	Used to calculate BRIEF descriptor.

	Args:
		img_shape: dimensions of the image to construct pixel pairs for
		num_pairs: number of pixel pairs to construct

	Returns x-coordinates and y-coordinates of paired pixels.
	'''
	dim_x, dim_y = img_shape[1], img_shape[0]
	x = np.sqrt(dim_x-1) * np.random.randn(num_pairs, 2) + (dim_x-1)/2
	x = np.clip(x, 0, dim_x-2).astype("int")

	y = np.sqrt(dim_y-1) * np.random.randn(num_pairs, 2) + (dim_y-1)/2
	y = np.clip(y, 0, dim_y-2).astype("int")
	return x,y


def brief_encode(img, x, y):
	'''
	Calculate multi-channel BRIEF descriptor for an image.
	Args:
		img: image to calculate descriptor for
		x: x-coordinates of pixel pairs [from pair_pixels()]
		y: y-coordinates of pixel pairs [from pair_pixels()]
	Return multi-channel BRIEF descriptor for the image.
	'''

	# construct BRIEF descriptor for each channel
	channel_descriptors = []
	for c in range(img.shape[2]):
		descriptor = (img[y[:,0], x[:,0], c]  >  img[y[:,1], x[:,1], c]).astype("uint8")
		channel_descriptors.append(descriptor)

	# connect channel descriptors
	channel_descriptors = np.array(channel_descriptors)
	full_descriptor = np.concatenate(channel_descriptors[:])
	return full_descriptor


def calc_weight_brief(camera_prediction, particle_prediction):
	'''
	Calculate particle weight (from ParticleFilter) using BRIEF descriptors.
	Args:
		  camera_prediction: prediction tensor for the camera image
		particle_prediction: prediction tensor for the particle area
	Returns particle weight.
	'''
	x, y = pair_pixels(camera_prediction.shape, 64)
	camera_descriptor = brief_encode(camera_prediction, x, y)
	particle_descriptor = brief_encode(particle_prediction, x, y)

	xnor = (camera_descriptor[:] == particle_descriptor[:]).astype("uint8")
	weight = np.sum(xnor)/len(xnor)
	return weight


def calc_weight_obj(camera_prediction, particle_prediction):
	channels = camera_prediction.shape[2]
	camera_sum, particle_sum = np.sum(camera_prediction), np.sum(particle_prediction)
	total_obj_ratio = max(camera_sum/particle_sum, particle_sum/camera_sum)
	
	class_totals_camera = np.array([np.sum(camera_prediction[:,:,c]) for c in range(channels)])
	class_totals_camera /= camera_sum
	
	class_totals_particle = np.array([np.sum(particle_prediction[:,:,c]) for c in range(channels)])
	class_totals_particle /= particle_sum

	error = np.sum(np.abs(class_totals_camera - class_totals_particle)) * np.log(total_obj_ratio)
	weight = 1/(error+1)
	return weight


def calc_weight_cossim(camera_prediction, particle_prediction):
	'''
	Calculate particle weight by constructing vectors of class certainty sums
	and applying Error Sum of Squares.

	Args:
		  camera_prediction: prediction tensor for the camera image
		particle_prediction: prediction tensor for the particle area
	Returns particle weight.
	'''

	channels = camera_prediction.shape[2]

	camera_prediction[camera_prediction < 1/channels] = 0
	particle_prediction[particle_prediction < 1/channels] = 0

	class_sums_camera = [np.sum(camera_prediction[:,:,c]) for c in range(channels)]
	class_sums_particle = [np.sum(particle_prediction[:,:,c]) for c in range(channels)]

	cam_norm = np.linalg.norm(class_sums_camera)
	particle_norm = np.linalg.norm(class_sums_particle)

	weight = np.dot(class_sums_camera, class_sums_particle) / (cam_norm*particle_norm)
	return weight


def calc_weight_sse(camera_prediction, particle_prediction):
	'''
	Calculate particle weight by constructing vectors of class certainty sums
	and applying Error Sum of Squares.

	Args:
		  camera_prediction: prediction tensor for the camera image
		particle_prediction: prediction tensor for the particle area
	Returns particle weight.
	'''

	channels = camera_prediction.shape[2]

	camera_prediction[camera_prediction < 1/channels] = 0
	particle_prediction[particle_prediction < 1/channels] = 0

	class_sums_camera = np.array([np.sum(camera_prediction[:,:,c]) for c in range(channels)])
	class_sums_particle = np.array([np.sum(particle_prediction[:,:,c]) for c in range(channels)])

	sum_squared_errors = np.sum(np.square(class_sums_camera - class_sums_particle))

	weight = 1/(sum_squared_errors+1)
	return weight