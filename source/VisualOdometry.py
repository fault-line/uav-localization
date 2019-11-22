import numpy as np
import cv2


class VisualOdometry:
	''' Implementation of Semi-Direct Monocular Visual Odometry.'''

	def __init__(self, camera_path, last_frame):
		'''
		Initialize VisualOdometry object.
		Args:
			camera_path: path to camera images
			 last_frame: number of camera images in the data set
		'''
		self.camera_path = camera_path
		self.last_frame = last_frame
		self.frame_counter = 1

		self.prev_img = None
		self.prev_features = None

		self.next_img, self.next_features = self.detect_features()


	def reset_frames(self):
		'''
		Reset frame counter.
		'''
		self.frame_counter = 1


	def process_next_frame(self):
		'''
		Perform keypoint detection on the next camera image (indicated by frame_counter).
		Then, track keypoints from previous image to the next.
		Returns mean optical flow of tracked keypoints.
		'''

		# advance one frame
		self.prev_img, self.prev_features = self.next_img, self.next_features
		self.frame_counter += 1
		if self.frame_counter > self.last_frame:
			print("Reached end of flight.")
			return -1

		# calculate transformation
		self.next_img, self.next_features = self.detect_features()
		mean_x, mean_y = self.track_features()
		#print("Mean optical flow: ({:.3f}, {:.3f})".format(mean_x, mean_y))

		#rotation_matrix, translation_vector = self.get_transformation()

		if len(self.next_features < 1000):
			#frame_counter += 1
			self.next_img, self.next_features = self.detect_features()
		return mean_x, mean_y


	def detect_features(self, verbose=False):
		'''
		Detect keypoints (FAST/ORB) in current camera image.
		Returns camera image converted to grayscale and list of detected keypoint coordinates (y,x).
		'''

		# get image
		img_filename = "{}{:03d}.jpg".format(self.camera_path, self.frame_counter)
		img = cv2.imread(img_filename)
				
		# turn to grayscale
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# use FAST detector
		#fast = cv2.FastFeatureDetector_create()
		#features = fast.detect(img, None)

		# use ORB detector
		orb = cv2.ORB_create(4000)
		feature_objects = orb.detect(img, None)

		features = np.array([obj.pt for obj in feature_objects], dtype=np.float32)
		return img, features


	def track_features(self):
		'''
		Track points from one camera image to the next with Lucas-Kanade method
		and calculate optical flow of tracked points.

		Returns mean optical flow of all tracked points.
		'''

		# calculate optical flow with iterative Lucas-Kanade method with pyramids
		self.next_features, status, errors = cv2.calcOpticalFlowPyrLK(self.prev_img,
			self.next_img, self.prev_features, self.next_features)

		# only retain successfully tracked keypoints
		status = status.flatten()
		self.prev_features = self.prev_features[status == 1]
		self.next_features = self.next_features[status == 1]
		print("Featues:", len(self.prev_features))
		
		# calc average optical flow
		feature_flow = self.next_features - self.prev_features
		mean_x = np.mean(feature_flow[:,1])/2
		mean_y = np.mean(feature_flow[:,0])/2
		return mean_x, mean_y


	def get_transformation(self):
		'''
		Find transformation between camera images. CURRENTLY UNUSED.
		'''

		H, mask = cv2.findHomography(self.next_features, self.prev_features, cv2.RANSAC)
		_, rot_matrix, trans_vector, mask = cv2.recoverPose(E, self.next_features, self.prev_features,
								focal=self.focal, pp=self.pp)
		return rot_matrix, trans_vector



if __name__ == "__main__":

	vo = VisualOdometry(camera_path="../data/camera_feed/", last_frame=156)
	for i in range(50):
		print("Frame {}:".format(i))
		vo.process_next_frame(verbose=True)
		#print("POINTS:", len(vo.prev_features), len(vo.next_features))