import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
cv2.ocl.setUseOpenCL(False)
import warnings
warnings.filterwarnings('ignore')
from collections import Counter
import os
import json
from video2panorama.utils import most_common_element, get_movement_direction_AND_speed, fill, trim_black_rows_or_columns



class Video2Panorama():

    def __init__(self, 
                hyperparameters_path=None,
                feature_extraction_algo=None,
                ratio=None,
                reprojectionThresh=None,
                adaptive_shift_pixels=None,
                non_dominant_direction_pixels_freedom_degree=None):
        
        if hyperparameters_path is None:
            self.hyperparameters_path = os.path.join(os.path.dirname(__file__), 'hyperparameters.json')
        else:
            self.hyperparameters_path = hyperparameters_path
        with open(self.hyperparameters_path, 'r') as file:
            hyperparameters = json.load(file)
        
        self.feature_extraction_algo = hyperparameters['feature_extraction_algo'] if feature_extraction_algo is None else feature_extraction_algo
        self.ratio = hyperparameters['ratio'] if ratio is None else ratio
        self.reprojectionThresh = hyperparameters['reprojectionThresh'] if reprojectionThresh is None else reprojectionThresh
        self.adaptive_shift_pixels = hyperparameters['adaptive_shift_pixels'] if adaptive_shift_pixels is None else adaptive_shift_pixels
        self.non_dominant_direction_pixels_freedom_degree = hyperparameters['non_dominant_direction_pixels_freedom_degree'] if non_dominant_direction_pixels_freedom_degree is None else non_dominant_direction_pixels_freedom_degree

    def _get_movement_direction(self, video_path):
        # Replace 'path_to_video' with your video file path
        direction, speed = get_movement_direction_AND_speed(video_path, 4)
        print('Movement direction is: ', direction)
        print('Movement speed, as optical flux vector average is: ', speed)

        # Create a VideoCapture object
        cap = cv2.VideoCapture(video_path)

        # fint out the number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('Total frames: ', total_frames)

        # adaptive adapting the parameters to number of frames
        begin_idx = min(10, total_frames)
        end_idx = total_frames
        #hop = 8
        # adaptive hop - has been set experimentally
        # when speed increases, the hop has to decrease
        adaptive_hop = round(164/speed) # The coefficient 164 was chosen based on experiments conducted at different speeds.

        hop = adaptive_hop
        print('Adaptive hop: ', adaptive_hop)

        # invert video
        if direction == 'ds' or direction == 'js':
            buffer = end_idx
            end_idx = begin_idx
            begin_idx = buffer  -1
            hop = -hop

        cap.set(cv2.CAP_PROP_POS_FRAMES, begin_idx)
        # ret will be a boolean – indicating whether something was returned or not; frame will be the returned frame.
        ret1, frame1 = cap.read()
        query_photo = frame1
        query_photo = cv2.cvtColor(query_photo,cv2.COLOR_BGR2RGB)
        begin_idx = begin_idx + 2 * hop

        return cap, begin_idx, end_idx, hop, direction, query_photo

    def _compute_homography_matrix(self, matches, keypoints_train_img, keypoints_query_img):
        # Transform the tuples that give the coordinates into an ndarray, where each row represents a coordinate.
        keypoints_train_img = np.float32([keypoint.pt for keypoint in keypoints_train_img])
        keypoints_query_img = np.float32([keypoint.pt for keypoint in keypoints_query_img])

        # RANSAC algorithm can only be applied if there are at least 4 common points
        if len(matches) > 4:
            # The characteristic points between images are transformed into an ndarray to be accepted by cv2.findHomography
            # Now, the points that match are selected from the previously formed ndarray of points
            points_train = np.float32([keypoints_train_img[m.queryIdx] for m in matches])
            points_query = np.float32([keypoints_query_img[m.trainIdx] for m in matches])
            
            # Using the RANSAC algorithm to find the homography matrix
            # reprojThresh controls which points are considered inliers - ||pi - pi'|| < reprojThresh, where pi' is the projection of pi in the new system (if it's smaller, it's an inlier)
            # *CHANGE* - here I will calculate reprojThresh relative to the image size - I need to find a formula
            (Homography_Matrix, _) = cv2.findHomography(points_train, points_query, cv2.RANSAC, self.reprojectionThresh)
            return Homography_Matrix
        else:
            # continue to the next loop if entering the else clause
            print("Couldn't compute homography matrix.")
            return None

    def _find_homography_matrix(self, train_photo, query_photo, descriptor, bf, direction):
        train_photo = cv2.cvtColor(train_photo,cv2.COLOR_BGR2RGB)
        train_photo_gray = cv2.cvtColor(train_photo, cv2.COLOR_RGB2GRAY)
        query_photo_gray = cv2.cvtColor(query_photo, cv2.COLOR_RGB2GRAY)

        if direction == 'js' or direction == 'sj':
            # 😨🕸️ some cameras have the specific problem of padding black lines on the orizontal axis
            train_photo = np.copy(train_photo[:-2, :, :])
            query_photo = np.copy(query_photo[:-2, :, :])

        keypoints_train_img, features_train_img = descriptor.detectAndCompute(train_photo_gray, None)
        keypoints_query_img, features_query_img = descriptor.detectAndCompute(query_photo_gray, None)
        # Each feature vector has a dimension of 128 for SIFT and 64 for SURF, for example
        # A KeyPoint object has -> x, y (coordinates), size, angle, response, octave (the octave is the resolution in the octave-pyramid system)

        # compute the raw matches and initialize the list of actual matches
        # returns the nearest k neighbours from which the best match is selected, but only if it also respects the Lowe criterium
        rawMatches = bf.knnMatch(features_train_img, features_query_img, k=2)

        # Iterate through the matches to filter them and keep only the credible ones (based on LOWE's RATIO TEST)
        # Apply LOWE's RATIO TEST -> check if the ratio of distances between 2 descriptors is below a certain threshold
        # nearest_neighbor/second_nearest_neighbor < ratio
        # Logic: there should be a significant difference between the best match and the second. If not, they probably come from a repetitive pattern.
        matches = [nearest for nearest,second_nearest in rawMatches if nearest.distance/second_nearest.distance < self.ratio]

        # afisez 100 de matchuri # **** to display this opnly if a parmaeter is verboise
        mapped_features_image_knn = cv2.drawMatches(train_photo, keypoints_train_img, query_photo, keypoints_query_img, np.random.choice(matches,100),
                                None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        Homography_Matrix = self._compute_homography_matrix(matches, keypoints_train_img, keypoints_query_img)
        return Homography_Matrix

    def _image_stitching(self, query_photo, train_photo, Homography_Matrix, direction, iteration):
        if direction == 'sd' or direction == 'ds': # if movement is horizontal
            # sum the image widths
            width = query_photo.shape[1] + train_photo.shape[1]

            # for width it is the maximum of the heights
            height = max(query_photo.shape[0], train_photo.shape[0])
            height += self.non_dominant_direction_pixels_freedom_degree

            # Here the perspective transformation is applied to the second image -> essentially performing the homographic transformation
            # The resulting image will be the next frame transformed into the new coordinates
            result = cv2.warpPerspective(train_photo, Homography_Matrix,  (width, height))

            # The second transformed image is kept, over which the original, unaltered first image is initially added
            # result[0:query_photo.shape[0], 0:query_photo.shape[1]] = query_photo
            ##### PROBLEM WITH THE ABOVE COMMENTED CODE:
            # The problem is that when the previous frame overlaps with the distorted version of the first frame, it overwrites part of it because it is black.
            # SOLUTION: Create a binary mask so that only pixels different from zero are copied over the modified image.
            # Shift the mask to the right and add it to itself - this way, I hope to eliminate the artifact.
            mask = np.any(query_photo != 0, axis=2)
            # The numebr of shifted pixels increases adaptively
            if iteration % 5 == 0:
                self.adaptive_shift_pixels += 1
            extended_mask = np.hstack((mask[:, self.adaptive_shift_pixels:], np.ones((mask.shape[0], self.adaptive_shift_pixels), dtype=bool)))
            extended_mask = fill(extended_mask)
            extended_mask_3d = np.repeat(extended_mask[:, :, np.newaxis], 3, axis=2)
            

            # Apply the mask to copy only non-zero pixels from query_photo to the corresponding position in result
            result[0:query_photo.shape[0], 0:query_photo.shape[1]][extended_mask_3d] = query_photo[extended_mask_3d]
        elif direction == 'sj' or direction == 'js': # daca miscarea e pe vericala
            # sum image widths
            width = max(query_photo.shape[1], train_photo.shape[1])
            width += self.non_dominant_direction_pixels_freedom_degree

            # hor height select the maximum of heights
            height = query_photo.shape[0] + train_photo.shape[0]

            # Here the perspective transformation is applied to the second image -> essentially performing the homographic transformation
            # The resulting image will be the next frame transformed into the new coordinates
            result = cv2.warpPerspective(train_photo, Homography_Matrix,  (width, height))


            mask = np.any(query_photo != 0, axis=2)
            # The numebr of shifted pixels increases adaptively
            if iteration % 5 == 0:
                self.adaptive_shift_pixels += 1

            extended_mask = np.vstack((mask[self.adaptive_shift_pixels:, :], np.ones((self.adaptive_shift_pixels, mask.shape[1]), dtype=bool)))
            extended_mask = fill(extended_mask)
            extended_mask_3d = np.repeat(extended_mask[:, :, np.newaxis], 3, axis=2)

            # Apply the mask to copy only non-zero pixels from query_photo to the corresponding position in result
            result[0:query_photo.shape[0], 0:query_photo.shape[1]][extended_mask_3d] = query_photo[extended_mask_3d]

        else:
            print('Issue in movement estimation step. Direction should have been one of: "sd", "ds", "js" or "sj"')
        # Prepare variables for the next loop
        query_photo = np.copy(result)
        return query_photo
    
    def _trim_image(self, query_photo, direction):
        query_photo = trim_black_rows_or_columns(query_photo, 'col')
        query_photo = trim_black_rows_or_columns(query_photo, 'row')
        return query_photo

    def _get_feature_extraction_algo(self):
        if self.feature_extraction_algo == 'sift':
            descriptor, bf = cv2.SIFT_create(), cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        elif self.feature_extraction_algo == 'orb':
            descriptor, bf = cv2.ORB_create(), cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        return descriptor, bf

    def convert(self, input_path=None, output_path=None):
        if input_path is None or output_path is None:
            raise ValueError('both input_path and output_path have to be passed')
        
        cap, begin_idx, end_idx, hop, direction, query_photo = self._get_movement_direction(input_path)

        # select the feature extract method for creating the descriptors
        # instantiate object that performs matching
        descriptor, bf = self._get_feature_extraction_algo()

        for i, idx in enumerate(tqdm(range(begin_idx, end_idx, hop))):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            # ret va fi boolean - inidc
            ret2, frame2 = cap.read()
            train_photo = frame2
            train_photo = cv2.cvtColor(train_photo,cv2.COLOR_BGR2RGB)

            # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            '''
            plt.figure(figsize=(20,10))
            plt.axis('off')
            plt.imshow(query_photo)
            plt.show()

            plt.figure(figsize=(20,10))
            plt.axis('off')
            plt.imshow(train_photo)
            plt.show()
            '''
            # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

            Homography_Matrix = self._find_homography_matrix(train_photo, query_photo, descriptor, bf, direction)
            if Homography_Matrix is None:
                continue

            query_photo = self._image_stitching(query_photo, train_photo, Homography_Matrix, direction, iteration=i)  
        
        query_photo = self._trim_image(query_photo, direction)

        query_photo = cv2.cvtColor(query_photo, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, query_photo)


