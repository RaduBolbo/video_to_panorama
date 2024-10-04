import cv2
import numpy as np
import imageio
cv2.ocl.setUseOpenCL(False) # this prevents errors for some builds
import warnings
warnings.filterwarnings('ignore')
from collections import Counter


def most_common_element(lst):
    if not lst:
        return None
    counter = Counter(lst)
    return counter.most_common(1)[0][0]


def get_movement_direction_AND_speed(video_path, hop):
    '''
    Returns 'sd' (left-right), 'ds' (right-left), 'js' (down-up) or 'sj' (up-down) depending on the movement direction.
    Using a voting system. To speed up, the algorithm jumps from hop to hop frames.
    '''
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Can't open the file.")
        return []

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Couldn't read any frames.")
        return []

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 1
    directions = []
    magnitudini = []

    while current_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        '''
        # cv2.calcOpticalFlowFarneback => computes optical flux with Gunnar Farneback's method
        Parametrii:
        0.5: The image scale parameter for the pyramid (or simple) layer. It is used to scale down the image each pyramid level. In this case, each level is half the previous level.
        3: The number of levels in the pyramid. A value of 3 indicates the use of three levels.
        15: The window size. It's the averaging window size; larger values increase the algorithm's robustness to noise and give smoother flow, but they also decrease motion details.
        3: The number of iterations to perform at each pyramid level.
        5: The size of the pixel neighborhood used to find polynomial expansion in each pixel.
        1.2: The standard deviation of the Gaussian used to smooth derivatives for polynomial expansion.
        '''
        # computes optical flux with Gunnar Farneback's method
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # ... = all elements in all directions, but an exception is made for the last dimension, from which only 0 is taken.
        # flow will contain, for each pixel in the image, how much it has shifted horizontally (in plane 0) and vertically (in plane 1).
        # ! The vector field can be obtained by combining the two planes (:, :, 0) and (:, :, 1).
        horz_movement = flow[..., 0].mean()
        vert_movement = flow[..., 1].mean()

        if abs(horz_movement) > abs(vert_movement):
            direction = 'ds' if horz_movement > 0 else 'sd'
        else:
            direction = 'js' if vert_movement > 0 else 'sj'
        directions.append(direction)

        # Estimate the velocity and translate it to polar coordinates
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # computing magnitude averages
        magnitudini.append(np.mean(magnitude))

        prev_gray = gray
        current_frame += hop

    cap.release()
    return most_common_element(directions), np.mean(magnitudini)

def fill(extended_mask):
    '''
    Morphological image closing
    '''
    extended_mask_8bit = (extended_mask * 255).astype(np.uint8)

    # define the filter
    kernel_size = 7  # kernel size
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # morphological closing
    closed_mask = cv2.morphologyEx(extended_mask_8bit, cv2.MORPH_CLOSE, kernel)

    # convert back to boolean
    closed_mask_bool = closed_mask.astype(bool)
    return closed_mask_bool

def trim_black_rows_or_columns(image, row_or_column='row'):
    """
    Trims black rows from the top and bottom or black columns from the left and right of the image.
    Assumes that black is represented by 0 intensity for grayscale and [0, 0, 0] for color images in BGR format.
    """
    # Check if the row/column is black. If all pixels in a row/column are black, their sum will be 0.
    axis_to_check = 1 if row_or_column == 'row' else 0
    is_black = np.all(image == 0, axis=axis_to_check)

    # Find the indices of rows/columns that are not black
    non_black_indices = np.where(~is_black)[0]

    # If there are no non-black rows/columns, return the original image
    if not len(non_black_indices):
        return image

    # Crop the image to remove black rows/columns
    if row_or_column == 'row':
        cropped_image = image[non_black_indices[0]:non_black_indices[-1] + 1, :]
    else:  # 'col'
        cropped_image = image[:, non_black_indices[0]:non_black_indices[-1] + 1]

    return cropped_image


