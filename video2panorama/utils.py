import cv2
import numpy as np
import imageio
cv2.ocl.setUseOpenCL(False)
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
    Reruneaza 'lr', 'rl', 'js' sau 'sj' in fucntie d edirectia prepondereta a miscarii.
    Se utilizeaza siste de votare. Se sare din hop in hop cadre.
    '''
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cu se poate deschide fisierul.")
        return []

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Nu se gasesc cadre.")
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
        # cv2.calcOpticalFlowFarneback => calculeaza fluxul optic cu metoda  Gunnar Farneback's
        Parametrii:
        0.5: The image scale parameter for the pyramid (or simple) layer. It is used to scale down the image each pyramid level. In this case, each level is half the previous level.
        3: The number of levels in the pyramid. A value of 3 indicates the use of three levels.
        15: The window size. It's the averaging window size; larger values increase the algorithm's robustness to noise and give smoother flow, but they also decrease motion details.
        3: The number of iterations to perform at each pyramid level.
        5: The size of the pixel neighborhood used to find polynomial expansion in each pixel.
        1.2: The standard deviation of the Gaussian used to smooth derivatives for polynomial expansion.
        '''
        # se calculeaza fluxul optic cu metoda  Gunnar Farneback's
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # ... = toate elementele de pe toate directiile, dar s epune exceptie pe ultima dimensiune de unde s ia doar 0.
        # flow va contine pentru fiecare pixel din imagine cat de mult s-a deplasat pe orizontala (in planul 0) si pe verticala (in pl1).
        # ! Campul de vectori se poate obtine combinand cele doua plane (:, : 0) si (:, :, 1)
        horz_movement = flow[..., 0].mean()
        vert_movement = flow[..., 1].mean()

        if abs(horz_movement) > abs(vert_movement):
            # daca vectorii de 
            direction = 'ds' if horz_movement > 0 else 'sd'
        else:
            direction = 'js' if vert_movement > 0 else 'sj'
        directions.append(direction)

        # de asemenea, vreau estimarea vitezei
        # traduc la coordonate poalre
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # calculez media magnitudinilor
        magnitudini.append(np.mean(magnitude))

        prev_gray = gray
        current_frame += hop

    cap.release()
    return most_common_element(directions), np.mean(magnitudini)

def fill(extended_mask):
    '''
    Practic doar face inchiderea morfologica
    '''
    extended_mask_8bit = (extended_mask * 255).astype(np.uint8)

    # definesc filtrul
    kernel_size = 7  # dimensiune kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # inchidere morfologica
    closed_mask = cv2.morphologyEx(extended_mask_8bit, cv2.MORPH_CLOSE, kernel)

    # conversie inapoi la boolean
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