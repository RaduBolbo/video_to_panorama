import argparse
from panorama import Video2Panorama
import os

def main(recipe):
    """
    This is the main function for the application. It instantiates the Image2Panorama class and calls the specific methods for conversion.

    Parameters
    ----------
    recipe : dict
        The configuration dictionary containing all the necessary parameters for the application
    """
    tts = Video2Panorama(recipe['hyperparameters_path'])
    tts.convert(recipe['input_path'], recipe['output_path'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video2Panorama parser')

    # parameters for convert metod
    parser.add_argument('--input_path', '-i', type=str, default='',
        help='Path to the input video to be transformed into image.')
    parser.add_argument('--output_path', '-o', type=str, default='',
        help='Path of the output file where the panoramic image to be saved.')
    
    # parameters for isntantiate method
    parser.add_argument('--hyperparameters_path', '-h', type=str, default=os.path.join(os.path.dirname(__file__), 'hyperparameters.json'),
        help='Path to a .json file containing hyperparameters for the algorithm.')

    args = parser.parse_args()
    main(args.categories)