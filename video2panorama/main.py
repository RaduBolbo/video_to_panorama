import argparse
from video2panorama.panorama import Video2Panorama
import os

def main():
    """
    This is the main function for the application. It instantiates the Video2Panorama class
    and calls the specific methods for conversion.
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Video2Panorama parser')

    # parameters for convert method
    parser.add_argument('--input_path', '-i', type=str, default='',
        help='Path to the input video to be transformed into image.')
    parser.add_argument('--output_path', '-o', type=str, default='',
        help='Path of the output file where the panoramic image will be saved.')
    
    # parameters for instantiate method
    parser.add_argument('--hyperparameters_path', '-hyp', type=str, default=os.path.join(os.path.dirname(__file__), 'hyperparameters.json'),
        help='Path to a .json file containing hyperparameters for the algorithm.')

    args = parser.parse_args()

    recipe = {
        'input_path': args.input_path,
        'output_path': args.output_path,
        'hyperparameters_path': args.hyperparameters_path
    }

    converter = Video2Panorama(recipe['hyperparameters_path'])
    converter.convert(recipe['input_path'], recipe['output_path'])

