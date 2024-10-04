<center><img src="data/logo_transparent.png" alt="video2panorama" /></center>

-----------

# video2panorama

**video2panorama** is a Python-based tool designed to transform video sequences into panoramic images. It automates the stitching process by analyzing frames from input video files and producing panoramic outputs. 


# How to install

**1. install with pip**

For users who want to install the package directly from PyPI:

```bash
pip install video2panorama
```

**2. Cloning the repo and installing locally**

For those who prefer to work with the latest code or contribute to the project, you can clone the GitHub repository and install it locally.

Clone the repo:
```bash
git clone https://github.com/RaduBolbo/video_to_panorama.git
```

Navigate into the cloned directory:
```bash
cd video_to_panorama
```

Create a conda environemnt:
```bash
conda create --name v2p python==3.10
```

Activate the conda environemnt:
```bash
conda activate v2p 
```

Install the package:
```bash
pip install .
```


# How to use

**2. Using the terminal**

Run the following command in the terimanl:
```bash
video2panorama  --input_path path/to/your/video.mp4 --output_path path/to/the/output/panoramic_image.png
```

Additionally, if you wantto try other hyperparameters, you can include them in a .json file whos path you can send to the comman:
```bash
video2panorama  --input_path path/to/your/video.mp4 --output_path path/to/the/output/panoramic_image.png --hyperparameters_path path/to/your/config/file.json
```

The config file should have the following format:
```json
{
    "feature_extraction_algo" : "sift",
    "ratio" : 0.75,
    "reprojectionThresh" : 2,
    "adaptive_shift_pixels" : 2,
    "non_dominant_direction_pixels_freedom_degree" : 0
}
```
The hyperparameters are explained in ****



**2. Using the Python Api**

```python
from video2panorama.panorama import Video2Panorama

tts = Video2Panorama()
tts.convert('path/to/your/video.mp', 'path/to/the/output/panoramic_image.png')
```

If you want to overwrite the default parameters you can specify them in the constructor:
```python
tts = Video2Panorama(feature_extraction_algo="sift",
                    ratio=0.75,
                    reprojectionThresh=2,
                    adaptive_shift_pixels=2,
                    non_dominant_direction_pixels_freedom_degree=0
                    )
```
