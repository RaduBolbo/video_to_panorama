import os

# run as API

from video2panorama.panorama import Video2Panorama

tts = Video2Panorama('video2panorama/hyperparameters.json')
tts.convert('data/video_horiz_sd.mp4', 'data/output_sd.png')
tts.convert('data/video_horiz_ds.mp4', 'data/output_ds.png')
tts.convert('data/video_vert_js.mp4', 'data/output_js.png')
tts.convert('data/video_vert_sj.mp4', 'data/output_sj.png')

# run as CLI

os.system('python video2panorama/main.py -i data/video_horiz_sd.mp4 -o data/output.png -h video2panorama/hyperparameters.json')

