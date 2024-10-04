import os
from video2panorama.panorama import Video2Panorama

# run as API



tts = Video2Panorama()
tts.convert('data/video_horiz_sd.mp4', 'data/output_sd.png')
tts.convert('data/video_horiz_ds.mp4', 'data/output_ds.png')
tts.convert('data/video_vert_js.mp4', 'data/output_js.png')
tts.convert('data/video_vert_sj.mp4', 'data/output_sj.png')

# run as CLI

os.system('video2panorama -i data/video_horiz_sd.mp4 -o data/output_cli_sd.png')
os.system('video2panorama -i data/video_horiz_ds.mp4 -o data/output_cli_ds.png')
os.system('video2panorama -i data/video_vert_js.mp4 -o data/output_cli_js.png')
os.system('video2panorama -i data/video_vert_sj.mp4 -o data/output_cli_sj.png')


