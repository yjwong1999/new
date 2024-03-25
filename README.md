# Efficient YOLOv8 Inferencing using Multithreading

Efficient YOLOv8 inference depends not only on GPU specifications but also on CPU processing. However, the significance of fully utilizing the CPU is often overlooked. In fact, leveraging the CPU is crucial because it plays an essential role in the I/O aspect of model deployment (specifically, reading input frames and plotting the outputs). In this repository, we explore how to utilize CPU multi-threading to enhance inference speed.

## Setup
Conda environment
```bash
conda create --name new python=3.8.10 -y
conda activate new

git clone https://github.com/yjwong1999/new.git
cd new
```

Install dependencies
```bash
pip3 install torch torchvision torchaudio
pip install ultralytics==8.1.24
pip install pip install pafy==0.5.5
pip install youtube-dl==2021.12.17

pip install scikit-learn==1.3.2
pip install loguru==0.7.2
pip install gdown==4.6.1
pip install ftfy==6.1.1
pip install regex==2023.6.3
pip install filterpy==1.4.5
pip install lapx==0.5.4
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

## Find port number connected to camera
```bash
python3 find_port.py
```

## If you are doing multi-stream
```
1. List all the sources in source.streams
2. If you are doing tracking + geofencing, list the geofencing roi xyxy in geofencing.streams
```

## Install VLC player to simulate rtsp streaming
```bash
sudo snap install vlc
```

## Detection
Single stream detection
```bash
python3 single_detect.py --webcam
python3 single_detect.py --camera 0
python3 single_detect.py --video-file sample_video.mp4
python3 single_detect.py --rtsp "rtsp://192.168.1.136:8554/"
python3 single_detect.py --youtube "http://www.youtube.com/watch?v=q0kPBRIPm6o"
```

Multi stream detection
```bash
python3 multi_detect.py
```

## Tracking
Single stream tracking
```bash
# Example (without geofencing)
python3 single_track.py --webcam
python3 single_track.py --camera 0
python3 single_track.py --video-file sample_video.mp4
python3 single_track.py --rtsp "rtsp://192.168.1.136:8554/"
python3 single_track.py --youtube "http://www.youtube.com/watch?v=q0kPBRIPm6o"

# Example (with geofencing)
python3 single_track.py -video-file sample_video.mp4 --roi-xyxy 0.6,0.4,0.9,0.8
```

Multi stream tracking
```bash
# without geofencing
python3 multi_track.py

# with geofencing
python3 multi_track.py --geofencing
```

## TODO
- [ ] cannot play youtube yet
- [ ] drive handling fails for multiple source
- [ ] no error warning when the video source is not available, not sure this will happen for other source types onot
- [ ] the dummy handler in multi_track.py will post() today, should post tmr only

## Citation
```
@software{Wong_Efficient_YOLOv8_Inferencing_2024,
  author = {Wong, Yi Jie},
  doi = {10.5281/zenodo.10792741},
  month = mar,
  title = {{Efficient YOLOv8 Inferencing using Multithreading}},
  url = {https://github.com/yjwong1999/efficient_yolov8_inference},
  version = {1.0.0},
  year = {2024}}
```


## Acknowledgement
1. ultralytics official repo [[ref]](https://github.com/ultralytics/ultralytics)
2. tips for effecient single-stream detection (multithread, resize frame, skipping frame) [[ref]](https://blog.stackademic.com/step-by-step-to-surveillance-innovation-pedestrian-detection-with-yolov8-and-python-opencv-dbada14ca4e9)
3. multi-thread for multi-stream detection [[ref]](https://ultralytics.medium.com/object-tracking-across-multiple-streams-using-ultralytics-yolov8-7934618ddd2)
4. Tracking with Ultralytics YOLO (how to handle the results) [[ref]](https://docs.ultralytics.com/modes/track/#plotting-tracks-over-time)
