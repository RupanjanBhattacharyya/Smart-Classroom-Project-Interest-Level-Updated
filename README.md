Step 1: Create a new environment: conda create -n affectnet-env python=3.7.16

Step 2: Activate the environment (don't continue to the following step without doing this!):  conda activate affectnet-env

Step 3: Install dependencies: pip install -r requirements.txt

To run a test on web camera: python inference.py --cam 0

To run a test on video file: python inference.py --video path/to/video.mp4
