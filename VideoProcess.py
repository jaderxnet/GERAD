'''
Create Python class to process each video  in processedVideos.csv
1 - Read csv and select the last no processed video;
2 - Change video status to Processing;
3 - Download video in a folder with the id name;
4 - Read each frame;
5 - Apply 2 types of Neural Networks in each frame
    A - Media Pipe
    B - Yolo8 

6 - Calculate the difference betwen inferences
7 - Save a file by video with all the inferences frames
    A - Media Pipe Landmarks 
    B - Yolo8 Landmarks 
    C - Frame
    D - FPS
7 - Save a file with sequences:
    A - Initial and final frame 
    B - Media Calculated difference ;
    C - Difference threshold;
    D - LOW or HI
6 - Save a picture in the first frame by sequence;
'''


import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import pandas as pd
# !pip install git+https://github.com/ytdl-org/youtube-dl.git@master
from yt_dlp import YoutubeDL
from datetime import datetime
#!pip install "opencv-python-headless<4.3"
import cv2
# Dowload python > 3.9 version in https://www.python.org/downloads
#!pip install -q mediapipe==0.10.0
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
print("MEDIAPIPE: ", mp.__version__)


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


inputFilePath = "processedVideos.csv"

videosTable = pd.read_csv(inputFilePath, sep=';')
print("Lendo: ", inputFilePath)
print(inputFilePath)

print("Table: ", len(videosTable))

filtered = videosTable["status"] == "Processing"
print("Filtered: ", len(filtered))
indexesProcessing = videosTable.index[filtered]
if (len(indexesProcessing) > 0):
    print("PROCESSING THE INDEX VIDEO: ", indexesProcessing.tolist()[0])
else:
    filtered = videosTable["status"] == "Ready"
    print("Filtered: ", len(filtered))
    indexesToProcess = videosTable.index[filtered]
    print("indexesToProcess: ", len(indexesToProcess))
    if (len(indexesToProcess) < 1):
        print("ALL VIDEOS PROCESSED!")
    else:
        print("Index to update: ", indexesToProcess.tolist()[0])
        videosTable.at[indexesToProcess.tolist()[0], 'status'] = 'Processing'
        print("Updated Table: ", videosTable)
        # videosTable.to_csv(inputFilePath, index=False, sep=';')
        selectedVideo = videosTable.loc[indexesToProcess.tolist()[0]]
        print("Selected Video: ", selectedVideo)
        options = {
            # 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'
            # This will select the specific resolution typed here
            "format": "mp4[height=1080]",
            'no_check_certificate': True,
            # "%(id)s/%(id)s-%(title)s.%(ext)s"
            "outtmpl": "videos/%(id)s/%(id)s.%(ext)s"
        }
        # with YoutubeDL(options) as ydl:
        #    ydl.download([selectedVideo['url']])
        #   print('Download Concluído: ', selectedVideo['id'])
        # see OUTPUT TEMPLATE in https://github.com/ytdl-org/youtube-dl

        # https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/python
        # https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/index#models
        model_path_lite = '/Users/jaderxnet/Documents/GitHub/GERAD/neuralNetorks/pose_landmarker_lite.task'
        model_path_Heavy = '/Users/jaderxnet/Documents/GitHub/GERAD/neuralNetorks/pose_landmarker_heavy.task'

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult

        # Create a pose landmarker instance with the video mode:
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path_Heavy),
            running_mode=VisionRunningMode.VIDEO)

        # Use OpenCV’s VideoCapture to load the input video.
        cap = cv2.VideoCapture("videos/" + selectedVideo['id']
                               + "/" + selectedVideo['id']
                               + ".mp4")
        if (cap.isOpened() == False):
            print("Error opening video stream or file")

        # Read until video is completed
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()

            if ret == True:

                # Display the resulting frame
                # cv2.imshow('Video Frame', frame)

                # Convert the frame received from OpenCV to a MediaPipe’s Image object.
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=frame)
                with PoseLandmarker.create_from_options(options) as landmarker:
                    # The landmarker is initialized. Use it here.
                    # ...
                    frame_timestamp_ms = selectedVideo['fps']
                    # Perform pose landmarking on the provided single image.
                    # The pose landmarker must be created with the video mode.
                    pose_landmarker_result = landmarker.detect_for_video(
                        mp_image, frame_timestamp_ms)
                    quantidade_poses = len(
                        pose_landmarker_result.pose_landmarks)
                    print(quantidade_poses)
                    if quantidade_poses > 0:
                        # print("Pose:", pose_landmarker_result)
                        annotated_image = draw_landmarks_on_image(
                            mp_image.numpy_view(), pose_landmarker_result)
                        annotated_image_rgb = cv2.cvtColor(
                            annotated_image, cv2.COLOR_RGB2BGR)
                        # print("Image: ", annotated_image)
                        # cv2.imshow("MediaPipe Fame", annotated_image_rgb)
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            # Break the loop
            else:
                break

        # When everything done, release the video capture object
        cap.release()

        # Closes all the frames
        cv2.destroyAllWindows()

        # Load the frame rate of the video using OpenCV’s CV_CAP_PROP_FPS
        # You’ll need it to calculate the timestamp for each frame.

        # Loop through each frame in the video using VideoCapture#read()
