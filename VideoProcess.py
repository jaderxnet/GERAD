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

from ultralytics import YOLO
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import mediapipe as mp
import os.path
import json
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import pandas as pd
# !pip install git+https://github.com/ytdl-org/youtube-dl.git@master
from yt_dlp import YoutubeDL
from datetime import datetime
#!pip install "opencv-python-headless<4.3"
import cv2

# YOLOv8 pip install ultralytics
import ultralytics
ultralytics.checks()
# Dowload python > 3.9 version in https://www.python.org/downloads
#!pip install -q mediapipe==0.10.0
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


def average_distance(keypoints1, keypoints2):
    averege = 0
    for point1, point2 in zip(keypoints1, keypoints2):
        dist = np.linalg.norm(point1[0:2]-point2[0:2])
        if (len(point1) > 2 and len(point2) > 2):
            dist *= point1[-1]*point2[-1]
        averege += dist
    return averege


inputFilePath = "processedVideos.csv"

# options
save_video = False
save_file = False
debug_yolo = False
debug_mediaPipe = True
print_all = False
print_resume = True
download_video = False
save_file = False
display_frame = False
display_mediapipe = False
display_yolo = False

videosTable = pd.read_csv(inputFilePath, sep=';')
filtered = videosTable["status"] == "Processing"

if print_all:
    print("Lendo: ", inputFilePath)
    print(inputFilePath)
    print("Table: ", len(videosTable))
    print("Filtered: ", len(filtered))


indexesProcessing = videosTable.index[filtered]
if (len(indexesProcessing) > 0):
    print("PROCESSING THE INDEX VIDEO: ", indexesProcessing.tolist()[0])
else:
    filtered = videosTable["status"] == "Ready"
    indexesToProcess = videosTable.index[filtered]
    if print_all:
        print("Filtered: ", len(filtered))
        print("indexesToProcess: ", len(indexesToProcess))
    if (len(indexesToProcess) < 1):
        print("ALL VIDEOS PROCESSED!")
    else:

        videosTable.at[indexesToProcess.tolist()[0], 'status'] = 'Processing'
        # videosTable.to_csv(inputFilePath, index=False, sep=';')
        selectedVideo = videosTable.loc[indexesToProcess.tolist()[0]]
        if print_all:
            print("Index to update: ", indexesToProcess.tolist()[0])
            print("Updated Table: ", videosTable)
            print("Selected Video: ", selectedVideo)
        options = {
            # 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'
            # This will select the specific resolution typed here
            "format": "mp4[height=1080]",
            'no_check_certificate': True,
            # "%(id)s/%(id)s-%(title)s.%(ext)s"
            "outtmpl": "videos/%(id)s/%(id)s.%(ext)s"
        }
        if download_video:
            with YoutubeDL(options) as ydl:
                ydl.download([selectedVideo['url']])
            if print_all:
                print('Download Concluído: ', selectedVideo['id'])
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

        # Load a model
        model_file = 'yolov8n-pose.pt'
        model = YOLO(model_file)  # load an official model
        # model = YOLO('path/to/best.pt')  # load a custom model

        # Use OpenCV’s VideoCapture to load the input video.
        cap = cv2.VideoCapture("videos/" + selectedVideo['id']
                               + "/" + selectedVideo['id']
                               + ".mp4")

        outputFilePath = "/Users/jaderxnet/Documents/GitHub/GERAD/videos/" + \
            selectedVideo['id'] + "/" + selectedVideo['id']+".txt"
        file1 = None
        if save_file:
            file1 = open(outputFilePath, "w")

        if (cap.isOpened() == False):
            print("Error opening video stream or file")

        dictionary = {}
        # dictionary with video id on youtube
        dictionary[selectedVideo["id"]] = {}
        dictionary[selectedVideo["id"]]["duration"] = selectedVideo["duration"]
        dictionary[selectedVideo["id"]]["fps"] = selectedVideo["fps"]
        # frames have the dictionary with frame id
        dictionary[selectedVideo["id"]]["frames"] = {}

        # record video
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = None
        if save_video:
            out = cv2.VideoWriter('videos/'+selectedVideo["id"]+'/Out2' + selectedVideo["id"]+'.mp4', fourcc, 20.0,
                                  (selectedVideo["width"], selectedVideo["height"]))

        if print_all:
            print(dictionary)
        print_count = 0
        frames_count = 0
        # Read until video is completed
        frame_information = ""
        while (cap.isOpened()):
            # Capture frame-by-frame

            ret, frame = cap.read()

            if ret == True:
                frames_count += 1
                # frames have the dictionary with frame id
                dictionary[selectedVideo["id"]
                           ]["frames"][frames_count] = {}
                # Display the resulting frame
                if display_frame:
                    cv2.imshow('Video Frame', frame)
                # Predict with the model

                results = model(frame)  # predict on an image
                quantidade_poses_yolo = len(results[0].keypoints)
                if print_all or debug_yolo:
                    print("Quant Poses YOLO: ", quantidade_poses_yolo)
                    print("Quant restults: ", len(results))
                dictionary[selectedVideo["id"]
                           ]["frames"][frames_count]["YOLO"] = {}
                dictionary[selectedVideo["id"]
                           ]["frames"][frames_count]["YOLO"]["neural_network_file"] = model_file
                dictionary[selectedVideo["id"]
                           ]["frames"][frames_count]["YOLO"]["poses_count"] = quantidade_poses_yolo

                if quantidade_poses_yolo > 0:
                    results_json = results[0].tojson(normalize=True)
                    results_json = json.loads(results_json)
                    if print_all or debug_yolo:
                        print("TO JSON: ", results_json)
                        print("X: ", results_json[0]["keypoints"]["x"])
                    dictionary[selectedVideo["id"]
                               ]["frames"][frames_count]["YOLO"]["keypoints"] = []
                    list_poses_yolo = []
                    for result in results_json:
                        pose_yolo = np.stack((np.array((result["keypoints"]["x"])), np.array((
                            result["keypoints"]["y"])), np.array((result["keypoints"]["visible"]))), axis=1)
                        if print_all or debug_yolo:
                            print("List Poses: ", pose_yolo)
                        list_poses_yolo.append(pose_yolo)
                    dictionary[selectedVideo["id"]
                               ]["frames"][frames_count]["YOLO"]["keypoints"] = list_poses_yolo

                if print_all or debug_yolo:
                    print("Yolo Results: ",  results)

                    print("Keypoints Results: ",
                          results[0].tojson(normalize=True))

                res_plotted = results[0].plot()
                if display_yolo:
                    cv2.imshow("result", res_plotted)

                # Convert the frame received from OpenCV to a MediaPipe’s Image object.
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=frame)

                # MediaPipe
                with PoseLandmarker.create_from_options(options) as landmarker:
                    # The landmarker is initialized. Use it here.
                    # ...
                    frame_timestamp_ms = selectedVideo['fps']
                    # Perform pose landmarking on the provided single image.
                    # The pose landmarker must be created with the video mode.
                    pose_landmarker_result = landmarker.detect_for_video(
                        mp_image, frame_timestamp_ms)
                    quantidade_poses_mediapipe = len(
                        pose_landmarker_result.pose_landmarks)
                    if print_all or debug_mediaPipe:
                        print("Quant Poses Mediapipe",
                              quantidade_poses_mediapipe)
                    # mediapipe have the dictionary  neural networks detections
                    dictionary[selectedVideo["id"]
                               ]["frames"][frames_count]["mediapipe"] = {}
                    dictionary[selectedVideo["id"]
                               ]["frames"][frames_count]["mediapipe"]["poses_count"] = quantidade_poses_mediapipe
                    dictionary[selectedVideo["id"]
                               ]["frames"][frames_count]["mediapipe"]["neural_network_file"] = model_path_Heavy
                    # landmarks have the landmarks

                    if print_all or debug_mediaPipe:
                        print(pose_landmarker_result)
                        print("Quant Poses:", quantidade_poses_mediapipe)

                    if quantidade_poses_mediapipe > 0:
                        dictionary[selectedVideo["id"]
                                   ]["frames"][frames_count]["mediapipe"]["keypoints"] = []
                        list_poses = []
                        for normalize_landmark in pose_landmarker_result.pose_landmarks[0]:
                            list_poses.append([normalize_landmark.x,
                                               normalize_landmark.y, normalize_landmark.visibility])
                        dictionary[selectedVideo["id"]]["frames"][frames_count
                                                                  ]["mediapipe"]["keypoints"].append(np.array((list_poses)))
                        if print_all or debug_mediaPipe:
                            print("Media PipePoses:", dictionary[selectedVideo["id"]]["frames"][frames_count
                                                                                                ]["mediapipe"]["keypoints"])
                    annotated_image = draw_landmarks_on_image(
                        res_plotted, pose_landmarker_result)
                    annotated_image_rgb = cv2.cvtColor(
                        annotated_image, cv2.COLOR_RGB2BGR)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (10, 30)
                    fontScale = 0.8
                    fontColor = (255, 255, 255)
                    thickness = 1
                    lineType = 2

                    cv2.putText(annotated_image_rgb, frame_information,
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                thickness,
                                lineType)
                    if print_all:
                        print("Image: ", annotated_image)
                    if display_mediapipe:
                        cv2.imshow("MediaPipe Fame", annotated_image_rgb)
                    if save_video:
                        out.write(annotated_image_rgb)
                # Press Q on keyboard to  exit
                # Compare distance

                minor_distance = 100
                minor_yolo_distance_index = -1
                if (dictionary[selectedVideo["id"]
                               ]["frames"][frames_count]["mediapipe"]["poses_count"] > 0
                   and dictionary[selectedVideo["id"]
                                  ]["frames"][frames_count]["YOLO"]["poses_count"] > 0):
                    index = 0

                    # print("MEDIAPIPE:", dictionary[selectedVideo["id"]]["frames"
                    #                                                    ][frames_count]["mediapipe"]["keypoints"][0])
                    for yolo_pose in dictionary[selectedVideo["id"]]["frames"][frames_count]["YOLO"]["keypoints"]:
                        # print("YOLO:", yolo_pose)

                        distance = average_distance(yolo_pose,
                                                    dictionary[selectedVideo["id"]]["frames"
                                                                                    ][frames_count]["mediapipe"]["keypoints"][0])
                        if (minor_distance > distance):
                            minor_distance = distance
                            minor_yolo_distance_index = index
                        index += 1
                        # print("Index: ", index, "Media: ", distance)
                frame_information = "| Frame" + f'{frames_count:06}'+"| Quant Poses YOLO: " + f'{quantidade_poses_yolo:02}'+"| Quant Poses Mediapipe" + \
                    f'{quantidade_poses_mediapipe:02}'+"| Indice: " + f'{minor_yolo_distance_index:02}' + \
                    "| Menor Distancia Euclidiana Média: " + \
                    f'{minor_distance:03.15f}'
                if print_resume:
                    print(
                        # "\033[K",
                        frame_information, end="\r")

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            # Break the loop
            else:
                break

        # When everything done, release the video capture object
        cap.release()
        if save_video:
            out.release()
        # print(dictionary)
        dictionary[selectedVideo["id"]
                   ]["frames_count"] = frames_count-1
        # Closes all the frames
        cv2.destroyAllWindows()

        # Load the frame rate of the video using OpenCV’s CV_CAP_PROP_FPS
        # You’ll need it to calculate the timestamp for each frame.

        # Loop through each frame in the video using VideoCapture#read()


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


if save_file:
    file1.write(str(dictionary))
    file1.close()
