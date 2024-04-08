'''
Create Python class to manage video process for each 
video  in processedVideos.csv based in:
1 - Read input csv and select the last no processed video;
2 - Change video status to Processing;
3 - Use the class ManagerSingleton to processing each video control; 
4 - Use class VideoSplitImage to download video in a folder with the video id name;
5 - Change class VideoSplitImage to Read each frame;
6 - Create class NeuralNetwork to manage diferents Neural Networks
7 - Apply 2 types of Neural Networks in each frame of video
    A - Media Pipe
    B - Yolo8 
8 - Create class CompareKeypoints to use diferents wheys to calculate keypoints
9 - Calculate the difference betwen inferences
10 - Save a file output by video with all the inferences frames
    A - Youtube ID
    B - Duration
    V - FPS
    D - Frames
        1 - Frame ID
        2 - Yolo8 Landmarks
        3 - Media Pipe Landmarks
        4 OK - (EPDNVP)EndPoint Diference Normalized Euclidian distance sum multiply by visivle product
        5 - (EPDNMVP) Normalized Euclidian distance multiply by visivle product media
        6 - (EPDNM) Normalized Euclidian distance media
        7 OK - (EPE)EndPoint Error (EPE) - Pixel Euclidian distance media
    E - Media of Normalized Euclidian distance sum multiply by visivle product
    F - Media of Normalized Euclidian distance media
    G - Media of Normalized Euclidian distance multiply by visivle product media
    H - Media of EndPoint Error (EPE) - Pixel Euclidian distance media    
11 - Save a file with sequences:
    A - Initial and final frame 
    B - Media Calculated difference ;
    C - Difference threshold;
    D - LOW or HI
12 - Save a picture in the first frame by sequence;
'''

from ultralytics import YOLO
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import mediapipe as mp
import os.path
import os
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


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


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


def normalize_distance_visible(keypoints1, keypoints2):
    sum_distance = 0
    for point1, point2 in zip(keypoints1, keypoints2):
        dist = np.linalg.norm(point1[0:2]-point2[0:2])
        if (len(point1) > 2 and len(point2) > 2):
            dist *= point1[-1]*point2[-1]
        sum_distance += dist
    return sum_distance


def normalize_distance_media(keypoints1, keypoints2):
    sum_distance = 0
    for point1, point2 in zip(keypoints1, keypoints2):
        dist = np.linalg.norm(point1[0:2]-point2[0:2])
        sum_distance += dist
    sum_distance /= 17
    return sum_distance


def average_pixel_distance(keypoints1, keypoints2):
    averege = 0
    count = 0
    # print("Keys 1:", keypoints1, " Keys 2: ", keypoints2)
    for point1, point2 in zip(keypoints1, keypoints2):
        dist = np.linalg.norm(point1[0:2] - point2[0:2])
        # print("dist ", dist)
        averege += dist
        count += 1
        # print("average ", count, " - ", averege)
    averege = averege/17
    # print("average /17 - ", averege)
    return averege


SCREEN_DIMENSIONS = (1920, 1080)


def to_pixel_coords(relative_coords):
    # print("to_pixel_coords: ", relative_coords)
    result = []
    for coord in relative_coords:
        result.append(coord[1:3] * SCREEN_DIMENSIONS)
    return result


# options
save_video = True
save_file = True
debug_yolo = False
debug_mediaPipe = False
print_all = False
print_resume = True
download_video = True
display_frame = False

inputFilePath = "processedVideos.csv"
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
        for indexToProcess in indexesToProcess.tolist():
            videosTable.at[indexToProcess, 'status'] = 'Processing'
            videosTable.to_csv(inputFilePath, index=False, sep=';')
            selectedVideo = videosTable.loc[indexToProcess]
            if print_all:
                print("Index to update: ", indexToProcess)
                print("Updated Table: ", videosTable)
                print("Selected Video: ", selectedVideo)
            options = {
                'format': 'bestvideo[ext=mp4]',
                # This will select the specific resolution typed here
                # "format": "mp4[height=1080]",
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
            model_path_lite = '/Users/jaderxnet/Documents/GitHub/GERAD/neuralNetworks/mediapipe/pose_landmarker_lite.task'
            model_path_Heavy = '/Users/jaderxnet/Documents/GitHub/GERAD/neuralNetworks/mediapipe/pose_landmarker_heavy.task'

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
            model_file = 'neuralNetworks/YOLO/yolov8n-pose.pt'
            model = YOLO(model_file)  # load an official model
            # model = YOLO('path/to/best.pt')  # load a custom model

            # Use OpenCV’s VideoCapture to load the input video.
            cap = cv2.VideoCapture("videos/" + selectedVideo['id']
                                   + "/" + selectedVideo['id']
                                   + ".mp4")

            outputFilePath = "/Users/jaderxnet/Documents/GitHub/GERAD/videos/" + \
                selectedVideo['id']
            file1 = None
            if save_file:
                # os.mkdir(outputFilePath)
                outputFilePath += "/" + selectedVideo['id']+".txt"
                file1 = open(outputFilePath, "w")

            if (cap.isOpened() == False):
                print("Error opening video stream or file")

            dictionary = {}
            # dictionary with video id on youtube
            dictionary[selectedVideo["id"]] = {}
            dictionary[selectedVideo["id"]
                       ]["duration"] = selectedVideo["duration"]
            dictionary[selectedVideo["id"]]["fps"] = selectedVideo["fps"]
            # frames have the dictionary with frame id
            # (EPDNVP)EndPoint Diference Normalized Euclidian distance sum multiply by visivle product
            media_distance_EPDNVP = 0
            media_distance_EPDNM = 0
            media_distance_EPDNMVP = 0
            media_count = 0
            # (EPE)EndPoint Error (EPE) - Pixel Euclidian distance media
            media_distance_EPE = 0

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
                    # if display_frame:
                    #    cv2.imshow('Video Frame', frame)
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
                    # if display_yolo:
                    #    cv2.imshow("result", res_plotted)

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

                    # Press Q on keyboard to  exit
                    # Compare distance

                    minor_distance_EPDNVP = 100000
                    minor_distance_EPDNM = 100000
                    minor_distance_EPDNMVP = 100000
                    minor_distance_EPE = 100000
                    minor_yolo_distance_index_EPDNVP = -1
                    minor_yolo_distance_index_EPDNM = -1
                    minor_yolo_distance_index_EPDNMVP = -1
                    minor_yolo_distance_index_EPE = -1
                    if (dictionary[selectedVideo["id"]
                                   ]["frames"][frames_count]["mediapipe"]["poses_count"] > 0
                        and dictionary[selectedVideo["id"]
                                       ]["frames"][frames_count]["YOLO"]["poses_count"] > 0):
                        index = 0

                        # print("MEDIAPIPE:", dictionary[selectedVideo["id"]]["frames"
                        # filter indexes metch to yolo from 33 mediapipe keypoints
                        #                                                 ][frames_count]["mediapipe"]["keypoints"][0])
                        filter_indices = [0, 2, 5, 7, 8, 11, 12,
                                          13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

                        for yolo_pose in dictionary[selectedVideo["id"]]["frames"][frames_count]["YOLO"]["keypoints"]:
                            # print("YOLO:", yolo_pose)

                            distance_EPDNVP = normalize_distance_visible(yolo_pose,
                                                                         dictionary[selectedVideo["id"]]["frames"
                                                                                                         ][frames_count]["mediapipe"]["keypoints"][0][filter_indices])
                            if (minor_distance_EPDNVP > distance_EPDNVP):
                                minor_distance_EPDNVP = distance_EPDNVP
                                minor_yolo_distance_index_EPDNVP = index

                            distance_EPDNMVP = distance_EPDNVP/17
                            if (minor_distance_EPDNMVP > distance_EPDNMVP):
                                minor_distance_EPDNMVP = distance_EPDNMVP
                                minor_yolo_distance_index_EPDNMVP = index

                            distance_EPDNM = normalize_distance_media(yolo_pose,
                                                                      dictionary[selectedVideo["id"]]["frames"
                                                                                                      ][frames_count]["mediapipe"]["keypoints"][0][filter_indices])
                            if (minor_distance_EPDNM > distance_EPDNM):
                                minor_distance_EPDNM = distance_EPDNM
                                minor_yolo_distance_index_EPDNM = index

                            # print("Index: ", index, "Media: ", distance)
                            distance_EPE = average_pixel_distance(to_pixel_coords(yolo_pose),
                                                                  to_pixel_coords(dictionary[selectedVideo["id"]]["frames"
                                                                                                                  ][frames_count]["mediapipe"]["keypoints"][0][filter_indices]))
                            if (minor_distance_EPE > distance_EPE):
                                minor_distance_EPE = distance_EPE
                                minor_yolo_distance_index_EPE = index
                            index += 1

                        # (EPDNVP)EndPoint Diference Normalized Euclidian distance sum multiply by visivle product
                        dictionary[selectedVideo["id"]
                                   ]["frames"][frames_count]["EPDNVP"] = minor_distance_EPDNVP
                        media_distance_EPDNVP += minor_distance_EPDNVP
                        dictionary[selectedVideo["id"]
                                   ]["frames"][frames_count]["EPDNMVP"] = minor_distance_EPDNMVP
                        media_distance_EPDNMVP += minor_distance_EPDNMVP
                        dictionary[selectedVideo["id"]
                                   ]["frames"][frames_count]["EPDNM"] = minor_distance_EPDNM
                        media_distance_EPDNM += minor_distance_EPDNM
                        dictionary[selectedVideo["id"]
                                   ]["frames"][frames_count]["EPE"] = minor_distance_EPE
                        media_distance_EPE += minor_distance_EPE
                        media_count += 1
                    # (EPDNM)Normalized Euclidian distance media
    #                dictionary[selectedVideo["id"]
    #                           ]["frames"][frames_count]["EPDNMVP"]
                    # (EPDNMVP)Normalized Euclidian distance multiply by visivle product media
    #                dictionary[selectedVideo["id"]
    #                           ]["frames"][frames_count]["EPDNM"]
                    # (EPE)EndPoint Error (EPE) - Pixel Euclidian distance media
    #                dictionary[selectedVideo["id"]
    #                           ]["frames"][frames_count]["EPE"]

                    frame_information = "| Frame" + f'{frames_count:06}' + \
                        "| YOLO: " + f'{quantidade_poses_yolo:02}' + \
                        "| Mediapipe: " + f'{quantidade_poses_mediapipe:02}' + \
                        "| EPDNVP INDEX : " + f'{minor_yolo_distance_index_EPDNVP:02}' + \
                        " : " + f'{minor_distance_EPDNVP:06.15f}' + \
                        "| EPDNMVP INDEX : " + f'{minor_yolo_distance_index_EPDNMVP:02}' + \
                        " : " + f'{minor_distance_EPDNMVP:06.15f}' + '\n' +\
                        "| EPDNM INDEX : " + f'{minor_yolo_distance_index_EPDNM:02}' + \
                        " : " + f'{minor_distance_EPDNM:06.15f}' +  \
                        "| EPE INDEX : " + f'{minor_yolo_distance_index_EPE:02}' + \
                        " : " + f'{minor_distance_EPE:06.15f}'
                    if (media_count > 0):
                        frame_information = frame_information + \
                            "| Media EPDNVP: " + f'{media_distance_EPDNVP/media_count:06.15f}' + '\n' +\
                            "| Media EPDNMVP: " + f'{media_distance_EPDNMVP/media_count:06.15f}' + \
                            "| Media EPDNM: " + f'{media_distance_EPDNM/media_count:06.15f}' + \
                            "| Media EPE: " + \
                            f'{media_distance_EPE/media_count:06.15f}'
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (10, 30)
                    fontScale = 0.8
                    fontColor = (255, 255, 255)
                    thickness = 1
                    lineType = 2
                    dy = 40
                    for i, line in enumerate(frame_information.split('\n')):
                        y = bottomLeftCornerOfText[1] + i*dy
                        cv2.putText(frame, line,
                                    (bottomLeftCornerOfText[0], y),
                                    font,
                                    fontScale,
                                    fontColor,
                                    thickness,
                                    lineType)
                        cv2.putText(res_plotted, line,
                                    (bottomLeftCornerOfText[0], y),
                                    font,
                                    fontScale,
                                    fontColor,
                                    thickness,
                                    lineType)
                        cv2.putText(annotated_image_rgb, line,
                                    (bottomLeftCornerOfText[0], y),
                                    font,
                                    fontScale,
                                    fontColor,
                                    thickness,
                                    lineType)
                    if print_resume:
                        print(
                            # "\033[K",
                            frame_information, end="\r")

                    if display_frame:
                        if (quantidade_poses_mediapipe > 0):
                            cv2.imshow("Fame", annotated_image_rgb)
                        else:
                            if quantidade_poses_yolo > 0:
                                cv2.imshow("Fame", res_plotted)
                            else:
                                cv2.imshow("Fame", frame)
                    if save_video:
                        if (quantidade_poses_mediapipe > 0):
                            out.write(annotated_image_rgb)
                        else:
                            if quantidade_poses_yolo > 0:
                                out.write(res_plotted)
                            else:
                                out.write(frame)

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
                       ]["frames_count"] = frames_count
            if (media_count > 0):
                dictionary[selectedVideo["id"]
                           ]["EPDNVP"] = media_distance_EPDNVP/media_count
                dictionary[selectedVideo["id"]
                           ]["EPDNMVP"] = media_distance_EPDNMVP/media_count
                dictionary[selectedVideo["id"]
                           ]["EPDNM"] = media_distance_EPDNM/media_count
                dictionary[selectedVideo["id"]
                           ]["EPE"] = media_distance_EPE/media_count
            # (EPDNM)Normalized Euclidian distance media
    #        dictionary[selectedVideo["id"]
    #                    ]["EPDNMVP"]
            # (EPDNMVP)Normalized Euclidian distance multiply by visivle product media
    #        dictionary[selectedVideo["id"]
    #                    ]["EPDNM"]
            # (EPE)EndPoint Error (EPE) - Pixel Euclidian distance media
    #        dictionary[selectedVideo["id"]
    #                    ]["EPE"]

            # Closes all the frames
            cv2.destroyAllWindows()

            # Load the frame rate of the video using OpenCV’s CV_CAP_PROP_FPS
            # You’ll need it to calculate the timestamp for each frame.

            # Loop through each frame in the video using VideoCapture#read()
            videosTable.at[indexToProcess, 'status'] = 'Processed'
            videosTable.to_csv(inputFilePath, index=False, sep=';')

            if save_file:
                file1.write(json.dumps(dictionary, indent=4, cls=NpEncoder))
                file1.close()
