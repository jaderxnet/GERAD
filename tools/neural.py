from ultralytics import YOLO
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import mediapipe as mp
import numpy as np


class NeuralNetwork:
    def __init__(self, model) -> None:
        self.model = model
        self.network = 'NAN'

    def load(self):
        pass

    def predict(self, frame):
        pass


class MediaPipe(NeuralNetwork):
    def __init__(self):
        model_path_lite = 'neuralNetworks/mediapipe/pose_landmarker_lite.task'
        model_path_Heavy = 'neuralNetworks/mediapipe/pose_landmarker_heavy.task'
        super().__init__(model_path_Heavy)
        self.load()

    def load(self):
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult

        # Create a pose landmarker instance with the video mode:
        mediaPipeOptionVideo = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model),
            running_mode=VisionRunningMode.IMAGE)

        self.network = PoseLandmarker.create_from_options(mediaPipeOptionVideo)
        return self.network

    def predict(self, frame):
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=frame)
        pose_landmarker_result = self.network.detect(
            mp_image)
        return pose_landmarker_result

    def convertMPImagge(self, frame):
        # Convert the frame received from OpenCV to a MediaPipe’s Image object.
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=frame)
        return mp_image

    def draw_landmarks_on_image(self, rgb_image, detection_result):

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

    def getAnnotatedImage(self, frame, pose_landmarker_result):
        annotated_image = self.draw_landmarks_on_image(
            frame, pose_landmarker_result)
        return annotated_image


class Yolo(NeuralNetwork):
    def __init__(self):
        # Load a model
        model_file = 'neuralNetworks/YOLO/yolov8n-pose.pt'
        # model = YOLO('path/to/best.pt')  # load a custom model
        super().__init__(model_file)
        self.load()

    def load(self):
        print("LOAD YOLO")
        self.network = YOLO(self.model)  # load an official model
        return self.network

    def predict(self, frame):
        results = self.network(frame)
        return results

# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/python
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/index#models
