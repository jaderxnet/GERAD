# import warnings
# warnings.filterwarnings("ignore")
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import json
import pandas as pd
import cv2
import shutil
import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold
from coder import NpEncoder, NpDecorder
from adapter import CocoConverter, BraceDataset


class VideoDataset(Dataset):
    '''Class Video Dataset Definition'''

    def __init__(self, df):
        assert len(df) > 0
        self.df = df

    def __len__(self):
        return len(self.df)

    @staticmethod
    def df_to_map_annotations(df):
        map_annotations = {}
        for id in df["video_id"]:
            map_annotations[id] = {}
        for ind, row in df.iterrows():
            map_annotations[row["video_id"]][row["frame"]] = {
                "file_name": f'{row["image_id"]}.jpg', "keypoints": row["keypoints"], "box": row["bbox"]}
        return map_annotations

    @staticmethod
    def drawRect(bbox, img):
        img = cv2.rectangle(img, (int(bbox[0]), int(
            bbox[1])), (int(bbox[0])+int(bbox[2]), int(
                bbox[1])+int(bbox[3])), (255, 0, 0), 2)
        return img

    @staticmethod
    def drawPoints(keypoints, img):
        '''
        Return a Image with the points painted
        '''
        # print("Keypoints: ", keypoints)
        for point in keypoints:
            # print("Keypoint: ", point[0], point[1])
            img = cv2.circle(img, (int(point[0]), int(point[1])), radius=10,
                             color=(0, 0, 255), thickness=-1)
        return img

    @staticmethod
    def loadFile(inputFilePath, pathVideos, DATASET_PATH):
        videosDataset = pd.read_csv(inputFilePath, sep=";")
        # Populate a map of lists with the dataset information data
        dataset = {"video_id": [], "frame": [],
                   "keypoints": [], "bbox": [], "image_id": [], "image_path": []}
        for index, row in videosDataset.iterrows():
            # Load a filtered file from map
            folder = file_name = id = row["video_id"]
            if (len(row["filteredAnnotation"]) > 0):
                fileAnnotation = open(row["filteredAnnotation"], "r")
            else:
                fileAnnotation = open(pathVideos + "/" + folder + "/" +
                                      id+".txt", "r")
            text = fileAnnotation.read().replace('\n', '')
            indicesToDataset = []
            videoData = json.loads(text, cls=NpDecorder)
            total_frame = videoData[id]["frames_count"]
            print("Total: ", total_frame)
            total_frame = len(videoData[id]["frames"])
            print("Filtered: ", total_frame)
            for frame in videoData[id]["frames"].keys():
                if (videoData[id]["frames"][frame]["YOLO"]["poses_count"] > 0 and
                        videoData[id]["frames"][frame]["mediapipe"]["poses_count"] > 0):
                    if (videoData[id]["frames"][frame][row["metric"]]["minor_distance"] <= row["threshold"]):
                        indicesToDataset.append(frame)
                        dataset["video_id"].append(id)
                        dataset["frame"].append(frame)
                        yoloKeypoints = videoData[id]["frames"][frame]["YOLO"]["keypoints"]
                        indexKeypointMinorDistance = videoData[id]["frames"][frame][row["metric"]]["index"]
                        keypoint = CocoConverter.to_pixel_coords(
                            np.array(yoloKeypoints[indexKeypointMinorDistance]), row["width"], row["height"])
                        dataset["keypoints"].append(keypoint)

                        dataset["bbox"].append(
                            BraceDataset.get_box_from_keypoints(np.array(keypoint), box_border=20))
                        dataset["image_id"].append(id+"-"+frame)
                        dataset["image_path"].append(
                            HOME_DIR+"/"+DATASET_PATH+"/"+id+"/"+frame+".jpg")
                    else:
                        print("Minor: ", videoData[id]["frames"][frame][row["metric"]]
                              ["minor_distance"], "Threshold", row["threshold"])
                else:
                    print("Poses: ", videoData[id]["frames"]
                          [frame]["YOLO"]["poses_count"])
            print("Id: ", id, "Count: ", len(indicesToDataset),
                  "dataset:", len(dataset["frame"]))
        return dataset

    @staticmethod
    def createDatasetFoldersAndImages(ORIGINAL_PATH_VIDEOS, DATASET_PATH, drawAll, saveImages, printAll, oldVideoId):
        # cria as pastas
        indexFrame = 1
        cap = None
        try:
            os.mkdir(HOME_DIR)
        except:
            print(HOME_DIR, "Já existe!")
        try:
            os.mkdir(HOME_DIR+"/"+DATASET_PATH)
        except:
            print(HOME_DIR+"/"+DATASET_PATH, "Já existe!")
        try:
            os.mkdir(HOME_DIR+"/"+DATASET_PATH+"/train2017")
        except:
            print(HOME_DIR+"/"+DATASET_PATH+"/train2017", "Já existe!")
        try:
            os.mkdir(HOME_DIR+"/"+DATASET_PATH+"/val2017")
        except:
            print(HOME_DIR+"/"+DATASET_PATH+"/val2017", "Já existe!")
        try:
            os.mkdir(HOME_DIR+"/"+DATASET_PATH+"/annotations")
        except:
            print(HOME_DIR+"/"+DATASET_PATH+"/annotations", "Já existe!")

        if saveImages:
            for indexDataset in range(len(dataset["video_id"])):
                print("Index: ", indexDataset)
                try:
                    os.mkdir(HOME_DIR+"/" + DATASET_PATH+"/" +
                             dataset["video_id"][indexDataset])
                except:
                    print(HOME_DIR+"/" + DATASET_PATH+"/" +
                          dataset["video_id"][indexDataset]+"/"+DATASET_PATH, "Já existe!")
                if (oldVideoId == None or oldVideoId != dataset["video_id"][indexDataset]):
                    indexFrame = 1
                    cap = cv2.VideoCapture(ORIGINAL_PATH_VIDEOS + "/" + dataset["video_id"][indexDataset]
                                           + "/" +
                                           dataset["video_id"][indexDataset]
                                           + ".mp4")
                if (cap.isOpened() == False):
                    raise Exception("Error opening video stream or file")
                if printAll:
                    print("ID: ", dataset["video_id"][indexDataset],
                          " Frame: ", dataset["frame"][indexDataset])
                while (indexFrame < int(dataset["frame"][indexDataset])):
                    ret, frame = cap.read()
                    if printAll:
                        print("Frame: ", indexFrame, ret, indexFrame ==
                              int(dataset["frame"][indexDataset]))
                    indexFrame += 1
                ret, frame = cap.read()
                if printAll:
                    print("Frame: ", indexFrame, ret, indexFrame ==
                          int(dataset["frame"][indexDataset]))
                if ret == True:
                    if printAll:
                        print(frame.shape)
                        print("Save Image in: ",
                              dataset["image_path"][indexDataset])
                        print("Box: ", dataset["bbox"][indexDataset])
                    if drawAll:
                        frame = VideoDataset.drawRect(
                            dataset["bbox"][indexDataset], frame)
                        frame = VideoDataset.drawPoints(
                            dataset["keypoints"][indexDataset], frame)
                    status = cv2.imwrite(
                        dataset["image_path"][indexDataset], frame)
                    if not status:
                        print("D'nt Save Image in: ",
                              dataset["image_path"][indexDataset])
                indexFrame += 1
                oldVideoId = dataset["video_id"][indexDataset]

    @staticmethod
    def groupByFold(dataset, val_percentage):
        videoDataset = VideoDataset(pd.DataFrame(dataset))
        print(videoDataset.df)
        # Distribut in train/val
        total_count = len(videoDataset.df["frame"])
        val_percentage = 0.1
        total_train = total_count * (1 - val_percentage)
        total_val = total_count * (val_percentage)
        total_test = total_count * (val_percentage)
        print("Total count: ", total_count)
        print("Dataset antes", videoDataset.df)

        kf = GroupKFold(n_splits=5)
        videoDataset.df = videoDataset.df.reset_index(drop=True)
        videoDataset.df['fold'] = -1
        for fold, (train_idx, val_idx) in enumerate(kf.split(videoDataset.df, y=videoDataset.df.video_id.tolist(), groups=videoDataset.df.frame)):
            videoDataset.df.loc[val_idx, 'fold'] = fold
        return videoDataset

    def moveImages(self, SELECTED_FOLD, moveImages):
        if moveImages:
            # Divide entre treino e teste
            for i in range(len(videoDataset.df)):
                row = videoDataset.df.loc[i]
                if row.fold != SELECTED_FOLD:
                    shutil.copyfile(row.image_path, HOME_DIR+"/" +
                                    DATASET_PATH+"/train2017/"+row.image_id+".jpg")
                    row.image_path = HOME_DIR+"/" + DATASET_PATH+"/train2017/"+row.image_id+".jpg"
                else:
                    shutil.copyfile(f'{row.image_path}', HOME_DIR+"/" +
                                    DATASET_PATH+"/val2017/"+row.image_id+".jpg")
                    row.image_path = HOME_DIR+"/" + DATASET_PATH+"/val2017/"+row.image_id+".jpg"

            print(
                f'Number of training files: {len(os.listdir(f"{HOME_DIR}/{DATASET_PATH}/train2017/"))}')
            print(
                f'Number of validation files: {len(os.listdir(f"{HOME_DIR}/{DATASET_PATH}/val2017/"))}')

    @staticmethod
    def save_annot_json(json_annotation, filename):
        with open(filename, 'w') as f:
            output_json = json.dumps(json_annotation, cls=NpEncoder)
            f.write(output_json)


if __name__ == '__main__':
    print(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    '''
    Main used to converter automatic videos and your annotations to cocodataset
    '''
    # Converter List Videos and your Annotation to COCO like Dataset
    # STEPS:
    # 1 - Load File With a list of videos IDs, Metric, thressholds
    # video file path and path to filtered annotations of full dataset
    HOME_DIR = 'datasetFrevo3'
    DATASET_PATH = 'dataset3'
    # Load File csv with dataset list
    inputFilePath = "ListsInfo/videofrevodataset220240514140423.csv"
    ORIGINAL_PATH_VIDEOS = "videoTest2"

    dataset = VideoDataset.loadFile(
        inputFilePath, ORIGINAL_PATH_VIDEOS, DATASET_PATH)

    print("TAM Dataset : ", len(dataset["video_id"]))

    # Create Frames by video
    # Use OpenCV’s VideoCapture to load the input video.

    drawAll = False
    saveImages = True
    printAll = False
    oldVideoId = None

    # Print size of complet dataset

    VideoDataset.createDatasetFoldersAndImages(
        ORIGINAL_PATH_VIDEOS, DATASET_PATH, drawAll, saveImages, printAll, oldVideoId)

    val_percentage = 0.2
    videoDataset = VideoDataset.groupByFold(dataset, val_percentage)

    print("TAM Dataset : ", videoDataset)

    # Dataset depois Group
    SELECTED_FOLD = 4
    moveImages = True

    videoDataset.moveImages(SELECTED_FOLD, moveImages)
    # Erro ao copiar arquivos da segunda pasta

    # salva o arquivo json

    train_dataset = CocoConverter("2024", "1", "Video Dataset Train", "Jader Abreu",
                                  "train", datetime.datetime.now().strftime("%x"), "train", "train", "train")

    val_dataset = CocoConverter("2024", "1", "Video Dataset Test", "Jader Abreu",
                                "test", datetime.datetime.now().strftime("%x"), "test", "test", "test")

    # Convert COTS dataset to JSON COCO
    train_annot_json = train_dataset.build(
        VideoDataset.df_to_map_annotations(videoDataset.df[videoDataset.df.fold != SELECTED_FOLD]))
    val_annot_json = val_dataset.build(
        VideoDataset.df_to_map_annotations(videoDataset.df[videoDataset.df.fold == SELECTED_FOLD]))
    # print(val_annot_json)
    # Save converted annotations
    VideoDataset.save_annot_json(train_annot_json,
                                 f"{HOME_DIR}/{DATASET_PATH}/annotations/train.json")
    VideoDataset.save_annot_json(
        val_annot_json, f"{HOME_DIR}/{DATASET_PATH}/annotations/valid.json")
    print(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
