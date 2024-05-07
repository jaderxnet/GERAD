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


if __name__ == '__main__':

    '''
    Main used to converter automatic videos and your annotations to cocodataset
    '''
    # Converter List Videos and your Annotation to COCO like Dataset
    # STEPS:
    # 1 - Load File With a list of videos IDs, Metric, thressholds
    # video file path and path to filtered annotations of full dataset
    inputFilePath = "ListsVideos/videodatasettest2.csv"
    pathVideos = "videoTest"
    ORIGINAL_PATH_VIDEOS = "videoTest"
    HOME_DIR = 'dataset2Video'
    VIDEOS_PATH = "videos"
    DATASET_PATH = 'images'
    SCREEN_DIMENSIONS = (1920, 1080)

    videosDataset = pd.read_csv(inputFilePath, sep=";")
    print("Videos Dataset: ", videosDataset.head())
    videoDatasetDict = videosDataset.set_index('video_id').T.to_dict('list')
    print("Dict: ", videoDatasetDict)

    dataset = {"video_id": [], "frame": [],
               "keypoints": [], "bbox": [], "image_id": [], "image_path": []}
    for index, row in videosDataset.iterrows():
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
                        np.array(yoloKeypoints[indexKeypointMinorDistance]), SCREEN_DIMENSIONS[0], SCREEN_DIMENSIONS[1])
                    dataset["keypoints"].append(keypoint)

                    dataset["bbox"].append(
                        BraceDataset.get_box_from_keypoints(np.array(keypoint), box_border=20))
                    dataset["image_id"].append(id+"-"+frame)
                    dataset["image_path"].append(
                        HOME_DIR+"/images/"+id+"/"+frame+".jpg")
                else:
                    print("Minor: ", videoData[id]["frames"][frame][row["metric"]]
                          ["minor_distance"], "Threshold", row["threshold"])
            else:
                print("Poses: ", videoData[id]["frames"]
                      [frame]["YOLO"]["poses_count"])
        print("Id: ", id, "Count: ", len(indicesToDataset),
              "dataset:", len(dataset["frame"]))

    # Create paht anda directorys COCOLike
    # cria as pastas
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

    # Create Frames by video
    # Use OpenCV’s VideoCapture to load the input video.

    indexFrame = 1
    drawAll = True
    saveImages = True
    moveImages = True
    printAll = False
    oldVideoId = None
    cap = None

    # Print size of complet dataset
    if saveImages:
        for indexDataset in range(len(dataset["video_id"])):
            print("Index: ", indexDataset)
            try:
                os.mkdir(HOME_DIR+"/" + DATASET_PATH+"/" +
                         dataset["video_id"][indexDataset])
            except:
                print(HOME_DIR+"/" + DATASET_PATH+"/" +
                      dataset["video_id"][indexDataset]+"/images", "Já existe!")
            if (oldVideoId == None or oldVideoId != dataset["video_id"][indexDataset]):
                indexFrame = 1
                cap = cv2.VideoCapture(ORIGINAL_PATH_VIDEOS + "/" + dataset["video_id"][indexDataset]
                                       + "/" +
                                       dataset["video_id"][indexDataset]
                                       + ".mp4")
            if (cap.isOpened() == False):
                raise Exception("Error opening video stream or file")
            if printAll:
                print("Index Frame: ", dataset["frame"][indexDataset])
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
                    frame = cv2.rectangle(frame, (int(dataset["bbox"][indexDataset][0]), int(
                        dataset["bbox"][indexDataset][1])), (int(dataset["bbox"][indexDataset][0])+int(dataset["bbox"][indexDataset][2]), int(
                            dataset["bbox"][indexDataset][1])+int(dataset["bbox"][indexDataset][3])), (255, 0, 0), 2)
                    frame = VideoDataset.drawPoints(
                        dataset["keypoints"][indexDataset], frame)
                status = cv2.imwrite(
                    dataset["image_path"][indexDataset], frame)
                if not status:
                    print("D'nt Save Image in: ",
                          dataset["image_path"][indexDataset])
            indexFrame += 1
            oldVideoId = dataset["video_id"][indexDataset]

    videoDataset = VideoDataset(pd.DataFrame(dataset))
    print(videoDataset.df)

    # Distribut in train/val
    total_count = len(videoDataset.df["frame"])
    val_percentage = 0.1
    test_percentage = 0.1
    total_train = total_count * (1 - val_percentage - test_percentage)
    total_val = total_count * (val_percentage)
    total_test = total_count * (val_percentage)
    print("Total count: ", total_count)
    print("Dataset antes", videoDataset.df)

    kf = GroupKFold(n_splits=5)
    videoDataset.df = videoDataset.df.reset_index(drop=True)
    videoDataset.df['fold'] = -1
    for fold, (train_idx, val_idx) in enumerate(kf.split(videoDataset.df, y=videoDataset.df.video_id.tolist(), groups=videoDataset.df.frame)):
        videoDataset.df.loc[val_idx, 'fold'] = fold

    print("Dataset depois Group", videoDataset.df)
    SELECTED_FOLD = 4

    # Erro ao copiar arquivos da segunda pasta
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

    # salva o arquivo json

    def save_annot_json(json_annotation, filename):
        with open(filename, 'w') as f:
            output_json = json.dumps(json_annotation, cls=NpEncoder)
            f.write(output_json)

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
    save_annot_json(train_annot_json,
                    f"{HOME_DIR}/{DATASET_PATH}/annotations/train.json")
    save_annot_json(
        val_annot_json, f"{HOME_DIR}/{DATASET_PATH}/annotations/valid.json")
