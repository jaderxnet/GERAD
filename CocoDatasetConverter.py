# import warnings
# warnings.filterwarnings("ignore")
import numpy as np
import ast
import os
import json
import pandas as pd
import cv2
import shutil
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class NpDecorder(json.JSONDecoder):
    def default(self, obj):
        if isinstance(obj, int):
            return np.integer(obj)
        if isinstance(obj, float):
            return np.floating(obj)
        if isinstance(obj, list):
            return np.ndarray(obj)
        return super(NpDecorder, self).default(obj)


# https://www.kaggle.com/code/remekkinas/yolox-training-pipeline-cots-dataset-lb-0-507/notebook

def to_int_Visibility(keypoints):
    keypoints_int_visibility = []
    for keypoint in keypoints:
        keypoints_int_visibility.append(int(keypoint[0]))
        keypoints_int_visibility.append(int(keypoint[1]))
        keypoints_int_visibility.append(2)
    if len(keypoints_int_visibility) != 51:
        raise Exception("Quantidade de Keypoints diferente de 51!")
    return keypoints_int_visibility


def dataset2coco(df, dest_path):

    global annotion_id
    annotion_id = 1
    annotations_json = {
        "info": [],
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    info = {
        "year": "2023",
        "version": "1",
        "description": "GERAD dataset - COCO format",
        "contributor": "Jader Abreu",
        "url": "https://github.com/jaderxnet/GERAD",
        "date_created": "2022-07-01T17:15:26-03:00"
    }
    annotations_json["info"] = info

    lic = {
        "id": 1,
        "url": "",
        "name": "Unknown"
    }
    annotations_json["licenses"].append(lic)

    for ann_row in df.itertuples():

        images = {
            "id": ann_row[0],
            "license": 1,
            "file_name": ann_row.image_id + '.jpg',
            "height": SCREEN_DIMENSIONS[1],
            "width": SCREEN_DIMENSIONS[0],
            "date_captured": "2023-07-10T15:17:26-03:00"
        }

        annotations_json["images"].append(images)

        bbox = ann_row.bbox
        b_width = bbox[2]
        b_height = bbox[3]

        # some boxes in COTS are outside the image height and width
        if (bbox[0] + bbox[2] > SCREEN_DIMENSIONS[0]):
            b_width = bbox[0] - SCREEN_DIMENSIONS[0]
        if (bbox[1] + bbox[3] > SCREEN_DIMENSIONS[1]):
            b_height = bbox[1] - SCREEN_DIMENSIONS[0]

        image_segmentation = {
            "segmentation": [],
            "num_keypoints": 17,
            "area": int(bbox[2] * bbox[3]),
            "iscrowd": 0,
            "keypoints": to_int_Visibility(ann_row.keypoints),
            "image_id": ann_row[0],
            "bbox": [bbox[0], bbox[1], b_width, b_height],
            "category_id": 1,
            "id": annotion_id
        }

        annotion_id += 1
        annotations_json["annotations"].append(image_segmentation)
        '''
        [{"supercategory": "person","id": 1,"name": "person","keypoints": ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"],"skeleton": [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]}]
        '''
    classes = {"supercategory": "person", "id": 1, "name": "person", "keypoints": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip",
                                                                                   "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"], "skeleton": [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]}

    annotations_json["categories"].append(classes)

    print(
        f"Dataset COTS annotation to COCO json format completed! Files: {len(df)}")
    return annotations_json


def get_bbox(annots):
    bboxes = [list(annot.values()) for annot in annots]
    return bboxes


def normalize_distance_visible(keypoints1, keypoints2):
    sum_distance = 0
    for point1, point2 in zip(keypoints1, keypoints2):
        dist = np.linalg.norm(point1[0:2]-point2[0:2])
        if (len(point1) > 2 and len(point2) > 2):
            dist *= point1[-1]*point2[-1]
        sum_distance += dist
    return sum_distance


def returnMinorDistanceIndexAndKeypoints(keypoint0, keypoints):
    minor_distance_EPDNVP = 100000
    minor_distance_index = -1
    index = 0
    for poseTest in keypoints:
        # print("YOLO:", yolo_pose)

        distance_EPDNVP = normalize_distance_visible(poseTest, keypoint0)
        if (minor_distance_EPDNVP > distance_EPDNVP):
            minor_distance_EPDNVP = distance_EPDNVP
            minor_distance_index = index
        index += 1

    return minor_distance_index, keypoints[minor_distance_index]


def get_bbox_from(keypoint):
    xmin = 1000000
    xmax = -1000000
    ymin = 1000000
    ymax = -1000000
    for x, y in keypoint:
        if (x < xmin):
            xmin = x
        if (x > xmax):
            xmax = x
        if (y < ymin):
            ymin = y
        if (y > ymax):
            ymax = y
    difx = xmax - xmin
    dify = ymax - ymin
    return [xmin, ymin, difx, dify]


def drawPoints(keypoints, img):
    # print("Keypoints: ", keypoints)
    for point in keypoints:
        # print("Keypoint: ", point[0], point[1])
        img = cv2.circle(img, (int(point[0]), int(point[1])), radius=10,
                         color=(0, 0, 255), thickness=-1)
    return img


SCREEN_DIMENSIONS = (1920, 1080)


def to_pixel_coords(relative_coords):
    # print("to_pixel_coords: ", relative_coords)
    result = []
    for coord in relative_coords:
        # print("Dim: ", SCREEN_DIMENSIONS, "Coord: ", coord,
        #      "Result: ", coord[0] * SCREEN_DIMENSIONS[0])
        result.append(
            np.array((coord[0] * SCREEN_DIMENSIONS[0], coord[1] * SCREEN_DIMENSIONS[1])))
    # print("Result: ", result)
    return result


print_all = False

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
    filtered = videosTable["status"] == "Processed"
    indexesToProcess = videosTable.index[filtered]
    if print_all:
        print("Filtered: ", len(filtered))
        print("indexesToProcess: ", len(indexesToProcess))
    if (len(indexesToProcess) < 1):
        print("ALL VIDEOS UNPROCESSED!")
    else:
        for indexToProcess in indexesToProcess.tolist():
            selectedVideo = videosTable.loc[indexToProcess]
            if print_all:
                print("Index to update: ", indexToProcess)
                print("Updated Table: ", videosTable)
                print("Selected Video: ", selectedVideo)

inputFilePath = "videodatasettest.csv"
videosDataset = pd.read_csv(inputFilePath, sep=";")

print("Videos Dataset: ", videosDataset.head())

# Carregar dados de planilha
fileName = "processedVideos"
ext = ".csv"
print("Read: " + fileName+ext)
df = pd.read_csv(fileName+ext, sep=';')
print(df.head())
filesNumber = 0
while (True):
    try:
        print("Read: " + fileName+str(filesNumber)+ext)
        inputCsv = pd.read_csv(fileName+str(filesNumber)+ext, sep=';')
        df = df._append(inputCsv, ignore_index=True)
        print(inputCsv.shape, " += ", df.shape)
        filesNumber += 1

    except:
        print("FileNotFoundError")
        break
print(df)
filtered = videosDataset["video_id"]
videoDatasetDict = videosDataset.set_index('video_id').T.to_dict('list')
print("Dict: ", videoDatasetDict)
df = df.loc[df['id'].isin(filtered)]
print("Filtered: \n", df)
TRAIN_PATH = 'dataset'
pathVideos = "videos/"
dataset = {"video_id": [], "frame": [],
           "keypoints": [], "bbox": [], "image_id": [], "image_path": []}
for id in df["id"]:

    folder = file_name = id
    f = open(pathVideos + folder + "/" + file_name+".txt", "r")
    text = f.read().replace('\n', '')
    indices = []
    videoData = json.loads(text, cls=NpDecorder)
    total_frame = videoData[id]["frames_count"]
    print("Total: ", total_frame)
    for frame in videoData[id]["frames"].keys():
        if (videoData[id]["frames"][frame]["YOLO"]["poses_count"] > 0 and
                videoData[id]["frames"][frame]["mediapipe"]["poses_count"] > 0):
            # print("Frame: ", x, " Poses YOLO ", videoData[id]["frames"][x]["YOLO"]["poses_count"],
            #      "Poses MediaPipe", videoData[id]["frames"][x]["mediapipe"]["poses_count"])
            filter_indices = [0, 2, 5, 7, 8, 11, 12,
                              13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
            if (videoData[id]["frames"][frame][videoDatasetDict[id][1]] <= videoDatasetDict[id][2]):
                indices.append(frame)
                dataset["video_id"].append(id)
                dataset["frame"].append(frame)

                mediaPipeKeypoints = np.array(
                    videoData[id]["frames"][frame]["mediapipe"]["keypoints"][0])[filter_indices]
                # print(mediaPipeKeypoints)
                # raise Exception("PAROU!")
                yoloKeypoints = videoData[id]["frames"][frame]["YOLO"]["keypoints"]
                keypointIndex, keypoint = returnMinorDistanceIndexAndKeypoints(
                    mediaPipeKeypoints, yoloKeypoints)
                # print("Keypoint: ", keypoint)
                keypoint = to_pixel_coords(np.array(keypoint))
                # print("Keypoint: ", keypoint)
                dataset["keypoints"].append(keypoint)

                dataset["bbox"].append(get_bbox_from(keypoint))
                dataset["image_id"].append(id+"-"+frame)
                dataset["image_path"].append(
                    TRAIN_PATH+"/images/"+id+"/"+frame+".jpg")
                # print("Index: ", videoDatasetDict[id][1], " Value Expected: ", videoDatasetDict[id]
                #      [2], " Value: ", videoData[id]["frames"][x][videoDatasetDict[id][1]])
                # raise Exception("PAROU!")
    print("Id: ", id, "Count: ", len(indices),
          "dataset:", len(dataset["frame"]))


# Filtrar os dados: selecionar apenas os dados que quiser
# df["num_bbox"] = df['annotations'].apply(lambda frame: str.count(frame, 'x'))
# df_train = df[df["num_bbox"] > 0]

# Convert anotações em listas de bboxes
# df_train['annotations'] = df_train['annotations'].progress_apply(
#    lambda frame: ast.literal_eval(frame))
# df_train['bboxes'] = df_train.annotations.progress_apply(get_bbox)

# Images resolution
# df_train["width"] = 1920
# df_train["height"] = 1080


# Cria uma coluna com lista de caminhos para a images
# dataset = dataset.progress_apply(get_path, axis=1)

# Cria uma coluna para dividir os diferentes
# tipos de dados que vamos agrupar e que vai
# ser usado na divisão entre treino e teste
# kf = GroupKFold(n_splits=5)
# df_train = df_train.reset_index(drop=True)
# df_train['fold'] = -1
# for fold, (train_idx, val_idx) in enumerate(kf.split(df_train, y=df_train.video_id.tolist(), groups=df_train.sequence)):
#    df_train.loc[val_idx, 'fold'] = fold


# cria o diretorio
HOME_DIR = 'dataset'
DATASET_PATH = '/images'
# cria as pastas
try:
    os.mkdir(HOME_DIR)
except:
    print(HOME_DIR, "Já existe!")
try:
    os.mkdir(HOME_DIR+DATASET_PATH)
except:
    print(HOME_DIR+DATASET_PATH, "Já existe!")
try:
    os.mkdir(HOME_DIR+DATASET_PATH+"/train2017")
except:
    print(HOME_DIR+DATASET_PATH+"/train2017", "Já existe!")
try:
    os.mkdir(HOME_DIR+DATASET_PATH+"/val2017")
except:
    print(HOME_DIR+DATASET_PATH+"/val2017", "Já existe!")
try:
    os.mkdir(HOME_DIR+DATASET_PATH+"/annotations")
except:
    print(HOME_DIR+DATASET_PATH+"/annotations", "Já existe!")


# Create Frames by video
# Use OpenCV’s VideoCapture to load the input video.

indexFrame = 0
drawAll = True
saveImages = False
moveImages = False
printAll = False
oldVideoId = None
cap = None

if saveImages:
    for indexDataset in range(len(dataset["video_id"])):
        print("Index: ", indexDataset)
        try:
            os.mkdir(HOME_DIR+DATASET_PATH+"/" +
                     dataset["video_id"][indexDataset])
        except:
            print(HOME_DIR+DATASET_PATH+"/images", "Já existe!")
        if (oldVideoId == None or oldVideoId != dataset["video_id"][indexDataset]):
            indexFrame = 0
            cap = cv2.VideoCapture("videos/" + dataset["video_id"][indexDataset]
                                   + "/" + dataset["video_id"][indexDataset]
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
                print("Save Image in: ", dataset["image_path"][indexDataset])
                print("Box: ", dataset["bbox"][indexDataset])
            if drawAll:
                frame = cv2.rectangle(frame, (int(dataset["bbox"][indexDataset][0]), int(
                    dataset["bbox"][indexDataset][1])), (int(dataset["bbox"][indexDataset][0])+int(dataset["bbox"][indexDataset][2]), int(
                        dataset["bbox"][indexDataset][1])+int(dataset["bbox"][indexDataset][3])), (255, 0, 0), 2)
                frame = drawPoints(
                    dataset["keypoints"][indexDataset], frame)
            status = cv2.imwrite(
                dataset["image_path"][indexDataset], frame)
            if not status:
                print("D'nt Save Image in: ",
                      dataset["image_path"][indexDataset])
        indexFrame += 1
        # if oldVideoId != None and oldVideoId != dataset["video_id"][indexDataset]:
        # raise Exception("PAROU!")
        oldVideoId = dataset["video_id"][indexDataset]

dataset = pd.DataFrame(dataset)
print(dataset)


# Distribut in train/val
total_count = len(dataset["frame"])
val_percentage = 0.1
test_percentage = 0.1
total_train = total_count * (1 - val_percentage - test_percentage)
total_val = total_count * (val_percentage)
total_test = total_count * (val_percentage)
print("Total count: ", total_count)


def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(
        list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets


# datasets = train_val_dataset(dataset, val_percentage)
print("Dataset antes", dataset)

kf = GroupKFold(n_splits=5)
dataset = dataset.reset_index(drop=True)
dataset['fold'] = -1
for fold, (train_idx, val_idx) in enumerate(kf.split(dataset, y=dataset.video_id.tolist(), groups=dataset.frame)):
    dataset.loc[val_idx, 'fold'] = fold

print("Dataset depois", dataset)
# raise Exception("PAROU!")
'''
# print(datasets)
print(len(dataset))
datasets = train_val_dataset(dataset)
print(len(datasets['train']))
print(len(datasets['val']))
# The original dataset is available in the Subset class
print(datasets['train'])
print(datasets['val'])

# dataloaders = {x: DataLoader(
#    datasets[x], 32, shuffle=True, num_workers=4) for x in ['train', 'val']}
# x, y = next(iter(dataloaders['train']))
# print(x.shape, y.shape)
# move images

'''
# seleciona o grupo para dividir para validação

SELECTED_FOLD = 4

if saveImages:

    # Divide entre treino e teste
    for i in range(len(dataset)):
        row = dataset.loc[i]
        if row.fold != SELECTED_FOLD:
            shutil.copyfile(row.image_path, HOME_DIR +
                            DATASET_PATH+"/train2017/"+row.image_id+".jpg")
        else:
            shutil.copyfile(f'{row.image_path}', HOME_DIR +
                            DATASET_PATH+"/val2017/"+row.image_id+".jpg")

    print(
        f'Number of training files: {len(os.listdir(f"{HOME_DIR}{DATASET_PATH}/train2017/"))}')
    print(
        f'Number of validation files: {len(os.listdir(f"{HOME_DIR}{DATASET_PATH}/val2017/"))}')

# salva o arquivo json


def save_annot_json(json_annotation, filename):
    with open(filename, 'w') as f:
        output_json = json.dumps(json_annotation, cls=NpEncoder)
        f.write(output_json)


annotion_id = 0

# Convert COTS dataset to JSON COCO
train_annot_json = dataset2coco(
    dataset[dataset.fold != SELECTED_FOLD], f"{HOME_DIR}{DATASET_PATH}/train2017/")
val_annot_json = dataset2coco(
    dataset[dataset.fold == SELECTED_FOLD], f"{HOME_DIR}{DATASET_PATH}/val2017/")

# Save converted annotations
save_annot_json(train_annot_json,
                f"{HOME_DIR}{DATASET_PATH}/annotations/train.json")
save_annot_json(
    val_annot_json, f"{HOME_DIR}{DATASET_PATH}/annotations/valid.json")
