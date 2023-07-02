#import warnings
#warnings.filterwarnings("ignore")

import ast
import os
import json
import pandas as pd
from sklearn.model_selection import GroupKFold

TRAIN_PATH = 'dataset'

# https://www.kaggle.com/code/remekkinas/yolox-training-pipeline-cots-dataset-lb-0-507/notebook
def dataset2coco(df, dest_path):

    global annotion_id

    annotations_json = {
        "info": [],
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": []
    }

    info = {
        "year": "2023",
        "version": "1",
        "description": "GERAD dataset - COCO format",
        "contributor": "Jader Abreu",
        "url": "https://github.com/jaderxnet/GERAD",
        "date_created": "2022-07-01T17:15:26-03:00"
    }
    annotations_json["info"].append(info)

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
            "height": ann_row.height,
            "width": ann_row.width,
            "date_captured": "2021-11-30T15:01:26+00:00"
        }

        annotations_json["images"].append(images)

        bbox_list = ann_row.bboxes

        for bbox in bbox_list:
            b_width = bbox[2]
            b_height = bbox[3]

            # some boxes in COTS are outside the image height and width
            if (bbox[0] + bbox[2] > 1280):
                b_width = bbox[0] - 1280
            if (bbox[1] + bbox[3] > 720):
                b_height = bbox[1] - 720

            image_segmentation = {
                "segmentation": [],
                "num_keypoints": 0,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "keypoints": [],
                "image_id": ann_row[0],
                "bbox": [bbox[0], bbox[1], b_width, b_height],
                "category_id": 0,
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


def get_path(row):
    row['image_path'] = f'{TRAIN_PATH}/train_images/video_{row.video_id}/{row.video_frame}.jpg'
    return row


# Carregar dados de planilha
df = pd.read_csv("/kaggle/input/tensorflow-great-barrier-reef/train.csv")
df.head(5)

# Filtrar os dados: selecionar apenas os dados que quiser
df["num_bbox"] = df['annotations'].apply(lambda x: str.count(x, 'x'))
df_train = df[df["num_bbox"] > 0]

# Convert anotações em listas de bboxes
df_train['annotations'] = df_train['annotations'].progress_apply(
    lambda x: ast.literal_eval(x))
df_train['bboxes'] = df_train.annotations.progress_apply(get_bbox)

# Images resolution
df_train["width"] = 1280
df_train["height"] = 720

# Cria uma coluna com lista de caminhos para a images
df_train = df_train.progress_apply(get_path, axis=1)

#Cria uma coluna para dividir os diferentes 
# tipos de dados que vamos agrupar e que vai
# ser usado na divisão entre treino e teste
kf = GroupKFold(n_splits=5)
df_train = df_train.reset_index(drop=True)
df_train['fold'] = -1
for fold, (train_idx, val_idx) in enumerate(kf.split(df_train, y=df_train.video_id.tolist(), groups=df_train.sequence)):
    df_train.loc[val_idx, 'fold'] = fold

df_train.head(5)

#cria o diretorio
HOME_DIR = 'dataset' 
DATASET_PATH = 'images'

#cria as pastas
os.mkdir( HOME_DIR+"/dataset")
os.mkdir( HOME_DIR+DATASET_PATH)
os.mkdir( HOME_DIR+DATASET_PATH+"/train2017")
os.mkdir( HOME_DIR+DATASET_PATH+"/val2017")
os.mkdir( HOME_DIR+DATASET_PATH+"/annotations")

#seleciona o grupo para dividir para validação
SELECTED_FOLD = 4

#Divide entre treino e teste
for i in tqdm(range(len(df_train))):
    row = df_train.loc[i]
    if row.fold != SELECTED_FOLD:
        copyfile(f'{row.image_path}', f'{HOME_DIR}{DATASET_PATH}/train2017/{row.image_id}.jpg')
    else:
        copyfile(f'{row.image_path}', f'{HOME_DIR}{DATASET_PATH}/val2017/{row.image_id}.jpg') 

print(f'Number of training files: {len(os.listdir(f"{HOME_DIR}{DATASET_PATH}/train2017/"))}')
print(f'Number of validation files: {len(os.listdir(f"{HOME_DIR}{DATASET_PATH}/val2017/"))}')

#salva o arquivo json
def save_annot_json(json_annotation, filename):
    with open(filename, 'w') as f:
        output_json = json.dumps(json_annotation)
        f.write(output_json)

annotion_id = 0

# Convert COTS dataset to JSON COCO
train_annot_json = dataset2coco(df_train[df_train.fold != SELECTED_FOLD], f"{HOME_DIR}{DATASET_PATH}/train2017/")
val_annot_json = dataset2coco(df_train[df_train.fold == SELECTED_FOLD], f"{HOME_DIR}{DATASET_PATH}/val2017/")

# Save converted annotations
save_annot_json(train_annot_json, f"{HOME_DIR}{DATASET_PATH}/annotations/train.json")
save_annot_json(val_annot_json, f"{HOME_DIR}{DATASET_PATH}/annotations/valid.json")