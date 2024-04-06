# import warnings
# warnings.filterwarnings("ignore")
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from pathlib import Path
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
    '''
        Class To encoder json files 
    '''

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class NpDecorder(json.JSONDecoder):
    '''
        Class To decoder json files 
    '''

    def default(self, obj):
        if isinstance(obj, int):
            return np.integer(obj)
        if isinstance(obj, float):
            return np.floating(obj)
        if isinstance(obj, list):
            return np.ndarray(obj)
        return super(NpDecorder, self)


# https://www.kaggle.com/code/remekkinas/yolox-training-pipeline-cots-dataset-lb-0-507/notebook

class CocoConverter:
    '''
        Class to converter a existent full boddy dataset in Coco keypoints dataset format like. 
    '''

    def __init__(self, year, version, description, contributor, url, date_created, lic_id, lic_url, lic_name):
        self.annotion_id = 1
        self.SCREEN_DIMENSIONS = (1920, 1080)
        self.annotations_json = {
            "info": [],
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }
        self.info = {
            "year": year,
            "version": version,
            "description": description,
            "contributor": contributor,
            "url": url,
            "date_created": date_created
        }
        self.annotations_json["info"] = self.info
        self.lic = {
            "id": lic_id,
            "url": lic_url,
            "name": lic_name
        }
        self.annotations_json["licenses"].append(self.lic)

    def build(self, df):
        '''
        Para cada linha vai ser gerada uma imagem com bbox, segmentação e anotação
        atributos importantes
        image: id, file_name, heifht, width
        segmentation:

        '''
        for video_id in df.keys():

            for frame_id in df[video_id].keys():

                images = {
                    "id": self.annotion_id,
                    "license": 1,
                    "file_name": df[video_id][frame_id]["file_name"],
                    "height": self.SCREEN_DIMENSIONS[1],
                    "width": self.SCREEN_DIMENSIONS[0],
                    "date_captured": self.info["date_created"]
                }

                self.annotations_json["images"].append(images)

                bbox = df[video_id][frame_id]["box"]
                b_width = bbox[2]
                b_height = bbox[3]

                # some boxes in COTS are outside the image height and width
                if (bbox[0] + bbox[2] > self.SCREEN_DIMENSIONS[0]):
                    b_width = bbox[0] - self.SCREEN_DIMENSIONS[0]
                if (bbox[1] + bbox[3] > self.SCREEN_DIMENSIONS[1]):
                    b_height = bbox[1] - self.SCREEN_DIMENSIONS[0]

                image_segmentation = {
                    "segmentation": [],
                    "num_keypoints": 17,
                    "area": int(bbox[2] * bbox[3]),
                    "iscrowd": 0,
                    "keypoints": self.to_int_Visibility(df[video_id][frame_id]["keypoints"]),
                    "image_id": self.annotion_id,
                    "bbox": [bbox[0], bbox[1], b_width, b_height],
                    "category_id": 1,
                    "id": self.annotion_id
                }

                self.annotion_id += 1
                self.annotations_json["annotations"].append(image_segmentation)
            '''
        [{"supercategory": "person","id": 1,"name": "person","keypoints": ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"],"skeleton": [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]}]
        '''
        classes = {"supercategory": "person", "id": 1, "name": "person", "keypoints": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip",
                                                                                       "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"], "skeleton": [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]}

        self.annotations_json["categories"].append(classes)

        print(
            f"Dataset COTS annotation to COCO json format completed! Files: {len(df)}")
        return self.annotations_json

    def to_int_Visibility(self, keypoints):
        keypoints_int_visibility = []
        for keypoint in keypoints:
            keypoints_int_visibility.append(int(keypoint[0]))
            keypoints_int_visibility.append(int(keypoint[1]))
            keypoints_int_visibility.append(2)
        if len(keypoints_int_visibility) != 51:
            raise Exception("Quantidade de Keypoints diferente de 51!")
        return keypoints_int_visibility

    def get_bbox(self, annots):
        bboxes = [list(annot.values()) for annot in annots]
        return bboxes

    def normalize_distance_visible(self, keypoints1, keypoints2):
        sum_distance = 0
        for point1, point2 in zip(keypoints1, keypoints2):
            dist = np.linalg.norm(point1[0:2]-point2[0:2])
            if (len(point1) > 2 and len(point2) > 2):
                dist *= point1[-1]*point2[-1]
            sum_distance += dist
        return sum_distance

    def returnMinorDistanceIndexAndKeypoints(self, keypoint0, keypoints):
        minor_distance_EPDNVP = 100000
        minor_distance_index = -1
        index = 0
        for poseTest in keypoints:
            # print("YOLO:", yolo_pose)

            distance_EPDNVP = self.normalize_distance_visible(
                poseTest, keypoint0)
            if (minor_distance_EPDNVP > distance_EPDNVP):
                minor_distance_EPDNVP = distance_EPDNVP
                minor_distance_index = index
            index += 1

        return minor_distance_index, keypoints[minor_distance_index]

    def get_bbox_from(self, keypoint):
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

    def drawPoints(self, keypoints, img):
        # print("Keypoints: ", keypoints)
        for point in keypoints:
            # print("Keypoint: ", point[0], point[1])
            img = cv2.circle(img, (int(point[0]), int(point[1])), radius=10,
                             color=(0, 0, 255), thickness=-1)
        return img

    def to_pixel_coords(self, relative_coords):
        # print("to_pixel_coords: ", relative_coords)
        result = []
        for coord in relative_coords:
            # print("Dim: ", SCREEN_DIMENSIONS, "Coord: ", coord,
            #      "Result: ", coord[0] * SCREEN_DIMENSIONS[0])
            result.append(
                np.array((coord[0] * self.SCREEN_DIMENSIONS[0], coord[1] * self.SCREEN_DIMENSIONS[1])))
        # print("Result: ", result)
        return result

    @staticmethod
    def save_annot_json(json_annotation, filename):
        with open(filename, 'w') as f:
            output_json = json.dumps(json_annotation, cls=NpEncoder)
            f.write(output_json)

    @staticmethod
    def plot_image_anotation(path, video_id, frame, points):
        from matplotlib import image
        from matplotlib import pyplot as plt
        import random
        import math

        print(frame)

        area = np.array([1920, 1080])
        print(points)
        print(points.shape)
        points = np.delete(points, 2, 1)
        print(points.shape)
        print(points)
        skeleton_ajusted = points*area
        print(skeleton_ajusted)
        print(points)

        # to read the image stored in the working directory
        data = image.imread(f"{path}/{video_id}/img-{frame:06}.jpg")

        # to draw a line from (200,300) to (500,100)
        count = 0
        # print(skeleton_ajusted.shape)
        # for coordinates in skeleton_ajusted:
        #    plt.plot(coordinates[0], coordinates[1],
        #             marker='v', color="yellow")

        for coordinates in points:
            plt.plot(coordinates[0], coordinates[1], marker='.', color="red")
        plt.imshow(data)
        plt.show()


class BraceDataset(Dataset):
    '''Class Brace Definition'''

    def __init__(self, sequences_path, df, sample_length=900, max_length=None):
        assert (sample_length is None) != (
            max_length is None), 'Choose sample length or max length but not both'
        assert len(df) > 0
        self.df = df
        self.sequences = []
        self.clips = []
        self.clip_labels = []
        self.max_length = max_length
        self.clip_label_map = dict(toprock=0, footwork=1, powermove=2)
        self.map_annotations = {}
        clip_paths_by_video = {}
        pose_jsons = list(Path(sequences_path).rglob('**/*.json'))

        for video_id in self.df.video_id.unique():
            video_paths = [
                p for p in pose_jsons if p.stem.startswith(video_id)]
            clip_paths_by_video[video_id] = {}

            for vp in video_paths:
                splits = vp.stem.replace(f'{video_id}_', '').split('_')
                clip_start, clip_end = (int(x) for x in splits[0].split('-'))
                clip_paths_by_video[video_id][(clip_start, clip_end)] = vp
        # print("DF: ", df)
        for seq_t in tqdm(self.df.itertuples(), total=len(self.df), desc='Loading BRACE'):
            # print("SEQ_T: ", seq_t)
            seq_clips = self.get_seq_clips(clip_paths_by_video, seq_t)
            # print("seq_clips: ", seq_clips)
            assert len(
                seq_clips) > 0, f'Did not find any segments for sequence {seq_t}'
            clips = []

            for p in seq_clips:
                clip, _, _ = self.load_clip(p)
                clips.append(clip)
                clip_label = None

                for cat in ('toprock', 'footwork', 'powermove'):
                    if cat in p.name:
                        assert clip_label is None, f'Trying to override clip labels for clip {p}'
                        clip_label = self.clip_label_map[cat]
                        break

                assert clip_label is not None
                self.clip_labels.append(clip_label)

            self.clips.extend(clips)
            seq = np.concatenate(clips, axis=0)
            self.sequences.append(seq)

        assert len(self.df) == len(self.sequences)
        self.clips = np.array(self.clips, dtype=object)
        self.clip_labels = np.array(self.clip_labels)

        if max_length is not None:
            max_seq_length = np.max([x.shape[0] for x in self.sequences])
            assert max_seq_length <= max_length, f'Found a sequence whose length {max_seq_length} is > than the max ' \
                                                 f'sequence allowed {max_length}. Adjust accordingly'
        else:
            avg_seq_length = np.mean([x.shape[0] for x in self.sequences])
            print(f'Going to sample {sample_length} frames from each sequence. '
                  f'Average sequence length is {avg_seq_length}')

        self.n_dancers = len(self.df.dancer_id.unique())
        self.sample_length = sample_length

    @staticmethod
    def get_seq_clips(clip_paths_by_video, seq_t):
        return [p for (start, end), p in sorted(clip_paths_by_video[seq_t.video_id].items(),
                                                key=lambda kv: kv[0][0])
                if seq_t.start_frame <= start <= seq_t.end_frame and seq_t.start_frame <= end <= seq_t.end_frame]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        seq_row = self.df.iloc[index]
        seq = np.array(self.sequences[index])
        metadata = seq_row.to_dict()
        """
        if self.sample_length is None:
            missing = self.max_length - seq.shape[0]

            if missing > 0:
                zeros = np.zeros((missing, *seq[0].shape))
                seq = np.concatenate((seq, zeros), axis=0)
        else:
            idx = np.linspace(
                0, seq.shape[0] - 1, self.sample_length, dtype=int)
            seq = seq[idx, ...]
        """
        return seq, metadata

    def load_clip(self, pose_path, img_ext='.png', broken_policy='skip_frame', policy_warning=False):
        with open(pose_path) as f:
            d = json.load(f)

        frame_ids = sorted(d.keys())
        video_id = frame_ids[0].split('/')[0]
        if (video_id not in self.map_annotations):
            self.map_annotations[video_id] = {}
        frame_numbers = [BraceDataset.get_frame_number_from_id(
            video_id, f_id, img_ext=img_ext) for f_id in frame_ids]
        clip = []

        for i, (frame_id, frame_number) in enumerate(zip(frame_ids, frame_numbers)):
            keypoints = np.array(d[frame_id]['keypoints'])
            try:
                # print(frame_id)
                if (frame_number not in self.map_annotations[video_id]):
                    self.map_annotations[video_id][frame_number] = {}
                self.map_annotations[video_id][frame_number]["file_name"] = frame_id
                self.map_annotations[video_id][frame_number]["keypoints"] = keypoints
                self.map_annotations[video_id][frame_number]["box"] = np.array(
                    d[frame_id]['box'])
                box = BraceDataset.get_box_from_keypoints(
                    keypoints, box_border=0)
                self.map_annotations[video_id][frame_number]["new_box"] = box
                norm_kpt = BraceDataset.normalise_keypoints(box, keypoints)
                self.map_annotations[video_id][frame_number]["normalized_keypoints"] = norm_kpt
            except AssertionError as e:
                if broken_policy == 'skip_frame':

                    if policy_warning:
                        print(
                            f'Got broken keypoints at frame {frame_id}. Skipping as per broken policy')
                    continue
                else:
                    raise e
            clip.append(norm_kpt)

        clip = np.stack(clip, axis=0)
        clip_id = pose_path.stem

        return clip, clip_id, video_id

    @staticmethod
    def get_frame_number_from_id(video_id, frame_id, img_ext='.png'):
        n = frame_id.replace(
            f'{video_id}/', '').replace('img-', '').replace(img_ext, '')

        try:
            return int(n)
        except ValueError:
            raise ValueError(f'Invalid frame id: {frame_id}')

    @staticmethod
    def get_box_from_keypoints(keypoints, box_border=100):
        box_x1, box_y1 = keypoints[:, :2].min(axis=0)
        box_x2, box_y2 = keypoints[:, :2].max(axis=0)
        box_x1 -= box_border
        box_y1 -= box_border
        box_x2 += box_border
        box_y2 += box_border
        w = box_x2 - box_x1
        h = box_y2 - box_y1
        assert w > 0 and h > 0, f'Invalid box: {box_x1}, {box_x2}, {box_y1}, {box_y2}'
        box = (box_x1, box_y1, w, h, 1)  # confidence score 1
        return box

    @staticmethod
    def normalise_keypoints(box, keypoints):
        x, y, w, h, _ = box
        xk = (keypoints[:, 0] - x) / w
        yk = (keypoints[:, 1] - y) / h
        nk = np.stack((xk, yk), axis=1)  # no need to stack the scores
        return nk

    @staticmethod
    def filter_set(sequences_df, set_df):

        pass


if __name__ == '__main__':
    import datetime

    '''
    Main used to adapter a dataset to cocodataset
    '''
    # Adapter to converter Brace Dataset in CocoDataset
    # STEPS:
    # 1 - Load Brace file dataset
# Ajust Brace Class to load Image Information
# path where you download and unzipped the keypoints
    sequences_path_ = Path(
        '/Users/jaderxnet/Documents/datasetBraceKeypoints')
    df_ = pd.read_csv(
        Path('/Users/jaderxnet/Documents/GitHub/brace/annotations/sequences.csv'))
    """
    train_df = pd.read_csv(
        '/Users/jaderxnet/Documents/GitHub/brace/annotations/sequences_train.csv')
    train_df = df_[df_.uid.isin(train_df.uid)]
    
    brace_train = BraceDataset(sequences_path_, train_df)
    # print(brace_train.map_annotations)
    print(
        f'Loaded BRACE training set! We got {len(brace_train)} training sequences')
    skeletons_train, metadata_train = brace_train.__getitem__(0)
    print(metadata_train)
    print(skeletons_train.shape)
    print(skeletons_train[442])
    """
    train_df = pd.read_csv(
        '/Users/jaderxnet/Documents/GitHub/brace/annotations/sequences_train.csv')
    train_df = df_[df_.uid.isin(train_df.uid)]

    brace_train = BraceDataset(sequences_path_, train_df)
    print(
        f'Loaded BRACE test set! We got {len(brace_train)} testing sequences')
    # skeletons_test, metadata_test = brace_train.__getitem__(0)
    # print(metadata_test)
    # print(skeletons_test.shape)
    # print(skeletons_test[442])
    # print(test_df)
    # print(brace_test.map_annotations["QwlEcwZiPaE"].keys())
    # 2 - Inflate one dataframe from each item from Brace
    """id = 6388
    for key in brace_train.map_annotations["J81VYjZvg80"][id].keys():
        print(key)
        print(brace_train.map_annotations["J81VYjZvg80"]
              [id][key])
    CocoConverter.plot_image_anotation("/Users/jaderxnet/Documents/GitHub/datasetBrace/images", "J81VYjZvg80",
                                       id, brace_train.map_annotations["J81VYjZvg80"][id]["keypoints"])
    #"""

    today = datetime.datetime.now()
    coco_train_from_brace = CocoConverter(
        "2024", "Coco Train From Brace Dataset", "Brace", today.strftime("%x"), "Train", "Train", "Train", "Train", "Train")
    coco_json_file = coco_train_from_brace.build(brace_train.map_annotations)
    # print(coco_json_file)
    # coco_train_from_brace.build(df)
    # 3 - Save CocoDataset file
    HOME_DIR = "datasetCocoBrace"
    try:
        os.mkdir("datasetCocoBrace/")
    except:
        print("datasetCocoBrace/", "Já existe!")
    try:
        os.mkdir("datasetCocoBrace/images/")
    except:
        print("datasetCocoBrace/images/", "Já existe!")
    try:
        os.mkdir("datasetCocoBrace/images/annotations/")
    except:
        print("datasetCocoBrace/images/annotations/", "Já existe!")
    DATASET_PATH = '/images'

    CocoConverter.save_annot_json(coco_json_file,
                                  f"{HOME_DIR}{DATASET_PATH}/annotations/train2017.json")
    # 4 - Test CocoDataset file
