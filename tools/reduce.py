import shutil
import os
import json
from pathlib import Path
from tqdm import tqdm


class DatasetReduce:
    '''
        Class to reduce size of Coco like dataset removing no essentials files. 
    '''

    def __init__(self, imagesPath, sequences_path, imagesReducedPath):
        self.imagesPath = imagesPath
        self.imagesReducedPath = imagesReducedPath
        self.files = []
        self.filesImages = []
        self.addDataset(sequences_path)

    def addDataset(self, sequences_path):
        video_paths = list(Path(sequences_path).rglob('*.json'))

        for video_path in video_paths:
            print(video_path)
            with open(video_path) as f:
                map = json.load(f)
            self.files.append(map)
            self.filesImages.extend(map["images"])
        print(self.files[0].keys())
        print(len(self.filesImages[0]))

    def printImageQuant(self):
        print(" Images Quant: ", len(self.filesImages))

    def createFolters(self, filePath):
        if not os.path.isdir(filePath):
            os.makedirs(filePath)
        # try:
        #    os.mkdir(self.imagesReducedPath)
        # except:
        #    print(self.imagesReducedPath, "Já existe!")

    def createDatasetReduced(self):
        count = 0
        for imageMap in tqdm(self.filesImages):
            # print(" Name: ", imageMap["file_name"])
            completFileNameOrigin = f'{self.imagesPath}/{imageMap["file_name"]}'
            completFileNameDestination = f'{self.imagesReducedPath}/{imageMap["file_name"]}'
            finalPath = os.path.join(
                *completFileNameDestination.split("/")[:-1])
            self.createFolters(finalPath)
            # print("Origin: ", completFileNameOrigin,
            #      " Destination: ", completFileNameDestination)
            shutil.copyfile(completFileNameOrigin, completFileNameDestination)
            # if (count > 10000):
            #    break
            # else:
            #    count += 1


if __name__ == '__main__':
    datasetR = DatasetReduce("../datasetBrace/datasetBraceImages",
                             "datasetCocoBrace/images/annotations",
                             "../datasetBrace/datasetBraceReduced")
    datasetR.printImageQuant()
    datasetR.createDatasetReduced()
