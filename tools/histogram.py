from logger import Logger
from singleton import VideoProcessorSingleton
from coder import NpEncoder
from metrics import MetricTipe
from matplotlib import colors
from enum import Enum
import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class HistogramProcess:
    def __init__(self, inputFilePath, outPutFolder, printOption=True) -> None:
        self.inputFilePath = inputFilePath
        self.outPutFolder = outPutFolder
        self.printOption = printOption
        self.logger = Logger(printOption=printOption)
        logging.basicConfig(filename=f'{outPutFolder}/histogram.log', filemode='w',
                            format='%(name)s - %(levelname)s - %(message)s')
        logging.warning('This will get logged to a file')

    def log(self, *string, end=""):
        logging.warning(string)
        return self.logger.print(string, end)

    def logError(self, *string):
        logging.error(string, exc_info=True)
        return self.logger.printError(string)

    def readInput(self):

        self.videosTable = pd.read_csv(self.inputFilePath, sep=';')
        self.logger.print("Lendo: ", self.inputFilePath)
        self.logger.print("Shape: ", self.videosTable.shape)

    def getIndexByStatus(self, status):
        singleton = VideoProcessorSingleton(self.videosTable["status"])
        indexesStatus = singleton.getIndexBy(status)
        self.logger.print("Total Size: ", len(self.videosTable))
        self.logger.print("Filtered " + status + ": ", len(indexesStatus))
        return indexesStatus

    def changeStatus(self, index, new_status):
        self.videosTable.at[index, 'status'] = new_status

    def updateCSVInput(self):
        self.videosTable.to_csv(self.inputFilePath, index=False, sep=';')

    def saveOutputFile(self, dictionary, id, text):
        # os.mkdir(outputFilePath)
        outputFilePath = self.outPutFolder + "/" + id + "/" + id+".txt"
        file = open(outputFilePath, "w")
        file.write(json.dumps(dictionary, indent=4, cls=NpEncoder))
        file.close()

    def getItemByIndex(self, index):
        return self.videosTable.loc[index]

    def plot_grph(self, valores, new_map, show, save, file_name, label):
        valuesScala = valores / float(max(valores))
        # indi = 0

        # my_cmap = plt.get_cmap("viridis")

        # print('Max: ' + str(max(valores)))

        cols = new_map(valuesScala)
        plot = plt.scatter(valores, valores, c=valores, cmap=new_map)
        plt.clf()
        plt.colorbar(plot)
        plt.xlabel('frames/s')
        plt.ylabel(label)
        plt.title(file_name)
        plt.bar(range(len(valores)), valores, color=cols)
        if (save):
            plt.savefig(self.outPutFolder + "/" + folder + "/" + label +
                        "_" + id+'.png', dpi=200)
        if (show):
            plt.show()


class Histogram:
    def __init__(self, idVideo, histogramTipe) -> None:
        self.idVideo = idVideo
        self.histogramTipe = histogramTipe
        self.frames = []
        self.valueList = []
        self.threshold = []

    def valuesNPArray(self):
        # if (self.histogramTipe == MetricTipe.EPE):
        #    print("AQUI: ", self.idVideo, self.histogramTipe, self.valueList)
        self.valueList = np.array(self.valueList)


class ColorDictType(Enum):
    DOWN = 1
    CENTER = 2


class ColorMap:
    def __init__(self, colorDictionaryType) -> None:
        self.colorDictionaryType = colorDictionaryType

    @staticmethod
    def scale(x):
        return np.interp(x=x, xp=[0, 255], fp=[0, 1])

    def getMap(self):
        if (self.colorDictionaryType == ColorDictType.DOWN):
            cdict = {
                'red': ((0.0, ColorMap.scale(0), ColorMap.scale(0)),
                        (1/10*1, ColorMap.scale(0), ColorMap.scale(0)),
                        (1/10*2, ColorMap.scale(210), ColorMap.scale(210)),
                        (1/10*3, ColorMap.scale(210), ColorMap.scale(210)),
                        (1/10*4, ColorMap.scale(210), ColorMap.scale(210)),
                        (1/10*5, ColorMap.scale(210), ColorMap.scale(210)),
                        (1/10*6, ColorMap.scale(210), ColorMap.scale(210)),
                        (1/10*7, ColorMap.scale(210), ColorMap.scale(210)),
                        (1/10*8, ColorMap.scale(210), ColorMap.scale(210)),
                        (1/10*9, ColorMap.scale(210), ColorMap.scale(210)),
                        (1.0, ColorMap.scale(210), ColorMap.scale(210))),
                'green': ((0.0, ColorMap.scale(220), ColorMap.scale(220)),
                          (1/10*1, ColorMap.scale(0), ColorMap.scale(0)),
                          (1/10*2, ColorMap.scale(0), ColorMap.scale(0)),
                          (1/10*3, ColorMap.scale(0), ColorMap.scale(0)),
                          (1/10*4, ColorMap.scale(0), ColorMap.scale(0)),
                          (1/10*5, ColorMap.scale(0), ColorMap.scale(0)),
                          (1/10*6, ColorMap.scale(0), ColorMap.scale(0)),
                          (1/10*7, ColorMap.scale(0), ColorMap.scale(0)),
                          (1/10*8, ColorMap.scale(0), ColorMap.scale(0)),
                          (1/10*9, ColorMap.scale(0), ColorMap.scale(0)),
                          (1.0, ColorMap.scale(0), ColorMap.scale(0))),
                'blue': ((0.0, ColorMap.scale(0), ColorMap.scale(0)),
                         (1/10*1, ColorMap.scale(255), ColorMap.scale(255)),
                         (1/10*2, ColorMap.scale(0), ColorMap.scale(0)),
                         (1/10*3, ColorMap.scale(0), ColorMap.scale(0)),
                         (1/10*4, ColorMap.scale(0), ColorMap.scale(0)),
                         (1/10*5, ColorMap.scale(0), ColorMap.scale(0)),
                         (1/10*6, ColorMap.scale(0), ColorMap.scale(0)),
                         (1/10*7, ColorMap.scale(0), ColorMap.scale(0)),
                         (1/10*8, ColorMap.scale(0), ColorMap.scale(0)),
                         (1/10*9, ColorMap.scale(0), ColorMap.scale(0)),
                         (1.0, ColorMap.scale(0), ColorMap.scale(0))),
            }
            new_cmap = colors.LinearSegmentedColormap(
                'new_cmap', segmentdata=cdict)
            return new_cmap
        if (self.colorDictionaryType == ColorDictType.CENTER):
            cdict2 = {
                'red': ((0.0, ColorMap.scale(210), ColorMap.scale(210)),
                        (1/10*1, ColorMap.scale(210), ColorMap.scale(210)),
                        (1/10*2, ColorMap.scale(210), ColorMap.scale(210)),
                        (1/10*3, ColorMap.scale(210), ColorMap.scale(210)),
                        (1/10*4, ColorMap.scale(210), ColorMap.scale(210)),
                        (1/10*5, ColorMap.scale(210), ColorMap.scale(210)),
                        (1/10*6, ColorMap.scale(0), ColorMap.scale(0)),
                        (1/10*7, ColorMap.scale(0), ColorMap.scale(0)),
                        (1/10*8, ColorMap.scale(0), ColorMap.scale(0)),
                        (1/10*9, ColorMap.scale(0), ColorMap.scale(0)),
                        (1.0, ColorMap.scale(0), ColorMap.scale(0))),
                'green': ((0.0, ColorMap.scale(0), ColorMap.scale(0)),
                          (1/10*1, ColorMap.scale(0), ColorMap.scale(0)),
                          (1/10*2, ColorMap.scale(0), ColorMap.scale(0)),
                          (1/10*3, ColorMap.scale(0), ColorMap.scale(0)),
                          (1/10*4, ColorMap.scale(0), ColorMap.scale(0)),
                          (1/10*5, ColorMap.scale(0), ColorMap.scale(0)),
                          (1/10*6, ColorMap.scale(220), ColorMap.scale(220)),
                          (1/10*7, ColorMap.scale(220), ColorMap.scale(220)),
                          (1/10*8, ColorMap.scale(220), ColorMap.scale(220)),
                          (1/10*9, ColorMap.scale(220), ColorMap.scale(220)),
                          (1.0, ColorMap.scale(220), ColorMap.scale(0))),
                'blue': ((0.0, ColorMap.scale(0), ColorMap.scale(0)),
                         (1/10*1, ColorMap.scale(0), ColorMap.scale(0)),
                         (1/10*2, ColorMap.scale(0), ColorMap.scale(0)),
                         (1/10*3, ColorMap.scale(0), ColorMap.scale(0)),
                         (1/10*4, ColorMap.scale(0), ColorMap.scale(0)),
                         (1/10*5, ColorMap.scale(0), ColorMap.scale(0)),
                         (1/10*6, ColorMap.scale(0), ColorMap.scale(0)),
                         (1/10*7, ColorMap.scale(0), ColorMap.scale(0)),
                         (1/10*8, ColorMap.scale(0), ColorMap.scale(0)),
                         (1/10*9, ColorMap.scale(0), ColorMap.scale(0)),
                         (1.0, ColorMap.scale(0), ColorMap.scale(0))),
            }
            new_cmap2 = colors.LinearSegmentedColormap(
                'new_cmap', segmentdata=cdict2)
            return new_cmap2


if __name__ == '__main__':
    inputFilePath = "ListsInfo/processedFrevo.csv"
    videosFolder = "videoTest"
    histogramProcess = HistogramProcess(
        inputFilePath, videosFolder, True)
    histogramProcess.readInput()
    idexexStatusProcessing = histogramProcess.getIndexByStatus("Processing")
    idexexStatusProcessed = histogramProcess.getIndexByStatus("Processed")
    if (len(idexexStatusProcessing) > 0):
        print("PROCESSING THE INDEX VIDEO: ",
              idexexStatusProcessing.tolist()[0])
    elif (len(idexexStatusProcessed) <= 0):
        print("ALL VIDEOS PROCESSED!")
    else:
        for indexProcessed in idexexStatusProcessed.tolist():
            selectedVideo = histogramProcess.getItemByIndex(indexProcessed)
            histogramProcess.log("Index to update: ", indexProcessed)
            histogramProcess.log("Updated Table: ", histogramProcess)
            histogramProcess.log("Selected Video: ", selectedVideo)

            folder = file_name = id = selectedVideo['id']
            outputFile = open(histogramProcess.outPutFolder + "/" +
                              folder + "/" + file_name+".txt", "r")
            text = outputFile.read().replace('\n', '')

            y = json.loads(text)
            metrics = {}
            metrics["EPDNVP"] = Histogram(id, MetricTipe.EPDNVP)
            metrics["EPDNMVP"] = Histogram(id, MetricTipe.EPDNMVP)
            metrics["EPDNM"] = Histogram(id, MetricTipe.EPDNM)
            metrics["EPE"] = Histogram(id, MetricTipe.EPE)
            metrics["EPDNVP Discord"] = Histogram(id, MetricTipe.EPDNVP)
            # the result is a Python dictionary:
            # print(y[file_name]["EPDNVP"])
            total_frame = y[id]["frames_count"]
            print("Total: ", id)
            for x in y[id]["frames"].keys():
                for metric in metrics.values():
                    metric.frames.append(x)
                if (y[id]["frames"][x]["YOLO"]["poses_count"] <= 0 and
                        y[id]["frames"][x]["mediapipe"]["poses_count"] <= 0):
                    metrics["EPDNVP Discord"].valueList.append(0)
                else:
                    if (y[id]["frames"][x]["YOLO"]["poses_count"] == 0 or
                            y[id]["frames"][x]["mediapipe"]["poses_count"] == 0):
                        metrics["EPDNVP Discord"].valueList.append(5)
                    else:
                        if ("minor_distance" in y[id]["frames"][x]["EPDNVP"]):
                            if (y[id]["frames"][x]["EPDNVP"]["minor_distance"] >= 1):
                                metrics["EPDNVP Discord"].valueList.append(
                                    y[id]["frames"][x]["EPDNVP"]["minor_distance"])
                        elif (y[id]["frames"][x]["EPDNVP"] >= 1):
                            metrics["EPDNVP Discord"].valueList.append(
                                y[id]["frames"][x]["EPDNVP"])
                        else:
                            metrics["EPDNVP Discord"].valueList.append(0)
                if ("EPDNVP" in y[id]["frames"][x]):
                    # print(y[id]["frames"][x]["EPDNVP"])
                    if ("minor_distance" in y[id]["frames"][x]["EPDNVP"]):
                        metrics["EPDNVP"].valueList.append(
                            y[id]["frames"][x]["EPDNVP"]["minor_distance"])
                    else:
                        metrics["EPDNVP"].valueList.append(
                            y[id]["frames"][x]["EPDNVP"])
                else:
                    metrics["EPDNVP"].valueList.append(0)
                if ("EPDNMVP" in y[id]["frames"][x]):
                    # print(y[id]["frames"][x]["EPDNVP"])
                    if ("minor_distance" in y[id]["frames"][x]["EPDNMVP"]):
                        metrics["EPDNMVP"].valueList.append(
                            y[id]["frames"][x]["EPDNMVP"]["minor_distance"])
                    else:
                        metrics["EPDNMVP"].valueList.append(
                            y[id]["frames"][x]["EPDNMVP"])
                else:
                    metrics["EPDNMVP"].valueList.append(0)
                if ("EPDNM" in y[id]["frames"][x]):
                    # print(y[id]["frames"][x]["EPDNM"])
                    if ("minor_distance" in y[id]["frames"][x]["EPDNM"]):
                        metrics["EPDNM"].valueList.append(
                            y[id]["frames"][x]["EPDNM"]["minor_distance"])
                    else:
                        metrics["EPDNM"].valueList.append(
                            y[id]["frames"][x]["EPDNM"])
                else:
                    metrics["EPDNM"].valueList.append(0)
                if ("EPE" in y[id]["frames"][x]):
                    # print(y[id]["frames"][x]["EPDNM"])
                    if ("minor_distance" in y[id]["frames"][x]["EPE"]):
                        metrics["EPE"].valueList.append(
                            y[id]["frames"][x]["EPE"]["minor_distance"])
                    else:
                        metrics["EPE"].valueList.append(
                            y[id]["frames"][x]["EPE"])
                else:
                    metrics["EPE"].valueList.append(0)
            '''
            indi = 0
            for z in valores:
                if indi > 6048 and indi < 6183:
                    print("V ", indi, z)
                indi += 1
            '''
            metrics["EPDNVP"].valuesNPArray()
            metrics["EPDNMVP"].valuesNPArray()
            metrics["EPDNM"].valuesNPArray()
            metrics["EPE"].valuesNPArray()
            metrics["EPDNVP Discord"].valuesNPArray()
            '''
            indi = 0
            for z in valores:
                if indi > 6048 and indi < 6183:
                    print("N ", indi, z)
                indi += 1
            '''

            valores_Similar = []
            for x in metrics["EPDNVP"].valueList:
                if x <= 0 or x > 1:
                    valores_Similar.append(0)
                else:
                    valores_Similar.append(1-x)

            colorMap1 = ColorMap(ColorDictType.DOWN)
            colorMap2 = ColorMap(ColorDictType.CENTER)

            # True, False, True
            histogramProcess.plot_grph(metrics["EPDNVP"].valueList, colorMap1.getMap(), False,
                                       True, file_name, 'EPDNVP')
            histogramProcess.plot_grph(np.array(valores_Similar), colorMap2.getMap(),
                                       False, True, file_name, 'EPDNVP Similarity')
            histogramProcess.plot_grph(metrics["EPDNVP Discord"].valueList, colorMap1.getMap(),
                                       False, True, file_name, 'EPDNVP Discordance')
            histogramProcess.plot_grph(metrics["EPDNMVP"].valueList, colorMap1.getMap(), False,
                                       True, file_name, 'EPDNMVP')
            histogramProcess.plot_grph(metrics["EPDNM"].valueList, colorMap1.getMap(),
                                       False, True, file_name, 'EPDNM')
            histogramProcess.plot_grph(metrics["EPE"].valueList, colorMap1.getMap(),
                                       False, True, file_name, 'EPE')
