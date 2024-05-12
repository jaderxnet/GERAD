from logger import Logger
from singleton import VideoProcessorSingleton
from coder import NpEncoder
from metrics import MetricTipe
from histogram import Histogram, ThresholdType
from matplotlib import colors
from enum import Enum
import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime


class filterProcess:
    def __init__(self, inputFilePath, outPutFolder, printOption=True) -> None:
        self.inputFilePath = inputFilePath
        self.outPutFolder = outPutFolder
        self.printOption = printOption
        self.videosTable = "NAN"
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
            plt.savefig(self.outPutFolder + "/" + id + "/" + label +
                        "_" + id+'.png', dpi=200)
        if (show):
            plt.show()

    def plot_grph_DataFrame(self, valoresTotais, valoresThreshold, new_map, show, save, file_name, label):
        valuesScala = valoresTotais / float(max(valoresTotais))
        # indi = 0

        # my_cmap = plt.get_cmap("viridis")

        # print('Max: ' + str(max(valores)))

        cols = new_map(valuesScala)
        plot = plt.scatter(
            valoresTotais, valoresTotais, c=valoresTotais, cmap=new_map)
        plt.clf()
        plt.colorbar(plot)
        plt.xlabel('frames/s')
        plt.ylabel(label)
        plt.title(file_name)
        framesX = [int(numeric_string)
                   for numeric_string in valoresThreshold["Frame"]]

        # print("Frames X: ", framesX)
        plt.bar(framesX, valoresThreshold["Value"], color=cols)

        if (save):
            plt.savefig(self.outPutFolder + "/" + id + "/" + label +
                        "_" + id+'.png', dpi=200)
        if (show):
            plt.show()

    def saveHistogramDataInCSV(self, histogram):
        if (type(histogram.dataFrame) is pd.DataFrame):
            histogram.dataFrame.to_csv(
                f'{self.outPutFolder}/{histogram.idVideo}/{histogram.idVideo}-{histogram.histogramTipe}.csv', sep='\t', encoding='utf-8')
            histogram.thresholdValues.to_csv(
                f'{self.outPutFolder}/{histogram.idVideo}/{histogram.idVideo}-{histogram.histogramTipe}-Threshold{histogram.threshold}.csv', sep='\t', encoding='utf-8')


if __name__ == '__main__':

    todayformated = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    print(todayformated)
    outputFilterdList = f'ListsInfo/videofrevodataset2{todayformated}.csv'
    inputFilePath = "ListsInfo/processedFrevo.csv"
    videosFolder = "videoTest2"

    filterProcess = filterProcess(
        inputFilePath, videosFolder, True)
    filterProcess.readInput()

    filteredList = pd.DataFrame()
    filteredList.index = filterProcess.videosTable.index
    # Assign the columns.
    filteredList[['video_id', 'hashtags']
                 ] = filterProcess.videosTable[['id', 'hashtags']]
    print(filteredList)
    idexexStatusProcessing = filterProcess.getIndexByStatus("Processing")
    idexexStatusProcessed = filterProcess.getIndexByStatus("Processed")
    if (len(idexexStatusProcessing) > 0):
        print("PROCESSING THE INDEX VIDEO: ",
              idexexStatusProcessing.tolist()[0])
    elif (len(idexexStatusProcessed) <= 0):
        print("ALL VIDEOS PROCESSED!")
    else:
        metrics = {}
        metrics["EPDNVP"] = Histogram(id, MetricTipe.EPDNVP)
        metrics["EPDNMVP"] = Histogram(id, MetricTipe.EPDNMVP)
        metrics["EPDNM"] = Histogram(id, MetricTipe.EPDNM)
        metrics["EPE"] = Histogram(id, MetricTipe.EPE)
        metrics["EPDNVP"].thresholdTipe = ThresholdType.MINOR_EQUAL
        metrics["EPDNVP"].threshold = 0.2
        metrics["EPDNMVP"].thresholdTipe = ThresholdType.MINOR_EQUAL
        metrics["EPDNMVP"].threshold = 0.05
        metrics["EPDNM"].thresholdTipe = ThresholdType.MINOR_EQUAL
        metrics["EPDNM"].threshold = 0.1
        metrics["EPE"].thresholdTipe = ThresholdType.MINOR_EQUAL
        metrics["EPE"].threshold = 200
        # select Metric File
        # metric;threshold;network;filteredAnnotation
        othesColumns = {'metric': [], 'threshold': [],
                        'network': [], 'filteredAnnotation': [],
                        'quantFilteredFrames': [], 'width': [], 'height': []}
        selectedMetric = metrics["EPDNVP"]

        for indexProcessed in idexexStatusProcessed.tolist():
            selectedVideo = filterProcess.getItemByIndex(indexProcessed)
            filterProcess.log("Index to update: ", indexProcessed)
            filterProcess.log("Updated Table: ", filterProcess)
            filterProcess.log("Selected Video: ", selectedVideo)

            folder = file_name = id = selectedVideo['id']
            print(indexProcessed, "Filter: ", id)
            outputFile = open(filterProcess.outPutFolder + "/" +
                              folder + "/" + file_name+".txt", "r")
            text = outputFile.read().replace('\n', '')

            framesDictionary = json.loads(text)
            finalDictionary = json.loads(text)

            thresoldFile = pd.read_csv(filterProcess.outPutFolder + "/" +
                                       str(folder) + "/" + str(file_name)+"-" +
                                       str(selectedMetric.histogramTipe)+"-Threshold" +
                                       str(selectedMetric.threshold)+".csv",
                                       usecols=["Frame", "Value"], sep="\t")

            framesFiltered = {
                key: framesDictionary[id]["frames"][str(key)] for key in thresoldFile["Frame"]}
            finalDictionary[id]["frames"] = framesFiltered
            quantFiltered = len(thresoldFile["Frame"])
            outputFilePath = filterProcess.outPutFolder + "/" + id + "/" + id+"Filtered.txt"
            othesColumns["metric"].append(selectedMetric.histogramTipe.name)
            othesColumns["threshold"].append(selectedMetric.threshold)
            othesColumns["network"].append("YOLO")
            othesColumns["filteredAnnotation"].append(outputFilePath)
            othesColumns["quantFilteredFrames"].append(quantFiltered)
            othesColumns["width"].append(selectedVideo['width'])
            othesColumns["height"].append(selectedVideo['height'])
            file = open(outputFilePath, "w")
            file.write(json.dumps(finalDictionary, indent=4, cls=NpEncoder))
            file.close()
            print('Filtered:', quantFiltered)
        othesColumns = pd.DataFrame(othesColumns)
        assert len(filteredList) == len(othesColumns)
        print("OTHER: ", othesColumns)
        filteredList = pd.concat(
            [filteredList, othesColumns], axis=1)
        print("CONCAT: ", filteredList)
        filteredList.to_csv(outputFilterdList, sep=';')
        print(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
