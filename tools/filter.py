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


class FilterProcess:
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
    inputFilePath = "ListsInfo/processedFrevo.csv"
    videosFolder = "videoTest"
    FilterProcess = FilterProcess(
        inputFilePath, videosFolder, True)
    FilterProcess.readInput()
    idexexStatusProcessing = FilterProcess.getIndexByStatus("Processing")
    idexexStatusProcessed = FilterProcess.getIndexByStatus("Processed")
    if (len(idexexStatusProcessing) > 0):
        print("PROCESSING THE INDEX VIDEO: ",
              idexexStatusProcessing.tolist()[0])
    elif (len(idexexStatusProcessed) <= 0):
        print("ALL VIDEOS PROCESSED!")
    else:
        for indexProcessed in idexexStatusProcessed.tolist():
            selectedVideo = FilterProcess.getItemByIndex(indexProcessed)
            FilterProcess.log("Index to update: ", indexProcessed)
            FilterProcess.log("Updated Table: ", FilterProcess)
            FilterProcess.log("Selected Video: ", selectedVideo)

            folder = file_name = id = selectedVideo['id']
            outputFile = open(FilterProcess.outPutFolder + "/" +
                              folder + "/" + file_name+".txt", "r")
            text = outputFile.read().replace('\n', '')

            framesDictionary = json.loads(text)
            finalDictionary = json.loads(text)
            metrics = {}
            metrics["EPDNVP"] = Histogram(id, MetricTipe.EPDNVP)
            metrics["EPDNMVP"] = Histogram(id, MetricTipe.EPDNMVP)
            metrics["EPDNM"] = Histogram(id, MetricTipe.EPDNM)
            metrics["EPE"] = Histogram(id, MetricTipe.EPE)
            metrics["EPDNVP"].thresholdTipe = ThresholdType.MINOR_EQUAL
            metrics["EPDNVP"].threshold = 0.8
            metrics["EPDNMVP"].thresholdTipe = ThresholdType.MINOR_EQUAL
            metrics["EPDNMVP"].threshold = 0.05
            metrics["EPDNM"].thresholdTipe = ThresholdType.MINOR_EQUAL
            metrics["EPDNM"].threshold = 0.1
            metrics["EPE"].thresholdTipe = ThresholdType.MINOR_EQUAL
            metrics["EPE"].threshold = 200
            # IS8r3wG8-Js-MetricTipe.EPDNM-Threshold0.1
            df = pd.read_csv(FilterProcess.outPutFolder + "/" +
                             str(folder) + "/" + str(file_name)+"-" +
                             str(metrics["EPE"].histogramTipe)+"-Threshold" +
                             str(metrics["EPE"].threshold)+".csv",
                             usecols=["Frame", "Value"], sep="\t")

            framesFiltered = {
                key: framesDictionary[id]["frames"][str(key)] for key in df["Frame"]}
            finalDictionary[id]["frames"] = framesFiltered
            outputFilePath = FilterProcess.outPutFolder + "/" + id + "/" + id+"Filtered.txt"
            file = open(outputFilePath, "w")
            file.write(json.dumps(finalDictionary, indent=4, cls=NpEncoder))
            file.close()
