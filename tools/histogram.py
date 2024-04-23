from logger import Logger
from singleton import VideoProcessorSingleton
from coder import NpEncoder
from matplotlib import colors
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

            indices = []
            valores_EPDNVP = []
            valores_EPDNMVP = []
            valores_EPDNM = []
            valores_EPE = []
            valores_EPDNVP_Discordancia = []
            # the result is a Python dictionary:
            # print(y[file_name]["EPDNVP"])
            total_frame = y[id]["frames_count"]
            print("Total: ", id)
            for x in y[id]["frames"].keys():
                indices.append(x)
                if (y[id]["frames"][x]["YOLO"]["poses_count"] <= 0 and
                        y[id]["frames"][x]["mediapipe"]["poses_count"] <= 0):
                    valores_EPDNVP_Discordancia.append(0)
                else:
                    if (y[id]["frames"][x]["YOLO"]["poses_count"] == 0 or
                            y[id]["frames"][x]["mediapipe"]["poses_count"] == 0):
                        valores_EPDNVP_Discordancia.append(5)
                    else:
                        if (y[id]["frames"][x]["EPDNVP"] >= 1):
                            valores_EPDNVP_Discordancia.append(
                                y[id]["frames"][x]["EPDNVP"])
                        else:
                            valores_EPDNVP_Discordancia.append(0)
                if ("EPDNVP" in y[id]["frames"][x]):
                    # print(y[id]["frames"][x]["EPDNVP"])
                    valores_EPDNVP.append(
                        y[id]["frames"][x]["EPDNVP"])
                else:
                    valores_EPDNVP.append(0)
                if ("EPDNMVP" in y[id]["frames"][x]):
                    # print(y[id]["frames"][x]["EPDNVP"])
                    valores_EPDNMVP.append(
                        y[id]["frames"][x]["EPDNMVP"])
                else:
                    valores_EPDNMVP.append(0)
                if ("EPDNM" in y[id]["frames"][x]):
                    # print(y[id]["frames"][x]["EPDNM"])
                    valores_EPDNM.append(
                        y[id]["frames"][x]["EPDNM"])
                else:
                    valores_EPDNM.append(0)
                if ("EPE" in y[id]["frames"][x]):
                    # print(y[id]["frames"][x]["EPDNM"])
                    valores_EPE.append(
                        y[id]["frames"][x]["EPE"])
                else:
                    valores_EPE.append(0)
            '''
            indi = 0
            for z in valores:
                if indi > 6048 and indi < 6183:
                    print("V ", indi, z)
                indi += 1
            '''
            valores_EPDNVP = np.array(valores_EPDNVP)
            valores_EPDNMVP = np.array(valores_EPDNMVP)
            valores_EPDNM = np.array(valores_EPDNM)
            valores_EPE = np.array(valores_EPE)
            valores_EPDNVP_Discordancia = np.array(valores_EPDNVP_Discordancia)
            '''
            indi = 0
            for z in valores:
                if indi > 6048 and indi < 6183:
                    print("N ", indi, z)
                indi += 1
            '''

            def inter_from_256(x):
                return np.interp(x=x, xp=[0, 255], fp=[0, 1])

            cdict = {
                'red': ((0.0, inter_from_256(0), inter_from_256(0)),
                        (1/10*1, inter_from_256(0), inter_from_256(0)),
                        (1/10*2, inter_from_256(210), inter_from_256(210)),
                        (1/10*3, inter_from_256(210), inter_from_256(210)),
                        (1/10*4, inter_from_256(210), inter_from_256(210)),
                        (1/10*5, inter_from_256(210), inter_from_256(210)),
                        (1/10*6, inter_from_256(210), inter_from_256(210)),
                        (1/10*7, inter_from_256(210), inter_from_256(210)),
                        (1/10*8, inter_from_256(210), inter_from_256(210)),
                        (1/10*9, inter_from_256(210), inter_from_256(210)),
                        (1.0, inter_from_256(210), inter_from_256(210))),
                'green': ((0.0, inter_from_256(220), inter_from_256(220)),
                          (1/10*1, inter_from_256(0), inter_from_256(0)),
                          (1/10*2, inter_from_256(0), inter_from_256(0)),
                          (1/10*3, inter_from_256(0), inter_from_256(0)),
                          (1/10*4, inter_from_256(0), inter_from_256(0)),
                          (1/10*5, inter_from_256(0), inter_from_256(0)),
                          (1/10*6, inter_from_256(0), inter_from_256(0)),
                          (1/10*7, inter_from_256(0), inter_from_256(0)),
                          (1/10*8, inter_from_256(0), inter_from_256(0)),
                          (1/10*9, inter_from_256(0), inter_from_256(0)),
                          (1.0, inter_from_256(0), inter_from_256(0))),
                'blue': ((0.0, inter_from_256(0), inter_from_256(0)),
                         (1/10*1, inter_from_256(255), inter_from_256(255)),
                         (1/10*2, inter_from_256(0), inter_from_256(0)),
                         (1/10*3, inter_from_256(0), inter_from_256(0)),
                         (1/10*4, inter_from_256(0), inter_from_256(0)),
                         (1/10*5, inter_from_256(0), inter_from_256(0)),
                         (1/10*6, inter_from_256(0), inter_from_256(0)),
                         (1/10*7, inter_from_256(0), inter_from_256(0)),
                         (1/10*8, inter_from_256(0), inter_from_256(0)),
                         (1/10*9, inter_from_256(0), inter_from_256(0)),
                         (1.0, inter_from_256(0), inter_from_256(0))),
            }
            new_cmap = colors.LinearSegmentedColormap(
                'new_cmap', segmentdata=cdict)

            cdict2 = {
                'red': ((0.0, inter_from_256(210), inter_from_256(210)),
                        (1/10*1, inter_from_256(210), inter_from_256(210)),
                        (1/10*2, inter_from_256(210), inter_from_256(210)),
                        (1/10*3, inter_from_256(210), inter_from_256(210)),
                        (1/10*4, inter_from_256(210), inter_from_256(210)),
                        (1/10*5, inter_from_256(210), inter_from_256(210)),
                        (1/10*6, inter_from_256(0), inter_from_256(0)),
                        (1/10*7, inter_from_256(0), inter_from_256(0)),
                        (1/10*8, inter_from_256(0), inter_from_256(0)),
                        (1/10*9, inter_from_256(0), inter_from_256(0)),
                        (1.0, inter_from_256(0), inter_from_256(0))),
                'green': ((0.0, inter_from_256(0), inter_from_256(0)),
                          (1/10*1, inter_from_256(0), inter_from_256(0)),
                          (1/10*2, inter_from_256(0), inter_from_256(0)),
                          (1/10*3, inter_from_256(0), inter_from_256(0)),
                          (1/10*4, inter_from_256(0), inter_from_256(0)),
                          (1/10*5, inter_from_256(0), inter_from_256(0)),
                          (1/10*6, inter_from_256(220), inter_from_256(220)),
                          (1/10*7, inter_from_256(220), inter_from_256(220)),
                          (1/10*8, inter_from_256(220), inter_from_256(220)),
                          (1/10*9, inter_from_256(220), inter_from_256(220)),
                          (1.0, inter_from_256(220), inter_from_256(0))),
                'blue': ((0.0, inter_from_256(0), inter_from_256(0)),
                         (1/10*1, inter_from_256(0), inter_from_256(0)),
                         (1/10*2, inter_from_256(0), inter_from_256(0)),
                         (1/10*3, inter_from_256(0), inter_from_256(0)),
                         (1/10*4, inter_from_256(0), inter_from_256(0)),
                         (1/10*5, inter_from_256(0), inter_from_256(0)),
                         (1/10*6, inter_from_256(0), inter_from_256(0)),
                         (1/10*7, inter_from_256(0), inter_from_256(0)),
                         (1/10*8, inter_from_256(0), inter_from_256(0)),
                         (1/10*9, inter_from_256(0), inter_from_256(0)),
                         (1.0, inter_from_256(0), inter_from_256(0))),
            }
            new_cmap2 = colors.LinearSegmentedColormap(
                'new_cmap', segmentdata=cdict2)

            def plot_grph(valores, new_map, show, save, file_name, label):
                map = valores / float(max(valores))
                # indi = 0

                # my_cmap = plt.get_cmap("viridis")

                print('Max: ' + str(max(valores)))

                cols = new_map(map)
                plot = plt.scatter(valores, valores, c=valores, cmap=new_map)
                plt.clf()
                plt.colorbar(plot)
                plt.xlabel('frames/s')
                plt.ylabel(label)
                plt.title(file_name)
                plt.bar(range(len(valores)), valores, color=cols)
                if (save):
                    plt.savefig(histogramProcess.outPutFolder + "/" + folder + "/" + label +
                                "_" + id+'.png', dpi=200)
                if (show):
                    plt.show()

            valores_Similar = []
            for x in valores_EPDNVP:
                if x <= 0 or x > 1:
                    valores_Similar.append(0)
                else:
                    valores_Similar.append(1-x)

            # True, False, True
            plot_grph(valores_EPDNVP, new_cmap, False,
                      True, file_name, 'EPDNVP')
            plot_grph(np.array(valores_Similar), new_cmap2,
                      False, True, file_name, 'EPDNVP Similarity')
            plot_grph(valores_EPDNVP_Discordancia, new_cmap,
                      False, True, file_name, 'EPDNVP Discordance')
            plot_grph(valores_EPDNMVP, new_cmap, False,
                      True, file_name, 'EPDNMVP')
            plot_grph(valores_EPDNM, new_cmap, False, True, file_name, 'EPDNM')
            plot_grph(valores_EPE, new_cmap, False, True, file_name, 'EPE')
