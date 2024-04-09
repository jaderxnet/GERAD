
import pandas as pd
from singleton import VideoProcessorSingleton
from logger import Logger


class VideosProcess:
    def __init__(self, inputFile, outputFile) -> None:
        self.inputFile = inputFile
        self.outputFile = outputFile
        self.logger = Logger(printOption=True)

    def readinput(self):

        self.videosTable = pd.read_csv(self.inputFile, sep=';')
        singleton = VideoProcessorSingleton(self.videosTable["status"])
        filtered = singleton.filterBy("Processing")
        self.logger.print("Lendo: ", self.inputFile)
        self.logger.print(self.inputFile)
        self.logger.print("Table: ", len(self.videosTable))
        self.logger.print("Filtered: ", len(filtered))


if __name__ == '__main__':
    inputFilePath = "/Users/jaderxnet/_DADOS/GitHub/GERAD/ListsInfo/processedFrevo.csv"
    outputFilePath = "processedData.csv"
    videosProcess = VideosProcess(inputFilePath, outputFilePath)
    videosProcess.readinput()
