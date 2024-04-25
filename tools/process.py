from metrics import MetricTipe, Metric
from video import VideoManipulation
from neural import Yolo, MediaPipe, NeuralNetwork
from extrator import OptionsVideoDownload, DownloadYoutube
from logger import Logger, PrintOption
from singleton import VideoProcessorSingleton
from coder import NpEncoder
import pandas as pd
import json
import numpy as np
import logging
import datetime


class VideosProcess:
    def __init__(self, inputFile, outPutFolder, printOption=PrintOption.ALL, download_video=False, saveFile=False) -> None:
        self.inputFile = inputFile
        self.outPutFolder = outPutFolder
        self.download_video = download_video
        self.saveFile = saveFile
        self.logger = Logger(printOption=printOption)
        logging.basicConfig(filename=f'{outPutFolder}/app{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.log', filemode='w',
                            format='%(name)s - %(levelname)s - %(message)s')
        logging.warning('This will get logged to a file')

    def log(self, *string, end="", printOption=PrintOption.ALL):
        logging.warning(string)
        return self.logger.print(string, end, printOption=printOption)

    def logError(self, *string):
        logging.error(string, exc_info=True)
        return self.logger.printError(string)

    def readInput(self):

        self.videosTable = pd.read_csv(self.inputFile, sep=';')
        self.logger.print("Lendo: ", self.inputFile)
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
        self.videosTable.to_csv(self.inputFile, index=False, sep=';')

    def saveOutputFile(self, id, text):
        if self.saveFile:
            # os.mkdir(outputFilePath)
            outputFilePath = self.outPutFolder + "/" + id + "/" + id+".txt"
            file = open(outputFilePath, "w")
            file.write(json.dumps(dictionary, indent=4, cls=NpEncoder))
            file.close()

    def getItemByIndex(self, index):
        return self.videosTable.loc[index]


if __name__ == '__main__':
    inputFilePath = "/Users/jaderxnet/_DADOS/GitHub/GERAD/ListsInfo/processedFrevo.csv"
    videosFolder = "videoTest"
    processControll = VideosProcess(
        inputFilePath, videosFolder, True, True, True)
    processControll.readInput()
    idexexStatusProcessing = processControll.getIndexByStatus("Processing")
    idexexStatusReady = processControll.getIndexByStatus("Ready")
    if (len(idexexStatusProcessing) > 0):
        print("PROCESSING THE INDEX VIDEO: ",
              idexexStatusProcessing.tolist()[0])
    elif (len(idexexStatusReady) <= 0):
        print("ALL VIDEOS PROCESSED!")
    else:
        # Load Neural Networks
        yolo = Yolo()
        mediaPipe = MediaPipe()
        for idexStatusReady in idexexStatusReady.tolist():
            processControll.changeStatus(idexStatusReady, 'Processing')
            # videosProcess.updateCSVInput()
            selectedVideo = processControll.getItemByIndex(idexStatusReady)
            processControll.log("Index to update: ", idexStatusReady)
            processControll.log("Updated Table: ", processControll)
            processControll.log("Selected Video: ", selectedVideo)

            download = DownloadYoutube(
                OptionsVideoDownload(1), outputVideoPath=videosFolder)
            result = download.getDownload([selectedVideo['url']])
            processControll.log('Download Concluído: ',
                                selectedVideo['id'], " - ", result)
            videoManipulation = VideoManipulation(
                videosFolder, videosFolder, selectedVideo["id"], selectedVideo["width"], selectedVideo["height"], True, True)
            if (videoManipulation.isOpened() == False):
                processControll.logError(
                    "Error opening video stream or file: " + selectedVideo["id"])
            else:
                dictionary = {}
                # dictionary with video id on youtube
                dictionary[selectedVideo["id"]] = {}
                dictionary[selectedVideo["id"]
                           ]["duration"] = selectedVideo["duration"]
                dictionary[selectedVideo["id"]]["fps"] = selectedVideo["fps"]

                # frames have the dictionary with frame id
                # (EPDNVP)EndPoint Diference Normalized Euclidian distance sum multiply by visivle product
                metricEPDNVP = Metric(
                    "EPDNVP", MetricTipe.EPDNVP)
                metricEPDNM = Metric("EPDNM", MetricTipe.EPDNM)
                metricEPDNMVP = Metric("EPDNMVP", MetricTipe.EPDNMVP)
                # (EPE)EndPoint Error (EPE) - Pixel Euclidian distance media
                metricEPE = Metric("EPE", MetricTipe.EPE)
                dictionary[selectedVideo["id"]]["frames"] = {}
                videoManipulation.writeVideo()
                processControll.log("Write Video")
                processControll.log(dictionary)
                totalCount = 0
                print_count = 0
                frames_count = 0
                # Read until video is completed
                frame_information = ""
                while (videoManipulation.isOpened()):
                    ret, frame = videoManipulation.readFrame()
                    if (ret == False):
                        processControll.logError(
                            f'Frame Reader ERROR! VIDEO: {selectedVideo["id"]} FRAME: {frames_count}')
                        break
                    else:
                        videoManipulation.display("frame", frame)
                        frames_count += 1
                        frame_information = "| Frame" + f'{frames_count:06}'
                        # frames have the dictionary with frame id
                        dictionary[selectedVideo["id"]
                                   ]["frames"][frames_count] = {}
                        yoloPredictResults = yolo.predict(frame)
                        quantidadePosesYolo = len(
                            yoloPredictResults[0].keypoints)
                        processControll.log(
                            "Quant Poses YOLO: ", quantidadePosesYolo)
                        processControll.log(
                            "Quant restults: ", len(yoloPredictResults))
                        dictionary[selectedVideo["id"]
                                   ]["frames"][frames_count]["YOLO"] = {}
                        dictionary[selectedVideo["id"]
                                   ]["frames"][frames_count]["YOLO"]["neural_network_file"] = yolo.model
                        dictionary[selectedVideo["id"]
                                   ]["frames"][frames_count]["YOLO"]["poses_count"] = quantidadePosesYolo
                        frame_information = frame_information + \
                            "| YOLO: " + f'{quantidadePosesYolo:02}'
                        if quantidadePosesYolo > 0:
                            results_json = yoloPredictResults[0].tojson(
                                normalize=True)
                            results_json = json.loads(results_json)
                            processControll.log("TO JSON: ", results_json)
                            processControll.log(
                                "X: ", results_json[0]["keypoints"]["x"])
                            dictionary[selectedVideo["id"]
                                       ]["frames"][frames_count]["YOLO"]["keypoints"] = []
                            list_poses_yolo = []
                            for result in results_json:
                                pose_yolo = np.stack((np.array((result["keypoints"]["x"])), np.array((
                                    result["keypoints"]["y"])), np.array((result["keypoints"]["visible"]))), axis=1)
                                processControll.log("List Poses: ", pose_yolo)
                                list_poses_yolo.append(pose_yolo)
                            dictionary[selectedVideo["id"]
                                       ]["frames"][frames_count]["YOLO"]["keypoints"] = list_poses_yolo

                        processControll.log(
                            "Yolo Results: ",  yoloPredictResults)
                        processControll.log("Keypoints Results: ",
                                            yoloPredictResults[0].tojson(normalize=True))

                        yoloAnnotatedImage = yoloPredictResults[0].plot()
                        videoManipulation.display("result", yoloAnnotatedImage)

                        # mpImage = mediaPipe.convertMPImagge(frame)
                        mediaPipePredictResults = mediaPipe.predict(frame)
                        quantidadePosesMediapipe = len(
                            mediaPipePredictResults.pose_landmarks)
                        processControll.log(
                            "Quant Poses Mediapipe: ", quantidadePosesMediapipe)
                        # mediapipe have the dictionary  neural networks detections
                        dictionary[selectedVideo["id"]
                                   ]["frames"][frames_count]["mediapipe"] = {}
                        dictionary[selectedVideo["id"]
                                   ]["frames"][frames_count]["mediapipe"]["poses_count"] = quantidadePosesMediapipe
                        dictionary[selectedVideo["id"]
                                   ]["frames"][frames_count]["mediapipe"]["neural_network_file"] = mediaPipe.model

                        # landmarks have the landmarks

                        processControll.log(mediaPipePredictResults)
                        processControll.log(
                            "Quant Poses:", quantidadePosesMediapipe)

                        if quantidadePosesMediapipe > 0:
                            dictionary[selectedVideo["id"]
                                       ]["frames"][frames_count]["mediapipe"]["keypoints"] = []
                            list_poses = []
                            for normalize_landmark in mediaPipePredictResults.pose_landmarks[0]:
                                list_poses.append([normalize_landmark.x,
                                                   normalize_landmark.y, normalize_landmark.visibility])
                            dictionary[selectedVideo["id"]]["frames"][frames_count
                                                                      ]["mediapipe"]["keypoints"].append(np.array((list_poses)))
                            processControll.log("Media PipePoses:", dictionary[selectedVideo["id"]]["frames"][frames_count
                                                                                                              ]["mediapipe"]["keypoints"])
                            mediaPipeAnnotatedImage = mediaPipe.getAnnotatedImage(
                                yoloAnnotatedImage, mediaPipePredictResults)
                            mediaPipeAnnotatedImageRGB = videoManipulation.convertImageToRGB(
                                mediaPipeAnnotatedImage)
                            videoManipulation.display(
                                "MediaPipe", mediaPipeAnnotatedImageRGB)
                            if (dictionary[selectedVideo["id"]
                                           ]["frames"][frames_count]["mediapipe"]["poses_count"] > 0
                                and dictionary[selectedVideo["id"]
                                               ]["frames"][frames_count]["YOLO"]["poses_count"] > 0):
                                index = 0

                                # print("MEDIAPIPE:", dictionary[selectedVideo["id"]]["frames"
                                # filter indexes metch to yolo from 33 mediapipe keypoints
                                #                                                 ][frames_count]["mediapipe"]["keypoints"][0])
                                filter_indices = [0, 2, 5, 7, 8, 11, 12,
                                                  13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

                                for yolo_pose in dictionary[selectedVideo["id"]]["frames"][frames_count]["YOLO"]["keypoints"]:
                                    # print("YOLO:", yolo_pose)
                                    otherDistance = metricEPDNVP.updateMinorDistance(
                                        yolo_pose, dictionary[selectedVideo["id"]]["frames"][frames_count]["mediapipe"]["keypoints"][0][filter_indices], index)

                                    metricEPDNMVP.updateMinorDistance(
                                        otherDistance=otherDistance, index=index)

                                    metricEPDNM.updateMinorDistance(yolo_pose,
                                                                    dictionary[selectedVideo["id"]]["frames"
                                                                                                    ][frames_count]["mediapipe"]["keypoints"][0][filter_indices], index=index)

                                    # print("Index: ", index, "Media: ", distance)
                                    metricEPE.updateMinorDistance(yolo_pose,
                                                                  dictionary[selectedVideo["id"]]["frames"
                                                                                                  ][frames_count]["mediapipe"]["keypoints"][0][filter_indices], index=index)
                                    index += 1
                                # (EPDNVP)EndPoint Diference Normalized Euclidian distance sum multiply by visivle product
                                dictionary[selectedVideo["id"]
                                           ]["frames"][frames_count]["EPDNVP"] = metricEPDNVP.minor_distance
                                metricEPDNVP.increaseValue()
                                dictionary[selectedVideo["id"]
                                           ]["frames"][frames_count]["EPDNMVP"] = metricEPDNMVP.minor_distance
                                metricEPDNMVP.increaseValue()
                                dictionary[selectedVideo["id"]
                                           ]["frames"][frames_count]["EPDNM"] = metricEPDNM.minor_distance
                                metricEPDNM.increaseValue()
                                dictionary[selectedVideo["id"]
                                           ]["frames"][frames_count]["EPE"] = metricEPE.minor_distance
                                metricEPE.increaseValue()
                                totalCount += 1
                                frame_information = frame_information + \
                                    "| Mediapipe: " + f'{quantidadePosesMediapipe:02}' + \
                                    "| EPDNVP INDEX : " + f'{metricEPDNVP.index:02}' + \
                                    " : " + f'{metricEPDNVP.minor_distance:06.15f}' + \
                                    "| EPDNMVP INDEX : " + f'{metricEPDNMVP.index:02}' + \
                                    " : " + f'{metricEPDNMVP.minor_distance:06.15f}' + '\n' +\
                                    "| EPDNM INDEX : " + f'{metricEPDNM.index:02}' + \
                                    " : " + f'{metricEPDNM.minor_distance:06.15f}' +  \
                                    "| EPE INDEX : " + f'{metricEPE.index:02}' + \
                                    " : " + \
                                    f'{metricEPE.minor_distance:06.15f}'
                                if (totalCount > 0):
                                    frame_information = frame_information + \
                                        "| Media EPDNVP: " + f'{metricEPDNVP.metricValue/totalCount:06.15f}' + '\n' +\
                                        "| Media EPDNMVP: " + f'{metricEPDNMVP.metricValue/totalCount:06.15f}' + \
                                        "| Media EPDNM: " + f'{metricEPDNM.metricValue/totalCount:06.15f}' + \
                                        "| Media EPE: " + \
                                        f'{metricEPE.metricValue/totalCount:06.15f}'
                            videoManipulation.putTextInFrame(
                                frame_information, mediaPipeAnnotatedImageRGB)
                        videoManipulation.putTextInFrame(
                            frame_information, yoloAnnotatedImage)
                        videoManipulation.putTextInFrame(
                            frame_information, frame)

                        processControll.log(
                            # "\033[K",
                            frame_information, end="\r", printOption=PrintOption.RESUME)

                        if (quantidadePosesMediapipe > 0):
                            videoManipulation.display(
                                "Frame", mediaPipeAnnotatedImageRGB)
                            videoManipulation.saveInVideo(
                                mediaPipeAnnotatedImageRGB)
                        else:
                            if quantidadePosesYolo > 0:
                                videoManipulation.display(
                                    "Frame", yoloAnnotatedImage)
                                videoManipulation.saveInVideo(
                                    yoloAnnotatedImage)
                            else:
                                videoManipulation.display("Frame", frame)
                                videoManipulation.saveInVideo(frame)

                        if videoManipulation.verifyVideoStop():
                            break
                # When everything done, release the video capture object
                videoManipulation.release()
                # print(dictionary)
                dictionary[selectedVideo["id"]
                           ]["frames_count"] = frames_count
                if (totalCount > 0):
                    dictionary[selectedVideo["id"]
                               ]["EPDNVP"] = metricEPDNVP.metricValue/totalCount
                    dictionary[selectedVideo["id"]
                               ]["EPDNMVP"] = metricEPDNMVP.metricValue/totalCount
                    dictionary[selectedVideo["id"]
                               ]["EPDNM"] = metricEPDNM.metricValue/totalCount
                    dictionary[selectedVideo["id"]
                               ]["EPE"] = metricEPE.metricValue/totalCount
                # (EPDNM)Normalized Euclidian distance media
        #        dictionary[selectedVideo["id"]
        #                    ]["EPDNMVP"]
                # (EPDNMVP)Normalized Euclidian distance multiply by visivle product media
        #        dictionary[selectedVideo["id"]
        #                    ]["EPDNM"]
                # (EPE)EndPoint Error (EPE) - Pixel Euclidian distance media
        #        dictionary[selectedVideo["id"]
        #                    ]["EPE"]

                # Closes all the frames
                videoManipulation.destroy()

                # Load the frame rate of the video using OpenCV’s CV_CAP_PROP_FPS
                # You’ll need it to calculate the timestamp for each frame.

                # Loop through each frame in the video using VideoCapture#read()
                processControll.changeStatus(idexStatusReady, 'Processed')
                processControll.updateCSVInput()
                processControll.saveOutputFile(selectedVideo['id'], dictionary)

            break
