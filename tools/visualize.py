from video import VideoManipulation
from histogram import Histogram, MetricTipe, ThresholdType
import pandas as pd

if __name__ == '__main__':
    id = "IS8r3wG8-Js"
    foldePath = "videoTest"

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
    # IS8r3wG8-Js-MetricTipe.EPDNM-Threshold0.1
    df = pd.read_csv(foldePath + "/" +
                     id + "/" + id+"-" +
                     str(metrics["EPDNVP"].histogramTipe)+"-Threshold" +
                     str(metrics["EPDNVP"].threshold)+".csv",
                     usecols=["Frame", "Value"], sep="\t")

    videoManipulation = VideoManipulation(
        "videoTest/IS8r3wG8-Js/OUTIS8r3wG8-Js.mp4", foldePath, "videoTest/IS8r3wG8-Js/OUTIS8r3wG8-Js.mp4", 1920, 1080, False, True, True)
    for frameFromFile in df["Frame"]:
        if videoManipulation.verifyVideoStop() == True:
            break
        if (videoManipulation.isOpened() == False):
            print(
                "Error opening video stream or file: " + videoManipulation.inputPath)
        else:
            # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
            videoManipulation.setNextFrame(frameFromFile)
            ret, frame = videoManipulation.readFrame()
            print("Frame From File: ", frameFromFile)
            if (ret == False):
                print(
                    f'Frame Reader ERROR! VIDEO: ', videoManipulation.inputPath)
                break
            else:
                videoManipulation.display("frame", frame)
    videoManipulation.release()
    videoManipulation.destroy()
