#!pip install "opencv-python-headless<4.3"
import cv2


class VideoManipulation:
    def __init__(self, inputPath, outputPath, filename, width, height, saveVideo=False, displayFrame=True, forceFileName=False) -> None:
        self.inputPath = inputPath
        self.outputPath = outputPath
        self.filename = filename
        self.width = width
        self.height = height
        self.saveVideo = saveVideo
        self.displayFrame = displayFrame
        self.readCapture(forceFileName)

    def readCapture(self, forceFileName=False):
        # Use OpenCV’s VideoCapture to load the input video.
        if (forceFileName):
            print("AQUI1" + self.filename)
            self.inputReder = cv2.VideoCapture(self.filename)
            print("AQUI2")
        else:
            self.inputReder = cv2.VideoCapture(
                self.outputPath+"/"+self.filename+"/"+self.filename+".mp4")

    def isOpened(self):
        return self.inputReder.isOpened()

    def writeVideo(self):
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        if self.saveVideo:
            self.outputWriter = cv2.VideoWriter(self.outputPath+"/"+self.filename+"/OUT"+self.filename+".mp4", fourcc, 20.0,
                                                (self.width, self.height))

    def setNextFrame(self, frame):
        self.inputReder.set(cv2.CAP_PROP_POS_FRAMES, frame-1)

    def readFrame(self):
        ret, frame = self.inputReder.read()
        # Display the resulting frame
        if ret and self.displayFrame:
            self.display('Video Frame', frame)
        return ret, frame

    def display(self, windowName, frame):
        if (self.displayFrame == True):
            cv2.imshow(windowName, frame)
            cv2.waitKey(1)

    def saveInVideo(self, frame):
        if (self.saveVideo):
            self.outputWriter.write(frame)

    def verifyVideoStop(self):
        return cv2.waitKey(25) & 0xFF == ord('q')

    def release(self):
        # When everything done, release the video capture object
        self.inputReder.release()
        if self.saveVideo:
            self.outputWriter.release()

    def destroy(self):
        # Closes all the frames
        cv2.destroyAllWindows()

    def convertImageToRGB(self, frame):
        annotated_image_rgb = cv2.cvtColor(
            frame, cv2.COLOR_RGB2BGR)
        return annotated_image_rgb

    def putTextInFrame(self, text, frame):
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 30)
        fontScale = 0.8
        fontColor = (255, 255, 255)
        thickness = 1
        lineType = 2
        dy = 40
        for i, line in enumerate(text.split('\n')):
            y = bottomLeftCornerOfText[1] + i*dy
            cv2.putText(frame, line,
                        (bottomLeftCornerOfText[0], y),
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)


if __name__ == '__main__':
    videoManipulation = VideoManipulation("videoTest/IS8r3wG8-Js/OUTIS8r3wG8-Js.mp4",
                                          "videoTest", "IS8r3wG8-Js", 1920, 1080, False, True)
    if (videoManipulation.isOpened() == False):
        print(
            "Error opening video stream or file: " + videoManipulation.inputPath)
    else:
        while (videoManipulation.isOpened() and videoManipulation.verifyVideoStop() == False):
            ret, frame = videoManipulation.readFrame()
            if (ret == False):
                print(
                    f'Frame Reader ERROR! VIDEO: ', videoManipulation.inputPath)
                break
            else:
                # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
                videoManipulation.display("frame", frame)
        videoManipulation.release()
        videoManipulation.destroy()
        # frames_count += 1
