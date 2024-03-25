# !pip install git+https://github.com/ytdl-org/youtube-dl.git@master
#!pip install --upgrade --force-reinstall git+https://github.com/ytdl-org/youtube-dl.git@master
from yt_dlp import YoutubeDL
#!pip install "opencv-python-headless<4.3"
import cv2
from tqdm.auto import tqdm
import os
import sys
import pandas as pd


class VideoSprit:

    def __init__(self, youtubeVideoLink, folderPath):
        self.youtubeVideoLink = youtubeVideoLink
        self.folderPath = folderPath
        self.options = {
            'format': 'bestvideo[ext=mp4]',
            # This will select the specific resolution typed here
            # "format": "mp4[height=1080]",
            'no_check_certificate': True,
            # "%(id)s/%(id)s-%(title)s.%(ext)s"
            "outtmpl": f"{self.folderPath}/%(id)s/%(id)s.%(ext)s"
        }
        self.videoInfoMap = self.infoVideoExtrator()

    def infoVideoExtrator(self):

        with YoutubeDL(self.options) as ydl:
            return ydl.extract_info(self.youtubeVideoLink, download=False)
            # see OUTPUT TEMPLATE in https://github.com/ytdl-org/youtube-dl
            """
            line = [
                index, "Ready",
                datetime.today().strftime('%Y%m%d'),
                row['hashtags'],
                youtubeInfoDictionary.get("id", None),
                row['Video'],
                youtubeInfoDictionary.get('title', None),
                youtubeInfoDictionary.get('description', None),
                youtubeInfoDictionary.get('upload_date', None),
                youtubeInfoDictionary.get('channel', None),
                youtubeInfoDictionary.get('duration', None),
                youtubeInfoDictionary.get('view_count', None),
                youtubeInfoDictionary.get('like_count', None),
                youtubeInfoDictionary.get('format', None),
                youtubeInfoDictionary.get('width', None),
                youtubeInfoDictionary.get('height', None),
                youtubeInfoDictionary.get('resolution', None),
                youtubeInfoDictionary.get('tbr', None),
                youtubeInfoDictionary.get('abr', None),
                youtubeInfoDictionary.get('acodec', None),
                youtubeInfoDictionary.get('vbr', None),
                youtubeInfoDictionary.get('fps', None),
                youtubeInfoDictionary.get('vcodec', None),
                youtubeInfoDictionary.get('container', None),
                youtubeInfoDictionary.get('filesize', None),
                youtubeInfoDictionary.get(
                    'filesize_approx', None),
                youtubeInfoDictionary.get('protocol', None),
                youtubeInfoDictionary.get('extractor', None),
                youtubeInfoDictionary.get('track', None),
                youtubeInfoDictionary.get('artist', None),
                youtubeInfoDictionary.get('genre', None),
                youtubeInfoDictionary.get('album', None),
                youtubeInfoDictionary.get('album_type', None),
                youtubeInfoDictionary.get('album_artist', None)]
            print(len(line), "Lines:")
            videosDetailedTable.loc[index] = line
            """

    def downloadVideo(self):
        with YoutubeDL(self.options) as ydl:
            ydl.download(self.youtubeVideoLink)

    def spritVideo(self):
        # Use OpenCV’s VideoCapture to load the input video.
        cap = cv2.VideoCapture(f"{self.folderPath}/" + self.videoInfoMap['id']
                               + "/" + self.videoInfoMap['id'] + ".mp4")
        if (cap.isOpened() == False):
            print("Error opening video stream or file")
        else:
            frames_count = 0
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print("Frames: ", length)
            # Read until video is completed
            try:
                os.mkdir(self.folderPath+"/images/")
            except:
                print(self.folderPath+"/images/")
            try:
                os.mkdir(self.folderPath+"/images/" +
                         self.videoInfoMap['id']+"/")
            except:
                print(self.folderPath+"/images/" +
                      self.videoInfoMap['id']+"/", "Já existe!")
            for i in tqdm(range(length), position=0, leave=True):
                # Capture frame-by-frame
                ret, frame = cap.read()
                frames_count = i + 1
                if ret == True:
                    # Convert the frame received from OpenCV to a MediaPipe’s Image object.
                    image_path = self.folderPath+"/images/" + \
                        self.videoInfoMap['id']+"/" + \
                        f'img-{frames_count:06}'+".jpg"
                    status = cv2.imwrite(image_path, frame)
                    if not status:
                        print("D'nt Save Image in: ",
                              image_path)
            # When everything done, release the video capture object
            cap.release()


if __name__ == '__main__':
    '''
    Arg1: CSV inputFile with a video list
    Arg2: path to dataset
    '''
    argvs = sys.argv
    inputFilePath = "processedVideos.csv"
    imagesPath = "tempdataset"
    if (argvs != None and len(argvs) > 0):
        for arg in argvs:
            print(arg)
            if (".csv" in arg):
                inputFilePath = arg
            else:
                imagesPath = arg

    videosTable = pd.read_csv(inputFilePath, sep=';')
    filtered = videosTable["status"] == "Processing"
    indexesProcessing = videosTable.index[filtered]
    if (len(indexesProcessing) > 0):
        print("PROCESSING THE INDEX VIDEO: ", indexesProcessing.tolist()[0])
    else:
        filtered = videosTable["status"] == "Ready"
        indexesToProcess = videosTable.index[filtered]
        print("Filtered: ", len(filtered))
        print("indexesToProcess: ", len(indexesToProcess))
        if (len(indexesToProcess) < 1):
            print("ALL VIDEOS PROCESSED!")
        else:
            for indexToProcess in indexesToProcess.tolist():
                videosTable.at[indexToProcess, 'status'] = 'Downloaded'
                videosTable.to_csv(inputFilePath, index=False, sep=';')
                selectedVideo = videosTable.loc[indexToProcess]
                video = VideoSprit(selectedVideo['url'], imagesPath)
                print(video.videoInfoMap["id"] +
                      " - " + video.videoInfoMap["title"], "  - ", selectedVideo["url"])
                video.downloadVideo()
                video.spritVideo()
                print(video.videoInfoMap["id"] +
                      " - " + video.videoInfoMap["title"], " Downloaded in ", imagesPath)
                # break
