# Create a python class with main
# this class read from csv file
# the csv struture is:
# - Youtube link
# - Dance types format like #hashtag
# Create a function to Stract youtube data info
# Create a function to create another csv file with:
# - Youtube ID;
# - Youtube Link;
# - Youtube Video Name;
# - Youtube Music Name;
# - Youtube Description;
# - Youtube Video best quality;
# - Dance #hashtags;
# - Status -> Ready --> Processing ---> Processed;
# - Status Date;

import sys
import pandas as pd
# !pip install git+https://github.com/ytdl-org/youtube-dl.git@master
from youtube_dl import YoutubeDL
from datetime import datetime
from tqdm import tqdm
from enum import Enum


class OptionsVideoDownload(Enum):
    DOWNLOAD_OPTION = 1
    BRACE_OPTION = 2


class DownloadYoutube:

    def __init__(self, optionEnum, download=False, outputVideoPath="videos"):
        self.optionEnum = optionEnum
        self.download = download
        self.outputVideoPath = outputVideoPath
        self.setOption()

    def setOption(self):
        if (self.optionEnum.value == 1):
            self.option = {
                'format': 'bestvideo[ext=mp4]',
                # This will select the specific resolution typed here
                # "format": "mp4[height=1080]",
                'no_check_certificate': True,
                # "%(id)s/%(id)s-%(title)s.%(ext)s"
                "outtmpl": f"{self.outputVideoPath}/%(id)s/%(id)s.%(ext)s"
            }
        elif (self.optionEnum.value == 2):
            self.option = {
                # 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'
                # "format": "mp4[height=1080]",
                "format": "bestvideo[ext=mp4]",
                "continue": True,
                'noplaylist': True,
                "outtmpl": f"{self.outputVideoPath}/%(id)s/%(id)s-%(title)s.%(ext)s"
            }

    def getInfo(self, url):
        with YoutubeDL(self.option) as ydl:
            ydl._ies = [ydl.get_info_extractor('Youtube')]
            youtubeInfoDictionary = ydl.extract_info(
                url, download=False)
            return youtubeInfoDictionary

    def getDownload(self, url):
        with YoutubeDL(self.option) as ydl:
            return ydl.download(url)


class VideoInfoExtrator:
    '''
        Class to extract Video Info from youtube basede in a csv 
        with links list and hashtags.
        Output: CSV wit info list from YOUTUBE 
    '''

    def __init__(self, inputFilePath, outputFilePath):
        self.inputFilePath = inputFilePath
        self.outputFilePath = outputFilePath
        self.videosTable = None
        self.videosDetailedTable = None

    def readInputFile(self):
        self.videosTable = pd.read_csv(self.inputFilePath, sep=";")
        print("Lendo: ", self.inputFilePath)
        print(self.videosTable)

    def getInfoFromYoutube(self):
        # Create a DataFrame object 33 columns
        columns = [
            # Control data 3
            'id_process', 'status', 'status_date', 'hashtags',
            # Youtube data 30
            'id', 'url', 'title', 'description', 'upload_date',
            'channel', 'duration', 'view_count',
            'like_count', 'format', 'width', 'height',
            'resolution', 'tbr', 'abr', 'acodec', 'vbr', 'fps',
            'vcodec', 'container', 'filesize', 'filesize_approx',
            'protocol', 'extractor', 'track', 'artist', 'genre',
            'album', 'album_type', 'album_artist'
        ]
        print(len(columns), "Columns:")
        self.videosDetailedTable = pd.DataFrame(columns=columns)

        for index, row in tqdm(self.videosTable.iterrows()):
            print("Visiting: ", row['Video'], row['hashtags'])
            options = {
                # 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'
                # "format": "mp4[height=1080]",
                "format": "bestvideo[ext=mp4]",
                "continue": True,
                'noplaylist': True,
                "outtmpl": "%(id)s/%(id)s-%(title)s.%(ext)s"
            }
            with YoutubeDL(options) as ydl:
                ydl._ies = [ydl.get_info_extractor('Youtube')]
                youtubeInfoDictionary = ydl.extract_info(
                    row['Video'], download=False)
                # see OUTPUT TEMPLATE in https://github.com/ytdl-org/youtube-dl
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
                self.videosDetailedTable.loc[index] = line

    def generateOutput(self):
        print(self.videosDetailedTable)
        self.videosDetailedTable.to_csv(
            self.outputFilePath, index=False, sep=';')
        """
        with open(outputFilePath, "a") as outputFile:
        outputFile.write("Video ID: ")
        outputFile.write(video_id)
        outputFile.write("\n")
        outputFile.write(str(info_dict))
        outputFile.write("\n")
        outputFile.write("\n")
        """


if __name__ == '__main__':

    argvs = sys.argv
    input
    inputFilePath = "ListsVideos/frevolist.csv"
    outputFilePath = "ListsInfo/processedFrevo.csv"

    videoInfoExtrator = VideoInfoExtrator(inputFilePath, outputFilePath)
    videoInfoExtrator.readInputFile()
    videoInfoExtrator.getInfoFromYoutube()
    videoInfoExtrator.generateOutput()
