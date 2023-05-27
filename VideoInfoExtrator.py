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

argvs = sys.argv
input
inputFilePath = "videos.csv"
outputFilePath = "processedVideos.csv"

if (argvs != None and len(argvs) > 0):
    for arg in argvs:
        print(arg)
        if (".csv" in arg):
            inputFilePath = arg
            break

videosTable = pd.read_csv(inputFilePath, sep=";")
print("Lendo: ", inputFilePath)
print(videosTable)

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
videosDetailedTable = pd.DataFrame(columns=columns)

for index, row in videosTable.iterrows():
    print("Visiting: ", row['Video'], row['hashtags'])
    options = {
        # 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'
        "format": "mp4[height=1080]",
        "outtmpl": "%(id)s/%(id)s-%(title)s.%(ext)s"
    }
    with YoutubeDL(options) as ydl:
        youtubeInfoDictionary = ydl.extract_info(row['Video'], download=False)
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
        print(len(line), "Lines:")
        videosDetailedTable.loc[index] = line

print(videosDetailedTable)
videosDetailedTable.to_csv(outputFilePath, index=False, sep=';')
'''with open(outputFilePath, "a") as outputFile:
    outputFile.write("Video ID: ")
    outputFile.write(video_id)
    outputFile.write("\n")
    outputFile.write(str(info_dict))
    outputFile.write("\n")
    outputFile.write("\n")'''
