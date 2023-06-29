import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
import json
# videos/Br8xTiOqvdA
# videos/cveTsmWApp8
# videos/FW-RjHXbyq8
# videos/xX0ZWd8KfbM
path = "videos/"
folder = file_name = id = "6jkn_yPV2mM"
f = open(path + folder + "/" + file_name+".txt", "r")
text = f.read().replace('\n', '')

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
new_cmap = colors.LinearSegmentedColormap('new_cmap', segmentdata=cdict)

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
new_cmap2 = colors.LinearSegmentedColormap('new_cmap', segmentdata=cdict2)


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
        plt.savefig(path + folder + "/" + label +
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
plot_grph(valores_EPDNVP, new_cmap, False, True, file_name, 'EPDNVP')
plot_grph(np.array(valores_Similar), new_cmap2,
          False, True, file_name, 'EPDNVP Similarity')
plot_grph(valores_EPDNVP_Discordancia, new_cmap,
          False, True, file_name, 'EPDNVP Discordance')
plot_grph(valores_EPDNMVP, new_cmap, False, True, file_name, 'EPDNMVP')
plot_grph(valores_EPDNM, new_cmap, False, True, file_name, 'EPDNM')
plot_grph(valores_EPE, new_cmap, False, True, file_name, 'EPE')
