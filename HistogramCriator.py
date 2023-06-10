import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
import json

path = "videos/"
file_name = "dNHUhaLqr7c"

f = open(path + file_name + "/" + file_name+".txt", "r")
text = f.read().replace('\n', '')

y = json.loads(text)

indices = []
valores = []

# the result is a Python dictionary:
# print(y[file_name]["EPDNVP"])
total_frame = y[file_name]["frames_count"]
print("Total: ", total_frame)
for x in y[file_name]["frames"].keys():
    indices.append(x)
    if ("EPDNVP" in y[file_name]["frames"][x]):
        # print(y[file_name]["frames"][x]["EPDNVP"])
        valores.append(
            y[file_name]["frames"][x]["EPDNVP"])
    else:
        valores.append(0)
'''
indi = 0
for z in valores:
    if indi > 6048 and indi < 6183:
        print("V ", indi, z)
    indi += 1
'''
valores = np.array(valores)
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


map = valores / float(max(valores))
indi = 0

my_cmap = plt.get_cmap("viridis")

print('Max: ' + str(max(valores)))

cols = new_cmap(map)
plot = plt.scatter(valores, valores, c=valores, cmap=new_cmap)
plt.clf()
plt.colorbar(plot)
plt.xlabel('frames/s')
plt.ylabel('EPDNVP')
plt.title(file_name)
plt.bar(range(len(valores)), valores, color=cols)

plt.savefig(path + file_name + "/" + file_name+'.png', dpi=200)
plt.show()
