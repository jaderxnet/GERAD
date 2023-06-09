from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

path = "videos/"
file_name = "dNHUhaLqr7c"

f = open(path + file_name + "/" + file_name+".txt", "r")
text = f.read().replace('\n', '')
# print(text)

# parse x:
y = json.loads(text)

histogram_map = {}
histogram_map["EPDNVP"] = {}

# the result is a Python dictionary:
# print(y[file_name]["EPDNVP"])
total_frame = y[file_name]["frames_count"]
for x in y[file_name]["frames"].keys():
    if ("EPDNVP" in y[file_name]["frames"][x]):
        # print(y[file_name]["frames"][x]["EPDNVP"])
        histogram_map["EPDNVP"][x] = y[file_name]["frames"][x]["EPDNVP"]
    else:
        histogram_map["EPDNVP"][x] = 0
# print(histogram_map)


df = pd.DataFrame(histogram_map)

df = df.reindex(histogram_map["EPDNVP"].keys())

x = np.arange(len(histogram_map["EPDNVP"].keys()))
y = list(histogram_map["EPDNVP"].values())
largura = 4.2
fig, ax = plt.subplots()
ax.bar(x+largura, y, largura, label="valores")
ax.set_xlabel('frames')
ax.set_xlabel('EPDNVP')
# ax.set_xticks
ax.legend()
plt.show()


'''
x_axis = np.arange(0, 10, 0.001)
# normal com média 2,232 e variância 1.88, conforme planilha
plt.plot(x_axis, norm.pdf(x_axis, 2.232, 1.88))
plt.show()
'''
