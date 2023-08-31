import json
import numpy as np
from itertools import groupby
import matplotlib.pyplot as plt

f = open("33-35.txt", "r")
name_idx = [0, 73, 146, 219]
line_count = 0
idx = 0
name = []
data = [[], [], [], []]
threashold = 0.4

def calc_avg(my_data, idx1, idx2_start, idx2_end, limit):
    lst = []
    cluster = 0
    obs_idx = idx2_start
    for obs_idx in range(idx2_start, idx2_end):
        # Omit lines with zeroes
        if len(my_data[idx1][obs_idx]) == 1 and my_data[idx1][obs_idx][0] == 0:
            continue
        else:
            a = my_data[idx1][obs_idx]
            cluster_licks = [x for x in a if x < limit]
            if len(cluster_licks) >= 4:
                lst.append(cluster_licks)
    return sum(map(len, lst)) / len(lst) if len(lst) > 0 else 0

# read the file and parse the data
while True:
    # Get next line from file
    line = f.readline()
 
    # if line is empty
    # end of file is reached
    if not line:
        break
    if line_count in name_idx:
        name.append(line.strip(' = [\n'))
        idx += 1
    else:
        data[idx-1].append(list(map(float, line.strip(' [],\n').split(","))))
    line_count += 1
 
f.close()

cond_idx = 0
ave_list = []
temp = []
for cond_idx in range(0, len(data)):
    temp.append(calc_avg(data, cond_idx, 0, 72, threashold))
ave_list.append(temp)
temp = []
ave_t = np.array(ave_list).T.tolist()

barWidth = 0.2
X1 = np.arange(len(ave_t[0]))
X2 = [x + barWidth for x in X1]
X3 = [x + barWidth for x in X2]
X4 = [x + barWidth for x in X3]

pre_cis_color = 'salmon'
post_cis_color = 'darkred'
pre_carvone_color = 'lightblue'
post_carvone_color = 'darkblue'

plt.bar(X1, ave_t[0], color=pre_carvone_color, width=barWidth, label='Pre-Carvone')
plt.bar(X2, ave_t[1], color=post_carvone_color, width=barWidth, label='Post-Carvone')
plt.bar(X3, ave_t[2], color=pre_cis_color, width=barWidth, label='Pre-cis')
plt.bar(X4, ave_t[3], color=post_cis_color, width=barWidth, label='Post-cis')

plt.xticks([r + barWidth for r in range(len(ave_t[0]))], ['Session'])
plt.ylabel('Average cluster size per block')

plt.legend()
plt.show()
