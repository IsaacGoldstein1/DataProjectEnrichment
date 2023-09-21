import json
import numpy as np
from itertools import groupby
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import pandas as pd
import seaborn as sns

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
print (ave_t)

pre_carv = np.array(ave_t[0])
post_carv = np.array(ave_t[2])
pre_cis = np.array(ave_t[1])
post_cis = np.array(ave_t[3])

# # Now, perform the Wilcoxon signed-rank test for carvone conditions
# carvone_wilcoxon_stat, carvone_p_value = wilcoxon(pre_carvone, post_carvone)

# # Perform the Wilcoxon signed-rank test for cis-hex conditions
# cis_hex_wilcoxon_stat, cis_hex_p_value = wilcoxon(pre_cis_hex, post_cis_hex)

# # Print the results
# print("Wilcoxon signed-rank test for carvone conditions:")
# print("Statistic:", carvone_wilcoxon_stat)
# print("P-value:", carvone_p_value)

# print("\nWilcoxon signed-rank test for cis-hex conditions:")
# print("Statistic:", cis_hex_wilcoxon_stat)
# print("P-value:", cis_hex_p_value)
barWidth = 0.15
spacing = 0.2  # Adjust this value to set the desired space between bars X2 and X3

X1 = np.arange(len(ave_t[0]))
X2 = [x + barWidth for x in X1]
X3 = [x + barWidth + spacing for x in X2]  # Add spacing between X2 and X3
X4 = [x + barWidth for x in X3]

pre_carv_color = 'salmon'
post_carv_color = 'darkred'
pre_cis_color = 'lightblue'
post_cis_color = 'darkblue'
fig = plt.figure(figsize=(20,12))

plt.bar(X1, ave_t[0], color=pre_carv_color, width=barWidth, label='Pre-Carvone')
plt.bar(X2, ave_t[2], color=post_carv_color, width=barWidth, label='Post-Carvone')
plt.bar(X3, ave_t[1], color=pre_cis_color, width=barWidth, label='Pre-cis-3-hexen-1-ol')
plt.bar(X4, ave_t[3], color=post_cis_color, width=barWidth, label='Post-cis-3-hexen-1-ol')
plt.xticks([r + barWidth for r in range(len(ave_t[0]))],
        ['session trials (30 per)'])
plt.ylabel('Average cluster size per block')  # Add the label for y-axis


averages = {
 	'Pre_Carv': [28.583, 33.299, 28.499],
 	'Post_Carv': [28.884, 33.194, 29.463],
 	'Pre_Cis': [41.499, 38.399, 34.993],
 	'Post_Cis': [42.318, 38.430, 33.891],
    }

df = pd.DataFrame(averages)

# Melt the DataFrame to long format

melted_df = pd.melt(df, var_name='group', value_name='Average Licks')

# Separate the 'Pre' and 'Post' test conditions
melted_df['Test'] = melted_df['group'].str.split('_').str[0]
melted_df['Odor'] = melted_df['group'].str.split('_').str[1]

# Create the bar plot using seaborn
g = sns.catplot(
 	x='Odor',
 	y='Average Licks',
 	hue='Test',
 	data=melted_df,
 	kind='bar',
 	errorbar='sd',
 	edgecolor='black',
 	errcolor='black',
 	errwidth=1.5,
 	capsize=0.1,
 	height=6,
 	aspect=1.2,
 	alpha=0.5
)
# Add the dot plot to the same figure
sns.stripplot(
    x='Odor',
    y='Average Licks',
    hue='Test',
    data=melted_df,
    dodge=True,  # Separate dots for 'Pre' and 'Post' conditions
    jitter=True,  # Add jitter to the dots for better visualization
    marker='o',  # Marker style for dots
    size=8,  # Size of the dots
    palette={'Pre': 'blue', 'Post': 'red'},  # Dot color mapping
    ax=g.ax,  # Use the same axes as the bar plot
)


plt.show()


==================================================================
