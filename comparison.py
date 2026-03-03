import matplotlib.pyplot as plt
import numpy as np
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8)) 

methods = ['Random','KCM + PCA', 'Graph Coloring']
min_distance = [114, 337.58, 344.75]
avg_distance = [397,675.71, 729.4]
score = [45258, 228107.68, 252541.24]

bar1 = np.arange(len(min_distance))
bar2 = [x + barWidth for x in bar1]
#bar3 = [x + barWidth for x in bar2] 

plt.bar(bar1, min_distance, color ='r', width = barWidth, edgecolor ='grey', label ='Min Distance')
plt.bar(bar2, avg_distance, color ='g', width = barWidth, edgecolor ='grey', label ='Avg Distance')
#plt.bar(bar3, score, color ='b', width = barWidth, edgecolor ='grey', label ='Score')

plt.xlabel('Methods', fontweight ='bold', fontsize = 15)
plt.ylabel('Values', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(min_distance))], methods)
plt.legend()
plt.show()