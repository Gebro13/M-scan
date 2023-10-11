# # This is a sample Python script.
#
# # Press Shift+F10 to execute it or replace it with your code.
# # Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# import numpy as np
# import openai
# xxx


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import t



aliccp_auc = [0.6605,0.6674,0.6693,0.6714,0.6694]
aliccp_c = [100,75,35,18,10]

cloud_auc = [0.7597,0.7608,0.7606,0.7599]
cloud_c = [0.1,1,5,10]

plt.rcParams['font.size'] = 15

fig = plt.figure(figsize=(8, 4))
gs = gridspec.GridSpec(1, 2)
  
# draw the first sub-graph
ax1 = fig.add_subplot(gs[0])
ax1.plot(aliccp_c,aliccp_auc,color ='#90baf4')  # feed data in the first graph
ax1.scatter(aliccp_c,aliccp_auc,color = '#90baf4')
ax1.set_xlabel('c (when '+chr(945)+' = 0.001)')
ax1.set_ylabel('AUC')
ax1.set_title('Aliccp')
ax1.set_xticks(np.linspace(0, 100, 6))
ax1.set_yticks(np.linspace(0.6595,0.6715,7))
# create another graph
ax2 = fig.add_subplot(gs[1])
ax2.plot(cloud_c,cloud_auc,color ='#90baf4')  # feed data in the second graph
ax2.scatter(cloud_c,cloud_auc,color = '#90baf4')
ax2.set_xlabel('c (when '+chr(945)+' = 0.1)')
ax2.set_ylabel('AUC')
ax2.set_title('Cloud')
ax2.set_xticks(np.linspace(0, 10, 11))
ax2.set_yticks(np.linspace(0.7596,0.7610,8))
plt.tight_layout()

plt.show()
plt.savefig('hyperparameter_c.pdf')
