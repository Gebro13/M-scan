# # This is a sample Python script.
#
# # Press Shift+F10 to execute it or replace it with your code.
# # Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# import numpy as np
# import openai


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import t

# aliccp_auc = [0.66,0.6714,0.6631,0.6537]
# aliccp_c = [0.0001,0.001,0.01,0.1]

# cloud_auc = [0.7591,0.7608,0.7595,0.7586]
# cloud_c = [0.01,0.1,1,10]

# plt.rcParams['font.size'] = 15

# fig = plt.figure(figsize=(8, 4))
# gs = gridspec.GridSpec(1, 2)
# print(chr(945))
# # 绘制第一个子图
# ax1 = fig.add_subplot(gs[0])
# ax1.bar([1,3,5,7],aliccp_auc,color = '#FFD9D9')  # 绘制第一个图的数据
# ax1.set_xlabel(chr(945)+'(when c = 18)')
# ax1.set_ylabel('AUC')
# ax1.set_title('Aliccp')
# ax1.set_xticks(np.linspace(1,7,4))
# ax1.set_xticklabels(['0.0001','0.001','0.01','0.1'])
# ax1.set_ylim(0.65,0.675)
# ax1.set_yticks(np.linspace(0.65,0.675,6))
# # 创建第二个图
# ax2 = fig.add_subplot(gs[1])
# ax2.bar([1,3,5,7],cloud_auc,color = '#FFD9D9')  # 绘制第一个图的数据
# ax2.set_xlabel(chr(945)+'(when c = 1)')
# ax2.set_ylabel('AUC')
# ax2.set_title('Cloud')
# ax2.set_xticks(np.linspace(1,7,4))
# ax2.set_xticklabels(['0.01','0.1','1','10'])
# ax2.set_ylim(0.755,0.765)
# ax2.set_yticks(np.linspace(0.755,0.765,6))

# plt.tight_layout()

# plt.show()
# plt.savefig('hyperparameter_alpha.pdf')


aliccp_auc = [0.6605,0.6674,0.6693,0.6714,0.6694]
aliccp_c = [100,75,35,18,10]

cloud_auc = [0.7597,0.7608,0.7606,0.7599]
cloud_c = [0.1,1,5,10]

plt.rcParams['font.size'] = 15

fig = plt.figure(figsize=(8, 4))
gs = gridspec.GridSpec(1, 2)

# 绘制第一个子图
ax1 = fig.add_subplot(gs[0])
ax1.plot(aliccp_c,aliccp_auc,color ='#90baf4')  # 绘制第一个图的数据
ax1.scatter(aliccp_c,aliccp_auc,color = '#90baf4')
ax1.set_xlabel('c (when '+chr(945)+' = 0.001)')
ax1.set_ylabel('AUC')
ax1.set_title('Aliccp')
ax1.set_xticks(np.linspace(0, 100, 6))
ax1.set_yticks(np.linspace(0.6595,0.6715,7))
# 创建第二个图
ax2 = fig.add_subplot(gs[1])
ax2.plot(cloud_c,cloud_auc,color ='#90baf4')  # 绘制第一个图的数据
ax2.scatter(cloud_c,cloud_auc,color = '#90baf4')
ax2.set_xlabel('c (when '+chr(945)+' = 0.1)')
ax2.set_ylabel('AUC')
ax2.set_title('Cloud')
ax2.set_xticks(np.linspace(0, 10, 11))
ax2.set_yticks(np.linspace(0.7596,0.7610,8))
plt.tight_layout()

plt.show()
plt.savefig('hyperparameter_c.pdf')


# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     #
#     # # Set your API key
#     openai.api_key = "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"
#     # Use the GPT-3 model
#     completion = openai.Completion.create(
#     engine = "text-davinci-003",
#     prompt = "用可爱的语气回答：那我跟朱大爷僵尸。",
#     max_tokens = 1024,
#     temperature = 0.9
#     )
#     # Print the generated text
#
#     print(completion.choices[0].text)
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
