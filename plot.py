# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# import matplotlib.pyplot as plt
# import numpy as np

# outputs = np.load("results.npy")
# sim_pairs = cosine_similarity(outputs)

# inner_similarity = []
# between_similarity = []
# for i in range(sim_pairs.shape[0]):
#     for j in range(sim_pairs.shape[0]):
#         if i == j:
#             continue
#         if i // 98 == j // 98:
#             inner_similarity.append(sim_pairs[i,j])
#         else:
#             between_similarity.append(sim_pairs[i,j])


# plt.style.use('_mpl-gallery')

# # plot:
# fig, ax = plt.subplots()

# ax.hist([inner_similarity, between_similarity], bins=10, linewidth=0.5, edgecolor="black", color=['blue', 'red'])

# ax.set(xlim=(0, 1))

# plt.show()

from util import bar_plot
import numpy as np
outputs = np.load("results_numscale15_upfactor2.npy")
n = 250
x_range = outputs[n].shape[0]
bar_plot(range(x_range), outputs[n,:])