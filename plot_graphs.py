import matplotlib.pyplot as plt
import numpy as np

data_centered_cn = np.load("data_centered_gibbs_cranknicolson.npy", allow_pickle=True)
data_centered_cn = data_centered_cn.item()
h_theta_cn = data_centered_cn["h_theta"]


data_rescale = np.load("data_rescale_tuned.npy", allow_pickle=True)
data_rescale = data_rescale.item()
h_theta_rescale = data_rescale["h_theta"]


data_centered_scaled = np.load("data_centered_gibbs_scalednc.npy", allow_pickle=True)
data_centered_scaled = data_centered_scaled.item()
h_theta_scaled = data_centered_scaled["h_theta"]



i = 4
plt.plot(h_theta_scaled[:, i], label="ScaledNC", alpha = 0.5)
plt.plot(h_theta_rescale[:, i], label="Resclae", alpha = 0.5)
plt.legend(loc="upper right")
plt.show()