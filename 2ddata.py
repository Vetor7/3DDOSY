import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cauchy


def add_noise(signal, snr_dB):
    # 计算信号功率
    signal_power = np.mean(np.abs(signal) ** 2)

    # 计算噪声功率
    snr_linear = 10 ** (snr_dB / 10)
    noise_power = signal_power / snr_linear

    # 生成具有所需信噪比的随机噪声
    noise = np.random.normal(scale=np.sqrt(noise_power), size=signal.shape) + \
            1j * np.random.normal(scale=np.sqrt(noise_power), size=signal.shape)

    # 将噪声添加到信号中
    noisy_signal = signal + noise

    return noisy_signal


dim = 2
N1 = 128
N2 = 128

max_J = 50

# Generate FID signals
J = np.ones([dim, 1]) * max_J  # Random number of harmonics

ph = np.random.uniform(0.0, 2 * np.pi, size=(dim, max_J))  # Random phase  # TODO
A = np.random.uniform(0.05, 1.0, size=(dim, max_J))  # Random amplitude
w = np.random.uniform(0.01, 0.99, size=(dim, max_J))  # Random frequency
sgm = np.random.uniform(10, 179.2, size=(dim, max_J))  # Random relaxation time

lor1 = 2 * A[..., None] * 1 / sgm[..., None] / (np.sqrt(2 * np.pi) / (sgm[..., None]) ** 2 + (
        2 * np.pi * w[..., None] - np.linspace(0, 1, N1) * 2 * np.pi) ** 2)
lor2 = 2 * A[..., None] * 1 / sgm[..., None] / (np.sqrt(2 * np.pi) / (sgm[..., None]) ** 2 + (
        2 * np.pi * w[..., None] - np.linspace(0, 1, N2) * 2 * np.pi) ** 2)
temp_l1l2 = np.matmul(lor1[0][:, :, np.newaxis], lor2[1][:, np.newaxis])
l1l2 = np.sum(temp_l1l2, axis=0)

plt.contour(l1l2, levels=40)
plt.show()
