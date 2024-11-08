from dataset import gen_3D_DOSY
import matplotlib.pyplot as plt
from config import parse_args
import torch
import numpy as np
from scipy.signal import find_peaks
from model import SimpleModel

# 解析参数
args = parse_args()
device = 'cpu'

# 初始化数据集和模型
dataset = gen_3D_DOSY(args)
model = SimpleModel(args.signal_dim, args.label_size).to(device)
model.load_state_dict(torch.load('result/best_model.pth'))
model.eval()

# 生成测试数据
S, label, Cif = dataset.__gen_signal__(0)
S_tensor = torch.tensor(S, dtype=torch.float32).to(device)

# 预测结果
with torch.no_grad():
    outputs = model(S_tensor.unsqueeze(0))
    predicted_label = outputs.squeeze(0).cpu().numpy()

# 准备可视化
def prepare_result_matrix(label, Cif, norm_factor=1):
    result = label * Cif.sum(0, keepdims=True) / norm_factor
    result = result.transpose(1, 2, 0)
    result /= np.max(result, keepdims=True)
    return result

# 生成频率和扩散系数坐标
frequency1 = np.linspace(0, args.Num_dim2, args.Num_dim2)
frequency2 = np.linspace(0, args.Num_dim1, args.Num_dim1)
diffusion_coefficient = np.linspace(0, args.label_size, args.label_size)
F1, F2, D = np.meshgrid(frequency1, frequency2, diffusion_coefficient, indexing='ij')

# 找到峰值位置
def find_peaks_result(result, threshold=0.05):
    summed_result = np.sum(result, axis=(0, 1))
    peaks, _ = find_peaks(summed_result, height=5, distance=1)

    for peak in peaks:  # 遍历每个峰值索引
        for offset in range(-2, 3):
            if offset != 0:  # 跳过自身
                idx = peak + offset
                if 0 <= idx < result.shape[2]:  # 检查是否在有效范围内
                    result[:, :, peak] += result[:, :, idx]

    result[result < np.tile(((np.max(result, axis=2)) * 0.7)[..., np.newaxis], [1, args.label_size])] = 0
    result[result < np.max(result, axis=(0, 1), keepdims=True) * threshold] = 0

    return peaks

# 绘制3D图和子图
def plot_3D_contours(fig, ax, result, peaks, contour_levels=40):
    for i in peaks:
        if i < D.shape[2] and i < result.shape[2]:
            Z = D[:, :, i] + result[:, :, i]
            contour = ax.contour(F1[:, :, i], F2[:, :, i], Z, levels=contour_levels, zdir='z', offset=i, cmap='viridis')
            plt.clabel(contour, inline=True, fontsize=8)

def plot_2D_slices(fig, ax_array, result, peaks, F1, F2):
    for idx, i in enumerate(peaks):
        Z = result[:, :, i]
        ax_array[idx].contour(F1[:, :, i], F2[:, :, i], Z, levels=40)
        ax_array[idx].set_xlabel('F1')
        ax_array[idx].set_ylabel('F2')
        ax_array[idx].set_title(f"Slice at Diffusion Coefficient: {diffusion_coefficient[i]/5:.2f}")

# 可视化真实标签结果
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('F1')
ax.set_ylabel('F2')
ax.set_zlabel('Diffusion Coefficient')
ax.set_zlim(0, 70)

real_result = prepare_result_matrix(label, Cif)
peaks = find_peaks_result(real_result)
plot_3D_contours(fig, ax, real_result, peaks)

fig3, ax3 = plt.subplots(1, len(peaks), figsize=(len(peaks)*8, 4))
plot_2D_slices(fig3, ax3, real_result, peaks, F1, F2)

# 可视化预测结果
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('F1')
ax.set_ylabel('F2')
ax.set_zlabel('Diffusion Coefficient')
ax.set_zlim(0, 70)

predicted_result = prepare_result_matrix(predicted_label, Cif, norm_factor=3)
peaks = find_peaks_result(predicted_result)
plot_3D_contours(fig, ax, predicted_result, peaks)

fig4, ax4 = plt.subplots(1, len(peaks), figsize=(8*len(peaks), 4))
plot_2D_slices(fig4, ax4, predicted_result, peaks, F1, F2)

plt.show()
