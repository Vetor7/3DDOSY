from scipy import interpolate

from dataset import gen_3D_DOSY
import matplotlib.pyplot as plt
from config import parse_args
import torch
import numpy as np
from scipy.signal import find_peaks
from model import SimpleModel
import scipy.io as scio

# 解析参数
args = parse_args()
device = 'cpu'
# 初始化数据集和模型
dataset = gen_3D_DOSY(args)
model = SimpleModel(args.signal_dim, args.label_size).to(device)
model.load_state_dict(torch.load('result/best_model.pth'))  # 加载最佳模型权重
model.eval()  # 切换到评估模式

NMRdata = scio.loadmat('QGC_net_input.mat')

# 预计算 linspace
new_b = np.linspace(0, np.max(NMRdata['b'][0]), 6)

# 转换 S 为 tensor，并移动到设备
S_tensor = torch.tensor(NMRdata['S'], dtype=torch.float32, device=device)

# 对 S_tensor 进行阈值处理
S_tensor[:, S_tensor[0] < 0.005] = 1e-6

# 获取 Diff, f1, f2 的形状
Diff, f1, f2 = S_tensor.shape

# Reshape S_tensor 为 (Diff, f1*f2)，然后转置
S_tensor = S_tensor.reshape(Diff, -1).transpose(0, 1)

# 生成插值函数并批量计算
NmrDatai = np.zeros([f1 * f2, 6])
b = NMRdata['b'][0]  # 重新获取 b 数据
for i in range(f1 * f2):
    f = interpolate.interp1d(b, S_tensor[i, :], fill_value='extrapolate')
    NmrDatai[i] = f(new_b)

# 将插值后的数据转为 tensor，并调整形状
test_input = torch.tensor(NmrDatai.T, dtype=torch.float32, device=device).reshape(6, f1, f2)

# 获取 Cif
Cif = NMRdata['S'][0, :, :]
# 进行预测

with torch.no_grad():
    outputs = model(test_input.unsqueeze(0))
    predicted_label = outputs.squeeze(0).cpu().numpy()  # 将预测结果转换为 NumPy 数组

frequency1 = np.linspace(0, f1, f1)  # 第一个频率维度
frequency2 = np.linspace(0, f2, f2)  # 第二个频率维度
diffusion_coefficient = np.linspace(0, args.label_size, args.label_size)  # 扩散系数维度
F1, F2, D = np.meshgrid(frequency1, frequency2, diffusion_coefficient, indexing='ij')

result = predicted_label * Cif[np.newaxis, ...] / 3
result = result.transpose(1, 2, 0)

summed_result = np.sum(result, axis=(0, 1))

peaks, _ = find_peaks(summed_result, 5, distance=1)
# 创建一个三维网格
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
result[result < np.tile(((np.max(result, axis=2)) * 0.7)[..., np.newaxis], [1, args.label_size])] = 0
for peak in peaks:  # 遍历每个峰值索引
    for offset in range(-2, 3):
        if offset != 0:  # 跳过自身
            idx = peak + offset
            if 0 <= idx < result.shape[2]:  # 检查是否在有效范围内
                result[:, :, peak] += result[:, :, idx]
result[result < np.max(result, axis=(0, 1), keepdims=True) * 0.05] = 0

ax.set_zlim(0, 70)
for i in peaks:
    # 确保 i 在 valid range 内
    if i < D.shape[2] and i < result.shape[2]:  # 确保不超出范围
        Z = D[:, :, i] + result[:, :, i]
        contour = ax.contour(F1[:, :, i], F2[:, :, i], Z, levels=40, zdir='z', offset=i, cmap='viridis')
        plt.clabel(contour, inline=True, fontsize=8)

ax.set_xlabel('F1')
ax.set_ylabel('F2')
ax.set_zlabel('Diffusion Coefficient')


fig4, ax4 = plt.subplots(1, len(peaks), figsize=(8, len(peaks) * 4))

for idx, i in enumerate(peaks):
    Z = result[:, :, i]
    contour = ax4[idx].contour(F1[:, :, i], F2[:, :, i], Z, levels=40)  # 使用 ax3[idx]
    ax4[idx].set_xlabel('F1')  # 设置当前子图的 X 轴标签
    ax4[idx].set_ylabel('F2')  # 设置当前子图的 Y 轴标签

plt.show()
