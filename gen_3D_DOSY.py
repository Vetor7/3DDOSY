import random
import scipy
from torch.utils.data import Dataset
import numpy as np


# np.random.seed(42)
class gen_3D_DOSY(Dataset):
    def __init__(self, args):
        super(gen_3D_DOSY, self).__init__()
        self.args = args

    def __len__(self):
        return self.args.num_samples

    def add_noise(self, input, snr):
        std_noise = np.max(input) / (2 * snr)
        noise_real = np.random.normal(loc=0.0, scale=std_noise, size=(self.args.Num_dim1, self.args.Num_dim2))
        noise_imag = np.random.normal(loc=0.0, scale=std_noise, size=(self.args.Num_dim1, self.args.Num_dim2))
        noise_out = noise_real + 1j * noise_imag
        return noise_out + input

    def get_DF(self, D, Sigma):
        d = np.linspace(0, self.args.max_D, self.args.label_size)
        D = D[:, np.newaxis]
        sqrt_2pi = np.sqrt(2 * np.pi)
        coef = 1 / (sqrt_2pi * Sigma)
        powercoef = -1 / (2 * Sigma ** 2)
        DF = coef * np.exp(powercoef * (d - D) ** 2)
        DF = DF / DF.max(axis=1, keepdims=True)
        return DF.T

    def get_D(self):
        while True:
            D = np.sort(np.random.rand(self.args.num_D) * self.args.max_D + self.args.base_D)
            if np.all(np.diff(D) > self.args.min_sep):
                return D

    def normalize_label(self, label):
        return label / np.sum(label, axis=0, keepdims=True)

    def get_label(self, DF, Ci_f):
        return np.sum(DF * Ci_f, axis=1)

    def get_Skernel(self):
        D_lab = np.linspace(0, self.args.max_D, self.args.label_size)
        b = np.linspace(0, self.args.max_b, self.args.signal_dim)
        return np.exp(-b[:, None] * D_lab)

    def get_signal(self, S_kernel, label):
        """Generate simulated signal by applying kernel to labels."""
        return np.sum(S_kernel * label, axis=1)

    def __gen_signal__(self, idx):
        Ci_f = np.zeros([self.args.num_D, self.args.Num_dim1, self.args.Num_dim2])
        for i in range(self.args.num_D):
            Ci_f[i] = self.get_Ci_f()  # (3, N1, N2)
        D = self.get_D()  # (3,)
        DF = self.get_DF(D, self.args.sig)  # (140,3)
        label = self.normalize_label(
            self.get_label(DF[..., np.newaxis, np.newaxis], Ci_f[np.newaxis, ...]))  # (140, N1, N2)
        S_kernel = self.get_Skernel()  # (6, 140)
        S = self.get_signal(S_kernel[..., np.newaxis, np.newaxis], label[np.newaxis, ...])  # (6, N1, N2)
        return S.astype('float32'), label.astype('float32'), Ci_f.astype('float32')

    def get_Ci_f(self):
        # Define simulation parameters
        dim = 2

        N1, N2 = self.args.Num_dim1, self.args.Num_dim2

        # Generate FID signals
        max_J = self.args.max_J
        J = np.random.randint(1, max_J + 1, size=(dim, 1))  # Random number of harmonics

        mask = np.zeros((dim, max_J), dtype=int)  # Create mask array
        for i in range(dim):
            mask[i, J[i, 0] - 1:] = 1  # Correctly index using J

        ph = np.random.uniform(0.0, 2 * np.pi, size=(dim, max_J))  # Random phase
        A = np.random.uniform(0.05, 1.0, size=(dim, max_J)) * mask  # Random amplitude with mask applied
        w = np.random.uniform(0.01, 0.99, size=(dim, max_J))  # Random frequency
        sgm = np.random.uniform(10, 179.2, size=(dim, max_J))  # Random relaxation time

        # Time axes
        t1 = np.arange(N1)
        t2 = np.arange(N2)

        # Pre-compute exponentials
        exp_ph = np.exp(1j * ph[..., None])  # Shape (dim, max_J, 1)
        exp_decay_t1 = np.exp(-t1 / sgm[..., None])  # Shape (dim, max_J, N1)
        exp_decay_t2 = np.exp(-t2 / sgm[..., None])  # Shape (dim, max_J, N2)
        exp_w_t1 = np.exp(1j * 2 * np.pi * w[..., None] * t1)  # Shape (dim, max_J, N1)
        exp_w_t2 = np.exp(1j * 2 * np.pi * w[..., None] * t2)  # Shape (dim, max_J, N2)

        # Calculate signals x1 and x2
        x1 = A[..., None] * exp_ph * exp_decay_t1 * exp_w_t1  # Shape (dim, max_J, N1)
        x2 = A[..., None] * exp_ph * exp_decay_t2 * exp_w_t2  # Shape (dim, max_J, N2)

        # Matrix multiplication and sum to get clean_xn
        xn_unit = np.matmul(x1[0][:, :, None], x2[1][:, None])  # Shape (max_J, N1, N2)
        clean_xn = np.sum(xn_unit, axis=0)  # Sum over harmonics, shape (N1, N2)

        # Add noise to FID signals
        noise_scale = 0
        noise = np.random.normal(loc=0.0, scale=noise_scale, size=(N1, N2))
        xx = noise + clean_xn
        xx = np.fft.fft(xx, axis=1)
        # xx_noise = self.add_noise(input=xx, snr=np.random.uniform(15, 60))
        Ci_f = np.abs(np.fft.fft(xx, axis=0))
        Ci_f = Ci_f / Ci_f.max()
        return Ci_f


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from config import parse_args
    from scipy.signal import find_peaks

    args = parse_args()
    dataset = gen_3D_DOSY(args)
    S, label, Cif = dataset.__gen_signal__(0)
    result = label * Cif.sum(0, keepdims=True)
    result = result.transpose(1, 2, 0)
    # Visualization
    x = np.linspace(0, args.Num_dim2, args.Num_dim2)
    y = np.linspace(0, args.Num_dim1, args.Num_dim1)
    X, Y = np.meshgrid(x, y)

    S = S.transpose(1, 2, 0).reshape(-1, 6)
    fig = plt.figure(1)
    plt.plot(S.T)
    plt.figure(2)
    contour = plt.contour(X, Y, result.sum(-1), cmap='viridis', levels=20)  # Use contourf for filled contours
    plt.colorbar(contour)  # Add a color bar to indicate magnitude
    
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Contour Plot of Ci_f Magnitude')

    frequency1 = np.linspace(0, args.Num_dim2, args.Num_dim2)  # 第一个频率维度
    frequency2 = np.linspace(0, args.Num_dim1, args.Num_dim1)  # 第二个频率维度
    diffusion_coefficient = np.linspace(0, args.label_size, args.label_size)  # 扩散系数维度

    # 创建一个三维网格
    F1, F2, D = np.meshgrid(frequency1, frequency2, diffusion_coefficient, indexing='ij')

    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')

    # 对前两个维度求和
    summed_result = np.sum(result, axis=(0, 1))

    # 找到峰值的位置
    peaks, _ = find_peaks(summed_result)

    for i in peaks:
        # 确保 i 在 valid range 内
        if i < D.shape[2] and i < result.shape[2]:  # 确保不超出范围
            Z = D[:, :, i] + result[:, :, i]
            contour = ax.contour(F1[:, :, i], F2[:, :, i], Z, levels=20, zdir='z', offset=i, cmap='viridis')
            plt.clabel(contour, inline=True, fontsize=8)

    ax.set_xlabel('Frequency 1')
    ax.set_ylabel('Frequency 2')
    ax.set_zlabel('Diffusion Coefficient')

    plt.show()
