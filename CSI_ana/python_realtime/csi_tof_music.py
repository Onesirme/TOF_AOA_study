# openwifi side info receive and display program
# Features:
# 1. CSI Phase (Wrapped)
# 2. CSI Phase (Unwrapped)
# 3. MUSIC Algorithm Distance Estimation (Real-time)

import socket
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks


# ==========================================
# 核心算法类：CSI处理与MUSIC测距
# ==========================================
class CSIProcessor:
    def __init__(self):
        # 802.11n 20MHz 参数
        self.N_FFT = 64
        self.BW = 20e6
        self.delta_f = self.BW / self.N_FFT  # 312.5 kHz
        self.c = 3e8  # 光速

        # 有效子载波定义 (OpenWiFi 标准)
        self.LTS_FREQ = np.array([
            0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1,
            1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1,
            0, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1,
            1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0
        ], dtype=complex)

        self.valid_indices = np.where(self.LTS_FREQ != 0)[0]
        self.full_indices = np.arange(64)

        # MUSIC 算法预计算参数
        self.L = 32  # 平滑窗口长度 (N/2)
        self.M = self.N_FFT - self.L + 1  # 子数组数量
        self.num_signals = 3  # 假设信号源数量 (直射+反射)

        # 扫描距离范围 (0 到 15米)
        self.scan_dist = np.linspace(0, 15, 301)  # 0.05m 分辨率
        self.scan_tau = self.scan_dist / self.c

        # 预计算导向矩阵 (Steering Matrix) 以加速实时运算
        # shape: [L, num_scan_points]
        # 频率向量 (0 到 L-1) * delta_f
        freq_vec = np.arange(self.L) * self.delta_f
        # A = exp(-j * 2 * pi * f * tau)
        self.steering_matrix = np.exp(-1j * 2 * np.pi * freq_vec[:, None] * self.scan_tau[None, :])

    def interpolate_csi(self, csi_raw):
        """
        使用 PCHIP 算法填充 DC 和 Guard Bands
        对应 MATLAB: interp1(..., 'pchip', 'extrap')
        """
        # 提取有效数据
        valid_data = csi_raw[self.valid_indices]

        # 分别对实部和虚部进行 PCHIP 插值
        # scipy 的 PchipInterpolator 默认支持外推
        pchip_real = PchipInterpolator(self.valid_indices, np.real(valid_data))
        pchip_imag = PchipInterpolator(self.valid_indices, np.imag(valid_data))

        filled_real = pchip_real(self.full_indices)
        filled_imag = pchip_imag(self.full_indices)

        return filled_real + 1j * filled_imag

    def run_music(self, csi_filled):
        """
        运行 MUSIC 算法估计距离
        csi_filled: 已经插值填满的 64 点 CSI 数据
        """
        # 1. 频率平滑 (Spatial/Frequency Smoothing)
        # 构建协方差矩阵 R
        # 向量化构建：使用 stride_tricks 或者简单的列表推导
        # 这里为了清晰使用循环，Python 3.10+ 中这种小循环还算快，也可以优化
        sub_arrays = np.array([csi_filled[i: i + self.L] for i in range(self.M)])
        # R = E[x * x^H]
        # transform to matrix multiplication: R = (X.T @ X.conj()) / M is not quite right for smoothing
        # Standard smoothing: sum(x_i * x_i^H)
        R = np.zeros((self.L, self.L), dtype=complex)
        for x in sub_arrays:
            x = x.reshape(-1, 1)
            R += x @ x.conj().T
        R /= self.M

        # 2. 特征值分解
        # eigh 用于 Hermitian 矩阵，比 eig 更快更准
        eigvals, eigvecs = np.linalg.eigh(R)

        # 3. 分离噪声子空间
        # eigh 返回的特征值默认是从小到大排序的
        # 取最小的 (L - num_signals) 个特征向量构成噪声空间
        noise_subspace = eigvecs[:, :self.L - self.num_signals]

        # 4. 计算伪谱 (向量化计算)
        # P = 1 / (a^H * Un * Un^H * a)
        # 令 Qn = Un * Un^H (投影矩阵)
        Qn = noise_subspace @ noise_subspace.conj().T

        # 分母 = diag(A^H * Qn * A)
        # 为了速度，只计算对角线部分: sum(conj(A) * (Qn @ A), axis=0)
        temp = Qn @ self.steering_matrix
        denom = np.sum(np.conj(self.steering_matrix) * temp, axis=0)

        spectrum = 1.0 / np.abs(denom)

        # 归一化并转 dB
        spectrum_db = 10 * np.log10(spectrum / np.max(spectrum))

        return self.scan_dist, spectrum_db


# ==========================================
# 辅助类：LTS 检测
# ==========================================
class LTSPreambles:
    def __init__(self):
        # 802.11a/g/n LTS
        self.LTS_FREQ = np.array([
            0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1,
            1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1,
            0, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1,
            1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0
        ], dtype=complex)
        self.valid_subcarriers = np.where(self.LTS_FREQ != 0)[0]
        self.LTS_TIME = np.fft.ifft(np.fft.ifftshift(self.LTS_FREQ))
        LTS_TIME_GI = self.LTS_TIME[-32:]
        self.LTS_WITH_GI = np.concatenate([LTS_TIME_GI, self.LTS_TIME, self.LTS_TIME])
        self.LTS_TEMPLATE = self.LTS_WITH_GI
        self.template_norm = self.LTS_TEMPLATE / np.sqrt(np.sum(np.abs(self.LTS_TEMPLATE) ** 2))

    def detect_lts(self, iq_signal, threshold=0.7):
        if len(iq_signal) < len(self.LTS_TEMPLATE): return [], np.array([])
        iq_signal = np.asarray(iq_signal).flatten()

        # 快速相关运算
        # 这里为了演示简单保留原逻辑，实际可以用 scipy.signal.correlate 加速
        corr_length = len(iq_signal) - len(self.LTS_TEMPLATE) + 1
        correlations = np.zeros(corr_length)

        # 简单的滑动窗口归一化相关
        window_len = len(self.LTS_TEMPLATE)
        # 预计算能量以加速? 暂时保持原样以确保稳定性
        for i in range(corr_length):
            window = iq_signal[i:i + window_len]
            energy = np.sum(np.abs(window) ** 2)
            if energy == 0: continue
            window_norm = window / np.sqrt(energy)
            correlations[i] = np.abs(np.sum(window_norm * np.conj(self.template_norm)))

        lts_positions = []
        if np.max(correlations) > threshold:
            peak_thresh = threshold * np.max(correlations)
            # 简单的峰值查找
            peaks, _ = find_peaks(correlations, height=peak_thresh, distance=50)
            lts_positions = peaks.tolist()

        return lts_positions, correlations

    def compute_csi(self, iq_signal, lts_position):
        if lts_position + 160 + 63 >= len(iq_signal): return None
        # 提取LTF
        ltf1 = iq_signal[lts_position + 32:lts_position + 96]
        ltf2 = iq_signal[lts_position + 96:lts_position + 160]
        # FFT
        fft1 = np.fft.fftshift(np.fft.fft(ltf1))
        fft2 = np.fft.fftshift(np.fft.fft(ltf2))
        avg_fft = (fft1 + fft2) / 2
        # 计算CSI
        csi = np.zeros(64, dtype=complex)
        np.divide(avg_fft, self.LTS_FREQ, out=csi, where=self.LTS_FREQ != 0)
        return csi


# ==========================================
# 实时分析与绘图类
# ==========================================
class RealTimeIQAnalyzer:
    def __init__(self):
        self.lts_detector = LTSPreambles()
        self.csi_processor = CSIProcessor()  # 引入处理器
        self.setup_plots()
        self.frame_count = 0
        self.last_update_time = time.time()

    def setup_plots(self):
        plt.ion()
        # 增加高度以容纳第三个图
        self.fig = plt.figure(figsize=(14, 12))

        # 三个子图
        self.ax_phase_wrap = self.fig.add_subplot(3, 1, 1)
        self.ax_phase_unwrap = self.fig.add_subplot(3, 1, 2)
        self.ax_music = self.fig.add_subplot(3, 1, 3)

        self.lines = {}

        # 1. Wrapped Phase
        self.lines['rx0_wrap'], = self.ax_phase_wrap.plot([], [], 'r.', label='RX0')
        self.lines['rx1_wrap'], = self.ax_phase_wrap.plot([], [], 'b.', label='RX1')
        self.lines['rx0_fit_wrap'], = self.ax_phase_wrap.plot([], [], 'r-', alpha=0.3)  # 显示PCHIP结果

        # 2. Unwrapped Phase
        self.lines['rx0_unwrap'], = self.ax_phase_unwrap.plot([], [], 'r-', label='RX0')
        self.lines['rx1_unwrap'], = self.ax_phase_unwrap.plot([], [], 'b-', label='RX1')
        self.text_slope = self.ax_phase_unwrap.text(0.02, 0.9, '', transform=self.ax_phase_unwrap.transAxes,
                                                    bbox=dict(facecolor='white', alpha=0.8))

        # 3. MUSIC Spectrum (新增)
        self.lines['rx0_music'], = self.ax_music.plot([], [], 'r-', linewidth=2, label='RX0 Dist')
        self.lines['rx1_music'], = self.ax_music.plot([], [], 'b-', linewidth=2, label='RX1 Dist')
        self.lines['rx0_peak'], = self.ax_music.plot([], [], 'rv')  # 峰值标记
        self.lines['rx1_peak'], = self.ax_music.plot([], [], 'bv')
        self.text_dist = self.ax_music.text(0.02, 0.9, '', transform=self.ax_music.transAxes,
                                            bbox=dict(facecolor='wheat', alpha=0.8))

        self.setup_appearance()

    def setup_appearance(self):
        # Wrapped
        self.ax_phase_wrap.set_title('CSI Phase (Wrapped) & PCHIP Interpolation')
        self.ax_phase_wrap.set_ylim(-180, 180)
        self.ax_phase_wrap.set_ylabel('Phase (deg)')
        self.ax_phase_wrap.grid(True, alpha=0.3)
        self.ax_phase_wrap.legend(loc='upper right')

        # Unwrapped
        self.ax_phase_unwrap.set_title('CSI Phase (Unwrapped)')
        self.ax_phase_unwrap.set_ylabel('Phase (deg)')
        self.ax_phase_unwrap.grid(True, alpha=0.3)

        # MUSIC
        self.ax_music.set_title('MUSIC Algorithm Distance Estimation (Super-Resolution)')
        self.ax_music.set_xlabel('Distance (meters)')
        self.ax_music.set_ylabel('Spectrum (dB)')
        self.ax_music.set_xlim(0, 15)  # 显示 0-15米
        self.ax_music.set_ylim(-20, 0.5)
        self.ax_music.grid(True, which='both', alpha=0.3)
        self.ax_music.legend(loc='upper right')

    def process_and_update(self, csi0, csi1):
        if csi0 is None or csi1 is None: return

        # === 1. PCHIP 插值填充 (针对 MUSIC) ===
        # 这一步非常关键，MUSIC 需要完整的 64 个点
        csi0_filled = self.csi_processor.interpolate_csi(csi0)
        csi1_filled = self.csi_processor.interpolate_csi(csi1)

        valid_idx = self.lts_detector.valid_subcarriers

        # === 2. 准备绘图数据 (Phase) ===
        # Wrapped
        ph0_wrap = np.angle(csi0[valid_idx]) * 180 / np.pi
        ph1_wrap = np.angle(csi1[valid_idx]) * 180 / np.pi
        # 展示拟合后的相位曲线，看看PCHIP效果
        ph0_filled_wrap = np.angle(csi0_filled) * 180 / np.pi

        # Unwrapped (只对有效子载波做解缠，或者对填充后的做解缠)
        # 为了斜率计算准确，通常对有效子载波做解缠
        ph0_unwrap = np.unwrap(np.angle(csi0[valid_idx])) * 180 / np.pi
        ph1_unwrap = np.unwrap(np.angle(csi1[valid_idx])) * 180 / np.pi

        # 计算斜率
        slope0 = np.polyfit(valid_idx, ph0_unwrap, 1)[0]
        slope1 = np.polyfit(valid_idx, ph1_unwrap, 1)[0]

        # === 3. 运行 MUSIC 算法 ===
        dist_axis, spec0 = self.csi_processor.run_music(csi0_filled)
        _, spec1 = self.csi_processor.run_music(csi1_filled)

        # 找峰值距离
        idx0 = np.argmax(spec0)
        dist0 = dist_axis[idx0]
        idx1 = np.argmax(spec1)
        dist1 = dist_axis[idx1]

        # === 4. 更新图表 ===
        # Wrapped
        self.lines['rx0_wrap'].set_data(valid_idx, ph0_wrap)
        self.lines['rx1_wrap'].set_data(valid_idx, ph1_wrap)
        self.lines['rx0_fit_wrap'].set_data(np.arange(64), ph0_filled_wrap)  # 画出填充后的全貌

        # Unwrapped
        self.lines['rx0_unwrap'].set_data(valid_idx, ph0_unwrap)
        self.lines['rx1_unwrap'].set_data(valid_idx, ph1_unwrap)
        self.text_slope.set_text(f'Slope RX0: {slope0:.2f}\nSlope RX1: {slope1:.2f}')
        self.ax_phase_unwrap.relim()
        self.ax_phase_unwrap.autoscale_view()

        # MUSIC Spectrum
        self.lines['rx0_music'].set_data(dist_axis, spec0)
        self.lines['rx1_music'].set_data(dist_axis, spec1)
        self.lines['rx0_peak'].set_data([dist0], [spec0[idx0]])  # 注意 set_data 接受序列
        self.lines['rx1_peak'].set_data([dist1], [spec1[idx1]])

        self.text_dist.set_text(f'Est Dist RX0: {dist0:.2f} m\nEst Dist RX1: {dist1:.2f} m')

        # FPS update
        now = time.time()
        fps = 1.0 / (now - self.last_update_time) if now > self.last_update_time else 0
        self.last_update_time = now
        self.fig.suptitle(f'Real-time CSI Analysis - FPS: {fps:.1f}')

        plt.pause(0.001)


# ==========================================
# 主程序逻辑
# ==========================================
def parse_iq(iq, iq_len):
    # 标准 OpenWifi 解析逻辑
    num_dma = 1 + iq_len
    num_int16 = num_dma * 4
    num_trans = len(iq) // num_int16
    iq = iq[:num_trans * num_int16].reshape([num_trans, num_int16])

    timestamp = iq[:, 0] + (iq[:, 1] << 16) + (iq[:, 2].astype(np.uint64) << 32) + (iq[:, 3].astype(np.uint64) << 48)
    iq0 = (iq[:, 4::4].astype(np.int16) + 1j * iq[:, 5::4].astype(np.int16)).flatten()
    iq1 = (iq[:, 6::4].astype(np.int16) + 1j * iq[:, 7::4].astype(np.int16)).flatten()
    return timestamp, iq0, iq1


def main():
    UDP_IP = "192.168.2.177"
    UDP_PORT = 4000
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)

    analyzer = RealTimeIQAnalyzer()
    print("Starting MUSIC Algorithm Real-time Display...")

    try:
        while True:
            data, _ = sock.recvfrom(32768)  # 增大buffer防止丢包
            if len(data) == 0: continue

            iq_u16 = np.frombuffer(data, dtype='uint16')
            _, iq0_stream, iq1_stream = parse_iq(iq_u16, 4095)

            # 为了演示，只取流中的一段进行检测（实际应用可能需要更复杂的Buffer管理）
            # 假设数据流足够长
            if len(iq0_stream) > 3000:
                segment0 = iq0_stream[0:3000]
                segment1 = iq1_stream[0:3000]

                # 检测 LTS
                lts_pos0, _ = analyzer.lts_detector.detect_lts(segment0)
                lts_pos1, _ = analyzer.lts_detector.detect_lts(segment1)

                # 寻找匹配的帧 (简单起见取第一个)
                if lts_pos0 and lts_pos1:
                    # 计算 CSI
                    csi0 = analyzer.lts_detector.compute_csi(segment0, lts_pos0[0])
                    csi1 = analyzer.lts_detector.compute_csi(segment1, lts_pos1[0])

                    # 更新显示
                    analyzer.process_and_update(csi0, csi1)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        sock.close()
        plt.close('all')


if __name__ == "__main__":
    main()