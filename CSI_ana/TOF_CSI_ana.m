clear; close all; clc;

% 802.11a/g/n Long Training Sequence (频域)
LTS_FREQ = [0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, ...
            1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, ...
            0, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, ...
            1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0];

% 生成时域LTS模板
LTS_TIME = ifft(ifftshift(LTS_FREQ));  % 64样本

LTS_TIME_GI = LTS_TIME(end-31:end);  % 提取最后32个样本作为GI
LTS_WITH_GI = [LTS_TIME_GI, LTS_TIME, LTS_TIME];  % 水平连接，160样本(GI2 + 2×LTS)
LTS_TEMPLATE = LTS_WITH_GI;

%% 读取数据
% RX1_RX2_1     -70度    RX1_RX2_2   -60度   RX1_RX2_2    -70度
% RX2_RX1_1     -5度    RX2_RX1_2   -6度   RX2_RX1_3    -7度
% RXMID1        2度    RXMID2      7度   RXMID3       10度

filename = 'move.txt';
fprintf('Reading data from %s...\n', filename);

%% 读取文件头信息
fid = fopen(filename, 'r');
frame_count = 0;
samples_per_frame = 0;

% 解析文件头
while true
    line = fgetl(fid);
    if ~ischar(line) || isempty(line)
        break;
    end
    
    if startsWith(line, '# Total frames:')
        frame_count = sscanf(line, '# Total frames: %d');
    elseif startsWith(line, '# Samples per frame:')
        samples_per_frame = sscanf(line, '# Samples per frame: %d');
    elseif ~startsWith(line, '#')
        break;
    end
end

fclose(fid);

if frame_count == 0 || samples_per_frame == 0
    error('Could not read file header information');
end

fprintf('Found %d frames, %d samples per frame\n', frame_count, samples_per_frame);

%% 读取数据
data = readmatrix(filename, 'FileType', 'text', 'NumHeaderLines', 6);

%% 提取数据
frames = cell(frame_count, 1);

for i = 1:frame_count
    frame_data = data(data(:,1) == i-1, :);  % Frame indices start from 0
    frames{i} = frame_data;
end
% CSI0和CSI1的值
Csi0 = zeros(frame_count,64)+1i*ones(frame_count,64);
Csi1 = zeros(frame_count,64)+1i*ones(frame_count,64);


fprintf('Data loaded successfully!\n');

%% 基本参数(wifi通道)
threshold = 0.6;  % LTS检测阈值

% 存储每一帧的CSI相位差中值和角度
phase_diff_medians = zeros(1, frame_count);
angles = zeros(1, frame_count);  % 存储计算的角度
valid_frames = false(1, frame_count);  % 标记哪些帧有有效的LTS检测

% WiFi 通道参数
fc = 2.412e9;  
c = physconst('light');      
lambda = c / fc;  % 波长
d = 0.06;   
t_delay0 = zeros(1, frame_count);
t_delay1 = zeros(1, frame_count);

% 找到有效子载波索引
valid_subcarriers = find(LTS_FREQ ~= 0);

fprintf('WiFi Channel 44 Parameters:\n');
fprintf('  Center Frequency: %.2f GHz\n', fc/1e9);
fprintf('  Wavelength: %.4f m\n', lambda);
fprintf('  Antenna Spacing: %.3f m\n', d);
fprintf('  d/lambda ratio: %.3f\n\n', d/lambda);

%% 对每一帧进行处理提取CSI
for frame_idx = 1:frame_count
    fprintf('\n=== Analyzing Frame %d ===\n', frame_idx);
    
    frame_data = frames{frame_idx};
    sample_idx = frame_data(:, 2);
    rx0_i = frame_data(:, 3);
    rx0_q = frame_data(:, 4);
    rx1_i = frame_data(:, 5);
    rx1_q = frame_data(:, 6);
    
    % 创建复数IQ信号
    iq0 = rx0_i + 1j * rx0_q;
    iq1 = rx1_i + 1j * rx1_q;
    
    % 检测LTS位置
    [lts_pos0, corr0] = detect_lts_matlab(iq0, LTS_TEMPLATE, threshold);
    [lts_pos1, corr1] = detect_lts_matlab(iq1, LTS_TEMPLATE, threshold);
    
    fprintf('RX0: %d LTS detected at positions: ', length(lts_pos0));
    if ~isempty(lts_pos0)
        fprintf('%d ', lts_pos0);
    end
    fprintf('\n');
    
    fprintf('RX1: %d LTS detected at positions: ', length(lts_pos1));
    if ~isempty(lts_pos1)
        fprintf('%d ', lts_pos1);
    end
    fprintf('\n');
    
    % 如果两个通道都检测到LTS，计算CSI相位差
    if ~isempty(lts_pos0) && ~isempty(lts_pos1)
        valid_frames(frame_idx) = true;
        
        % 使用第一个检测到的LTS位置
        lts_temp0 = lts_pos0(1);
        lts_temp1 = lts_pos1(1);
        
        % 提取LTF部分（两个LTS符号）
        LTF_IQ0 = iq0(lts_temp0+32:lts_temp0+32+63);
        LTF_IQ1 = iq1(lts_temp1+32:lts_temp1+32+63);
        LTF_IQ0_2 = iq0(lts_temp0+64+32:lts_temp0+64+32+63);
        LTF_IQ1_2 = iq1(lts_temp1+64+32:lts_temp1+64+32+63);
        
        % 计算FFT并平均
        fft_LTF_IQ0 = (fftshift(fft(LTF_IQ0)) + fftshift(fft(LTF_IQ0_2))) / 2;
        fft_LTF_IQ1 = (fftshift(fft(LTF_IQ1)) + fftshift(fft(LTF_IQ1_2))) / 2;
        
        % 计算CSI
        IQ0_csi = zeros(1, 64);
        IQ1_csi = zeros(1, 64);
        for i = 1:64
            if LTS_FREQ(i) ~= 0  % 只在有效子载波上计算CSI
                IQ0_csi(i) = fft_LTF_IQ0(i) / LTS_FREQ(i);
                IQ1_csi(i) = fft_LTF_IQ1(i) / LTS_FREQ(i);
            else
                IQ0_csi(i) = 0;  % 无效子载波设为0
                IQ1_csi(i) = 0;
            end
        end
        

        
        % 提取有效子载波的CSI
%         valid_IQ0_csi = IQ0_csi(valid_subcarriers);
%         valid_IQ1_csi = IQ1_csi(valid_subcarriers);
        
        Csi0(frame_idx,:) = IQ0_csi(:);
        Csi1(frame_idx,:) = IQ1_csi(:);

        % CSI解缠绕
%         unwrap_phase0 = unwrap(angle(valid_IQ0_csi));
%         unwrap_phase1 = unwrap(angle(valid_IQ1_csi));
        
%        %% 斜率计算frame_idx

%         p =  polyfit(valid_subcarriers*312500, unwrap_phase0*c, 1);
%         t_delay0(frame_idx)=p(1);
%         p =  polyfit(valid_subcarriers*312500, unwrap_phase1*c, 1);
%         t_delay1(frame_idx)=p(1);

    end
end
full_range = 1:64;

% 创建填充后的矩阵
Csi0_filled = zeros(frame_count, 64);
Csi1_filled = zeros(frame_count, 64);

for k = 1:frame_count
    % 1. 提取当前帧的有效数据 (原始 52 个点)
    y0_valid = Csi0(k, valid_subcarriers);
    y1_valid = Csi1(k, valid_subcarriers);
    
    % 2. 核心操作：插值 + 外推 ('extrap')
    % 使用 'pchip' (Piecewise Cubic Hermite Interpolating Polynomial)
    % 它在保持波形形状的同时，能够比较平稳地向外延伸，比 'linear' 平滑，比 'spline' 稳定
    Csi0_filled(k, :) = interp1(valid_subcarriers, y0_valid, full_range, 'pchip', 'extrap');
    Csi1_filled(k, :) = interp1(valid_subcarriers, y1_valid, full_range, 'pchip', 'extrap');
end

% 我们只在有效数据的范围内进行插值（填补中间的洞，如DC）
% 不要去拟合左右两边的 Guard Bands (1-6 和 60-64)，那样会引入误差
interp_range = min(valid_subcarriers) : max(valid_subcarriers);

% 创建填充后的CSI矩阵
Csi0_filled = Csi0;
Csi1_filled = Csi1;

% 2. 对每一帧进行插值处理
for k = 1:frame_count
    % 提取当前帧的有效数据
    y0_valid = Csi0(k, valid_subcarriers);
    y1_valid = Csi1(k, valid_subcarriers);
    
    % 对复数数据进行线性插值 ('linear' 或 'spline')
    % interp1 支持复数，会自动同时拟合实部和虚部
    Csi0_filled(k, interp_range) = interp1(valid_subcarriers, y0_valid, interp_range, 'linear');
    Csi1_filled(k, interp_range) = interp1(valid_subcarriers, y1_valid, interp_range, 'linear');
end


% 注意：确保上面的 Csi0_filled 是由 'pchip' + 'extrap' 生成的，
% 并且没有被后面的代码覆盖。

fprintf('\nRunning MUSIC Algorithm for ToF estimation...\n');

% --- 1. MUSIC 参数设置 ---
subcarrier_spacing = 312.5e3; % 312.5 kHz (20MHz / 64)
N = 64;                       % 子载波总数
bandwidth = subcarrier_spacing * N;
L = 32;                       % 平滑窗口长度 (通常取 N/2)
M = N - L + 1;                % 子数组数量
scan_dist = 0:0.05:20;        % 扫描距离范围：0 到 20米，步长 0.05米
scan_tau = scan_dist / c;     % 对应的时延 (ToF)

% 定义频率向量 (相对于中心频率的偏移，或者直接用 0:63)
% 为了计算方便，这里使用 0 到 63 的相对频率索引
freq_vec = (0:L-1)' * subcarrier_spacing; 

% 存储结果
music_spectrum_db = zeros(length(scan_tau), 1);

% --- 2. 选择一帧进行分析 (例如第 20 帧) ---
target_frame = 8;
csi_sample = Csi0_filled(target_frame, :).'; % 转置为列向量 (64x1)

% --- 3. 频率平滑 (Spatial/Frequency Smoothing) ---
% 这是处理单快拍(Single Snapshot)且信号相干(多径)的关键步骤
Rx = zeros(L, L);
for i = 1:M
    % 提取长度为 L 的子向量
    sub_vec = csi_sample(i : i+L-1);
    % 累加协方差矩阵
    Rx = Rx + (sub_vec * sub_vec');
end
Rx = Rx / M;

% --- 4. 特征值分解 ---
[EigenVecs, EigenVals] = eig(Rx);
% 提取特征值的对角线元素并排序
evals = diag(EigenVals);
[evals_sorted, idx] = sort(abs(evals), 'ascend');
evecs_sorted = EigenVecs(:, idx);

% --- 5. 分离噪声子空间 ---
% 假设主要路径数 (Signal Subspace Dimension)。
% 室内环境通常设为 1 (仅直射) 或 3 (直射+最强反射)
% 特征值最小的那部分对应噪声空间
num_signals = 3; 
noise_subspace = evecs_sorted(:, 1 : L - num_signals);

% --- 6. 构造伪谱 (Pseudospectrum) ---
fprintf('Scanning distances...\n');
for i = 1:length(scan_tau)
    tau = scan_tau(i);
    
    % 构造导向向量 (Steering Vector)
    % a(tau) = exp(-j * 2*pi * f * tau)
    a_vec = exp(-1j * 2 * pi * freq_vec * tau);
    
    % MUSIC 公式: P = 1 / (a' * En * En' * a)
    % 利用正交性：信号导向向量与噪声子空间正交，分母趋近0，谱峰极大
    denom = a_vec' * (noise_subspace * noise_subspace') * a_vec;
    music_spectrum_db(i) = 10 * log10(1 / abs(denom));
end

% --- 7. 归一化谱 ---
music_spectrum_db = music_spectrum_db - max(music_spectrum_db);

figure('Name', 'MUSIC ToF Estimation', 'Color', 'w');

plot(scan_dist, music_spectrum_db, 'b-', 'LineWidth', 2);
xlabel('Distance (meters)');
ylabel('Normalized Pseudospectrum (dB)');
title(sprintf('MUSIC Distance Estimation (Frame %d)', target_frame));
grid on;
xlim([0 15]); % 显示 0-15米范围

% 标记最大峰值
[pks, locs] = findpeaks(music_spectrum_db, 'NPeaks', 1, 'SortStr', 'descend');
if ~isempty(locs)
    peak_dist = scan_dist(locs(1));
    hold on;
    plot(peak_dist, pks(1), 'rv', 'MarkerFaceColor', 'r');
    text(peak_dist, pks(1)+1, sprintf('  %.2f m', peak_dist), 'Color', 'red', 'FontWeight', 'bold');
    fprintf('Estimated Main Path Distance: %.2f meters\n', peak_dist);
else
    fprintf('No distinct peak found.\n');
end

% 简单的 IFFT 结果对比 (画在小图里)
axes('Position',[.6 .6 .25 .25]);
box on;
cir_ifft = abs(ifft(csi_sample));
plot(0:length(cir_ifft)-1, cir_ifft, 'k-');
title('Standard IFFT (Low Res)');
xlabel('Index');


