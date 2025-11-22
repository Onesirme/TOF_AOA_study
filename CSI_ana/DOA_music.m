% MATLAB script to analyze WiFi signals with multiple DOA estimation algorithms
% Implements MUSIC, Capon (MVDR), ESPRIT, and phase difference methods

clear; close all; clc;

% 802.11a/g/n Long Training Sequence (频域)
LTS_FREQ = [0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, ...
            1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, ...
            0, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, ...
            1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0];

% 生成时域LTS模板
LTS_TIME = ifft(ifftshift(LTS_FREQ));
LTS_TIME_GI = LTS_TIME(end-31:end);
LTS_WITH_GI = [LTS_TIME_GI, LTS_TIME, LTS_TIME];
LTS_TEMPLATE = LTS_WITH_GI;

% WiFi 44通道参数
fc = 5.22e9;  % 中心频率
c = 3e8;      % 光速
lambda = c / fc;
d = 0.022;    % 天线间距 2.2 cm
M = 2;        % 天线数量

fprintf('=== WiFi Channel 44 Parameters ===\n');
fprintf('Center Frequency: %.2f GHz\n', fc/1e9);
fprintf('Wavelength: %.4f m\n', lambda);
fprintf('Antenna Spacing: %.3f m (%.2f λ)\n', d, d/lambda);
fprintf('Number of Antennas: %d\n', M);

% 检查天线间距与波长的关系
d_over_lambda = d / lambda;
fprintf('\n=== Antenna Spacing Analysis ===\n');
fprintf('d/λ = %.3f\n', d_over_lambda);
if d_over_lambda > 0.5
    fprintf('WARNING: d/λ > 0.5, phase ambiguity may occur!\n');
    fprintf('Maximum unambiguous angle range is limited.\n');
    fprintf('Phase wrapping at ±%.1f degrees\n', asind(lambda/(2*d)));
else
    fprintf('d/λ < 0.5, full ±90° range is unambiguous.\n');
end
fprintf('\n');

% 读取数据
filename = 'IQsignal.txt';
fprintf('Reading data from %s...\n', filename);

fid = fopen(filename, 'r');
frame_count = 0;
samples_per_frame = 0;

while true
    line = fgetl(fid);
    if ~ischar(line) || isempty(line), break; end
    if startsWith(line, '# Total frames:')
        frame_count = sscanf(line, '# Total frames: %d');
    elseif startsWith(line, '# Samples per frame:')
        samples_per_frame = sscanf(line, '# Samples per frame: %d');
    elseif ~startsWith(line, '#'), break; end
end
fclose(fid);

if frame_count == 0 || samples_per_frame == 0
    error('Could not read file header information');
end

fprintf('Found %d frames, %d samples per frame\n\n', frame_count, samples_per_frame);

data = readmatrix(filename, 'FileType', 'text', 'NumHeaderLines', 6);
frames = cell(frame_count, 1);

for i = 1:frame_count
    frame_data = data(data(:,1) == i-1, :);
    frames{i} = frame_data;
end

% 存储结果
results = struct();
results.phase_diff = zeros(1, frame_count);
results.music = zeros(1, frame_count);
results.capon = zeros(1, frame_count);
results.esprit = zeros(1, frame_count);
results.valid = false(1, frame_count);

threshold = 0.6;

% 处理每一帧
for frame_idx = 1:frame_count
    fprintf('=== Frame %d ===\n', frame_idx);
    
    frame_data = frames{frame_idx};
    rx0_i = frame_data(:, 3);
    rx0_q = frame_data(:, 4);
    rx1_i = -frame_data(:, 5);
    rx1_q = -frame_data(:, 6);
    
    iq0 = rx0_i + 1j * rx0_q;
    iq1 = rx1_i + 1j * rx1_q;
    
    % 检测LTS
    [lts_pos0, ~] = detect_lts_matlab(iq0, LTS_TEMPLATE, threshold);
    [lts_pos1, ~] = detect_lts_matlab(iq1, LTS_TEMPLATE, threshold);
    
    if isempty(lts_pos0) || isempty(lts_pos1)
        fprintf('  No valid LTS detected\n\n');
        continue;
    end
    
    results.valid(frame_idx) = true;
    
    % 提取LTS并计算CSI
    lts_start0 = lts_pos0(1);
    lts_start1 = lts_pos1(1);
    
    % 提取两个LTS符号
    ltf0_1 = iq0(lts_start0+32:lts_start0+95);
    ltf0_2 = iq0(lts_start0+96:lts_start0+159);
    ltf1_1 = iq1(lts_start1+32:lts_start1+95);
    ltf1_2 = iq1(lts_start1+96:lts_start1+159);
    
    % FFT并平均
    fft0 = (fftshift(fft(ltf0_1)) + fftshift(fft(ltf0_2))) / 2;
    fft1 = (fftshift(fft(ltf1_1)) + fftshift(fft(ltf1_2))) / 2;
    
    % 计算CSI
    valid_idx = find(LTS_FREQ ~= 0);
    csi0 = fft0(valid_idx) ./ LTS_FREQ(valid_idx).';
    csi1 = fft1(valid_idx) ./ LTS_FREQ(valid_idx).';
    
    % 构造阵列数据矩阵 (M x N_subcarriers)
    X = [csi0.'; csi1.'];
    
    % === 1. 相位差法 ===
    avg_csi0 = mean(csi0);
    avg_csi1 = mean(csi1);
    phase_diff = angle(avg_csi0) - angle(avg_csi1);
    
    % 相位差归一化到[-π, π]
    if abs(phase_diff) > pi
        phase_diff = phase_diff - sign(phase_diff) * 2 * pi;
    end
    
    % 检查是否有180度相位翻转（d/λ > 0.5时可能发生）
    % 如果计算的角度接近±90度，可能是发生了180度翻转
    theta_phase_raw = asin(phase_diff * lambda / (2 * pi * d)) * 180 / pi;
    
    % 尝试180度翻转纠正
    phase_diff_flipped = phase_diff + pi;
    if phase_diff_flipped > pi
        phase_diff_flipped = phase_diff_flipped - 2 * pi;
    end
    theta_phase_flipped = asin(phase_diff_flipped * lambda / (2 * pi * d)) * 180 / pi;
    
    % 选择更合理的角度（倾向于选择绝对值较小的）
    if abs(theta_phase_flipped) < abs(theta_phase_raw) && d/lambda > 0.5
        theta_phase = theta_phase_flipped;
        phase_diff = phase_diff_flipped;  % 使用翻转后的相位差
    else
        theta_phase = theta_phase_raw;
    end
    
    results.phase_diff(frame_idx) = theta_phase;
    
    % === 2. MUSIC算法 ===
    theta_music = music_doa(X, d, lambda, 1);  % 假设1个信号源
    results.music(frame_idx) = theta_music;
    
    % === 3. Capon (MVDR)算法 ===
    theta_capon = capon_doa(X, d, lambda);
    results.capon(frame_idx) = theta_capon;
    
    % === 4. ESPRIT算法 ===
    theta_esprit = esprit_doa(X, d, lambda, 1);  % 假设1个信号源
    results.esprit(frame_idx) = theta_esprit;
    
    fprintf('  Phase Diff: %7.2f°\n', theta_phase);
    fprintf('  MUSIC:      %7.2f°\n', theta_music);
    fprintf('  Capon:      %7.2f°\n', theta_capon);
    fprintf('  ESPRIT:     %7.2f°\n\n', theta_esprit);
    
    % 为第3帧生成详细分析
    if frame_idx == 3
        plot_doa_analysis(X, d, lambda, frame_idx, theta_phase, theta_music, theta_capon, theta_esprit);
        plot_csi_analysis(csi0, csi1, valid_idx, frame_idx);
    end
end

% 统计结果
valid_idx = find(results.valid);
if ~isempty(valid_idx)
    % 对结果进行相位unwrap处理，避免360度跳变
    fprintf('\n=== Phase Jump Detection ===\n');
    [results.phase_diff, jumps_pd] = unwrap_angles_with_detection(results.phase_diff, valid_idx, 'Phase Diff');
    [results.music, jumps_music] = unwrap_angles_with_detection(results.music, valid_idx, 'MUSIC');
    [results.capon, jumps_capon] = unwrap_angles_with_detection(results.capon, valid_idx, 'Capon');
    [results.esprit, jumps_esprit] = unwrap_angles_with_detection(results.esprit, valid_idx, 'ESPRIT');
    
    % 检测并纠正180度相位翻转（针对d/λ > 0.5的情况）
    if d/lambda > 0.5
        fprintf('\n=== 180° Phase Flip Detection (d/λ = %.3f > 0.5) ===\n', d/lambda);
        [results.phase_diff, flips_pd] = detect_phase_flip(results.phase_diff, valid_idx, d, lambda, 'Phase Diff');
        [results.music, flips_music] = detect_phase_flip(results.music, valid_idx, d, lambda, 'MUSIC');
        [results.capon, flips_capon] = detect_phase_flip(results.capon, valid_idx, d, lambda, 'Capon');
        [results.esprit, flips_esprit] = detect_phase_flip(results.esprit, valid_idx, d, lambda, 'ESPRIT');
    else
        flips_pd = [];
        flips_music = [];
        flips_capon = [];
        flips_esprit = [];
    end
    
    % 处理超出±80度范围的异常值
    fprintf('\n=== Large Angle Correction (>80°) ===\n');
    [results.phase_diff, outliers_pd, valid_pd] = correct_large_angles(results.phase_diff, valid_idx, 'Phase Diff');
    [results.music, outliers_music, valid_music] = correct_large_angles(results.music, valid_idx, 'MUSIC');
    [results.capon, outliers_capon, valid_capon] = correct_large_angles(results.capon, valid_idx, 'Capon');
    [results.esprit, outliers_esprit, valid_esprit] = correct_large_angles(results.esprit, valid_idx, 'ESPRIT');
    
    fprintf('\n=== Overall Statistics (All Valid Frames) ===\n');
    fprintf('Method         Mean      Std Dev   Min       Max       Valid Frames\n');
    fprintf('-------------------------------------------------------------------------\n');
    
    methods = {'phase_diff', 'music', 'capon', 'esprit'};
    labels = {'Phase Diff', 'MUSIC', 'Capon', 'ESPRIT'};
    valid_sets = {valid_pd, valid_music, valid_capon, valid_esprit};
    
    for i = 1:length(methods)
        valid_frames_method = valid_sets{i};
        if ~isempty(valid_frames_method)
            vals = results.(methods{i})(valid_frames_method);
            fprintf('%-12s  %7.2f°  %7.2f°  %7.2f°  %7.2f°  %d/%d\n', ...
                labels{i}, mean(vals), std(vals), min(vals), max(vals), ...
                length(valid_frames_method), length(valid_idx));
        else
            fprintf('%-12s  N/A (no valid data after filtering)\n', labels{i});
        end
    end
    
    fprintf('\n=== Corrected Average Angles (Excluding >80° outliers) ===\n');
    for i = 1:length(methods)
        valid_frames_method = valid_sets{i};
        if ~isempty(valid_frames_method)
            vals = results.(methods{i})(valid_frames_method);
            fprintf('%-12s: %7.2f° (based on %d frames)\n', labels{i}, mean(vals), length(valid_frames_method));
        else
            fprintf('%-12s: N/A (no valid data)\n', labels{i});
        end
    end
    
    % 绘制比较图
    figure('Position', [100, 100, 1400, 900]);
    
    % 保存原始数据用于对比
    results_raw = struct();
    results_raw.phase_diff = results.phase_diff;
    results_raw.music = results.music;
    results_raw.capon = results.capon;
    results_raw.esprit = results.esprit;
    
    % 有效帧集合
    valid_sets = {valid_pd, valid_music, valid_capon, valid_esprit};
    flip_sets = {flips_pd, flips_music, flips_capon, flips_esprit};
    
    % 绘制原始结果（有跳变）和纠正后结果
    for method_idx = 1:4
        subplot(2,2,method_idx);
        method = methods{method_idx};
        label = labels{method_idx};
        valid_frames_method = valid_sets{method_idx};
        
        % 获取原始值（重新计算以显示跳变）
        raw_vals = zeros(1, frame_count);
        for fi = valid_idx
            raw_vals(fi) = results_raw.(method)(fi);
        end
        
        % 重新应用unwrap来获取原始的跳变数据
        [~, ~, raw_before_unwrap] = unwrap_angles_with_detection(raw_vals, valid_idx, '', true);
        
        hold on;
        % 绘制原始数据（有跳变）
        plot(valid_idx, real(raw_before_unwrap(valid_idx)), 'r--o', 'LineWidth', 1.5, ...
             'MarkerSize', 6, 'DisplayName', 'Before Correction');
        % 绘制纠正后数据（排除>80度的异常点）
        if ~isempty(valid_frames_method)
            plot(valid_frames_method, real(results.(method)(valid_frames_method)), 'b-s', 'LineWidth', 2, ...
                 'MarkerSize', 6, 'DisplayName', 'After Correction');
        end
        
        % 标记跳变点
        if method_idx == 1
            jump_frames = jumps_pd;
            outlier_frames = outliers_pd;
            flip_frames = flips_pd;
        elseif method_idx == 2
            jump_frames = jumps_music;
            outlier_frames = outliers_music;
            flip_frames = flips_music;
        elseif method_idx == 3
            jump_frames = jumps_capon;
            outlier_frames = outliers_capon;
            flip_frames = flips_capon;
        else
            jump_frames = jumps_esprit;
            outlier_frames = outliers_esprit;
            flip_frames = flips_esprit;
        end
        
        % 标记相位跳变点（绿色三角）
        if ~isempty(jump_frames)
            for jf = jump_frames
                if any(valid_idx == jf) && any(valid_frames_method == jf)
                    plot(jf, real(results.(method)(jf)), 'g^', 'MarkerSize', 12, ...
                         'LineWidth', 2, 'MarkerFaceColor', 'g');
                end
            end
        end
        
        % 标记180度相位翻转点（青色五角星）
        if ~isempty(flip_frames)
            for ff = flip_frames
                if any(valid_idx == ff) && any(valid_frames_method == ff)
                    plot(ff, real(results.(method)(ff)), 'cp', 'MarkerSize', 14, ...
                         'LineWidth', 2, 'MarkerFaceColor', 'c');
                end
            end
        end
        
        % 标记大角度异常点（品红色菱形，带红色X表示被排除）
        if ~isempty(outlier_frames)
            for of = outlier_frames
                if any(valid_idx == of)
                    % 绘制异常点位置
                    plot(of, real(raw_before_unwrap(of)), 'md', 'MarkerSize', 12, ...
                         'LineWidth', 2, 'MarkerFaceColor', 'm');
                    % 添加红色X标记表示被排除
                    plot(of, real(raw_before_unwrap(of)), 'rx', 'MarkerSize', 14, ...
                         'LineWidth', 3);
                end
            end
        end
        
        % 添加±80度参考线
        yline(80, 'k--', 'LineWidth', 1, 'Alpha', 0.3);
        yline(-80, 'k--', 'LineWidth', 1, 'Alpha', 0.3);
        
        xlabel('Frame Index');
        ylabel('Estimated Angle (degrees)');
        title(sprintf('%s DOA Estimation', label));
        
        % 更新图例
        legend_entries = {'Before Correction'};
        if ~isempty(valid_frames_method)
            legend_entries{end+1} = 'After Correction';
        end
        if ~isempty(jump_frames)
            legend_entries{end+1} = '360° Jump';
        end
        if ~isempty(flip_frames)
            legend_entries{end+1} = '180° Flip';
        end
        if ~isempty(outlier_frames)
            legend_entries{end+1} = 'Outlier (Excluded)';
        end
        legend(legend_entries, 'Location', 'best');
        grid on;
    end
    
    sgtitle('DOA Estimation: Before and After Correction (Outliers Excluded from Average)');
    
    % 箱型图对比
    figure('Position', [100, 100, 1000, 500]);
    
    % 确保数据是实数，并且只包含有效帧
    subplot(1,2,1);
    % 找到所有方法中最大的有效帧数
    max_valid = max([length(valid_pd), length(valid_music), length(valid_capon), length(valid_esprit)]);
    data_box = nan(max_valid, 4);
    for i = 1:4
        valid_frames_method = valid_sets{i};
        if ~isempty(valid_frames_method)
            vals = real(results.(methods{i})(valid_frames_method));
            data_box(1:length(vals), i) = vals;
        end
    end
    boxplot(data_box, 'Labels', {'Phase Diff', 'MUSIC', 'Capon', 'ESPRIT'});
    ylabel('Estimated Angle (degrees)');
    title('DOA Distribution (After Correction, Outliers Excluded)');
    grid on;
    
    % 添加散点图显示数据分布
    subplot(1,2,2);
    hold on;
    colors = ['b', 'r', 'g', 'm'];
    for i = 1:4
        valid_data = data_box(~isnan(data_box(:,i)), i);
        if ~isempty(valid_data)
            scatter(i*ones(size(valid_data)), valid_data, 50, colors(i), 'filled', 'MarkerFaceAlpha', 0.6);
        end
    end
    boxplot(data_box, 'Labels', {'Phase Diff', 'MUSIC', 'Capon', 'ESPRIT'}, 'Colors', 'k');
    ylabel('Estimated Angle (degrees)');
    title('DOA Distribution with Data Points (Outliers Excluded)');
    grid on;
end

%% 辅助函数

function [lts_positions, correlations] = detect_lts_matlab(iq_signal, lts_template, threshold)
    if length(iq_signal) < length(lts_template)
        lts_positions = [];
        correlations = [];
        return;
    end
    
    if size(iq_signal, 1) > size(iq_signal, 2)
        iq_signal = iq_signal.';
    end
    if size(lts_template, 1) > size(lts_template, 2)
        lts_template = lts_template.';
    end
    
    template_norm = lts_template / sqrt(sum(abs(lts_template).^2));
    correlations = zeros(1, length(iq_signal) - length(lts_template) + 1);
    
    for i = 1:length(correlations)
        window = iq_signal(i:i + length(lts_template) - 1);
        window_norm = window / (sqrt(sum(abs(window).^2)) + 1e-10);
        correlations(i) = abs(sum(window_norm .* conj(template_norm)));
    end
    
    lts_positions = [];
    max_corr = max(correlations);
    
    if max_corr > threshold
        peak_threshold = threshold * max_corr;
        i = 1;
        while i <= length(correlations)
            if correlations(i) > peak_threshold
                local_max_idx = i;
                local_max_val = correlations(i);
                j = i + 1;
                while j <= min(i + length(lts_template) - 1, length(correlations)) && correlations(j) > peak_threshold
                    if correlations(j) > local_max_val
                        local_max_idx = j;
                        local_max_val = correlations(j);
                    end
                    j = j + 1;
                end
                lts_positions = [lts_positions, local_max_idx];
                i = j;
            else
                i = i + 1;
            end
        end
    end
end

function theta = music_doa(X, d, lambda, num_sources)
    % MUSIC算法估计DOA
    % X: M x N 阵列数据矩阵
    % d: 天线间距
    % lambda: 波长
    % num_sources: 信号源数量
    
    [M, N] = size(X);
    
    % 计算协方差矩阵
    Rxx = (X * X') / N;
    
    % 特征值分解
    [V, D] = eig(Rxx);
    [~, idx] = sort(diag(D), 'descend');
    V = V(:, idx);
    
    % 噪声子空间
    Un = V(:, num_sources+1:end);
    
    % 角度搜索
    theta_scan = -90:0.5:90;
    P_music = zeros(size(theta_scan));
    
    for i = 1:length(theta_scan)
        a = steering_vector(theta_scan(i), d, lambda, M);
        P_music(i) = 1 / (a' * (Un * Un') * a);
    end
    
    % 找到峰值
    [~, idx] = max(abs(P_music));
    theta = theta_scan(idx);
end

function theta = capon_doa(X, d, lambda)
    % Capon (MVDR)算法估计DOA
    
    [M, N] = size(X);
    
    % 计算协方差矩阵
    Rxx = (X * X') / N;
    
    % 添加对角加载以提高稳定性
    Rxx = Rxx + 1e-6 * eye(M);
    
    % 角度搜索
    theta_scan = -90:0.5:90;
    P_capon = zeros(size(theta_scan));
    
    for i = 1:length(theta_scan)
        a = steering_vector(theta_scan(i), d, lambda, M);
        P_capon(i) = 1 / real(a' * inv(Rxx) * a);
    end
    
    % 找到峰值
    [~, idx] = max(P_capon);
    theta = theta_scan(idx);
end

function theta = esprit_doa(X, d, lambda, num_sources)
    % ESPRIT算法估计DOA
    
    [M, N] = size(X);
    
    % 计算协方差矩阵
    Rxx = (X * X') / N;
    
    % 特征值分解
    [V, D] = eig(Rxx);
    [~, idx] = sort(diag(D), 'descend');
    Us = V(:, idx(1:num_sources));
    
    % 分割子阵
    Us1 = Us(1:M-1, :);
    Us2 = Us(2:M, :);
    
    % 计算旋转不变量
    Psi = pinv(Us1) * Us2;
    phi = eig(Psi);
    
    % 从相位计算角度
    phi = phi(1);  % 取第一个特征值
    theta = asin(angle(phi) * lambda / (2 * pi * d)) * 180 / pi;
end

function a = steering_vector(theta, d, lambda, M)
    % 生成导向矢量
    theta_rad = theta * pi / 180;
    k = 2 * pi / lambda;
    a = exp(-1j * k * d * (0:M-1)' * sin(theta_rad));
end

function plot_doa_analysis(X, d, lambda, frame_idx, theta_phase, theta_music, theta_capon, theta_esprit)
    % 绘制详细的DOA分析图
    
    [M, N] = size(X);
    Rxx = (X * X') / N;
    Rxx = Rxx + 1e-6 * eye(M);
    
    % 特征值分解
    [V, D] = eig(Rxx);
    [eigenvalues, idx] = sort(diag(D), 'descend');
    V = V(:, idx);
    Un = V(:, 2:end);  % 假设1个信号源
    
    % 角度扫描
    theta_scan = -90:0.1:90;
    P_music = zeros(size(theta_scan));
    P_capon = zeros(size(theta_scan));
    
    for i = 1:length(theta_scan)
        a = steering_vector(theta_scan(i), d, lambda, M);
        P_music(i) = 1 / real(a' * (Un * Un') * a);
        P_capon(i) = 1 / real(a' * inv(Rxx) * a);
    end
    
    % 归一化
    P_music = 10*log10(abs(P_music) / max(abs(P_music)));
    P_capon = 10*log10(P_capon / max(P_capon));
    
    figure('Position', [100, 100, 1400, 800]);
    
    % MUSIC谱
    subplot(2,2,1);
    plot(theta_scan, P_music, 'b-', 'LineWidth', 2); hold on;
    xline(theta_music, 'r--', 'LineWidth', 2);
    xlabel('Angle (degrees)');
    ylabel('Pseudo Spectrum (dB)');
    title(sprintf('MUSIC Spectrum (θ = %.2f°)', theta_music));
    grid on;
    xlim([-90 90]);
    
    % Capon谱
    subplot(2,2,2);
    plot(theta_scan, P_capon, 'g-', 'LineWidth', 2); hold on;
    xline(theta_capon, 'r--', 'LineWidth', 2);
    xlabel('Angle (degrees)');
    ylabel('Pseudo Spectrum (dB)');
    title(sprintf('Capon Spectrum (θ = %.2f°)', theta_capon));
    grid on;
    xlim([-90 90]);
    
    % 特征值分布
    subplot(2,2,3);
    stem(1:M, eigenvalues, 'filled', 'LineWidth', 2);
    xlabel('Eigenvalue Index');
    ylabel('Magnitude');
    title('Eigenvalue Distribution');
    grid on;
    
    % 结果比较
    subplot(2,2,4);
    methods = {'Phase Diff', 'MUSIC', 'Capon', 'ESPRIT'};
    angles = [theta_phase, theta_music, theta_capon, theta_esprit];
    bar(angles);
    set(gca, 'XTickLabel', methods);
    ylabel('Estimated Angle (degrees)');
    title('DOA Estimation Comparison');
    grid on;
    
    sgtitle(sprintf('Frame %d: DOA Analysis', frame_idx));
end

function plot_csi_analysis(csi0, csi1, valid_subcarrier_idx, frame_idx)
    % 绘制CSI分析图
    figure('Position', [100, 100, 1200, 800]);
    
    % CSI幅度
    subplot(3,2,1);
    plot(valid_subcarrier_idx, abs(csi0), 'bo-', 'LineWidth', 1.5); hold on;
    plot(valid_subcarrier_idx, abs(csi1), 'rs-', 'LineWidth', 1.5);
    xlabel('Subcarrier Index');
    ylabel('Magnitude');
    title('CSI Magnitude');
    legend('RX0', 'RX1');
    grid on;
    
    % CSI原始相位
    subplot(3,2,2);
    plot(valid_subcarrier_idx, angle(csi0)*180/pi, 'bo-', 'LineWidth', 1.5); hold on;
    plot(valid_subcarrier_idx, angle(csi1)*180/pi, 'rs-', 'LineWidth', 1.5);
    xlabel('Subcarrier Index');
    ylabel('Phase (degrees)');
    title('CSI Phase (Wrapped)');
    legend('RX0', 'RX1');
    grid on;
    
    % CSI解缠绕相位
    subplot(3,2,3);
    unwrap_phase0 = unwrap(angle(csi0));
    unwrap_phase1 = unwrap(angle(csi1));
    plot(valid_subcarrier_idx, unwrap_phase0*180/pi, 'bo-', 'LineWidth', 1.5); hold on;
    plot(valid_subcarrier_idx, unwrap_phase1*180/pi, 'rs-', 'LineWidth', 1.5);
    xlabel('Subcarrier Index');
    ylabel('Phase (degrees)');
    title('CSI Phase (Unwrapped)');
    legend('RX0', 'RX1');
    grid on;
    
    % 相位差（原始）
    subplot(3,2,4);
    phase_diff = angle(csi0) - angle(csi1);
    plot(valid_subcarrier_idx, phase_diff*180/pi, 'go-', 'LineWidth', 1.5);
    xlabel('Subcarrier Index');
    ylabel('Phase Difference (degrees)');
    title('Phase Difference (Wrapped)');
    grid on;
    yline(median(phase_diff)*180/pi, 'r--', 'LineWidth', 2);
    
    % 相位差（解缠绕）
    subplot(3,2,5);
    phase_diff_unwrap = unwrap_phase0 - unwrap_phase1;
    plot(valid_subcarrier_idx, phase_diff_unwrap*180/pi, 'mo-', 'LineWidth', 1.5);
    xlabel('Subcarrier Index');
    ylabel('Phase Difference (degrees)');
    title('Phase Difference (Unwrapped)');
    grid on;
    yline(median(phase_diff_unwrap)*180/pi, 'r--', 'LineWidth', 2);
    
    % 相位差统计
    subplot(3,2,6);
    histogram(phase_diff*180/pi, 30, 'FaceColor', 'g', 'EdgeColor', 'k', 'FaceAlpha', 0.7);
    hold on;
    histogram(phase_diff_unwrap*180/pi, 30, 'FaceColor', 'm', 'EdgeColor', 'k', 'FaceAlpha', 0.5);
    xlabel('Phase Difference (degrees)');
    ylabel('Count');
    title('Phase Difference Distribution');
    legend('Wrapped', 'Unwrapped');
    grid on;
    
    sgtitle(sprintf('Frame %d: CSI Analysis', frame_idx));
end

function [angles_unwrapped, jump_frames, angles_original] = unwrap_angles_with_detection(angles, valid_idx, method_name, silent)
    % 对角度序列进行unwrap处理，检测并打印360度跳变
    % angles: 完整的角度数组
    % valid_idx: 有效帧的索引
    % method_name: 方法名称（用于打印）
    % silent: 是否静默模式（不打印）
    
    if nargin < 4
        silent = false;
    end
    
    angles_unwrapped = angles;
    angles_original = angles;  % 保存原始值
    jump_frames = [];
    
    if length(valid_idx) < 2
        return;
    end
    
    % 只处理有效帧
    valid_angles = angles(valid_idx);
    original_angles = valid_angles;  % 保存原始值用于对比
    
    % 检测并修正360度跳变
    jump_count = 0;
    for i = 2:length(valid_angles)
        diff = valid_angles(i) - valid_angles(i-1);
        
        % 如果相邻角度差超过180度，可能是360度跳变
        if abs(diff) > 180
            jump_count = jump_count + 1;
            frame_num = valid_idx(i);
            jump_frames = [jump_frames, frame_num];
            
            if diff > 180
                correction = -360;
                valid_angles(i:end) = valid_angles(i:end) - 360;
            else  % diff < -180
                correction = 360;
                valid_angles(i:end) = valid_angles(i:end) + 360;
            end
            
            if ~silent
                fprintf('  %s: Frame %d detected phase jump (%.2f° -> %.2f°, corrected by %+d°)\n', ...
                    method_name, frame_num, original_angles(i), valid_angles(i), correction);
            end
        end
    end
    
    if ~silent && jump_count == 0
        fprintf('  %s: No phase jumps detected\n', method_name);
    elseif ~silent
        fprintf('  %s: Total %d phase jump(s) detected at frames: [%s]\n', ...
            method_name, jump_count, num2str(jump_frames));
    end
    
    angles_unwrapped(valid_idx) = valid_angles;
end

function [angles_corrected, outlier_frames, valid_frames] = correct_large_angles(angles, valid_idx, method_name)
    % 检测并排除绝对值大于80度的角度异常值
    % 这些异常值不会被纠正，而是被排除在统计之外
    
    angles_corrected = angles;
    outlier_frames = [];
    valid_frames = valid_idx;  % 初始化为所有有效帧
    
    if isempty(valid_idx)
        return;
    end
    
    % 只处理有效帧
    valid_angles = angles(valid_idx);
    
    % 检测大角度异常值
    outlier_count = 0;
    for i = 1:length(valid_angles)
        if abs(valid_angles(i)) > 80
            outlier_count = outlier_count + 1;
            frame_num = valid_idx(i);
            outlier_frames = [outlier_frames, frame_num];
            
            fprintf('  %s: Frame %d detected as outlier (%.2f°) - excluded from average\n', ...
                method_name, frame_num, valid_angles(i));
        end
    end
    
    % 从有效帧中排除异常值
    valid_frames = setdiff(valid_idx, outlier_frames);
    
    if outlier_count == 0
        fprintf('  %s: No large angles (>80°) detected\n', method_name);
    else
        fprintf('  %s: Total %d outlier(s) detected at frames: [%s] - excluded from statistics\n', ...
            method_name, outlier_count, num2str(outlier_frames));
    end
end