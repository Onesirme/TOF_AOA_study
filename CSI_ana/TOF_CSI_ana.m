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

fprintf('Data loaded successfully!\n');

%% 分析每一帧的LTS位置和CSI相位差
threshold = 0.6;  % LTS检测阈值

% 存储每一帧的CSI相位差中值和角度
phase_diff_medians = zeros(1, frame_count);
angles = zeros(1, frame_count);  % 存储计算的角度
valid_frames = false(1, frame_count);  % 标记哪些帧有有效的LTS检测

% WiFi 44通道参数
fc = 2.412e9;  
c = physconst('light');      
lambda = c / fc;  % 波长
d = 0.06;   
t_delay0 = zeros(1, frame_count);
t_delay1 = zeros(1, frame_count);

fprintf('WiFi Channel 44 Parameters:\n');
fprintf('  Center Frequency: %.2f GHz\n', fc/1e9);
fprintf('  Wavelength: %.4f m\n', lambda);
fprintf('  Antenna Spacing: %.3f m\n', d);
fprintf('  d/lambda ratio: %.3f\n\n', d/lambda);

%% 对每一帧进行处理
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
        
        % 找到有效子载波索引
        valid_subcarriers = find(LTS_FREQ ~= 0);
        
        % 提取有效子载波的CSI
        valid_IQ0_csi = IQ0_csi(valid_subcarriers);
        valid_IQ1_csi = IQ1_csi(valid_subcarriers);
        
        % CSI解缠绕
        unwrap_phase0 = unwrap(angle(valid_IQ0_csi));
        unwrap_phase1 = unwrap(angle(valid_IQ1_csi));
        

        % 计算相位差（解缠绕后）
        phase_diff = (unwrap_phase0 - unwrap_phase1);
        phase_diff_deg = phase_diff * 180 / pi;
        
        % 计算相位差的中值
        phase_diff_medians(frame_idx) = median(phase_diff_deg);
        
        % 使用平均CSI计算相位差（用于角度计算）
        % avg_phase_diff = angle(avg_IQ0_csi) - angle(avg_IQ1_csi);
        avg_phase_diff = mean(phase_diff);
        % 计算设备角度
        % 根据相位差公式: Δφ = (2πd/λ) * sin(θ)
        % 所以: θ = arcsin(Δφ * λ / (2πd))
        if abs(avg_phase_diff) > 2*pi
            avg_phase_diff = avg_phase_diff - sign(avg_phase_diff) *  pi;
        end
        
        % 角度计算：确保参数在[-1,1]范围内
        sin_theta = avg_phase_diff * lambda / (2 * pi * d);
        
        theta_rad = asin(sin_theta);
        theta_deg = (theta_rad) * 180 / pi; 
        
        angles(frame_idx) = theta_deg;
        
        fprintf('Frame %d:\n', frame_idx);
        fprintf('  CSI phase difference median = %.2f degrees\n', phase_diff_medians(frame_idx));
        fprintf('  Average phase difference = %.2f degrees\n', avg_phase_diff * 180 / pi);
        fprintf('  sin(theta) = %.4f\n', sin_theta);
        fprintf('  Estimated angle = %.2f degrees\n', theta_deg);
        %% 斜率计算frame_idx

        p =  polyfit(valid_subcarriers*312500, unwrap_phase0*c, 1);
        t_delay0(frame_idx)=p(1);
        p =  polyfit(valid_subcarriers*312500, unwrap_phase1*c, 1);
        t_delay1(frame_idx)=p(1);

            % CSI分析图
        if frame_idx == 2
            figure(1);
            subplot(311)
            plot(valid_subcarriers, angle(valid_IQ0_csi)*180/pi, "ro-", ...
                 valid_subcarriers, angle(valid_IQ1_csi)*180/pi, "bo-")
            title('CSI Phase (Valid Subcarriers)');
            legend('RX0', 'RX1');
            ylabel('Phase (deg)');
            grid on;
            
            subplot(312)
            plot(valid_subcarriers, unwrap_phase0*180/pi, "ro-", ...
                 valid_subcarriers, unwrap_phase1*180/pi, "bo-")
            title('Unwrapped CSI Phase');
            legend('RX0', 'RX1');
            ylabel('Phase (deg)');
            grid on;
            
            subplot(313)
            plot(valid_subcarriers, phase_diff_deg, "go-", 'LineWidth', 2)
            hold on;
            yline(phase_diff_medians(frame_idx), 'r--', 'LineWidth', 2, ...
                  'Label', sprintf('Median: %.2f°', phase_diff_medians(frame_idx)));
            title('Phase Difference (Unwrapped)');
            legend('Phase Difference', 'Median');
            xlabel('Subcarrier Index');
            ylabel('Phase Difference (deg)');
            grid on;
            
            sgtitle(sprintf('Frame %d: CSI Analysis (Channel 44, d=%.3fm)', frame_idx, d));
        end
    end
end

%% 绘制所有帧的CSI相位差中值
figure(3);
valid_frame_indices = find(valid_frames);
if ~isempty(valid_frame_indices)
    % 确保所有值都是实数
    phase_diff_medians = real(phase_diff_medians);
    angles = real(angles);
    
    subplot(2,1,1);
    plot(valid_frame_indices, phase_diff_medians(valid_frames), 'bo-', 'LineWidth', 2, 'MarkerSize', 6);
    xlabel('Frame Index');
    ylabel('CSI Phase Difference Median (degrees)');
    title('CSI Phase Difference Median Across Frames');
    grid on;
    
    % 计算所有有效帧的总体中值
    overall_median = median(phase_diff_medians(valid_frames), 'omitnan');
    hold on;
    yline(overall_median, 'r--', 'LineWidth', 2, 'Label', sprintf('Overall Median: %.2f°', overall_median));
    legend('Per-Frame Median', 'Overall Median', 'Location', 'best');
    
    % 绘制角度估计
    subplot(2,1,2);
    plot(valid_frame_indices, angles(valid_frames), 'go-', 'LineWidth', 2, 'MarkerSize', 6);
    xlabel('Frame Index');
    ylabel('Estimated Angle (degrees)');
    title('Estimated Device Angle Across Frames');
    grid on;
    
    % 计算平均角度
    avg_angle = mean(angles(valid_frames), 'omitnan');
    if isreal(avg_angle) && isfinite(avg_angle)
        hold on;
        yline(avg_angle, 'r--', 'LineWidth', 2, 'Label', sprintf('Average Angle: %.2f°', avg_angle));
        legend('Per-Frame Angle', 'Average Angle', 'Location', 'best');
    else
        fprintf('Warning: Average angle is not a finite real number: %.2f\n', avg_angle);
    end
    
    fprintf('\n=== Overall Results ===\n');
    fprintf('Number of valid frames: %d/%d\n', sum(valid_frames), frame_count);
    fprintf('Overall CSI phase difference median: %.2f degrees\n', overall_median);
    fprintf('Average estimated angle: %.2f degrees\n', avg_angle);
    
    % 显示角度分布统计
    angle_std = std(angles(valid_frames), 'omitnan');
    fprintf('Angle standard deviation: %.2f degrees\n', angle_std);
else
    fprintf('No valid frames found with LTS detected in both channels.\n');
end

%% 斜率绘制
figure;
subplot(211)
plot(1:frame_count,t_delay0);
subplot(212)
plot(1:frame_count,t_delay1);



%% LTS检测函数 (必须放在文件末尾)
function [lts_positions, correlations] = detect_lts_matlab(iq_signal, lts_template, threshold)
    % 检测IQ信号中的LTS序列
    if length(iq_signal) < length(lts_template)
        lts_positions = [];
        correlations = [];
        return;
    end
    
    % 确保输入是行向量
    if size(iq_signal, 1) > size(iq_signal, 2)
        iq_signal = iq_signal.';
    end
    if size(lts_template, 1) > size(lts_template, 2)
        lts_template = lts_template.';
    end
    
    % 归一化LTS模板
    template_norm = lts_template / sqrt(sum(abs(lts_template).^2));
    
    % 滑动相关
    correlations = zeros(1, length(iq_signal) - length(lts_template) + 1);
    for i = 1:length(correlations)
        window = iq_signal(i:i + length(lts_template) - 1);
        window_norm = window / (sqrt(sum(abs(window).^2)) + 1e-10);
        correlations(i) = abs(sum(window_norm .* conj(template_norm)));
    end
    
    % 查找峰值
    lts_positions = [];
    max_corr = max(correlations);
    
    if max_corr > threshold
        % 找到所有超过阈值的峰值
        peak_threshold = threshold * max_corr;
        i = 1;
        while i <= length(correlations)
            if correlations(i) > peak_threshold
                % 找到局部最大值
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