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