clc; clear;

%% 1. 站点参数（station01）
eta_STC = 270 / 1.6635e3;   % 组件效率 ≈0.1623
A_total = 74000;            % 组件总面积 m²
G_STC   = 1000;             % STC 辐照 W/m²
P_rated = 20000;            % 装机容量 kW
lat     = 38.18306;         % 纬度（station01）

%% 2. 读取 CSV（确保时间列为字符串）
T = readtable('station01.csv','TextType','string');

%% 3. 时间列处理（字符串切片 → datetime）
rawStr  = string(T.date_time);
rawStr  = replace(rawStr,'  ',' ');
parts   = split(rawStr,' ');

dateTxt = parts(:,1); timeTxt = parts(:,2);
dateP   = split(dateTxt,'-'); timeP = split(timeTxt,':');

my_year   = str2double(dateP(:,1));
my_month  = str2double(dateP(:,2));
my_day    = str2double(dateP(:,3));
my_hour   = str2double(timeP(:,1));
my_minute = str2double(timeP(:,2));
my_second = str2double(timeP(:,3));

T.date_time = datetime(my_year,my_month,my_day,my_hour,my_minute,my_second) + hours(8);
T.month     = my_month;
T.power     = T.power * 1000;  % MW → kW

%% 4. 理论发电功率 P_cs —— 使用太阳高度角估计
my_doy   = day(T.date_time,'dayofyear');
my_hourf = hour(T.date_time) + minute(T.date_time)/60;

delta = deg2rad(23.45) .* sin(deg2rad(360 * (284 + my_doy) ./ 365));
H     = deg2rad(15 .* (my_hourf - 12));
phi   = deg2rad(lat);
sin_h = sin(phi).*sin(delta) + cos(phi).*cos(delta).*cos(H);
sin_h(sin_h < 0) = 0;

G_cs = G_STC .* sin_h;
T.P_cs = eta_STC .* G_cs .* A_total ./ G_STC;

%% 4.1 实际功率截断
T.power_clipped = min(T.power, T.P_cs);

P = T.power_clipped;
t = T.date_time;

%% === 构造序列（输入为过去96点，输出为下1点）===
win = 96;
X = []; Y = [];
for i = win+1:length(P)
    if all(~isnan(P(i-win:i)))
        X(end+1,:) = P(i-win:i-1);  % 输入96点
        Y(end+1,1) = P(i);          % 输出第97点
    end
end

%% === 划分训练集和测试集（测试：2/5/8/11月最后7天）===
ts = t(win+1:end);
mon = month(ts); dayN = day(ts);
is_test = false(size(ts));

for m = [2 5 8 11]
    idx = find(mon == m);
    last7 = idx(end-7*96+1:end);  % 最后7天数据
    is_test(last7) = true;
end

XTrain = X(~is_test,:); YTrain = Y(~is_test);
XTest  = X(is_test,:);  YTest  = Y(is_test);

% 划分验证集（取训练集的后10%）
nTrain = size(XTrain,1);
nVal = round(0.1 * nTrain);
idxVal = (nTrain - nVal + 1):nTrain;

XVal = XTrain(idxVal, :);  YVal = YTrain(idxVal);
XTrain = XTrain(1:end-nVal, :);  YTrain = YTrain(1:end-nVal);

XValSeq = cellfun(@(x) x', num2cell(XVal, 2), 'UniformOutput', false);
YValSeq = num2cell(YVal);




%% === 构建 LSTM 模型 ===
inputSize = win;
numHidden = 64;
layers = [
    sequenceInputLayer(inputSize)
    lstmLayer(numHidden)
    fullyConnectedLayer(1)
    regressionLayer
];

opts = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 128, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'ValidationData',{XValSeq, YValSeq}, ...
    'ValidationFrequency', 50, ...
    'Verbose',false);

XTrainSeq = cellfun(@(x) x', num2cell(XTrain, 2), 'UniformOutput', false);  % 每行为一个序列，转置为列
YTrainSeq = num2cell(YTrain);  % 每个 cell 是标量

%% === 训练模型 ===


net = trainNetwork(XTrainSeq, YTrainSeq, layers, opts);

%% === 测试集预测 ===


XTestSeq = cellfun(@(x) x', num2cell(XTest, 2), 'UniformOutput', false);  % 转置为 96×1 列向量
YPred = predict(net, XTestSeq, 'MiniBatchSize', 1);
YPred = cell2mat(YPred);  

%% === 误差评估（仅限白天）===
Pcs = T.P_cs(win+1:end);
Pcs = Pcs(is_test);
is_day = Pcs > 0;

nMAE = mean(abs(YPred(is_day) - YTest(is_day))) / 20000;
nRMSE = sqrt(mean((YPred(is_day) - YTest(is_day)).^2)) / 20000;

fprintf('nMAE = %.4f, nRMSE = %.4f\n', nMAE, nRMSE);

%% === 绘图 ===
figure;
plot(YTest,'k','DisplayName','实际');
hold on;
plot(YPred,'r--','DisplayName','预测');
xlabel('样本序号'); ylabel('功率(kW)');
title('光伏功率预测 - 简化LSTM'); legend show; grid on;


%% === Store results in required prediction table format ===
ts_all = T.date_time(win+1:end);
ts_test = ts_all(is_test);

% Forecast start time = every 7-day block’s first time
forecast_start_time = NaT(size(ts_test));
samples_per_week = 96 * 7;
for i = 1:samples_per_week:length(ts_test)
    i_end = min(i+samples_per_week-1, length(ts_test));
    forecast_start_time(i:i_end) = ts_test(i);
end

%% === Save table with full datetime format including 00:00:00 ===
result_table = table;
result_table.StartTime = ts_test - days(1);  
result_table.ForecastTime   = ts_test;
result_table.ActualPower_MW = YTest / 1000;
result_table.PredictedPower_MW = YPred / 1000;

% 格式化时间为完整格式，包括 00:00:00
result_table.StartTime    = datestr(result_table.StartTime, 'yyyy/mm/dd HH:MM:ss');
result_table.ForecastTime = datestr(result_table.ForecastTime, 'yyyy/mm/dd HH:MM:ss');

% 保存为 CSV
writetable(result_table, 'lstm_prediction_result_en.csv');
fprintf('Prediction results saved to lstm_prediction_result_en.csv\n');












