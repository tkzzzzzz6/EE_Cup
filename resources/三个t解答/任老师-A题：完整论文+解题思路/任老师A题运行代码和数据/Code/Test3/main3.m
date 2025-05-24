clc; clear;

%% 1. 站点参数
eta_STC = 270 / 1.6635e3;   % 组件效率 ≈0.1623
A_total = 74000;            % 组件总面积 m²
G_STC   = 1000;             % STC 辐照 W/m²
P_rated = 20000;            % 装机容量 kW
lat     = 38.18306;

%% 2. 读取数据
T = readtable('station01.csv','TextType','string');
rawStr  = string(T.date_time);
rawStr  = replace(rawStr,'  ',' ');
parts   = split(rawStr,' ');
dateP   = split(parts(:,1),'-'); timeP = split(parts(:,2),':');
T.date_time = datetime(str2double(dateP(:,1)),str2double(dateP(:,2)), ...
    str2double(dateP(:,3)),str2double(timeP(:,1)), ...
    str2double(timeP(:,2)),str2double(timeP(:,3))) + hours(8);
T.power = T.power * 1000;

%% 3. 理论功率 P_cs
my_doy   = day(T.date_time,'dayofyear');
my_hourf = hour(T.date_time) + minute(T.date_time)/60;
delta = deg2rad(23.45) .* sin(deg2rad(360 * (284 + my_doy) ./ 365));
H     = deg2rad(15 .* (my_hourf - 12));
phi   = deg2rad(lat);
sin_h = sin(phi).*sin(delta) + cos(phi).*cos(delta).*cos(H);
sin_h(sin_h < 0) = 0;
G_cs = G_STC .* sin_h;
T.P_cs = eta_STC .* G_cs .* A_total ./ G_STC;
T.power_clipped = min(T.power, T.P_cs);

%% 4. 构建序列输入 (含NWP特征)
win = 96;
featureNames = {'nwp_globalirrad','nwp_directirrad','nwp_temperature','nwp_humidity', ...
    'nwp_windspeed','nwp_winddirection','nwp_pressure'};
F = T{:,featureNames};  % NWP特征
P = T.power_clipped;
t = T.date_time;

X = {}; Y = [];
for i = win+1:length(P)
    if all(~isnan(P(i-win:i))) && all(all(~isnan(F(i-win:i,:))))
        seq = [P(i-win:i-1), F(i-win+1:i,:)];  % 功率 + NWP滑窗
        X{end+1} = seq';  % [feature x timestep]
        Y(end+1,1) = P(i);
    end
end

%% 5. 训练集/验证集/测试集划分
ts = t(win+1:end);
mon = month(ts); dayN = day(ts);
is_test = false(size(ts));
for m = [2 5 8 11]
    idx = find(mon == m);
    last7 = idx(end-7*96+1:end);
    is_test(last7) = true;
end

X = X(~isnan(Y)); Y = Y(~isnan(Y));  % 清理NaN


XTest = X(is_test); YTest = Y(is_test);
XTrain = X(~is_test); YTrain = Y(~is_test);

% 取训练集后10%作为验证集
nTrain = numel(XTrain);
nVal = round(0.1 * nTrain);
XVal = XTrain(end-nVal+1:end); YVal = YTrain(end-nVal+1:end);
XTrain = XTrain(1:end-nVal);   YTrain = YTrain(1:end-nVal);


%% 6. LSTM模型定义
inputSize = size(XTrain{1},1);
layers = [
    sequenceInputLayer(inputSize)
    lstmLayer(64, 'OutputMode', 'last')
    fullyConnectedLayer(1)
    regressionLayer
];
opts = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 128, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'ValidationData',{XVal, YVal}, ...
    'ValidationFrequency', 50, ...
    'Verbose',false);

%% 7. 模型训练

net = trainNetwork(XTrain, YTrain, layers, opts);

%% 8. 测试集预测与误差
YPred = predict(net, XTest, 'MiniBatchSize', 1);
Pcs = T.P_cs(win+1:end);
Pcs = Pcs(is_test);
is_day = Pcs > 0;

nMAE = mean(abs(YPred(is_day) - YTest(is_day))) / P_rated;
nRMSE = sqrt(mean((YPred(is_day) - YTest(is_day)).^2)) / P_rated;
fprintf('nMAE = %.4f, nRMSE = %.4f\n', EMAE, ERMSE);

%% 9. 绘图
figure;
plot(YTest,'k'); hold on;
plot(YPred,'r--'); legend('实际','预测');
xlabel('样本'); ylabel('功率(kW)'); title('LSTM + NWP'); grid on;

%% === 10. 结果表格生成与保存 ===
ts_all = T.date_time(win+1:end);
ts_test = ts_all(is_test);

forecast_start_time = NaT(size(ts_test));
samples_per_week = 96 * 7;
for i = 1:samples_per_week:length(ts_test)
    i_end = min(i+samples_per_week-1, length(ts_test));
    forecast_start_time(i:i_end) = ts_test(i);
end

result_table = table;
result_table.StartTime          = forecast_start_time - days(1);  
result_table.ForecastTime       = ts_test;
result_table.ActualPower_MW     = YTest / 1000;
result_table.PredictedPower_MW  = YPred / 1000;

result_table.StartTime    = datestr(result_table.StartTime, 'yyyy/mm/dd HH:MM:ss');
result_table.ForecastTime = datestr(result_table.ForecastTime, 'yyyy/mm/dd HH:MM:ss');

writetable(result_table, 'lstm_prediction_result_with_nwp.csv');
fprintf('预测结果已保存至 lstm_prediction_result_with_nwp.csv\n');

%% === 11. 按全天空指数分类，计算不同天气下的RMSE ===
% 提取对应测试集的实际功率、预测功率和理论功率
P_actual = YTest;
P_pred   = YPred;
P_theory = Pcs;

% 计算全天空指数 Kt
Kt = P_actual ./ P_theory;
Kt(P_theory == 0) = NaN;  % 避免除零

% 分类阈值
is_clear   = Kt >= 0.7;
is_cloudy  = (Kt >= 0.4) & (Kt < 0.7);
is_overcast = Kt < 0.4;

% 分别计算 RMSE（仅限于有效时间点）
rmse_clear = sqrt(mean((P_actual(is_clear) - P_pred(is_clear)).^2, 'omitnan'));
rmse_cloudy = sqrt(mean((P_actual(is_cloudy) - P_pred(is_cloudy)).^2, 'omitnan'));
rmse_overcast = sqrt(mean((P_actual(is_overcast) - P_pred(is_overcast)).^2, 'omitnan'));

% 打印结果
fprintf('\n不同天气条件下的预测 RMSE (单位: kW)：\n');
fprintf('晴天（Kt ≥ 0.7）：       %.4f kW\n', rmse_clear);
fprintf('多云（0.4 ≤ Kt < 0.7）：%.4f kW\n', rmse_cloudy);
fprintf('阴天（Kt < 0.4）：       %.4f kW\n', rmse_overcast);
