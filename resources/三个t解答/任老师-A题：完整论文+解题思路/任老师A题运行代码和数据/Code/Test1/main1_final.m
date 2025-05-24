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

%% 5. 全年理论功率曲线
figure;
plot(T.date_time,T.P_cs,'Color',[0.2 0.4 0.6],'LineWidth',1.3);
xlabel('时间'); ylabel('理论功率 P_{cs} (kW)');
title('全年 15分钟 理论发电功率'); grid on;

%% 6. 季节分类 + 中位曲线
T.season = strings(height(T),1);
T.season(ismember(T.month,[12,1,2])) = "Winter";
T.season(ismember(T.month,[3,4,5]))  = "Spring";
T.season(ismember(T.month,[6,7,8]))  = "Summer";
T.season(ismember(T.month,[9,10,11]))= "Autumn";

T.t_hhmm = string(datestr(T.date_time,'HH:MM'));
seasonLUT = ["Winter", "Spring", "Summer", "Autumn"];
colors = [0.55 0.65 0.78;
          0.86 0.78 0.65;
          0.74 0.86 0.78;
          0.92 0.70 0.74];  

figure; hold on;
for i = 1:4
    s = seasonLUT(i);
    idx = T.season == s;
    [grp, tags] = findgroups(T.t_hhmm(idx));
    pMed = splitapply(@median, T.P_cs(idx), grp);
    tags = string(tags);
    tAxis = duration(tags,'InputFormat','hh:mm');
    plot(tAxis, pMed, 'DisplayName', s, 'Color', colors(i,:), 'LineWidth', 1.8);
end
xlabel('时刻'); ylabel('P_{cs}(kW)');
title('四季典型日中位理论功率曲线'); legend show; grid on;

%% 7. 每季典型日理论功率 vs 实际功率
typical = ["2018-12-21","2019-03-21","2019-06-11","2018-09-21"];
figure;
for k = 1:4
    subplot(2,2,k);
    mask = string(datestr(T.date_time,'yyyy-mm-dd')) == typical(k);
    plot(T.date_time(mask),T.P_cs(mask),'k','LineWidth',1.6,'DisplayName','理论');
    hold on;
    plot(T.date_time(mask),T.power_clipped(mask),'r--','LineWidth',1.6,'DisplayName','实际');
    xlabel('时间'); ylabel('功率 (kW)');
    title("典型日 · "+typical(k)); legend show; grid on;
end

%% 8. 全天空指数 K_t
T.Kt = T.power_clipped ./ T.P_cs;
T.Kt(T.P_cs == 0) = NaN;

figure;
plot(T.date_time, T.Kt, '.', 'Color', [0.4 0.4 0.7]);
xlabel('时间'); ylabel('全天空指数 K_t');
title('全天空指数时间序列'); grid on;

%% 9. 容量因子 CF_d 与 偏差分析
T.date_only = dateshift(T.date_time,'start','day');
[gd,days] = findgroups(T.date_only);

CF_d = splitapply(@(x) sum(x)/(P_rated*24), T.power_clipped, gd);
biasMean = splitapply(@mean, T.power_clipped - T.P_cs, gd);

figure;
plot(days, CF_d, 'Color', [0.6 0.5 0.8], 'LineWidth', 1.3);
xlabel('日期'); ylabel('容量因子 CF_d');
title('每日容量因子'); grid on;

figure;
plot(days, biasMean, 'Color', [0.9 0.4 0.4], 'LineWidth', 1.3);
xlabel('日期'); ylabel('平均偏差 (kW)');
title('每日平均功率偏差（实际 - 理论）'); grid on;

%% 10. 年度性能比 PR
PR = sum(T.power_clipped,'omitnan') / sum(T.P_cs,'omitnan');
fprintf('全年性能比 PR = %.4f\n', PR);

%% 11. 年度实际 vs 理论 功率对比图
figure;
plot(T.date_time, T.P_cs, 'k-', 'LineWidth', 1.3, 'DisplayName','理论');
hold on;
plot(T.date_time, T.power_clipped, 'Color', [0.8 0.3 0.3], 'LineWidth', 1.3, 'DisplayName','实际（截断）');
xlabel('时间'); ylabel('功率 (kW)');
title('全年 实际 vs 理论 发电功率'); legend show; grid on;
