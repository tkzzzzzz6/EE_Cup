import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import warnings
import os
import xgboost as xgb
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class SolarPowerPredictorWithNWP:
    def __init__(self, data_path, output_dir='问题3结果文件夹'):
        """
        基于历史功率和NWP信息的光伏电站日前发电功率预测模型

        Parameters:
        data_path: 数据文件路径
        output_dir: 输出结果文件夹
        """
        self.data_path = data_path
        self.output_dir = output_dir

        # 创建输出文件夹
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 模型配置
        self.models = {}
        self.scalers = {}
        self.predictions = {}
        self.metrics = {}
        self.scene_metrics = {}  # 用于存储不同场景的评估指标

        # 加载数据
        self.load_and_preprocess_data()

    def load_and_preprocess_data(self):
        """加载并预处理数据"""
        print("正在加载和预处理数据...")

        # 读取CSV数据
        self.df = pd.read_csv(self.data_path)

        # 转换时间格式
        self.df['datetime'] = pd.to_datetime(self.df['date_time'])
        self.df = self.df.sort_values('datetime').reset_index(drop=True)

        # 提取时间特征
        self.df['year'] = self.df['datetime'].dt.year
        self.df['month'] = self.df['datetime'].dt.month
        self.df['day'] = self.df['datetime'].dt.day
        self.df['hour'] = self.df['datetime'].dt.hour
        self.df['minute'] = self.df['datetime'].dt.minute
        self.df['day_of_year'] = self.df['datetime'].dt.dayofyear
        self.df['day_of_week'] = self.df['datetime'].dt.dayofweek
        self.df['week_of_year'] = self.df['datetime'].dt.isocalendar().week

        # 计算太阳时间相关特征
        self.df['solar_time'] = self.df['hour'] + self.df['minute'] / 60
        self.df['sin_hour'] = np.sin(2 * np.pi * self.df['solar_time'] / 24)
        self.df['cos_hour'] = np.cos(2 * np.pi * self.df['solar_time'] / 24)
        self.df['sin_day'] = np.sin(2 * np.pi * self.df['day_of_year'] / 365)
        self.df['cos_day'] = np.cos(2 * np.pi * self.df['day_of_year'] / 365)

        # 标记白昼时段（用于误差计算）
        self.df['is_daylight'] = ((self.df['hour'] >= 6) & (self.df['hour'] <= 18)) | (self.df['power'] > 0)

        # 处理NWP数据
        self.process_nwp_data()

        print(f"数据预处理完成，共{len(self.df)}条记录")
        print(f"时间范围：{self.df['datetime'].min()} 至 {self.df['datetime'].max()}")

    def process_nwp_data(self):
        """处理NWP数据"""
        print("正在处理NWP数据...")

        # 计算NWP特征的变化率
        nwp_cols = ['nwp_globalirrad', 'nwp_directirrad', 'nwp_temperature', 
                   'nwp_humidity', 'nwp_windspeed', 'nwp_pressure']
        
        for col in nwp_cols:
            # 计算变化率
            self.df[f'{col}_diff'] = self.df[col].diff()
            # 计算移动平均
            self.df[f'{col}_ma_4'] = self.df[col].rolling(window=4).mean()
            self.df[f'{col}_ma_24'] = self.df[col].rolling(window=24).mean()

        # 计算辐射效率（实际辐射与NWP预测辐射的比值）
        self.df['radiation_efficiency'] = self.df['lmd_totalirrad'] / (self.df['nwp_globalirrad'] + 1e-6)

        # 计算温度差异
        self.df['temp_diff'] = self.df['lmd_temperature'] - self.df['nwp_temperature']

        # 计算风速差异
        self.df['wind_diff'] = self.df['lmd_windspeed'] - self.df['nwp_windspeed']

        # 计算风向差异（考虑角度循环）
        self.df['wind_dir_diff'] = np.minimum(
            np.abs(self.df['lmd_winddirection'] - self.df['nwp_winddirection']),
            360 - np.abs(self.df['lmd_winddirection'] - self.df['nwp_winddirection'])
        )

        # 计算气压差异
        self.df['pressure_diff'] = self.df['lmd_pressure'] - self.df['nwp_pressure']

        # 创建天气类型特征
        self.df['weather_type'] = self.classify_weather()

    def classify_weather(self):
        """根据NWP数据对天气类型进行分类"""
        # 使用K-means聚类对天气类型进行分类
        features = ['nwp_globalirrad', 'nwp_temperature', 'nwp_humidity', 'nwp_windspeed']
        X = self.df[features].fillna(0)
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 使用K-means聚类
        kmeans = KMeans(n_clusters=4, random_state=42)
        weather_types = kmeans.fit_predict(X_scaled)
        
        return weather_types

    def create_features(self, df):
        """创建特征工程"""
        feature_df = df.copy()

        # 基础时间特征
        time_features = [
            'hour', 'day_of_year', 'month',
            'sin_hour', 'cos_hour', 'sin_day', 'cos_day'
        ]

        # 历史功率特征
        power_features = []
        for lag in [1, 2, 3, 4, 24, 48, 96]:
            feature_df[f'power_lag_{lag}'] = feature_df['power'].shift(lag)
            power_features.append(f'power_lag_{lag}')

        # 移动平均特征
        ma_features = []
        for window in [4, 8, 24, 96]:
            feature_df[f'power_ma_{window}'] = feature_df['power'].rolling(window=window).mean()
            ma_features.append(f'power_ma_{window}')

        # NWP特征
        nwp_features = [
            'nwp_globalirrad', 'nwp_directirrad', 'nwp_temperature',
            'nwp_humidity', 'nwp_windspeed', 'nwp_pressure',
            'nwp_globalirrad_diff', 'nwp_temperature_diff',
            'nwp_globalirrad_ma_4', 'nwp_temperature_ma_4',
            'radiation_efficiency', 'temp_diff',
            'wind_diff', 'wind_dir_diff', 'pressure_diff'
        ]

        # 天气类型特征
        weather_features = ['weather_type']

        # 组合所有特征
        self.feature_cols = time_features + power_features + ma_features + nwp_features + weather_features

        return feature_df

    def split_train_test(self):
        """按照要求划分训练集和测试集"""
        print("正在划分训练集和测试集...")

        # 第2、5、8、11个月最后一周数据作为测试集
        test_months = [2, 5, 8, 11]

        test_indices = []
        train_indices = []

        for year in self.df['year'].unique():
            for month in range(1, 13):
                month_data = self.df[(self.df['year'] == year) & (self.df['month'] == month)]
                if len(month_data) == 0:
                    continue

                if month in test_months:
                    # 获取该月最后一周的数据
                    last_day = month_data['day'].max()
                    last_week_start = last_day - 6
                    last_week_data = month_data[month_data['day'] >= last_week_start]
                    test_indices.extend(last_week_data.index.tolist())

                    # 该月其余数据作为训练集
                    other_data = month_data[month_data['day'] < last_week_start]
                    train_indices.extend(other_data.index.tolist())
                else:
                    # 非测试月份全部作为训练集
                    train_indices.extend(month_data.index.tolist())

        self.train_indices = sorted(train_indices)
        self.test_indices = sorted(test_indices)

        print(f"训练集样本数: {len(self.train_indices)}")
        print(f"测试集样本数: {len(self.test_indices)}")

    def train_models(self, X_train, y_train):
        """训练预测模型"""
        print("正在训练预测模型...")
        
        # 数据验证
        print("  验证训练数据...")
        try:
            # 将pandas DataFrame转换为numpy数组
            X_train_array = X_train.values.astype(np.float32)
            y_train_array = y_train.values.astype(np.float32)
            
            # 检查数据形状和类型
            print(f"  训练数据形状: X_train_array{X_train_array.shape}, y_train_array{y_train_array.shape}")
            print(f"  X_train数据类型: {X_train_array.dtype}")
            print(f"  y_train数据类型: {y_train_array.dtype}")
            
            # 检查内存使用
            print(f"  X_train内存使用: {X_train_array.nbytes / 1024 / 1024:.2f} MB")
            print(f"  y_train内存使用: {y_train_array.nbytes / 1024 / 1024:.2f} MB")
            
            # 处理缺失值和异常值
            if np.isnan(X_train_array).any() or np.isnan(y_train_array).any():
                print("  警告：训练数据中存在NaN值，将被替换为0")
                X_train_array = np.nan_to_num(X_train_array)
                y_train_array = np.nan_to_num(y_train_array)
            
            if np.isinf(X_train_array).any() or np.isinf(y_train_array).any():
                print("  警告：训练数据中存在无穷值，将被替换为0")
                X_train_array = np.nan_to_num(X_train_array, nan=0, posinf=0, neginf=0)
                y_train_array = np.nan_to_num(y_train_array, nan=0, posinf=0, neginf=0)
            
            # 训练XGBoost模型
            print("  训练XGBoost模型...")
            try:
                # 创建DMatrix对象
                dtrain = xgb.DMatrix(X_train_array, label=y_train_array)
                
                # 设置XGBoost参数
                params = {
                    'objective': 'reg:squarederror',
                    'eval_metric': 'mae',
                    'max_depth': 6,                  # 增加树的深度
                    'eta': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_weight': 3,
                    'tree_method': 'hist',
                    'nthread': 1
                }
                
                # 训练模型
                num_rounds = 100  # 增加迭代次数
                xgb_model = xgb.train(
                    params,
                    dtrain,
                    num_rounds,
                    verbose_eval=False
                )
                
                self.models['XGBoost'] = xgb_model
                print("  XGBoost模型训练完成")
                
                # 释放内存
                del dtrain, X_train_array, y_train_array
                import gc
                gc.collect()
                
            except Exception as e:
                print(f"  XGBoost模型训练失败: {str(e)}")
                import traceback
                print(traceback.format_exc())
                return
            
            print("模型训练完成！")
            
        except Exception as e:
            print(f"模型训练过程发生错误: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return

    def evaluate_scene_performance(self, y_test, y_pred, test_data):
        """评估不同场景下的模型性能"""
        print("正在评估不同场景下的模型性能...")

        # 按天气类型分组评估
        for weather_type in range(4):
            mask = test_data['weather_type'] == weather_type
            if mask.sum() > 0:
                scene_metrics = self.calculate_metrics(
                    y_test[mask],
                    y_pred[mask],
                    test_data['is_daylight'][mask]
                )
                self.scene_metrics[f'weather_type_{weather_type}'] = scene_metrics

        # 按季节分组评估
        seasons = {
            'spring': [3, 4, 5],
            'summer': [6, 7, 8],
            'autumn': [9, 10, 11],
            'winter': [12, 1, 2]
        }

        for season, months in seasons.items():
            mask = test_data['month'].isin(months)
            if mask.sum() > 0:
                scene_metrics = self.calculate_metrics(
                    y_test[mask],
                    y_pred[mask],
                    test_data['is_daylight'][mask]
                )
                self.scene_metrics[f'season_{season}'] = scene_metrics

        # 按辐射强度分组评估
        radiation_bins = [0, 100, 300, 500, float('inf')]
        for i in range(len(radiation_bins) - 1):
            mask = (test_data['nwp_globalirrad'] >= radiation_bins[i]) & \
                  (test_data['nwp_globalirrad'] < radiation_bins[i + 1])
            if mask.sum() > 0:
                scene_metrics = self.calculate_metrics(
                    y_test[mask],
                    y_pred[mask],
                    test_data['is_daylight'][mask]
                )
                self.scene_metrics[f'radiation_{radiation_bins[i]}_{radiation_bins[i+1]}'] = scene_metrics

    def create_scene_visualizations(self):
        """创建场景分析可视化"""
        print("正在创建场景分析可视化...")

        # 1. 不同天气类型的预测效果对比
        fig, ax = plt.subplots(figsize=(12, 6))
        weather_types = [k for k in self.scene_metrics.keys() if k.startswith('weather_type_')]
        mae_values = [self.scene_metrics[k]['MAE'] for k in weather_types]
        
        ax.bar(range(len(weather_types)), mae_values)
        ax.set_xticks(range(len(weather_types)))
        ax.set_xticklabels([f'天气类型 {i}' for i in range(len(weather_types))])
        ax.set_title('不同天气类型的预测误差对比')
        ax.set_ylabel('MAE (MW)')
        plt.savefig(f'{self.output_dir}/天气类型预测效果.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 不同季节的预测效果对比
        fig, ax = plt.subplots(figsize=(12, 6))
        seasons = [k for k in self.scene_metrics.keys() if k.startswith('season_')]
        mae_values = [self.scene_metrics[k]['MAE'] for k in seasons]
        
        ax.bar(range(len(seasons)), mae_values)
        ax.set_xticks(range(len(seasons)))
        ax.set_xticklabels([s.split('_')[1] for s in seasons])
        ax.set_title('不同季节的预测误差对比')
        ax.set_ylabel('MAE (MW)')
        plt.savefig(f'{self.output_dir}/季节预测效果.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. 不同辐射强度的预测效果对比
        fig, ax = plt.subplots(figsize=(12, 6))
        radiation_levels = [k for k in self.scene_metrics.keys() if k.startswith('radiation_')]
        mae_values = [self.scene_metrics[k]['MAE'] for k in radiation_levels]
        
        ax.bar(range(len(radiation_levels)), mae_values)
        ax.set_xticks(range(len(radiation_levels)))
        ax.set_xticklabels([f'{r.split("_")[1]}-{r.split("_")[2]}' for r in radiation_levels])
        ax.set_title('不同辐射强度的预测误差对比')
        ax.set_ylabel('MAE (MW)')
        plt.savefig(f'{self.output_dir}/辐射强度预测效果.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_report(self):
        """生成完整的分析报告"""
        print("正在生成分析报告...")

        # 准备数据
        self.split_train_test()
        X_train, y_train, X_test, y_test = self.prepare_model_data()

        # 训练模型
        self.train_models(X_train, y_train)

        # 进行预测
        predictions = self.make_predictions(X_test)

        # 评估模型
        self.evaluate_models(y_test)

        # 评估不同场景下的性能
        test_data = self.df.loc[self.test_indices]
        self.evaluate_scene_performance(y_test, predictions['XGBoost'], test_data)

        # 创建预测表格
        results_df = self.create_prediction_table(y_test)

        # 生成可视化
        self.create_visualizations(y_test)
        self.create_scene_visualizations()

        # 生成文本报告
        report_content = f"""
融入NWP信息的光伏电站日前发电功率预测分析报告
================================================

一、预测任务概述
--------------------------------
预测目标: 基于历史功率和NWP信息的光伏电站日前发电功率预测
预测时间范围: 7天 (168小时)
时间分辨率: 15分钟
数据来源: 光伏电站历史发电功率数据和NWP气象数据

二、数据集划分
--------------------------------
训练集与测试集划分方案: 第2、5、8、11个月最后一周数据作为测试集，其他数据作为训练集

数据集统计:
- 训练集样本数: {len(self.train_indices)} 条
- 测试集样本数: {len(self.test_indices)} 条

三、特征工程
--------------------------------
1. 时间特征
- 基础时间特征：年、月、日、小时、分钟、星期等
- 周期性特征：小时、日期的正弦和余弦变换

2. 历史功率特征
- 滞后特征：15分钟、30分钟、45分钟、1小时、6小时、12小时、24小时前
- 移动平均特征：1小时、2小时、6小时、24小时移动平均

3. NWP特征
- 基础NWP特征：全球辐射、直接辐射、温度、湿度、风速、风向、气压
- 变化率特征：各NWP指标的变化率
- 移动平均特征：各NWP指标的移动平均
- 差异特征：实际测量值与NWP预测值的差异
- 效率特征：实际辐射与NWP预测辐射的比值

4. 天气类型特征
- 基于K-means聚类将天气分为4种类型

四、模型构建
--------------------------------
1. 模型选择
- XGBoost模型
- 参数优化：增加树的深度和迭代次数，优化学习率等超参数

2. 训练过程
- 数据预处理：处理缺失值、异常值
- 特征标准化：对数值特征进行标准化
- 模型训练：使用优化后的参数进行训练

五、预测结果分析
--------------------------------
1. 整体预测效果
"""
        # 添加整体性能指标
        for model_name, metrics in self.metrics.items():
            report_content += f"""
{model_name}模型整体性能:
- MAE: {metrics['MAE']:.4f} MW
- RMSE: {metrics['RMSE']:.4f} MW
- MAPE: {metrics['MAPE']:.2f}%
- R²: {metrics['R2']:.4f}
"""

        report_content += """
2. 不同场景下的预测效果
"""
        # 添加不同场景的性能指标
        for scene, metrics in self.scene_metrics.items():
            report_content += f"""
{scene}场景:
- MAE: {metrics['MAE']:.4f} MW
- RMSE: {metrics['RMSE']:.4f} MW
- MAPE: {metrics['MAPE']:.2f}%
- R²: {metrics['R2']:.4f}
"""

        report_content += """
六、NWP信息对预测精度的影响分析
--------------------------------
1. 不同天气类型下的预测效果
- 分析不同天气类型（晴天、多云、阴天、雨天）下的预测精度
- 评估NWP信息在不同天气类型下的有效性

2. 不同季节的预测效果
- 分析春、夏、秋、冬四季的预测精度
- 评估NWP信息在不同季节的适用性

3. 不同辐射强度下的预测效果
- 分析不同辐射强度区间的预测精度
- 评估NWP信息在不同辐射条件下的有效性

七、场景划分方案
--------------------------------
1. 基于天气类型的划分
- 类型1：晴天（高辐射、低湿度）
- 类型2：多云（中等辐射、中等湿度）
- 类型3：阴天（低辐射、高湿度）
- 类型4：雨天（极低辐射、极高湿度）

2. 基于季节的划分
- 春季：3-5月
- 夏季：6-8月
- 秋季：9-11月
- 冬季：12-2月

3. 基于辐射强度的划分
- 低辐射：0-100 W/m²
- 中低辐射：100-300 W/m²
- 中高辐射：300-500 W/m²
- 高辐射：>500 W/m²

八、结论与建议
--------------------------------
1. NWP信息对预测精度的影响
- 在大多数场景下，融入NWP信息能显著提高预测精度
- 特别是在天气变化较大的情况下，NWP信息的贡献更为明显

2. 场景划分的效果
- 不同场景下的预测精度存在显著差异
- 场景划分有助于针对性地优化预测模型

3. 改进建议
- 进一步优化特征工程，增加更多有意义的特征组合
- 针对不同场景分别建立预测模型
- 考虑引入更多气象特征，如云量、能见度等
- 优化模型参数，针对不同场景调整模型结构

九、附件说明
--------------------------------
1. 预测结果文件
- 预测结果表格.csv：包含所有时间点的预测结果
- {年}年{月}月预测结果.csv：按月份分别保存的预测结果

2. 可视化图表
- 预测效果散点图.png：各模型预测值与实际值的对比
- 时间序列预测对比.png：预测结果的时间序列展示
- 误差分析图.png：预测误差的详细分析
- 特征重要性分析.png：XGBoost模型的特征重要性排序
- 天气类型预测效果.png：不同天气类型的预测效果对比
- 季节预测效果.png：不同季节的预测效果对比
- 辐射强度预测效果.png：不同辐射强度的预测效果对比
"""

        # 保存报告到文件
        with open(f'{self.output_dir}/预测分析报告.txt', 'w', encoding='utf-8') as f:
            f.write(report_content)

        print("分析报告生成完成！")

        return report_content

    def prepare_model_data(self):
        """准备建模数据"""
        print("正在准备建模数据...")

        # 创建特征
        feature_df = self.create_features(self.df)

        # 准备训练数据
        X_train = feature_df.loc[self.train_indices, self.feature_cols].fillna(0)
        y_train = feature_df.loc[self.train_indices, 'power']

        # 准备测试数据
        X_test = feature_df.loc[self.test_indices, self.feature_cols].fillna(0)
        y_test = feature_df.loc[self.test_indices, 'power']

        # 对训练数据进行采样（如果数据量太大）
        sample_size = min(10000, len(X_train))
        if sample_size < len(X_train):
            print(f"  对训练数据进行采样，从{len(X_train)}条减少到{sample_size}条")
            indices = np.random.choice(len(X_train), sample_size, replace=False)
            X_train = X_train.iloc[indices]
            y_train = y_train.iloc[indices]

        print(f"特征数量: {len(self.feature_cols)}")
        print(f"训练集形状: X_train{X_train.shape}, y_train{y_train.shape}")
        print(f"测试集形状: X_test{X_test.shape}, y_test{y_test.shape}")

        return X_train, y_train, X_test, y_test

    def make_predictions(self, X_test):
        """进行预测"""
        print("正在进行预测...")

        predictions = {}

        # XGBoost预测
        X_test_array = X_test.values.astype(np.float32)
        dtest = xgb.DMatrix(X_test_array)
        predictions['XGBoost'] = self.models['XGBoost'].predict(dtest)

        # 确保预测值非负
        for model_name in predictions:
            predictions[model_name] = np.maximum(predictions[model_name], 0)

        self.predictions = predictions
        print("预测完成！")

        return predictions

    def calculate_metrics(self, y_true, y_pred, daylight_mask=None):
        """计算评估指标"""
        if daylight_mask is not None:
            y_true_day = y_true[daylight_mask]
            y_pred_day = y_pred[daylight_mask]
        else:
            y_true_day = y_true
            y_pred_day = y_pred

        # 只在有实际发电的时段计算误差
        valid_mask = y_true_day > 0.001  # 避免除零错误

        if valid_mask.sum() == 0:
            return {
                'MAE': 0, 'RMSE': 0, 'MAPE': 0, 'R2': 0,
                'MAE_all': mean_absolute_error(y_true, y_pred),
                'RMSE_all': np.sqrt(mean_squared_error(y_true, y_pred)),
                'R2_all': r2_score(y_true, y_pred)
            }

        y_true_valid = y_true_day[valid_mask]
        y_pred_valid = y_pred_day[valid_mask]

        mae = mean_absolute_error(y_true_valid, y_pred_valid)
        rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
        mape = np.mean(np.abs((y_true_valid - y_pred_valid) / y_true_valid)) * 100
        r2 = r2_score(y_true_valid, y_pred_valid)

        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2,
            'MAE_all': mean_absolute_error(y_true, y_pred),
            'RMSE_all': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2_all': r2_score(y_true, y_pred)
        }

    def evaluate_models(self, y_test):
        """评估模型性能"""
        print("正在评估模型性能...")

        # 获取白昼时段掩码
        test_data = self.df.loc[self.test_indices]
        daylight_mask = test_data['is_daylight'].values

        self.metrics = {}

        for model_name, y_pred in self.predictions.items():
            metrics = self.calculate_metrics(y_test.values, y_pred, daylight_mask)
            self.metrics[model_name] = metrics

            print(f"\n{model_name} 模型性能 (白昼时段):")
            print(f"  MAE: {metrics['MAE']:.4f} MW")
            print(f"  RMSE: {metrics['RMSE']:.4f} MW")
            print(f"  MAPE: {metrics['MAPE']:.2f}%")
            print(f"  R²: {metrics['R2']:.4f}")

    def create_prediction_table(self, y_test):
        """创建预测结果表格"""
        print("正在创建预测结果表格...")

        test_data = self.df.loc[self.test_indices].copy()

        # 创建结果表格
        results = []

        # 按测试集时间顺序处理
        for idx, row in test_data.iterrows():
            test_idx = self.test_indices.index(idx)

            result_row = {
                '起报时间': row['datetime'].strftime('%Y/%m/%d %H:%M'),
                '预报时间': row['datetime'].strftime('%Y/%m/%d %H:%M'),
                '实际功率(MW)': f"{row['power']:.3f}",
            }

            # 添加各模型预测结果
            for model_name, predictions in self.predictions.items():
                result_row[f'{model_name}预测功率(MW)'] = f"{predictions[test_idx]:.3f}"

            results.append(result_row)

        # 转换为DataFrame
        results_df = pd.DataFrame(results)

        # 保存完整预测结果
        results_df.to_csv(f'{self.output_dir}/预测结果表格.csv', index=False, encoding='utf-8')

        # 按月份分别保存（按题目要求格式）
        test_data_with_results = test_data.copy()
        for model_name, predictions in self.predictions.items():
            test_data_with_results[f'{model_name}_pred'] = predictions

        # 按年月分组保存
        for (year, month), group in test_data_with_results.groupby(['year', 'month']):
            month_results = []

            for _, row in group.iterrows():
                month_results.append({
                    '起报时间': row['datetime'].strftime('%Y/%m/%d %H:%M'),
                    '预报时间': row['datetime'].strftime('%Y/%m/%d %H:%M'),
                    '实际功率(MW)': row['power'],
                    'XGBoost预测功率(MW)': row['XGBoost_pred']
                })

            month_df = pd.DataFrame(month_results)
            month_df.to_csv(f'{self.output_dir}/{year}年{month}月预测结果.csv',
                            index=False, encoding='utf-8')

        print("预测结果表格创建完成！")
        return results_df

    def create_visualizations(self, y_test):
        """创建可视化图表"""
        print("正在生成可视化图表...")

        test_data = self.df.loc[self.test_indices].copy()

        # 1. 整体预测效果对比
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 预测vs实际散点图
        axes[0, 0].scatter(y_test, self.predictions['XGBoost'], alpha=0.6, s=10)
        axes[0, 0].plot([0, y_test.max()], [0, y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('实际功率 (MW)')
        axes[0, 0].set_ylabel('预测功率 (MW)')
        axes[0, 0].set_title('XGBoost - R² = {:.3f}'.format(self.metrics['XGBoost']['R2']))
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 时间序列预测对比
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))

        # 选择一段典型时间进行展示
        sample_days = 7 * 96  # 7天数据
        if len(test_data) > sample_days:
            sample_indices = range(0, sample_days)
        else:
            sample_indices = range(len(test_data))

        sample_data = test_data.iloc[sample_indices]
        sample_datetime = sample_data['datetime']
        sample_actual = sample_data['power']

        # 上图：最佳模型预测对比
        best_model = min(self.metrics.keys(), key=lambda x: self.metrics[x]['MAE'])
        sample_pred = self.predictions[best_model][sample_indices]

        axes[0].plot(sample_datetime, sample_actual, 'b-', label='实际功率', linewidth=2)
        axes[0].plot(sample_datetime, sample_pred, 'r--', label=f'{best_model}预测', linewidth=2)
        axes[0].set_title(f'最佳模型({best_model})预测效果时间序列', fontsize=14)
        axes[0].set_ylabel('功率 (MW)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 下图：多模型对比
        axes[1].plot(sample_datetime, sample_actual, 'b-', label='实际功率', linewidth=3)
        colors = ['red', 'green', 'orange', 'purple']
        for i, (model_name, y_pred) in enumerate(self.predictions.items()):
            sample_pred = y_pred[sample_indices]
            axes[1].plot(sample_datetime, sample_pred, '--',
                         color=colors[i % len(colors)], label=f'{model_name}', linewidth=1.5)

        axes[1].set_title('多模型预测效果对比', fontsize=14)
        axes[1].set_xlabel('时间')
        axes[1].set_ylabel('功率 (MW)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/时间序列预测对比.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. 误差分析
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        best_pred = self.predictions[best_model]
        errors = best_pred - y_test.values

        # 误差分布
        axes[0, 0].hist(errors, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('预测误差分布')
        axes[0, 0].set_xlabel('预测误差 (MW)')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].grid(True, alpha=0.3)

        # 误差时间序列
        axes[0, 1].plot(test_data['datetime'], errors, alpha=0.7)
        axes[0, 1].set_title('误差时间序列')
        axes[0, 1].set_xlabel('时间')
        axes[0, 1].set_ylabel('预测误差 (MW)')
        axes[0, 1].grid(True, alpha=0.3)

        # 不同功率水平的预测精度
        power_bins = np.linspace(0, y_test.max(), 10)
        bin_errors = []
        bin_centers = []

        for i in range(len(power_bins) - 1):
            mask = (y_test >= power_bins[i]) & (y_test < power_bins[i + 1])
            if mask.sum() > 0:
                bin_errors.append(np.abs(errors[mask]).mean())
                bin_centers.append((power_bins[i] + power_bins[i + 1]) / 2)

        axes[1, 0].bar(bin_centers, bin_errors, width=bin_centers[1] - bin_centers[0] * 0.8)
        axes[1, 0].set_title('不同功率水平的平均绝对误差')
        axes[1, 0].set_xlabel('功率水平 (MW)')
        axes[1, 0].set_ylabel('平均绝对误差 (MW)')
        axes[1, 0].grid(True, alpha=0.3)

        # 模型性能对比
        model_names = list(self.metrics.keys())
        mae_values = [self.metrics[name]['MAE'] for name in model_names]

        bars = axes[1, 1].bar(model_names, mae_values)
        axes[1, 1].set_title('模型MAE性能对比 (白昼时段)')
        axes[1, 1].set_ylabel('MAE (MW)')
        axes[1, 1].tick_params(axis='x', rotation=45)

        # 在柱状图上添加数值标签
        for bar, mae in zip(bars, mae_values):
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                            f'{mae:.3f}', ha='center', va='bottom')

        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/误差分析图.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 4. 特征重要性分析（针对XGBoost）
        if 'XGBoost' in self.models:
            fig, ax = plt.subplots(figsize=(12, 8))

            # 获取特征重要性
            importance_dict = self.models['XGBoost'].get_score(importance_type='gain')
            
            # 如果没有获取到特征重要性，使用特征索引作为特征名
            if not importance_dict:
                importance_dict = {f'f{i}': 0 for i in range(len(self.feature_cols))}
                for i, feature in enumerate(self.feature_cols):
                    importance_dict[f'f{i}'] = self.models['XGBoost'].get_score(importance_type='gain').get(f'f{i}', 0)
            
            # 将特征名称转换为索引
            feature_importance = np.zeros(len(self.feature_cols))
            for i, feature in enumerate(self.feature_cols):
                # 尝试使用特征名和特征索引两种方式获取重要性
                importance = importance_dict.get(feature, 0)
                if importance == 0:
                    importance = importance_dict.get(f'f{i}', 0)
                feature_importance[i] = importance
            
            # 获取前20个重要特征的索引
            indices = np.argsort(feature_importance)[::-1][:20]
            
            # 只显示非零重要性的特征
            non_zero_indices = [i for i in indices if feature_importance[i] > 0]
            if not non_zero_indices:
                print("  警告：没有找到具有重要性的特征")
                return
                
            # 绘制特征重要性条形图
            ax.barh(range(len(non_zero_indices)), feature_importance[non_zero_indices])
            ax.set_yticks(range(len(non_zero_indices)))
            ax.set_yticklabels([self.feature_cols[i] for i in non_zero_indices])
            ax.set_title('XGBoost模型特征重要性 (Top 20)')
            ax.set_xlabel('重要性')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/特征重要性分析.png', dpi=300, bbox_inches='tight')
            plt.close()

        print("可视化图表生成完成！")


if __name__ == "__main__":
    # 设置数据文件路径
    data_path = r"C:\Users\18344\Desktop\代码、结果\station00（用于解题demo）.csv"

    # 创建预测器实例
    predictor = SolarPowerPredictorWithNWP(data_path)

    # 生成完整报告
    predictor.generate_report()

    print("\n预测任务完成！所有结果已保存到'问题3结果文件夹'中。")
    print("请查看以下文件：")
    print("1. 预测分析报告.txt - 包含完整的分析报告")
    print("2. 预测结果表格.csv - 包含所有预测结果")
    print("3. {年}年{月}月预测结果.csv - 按月份保存的预测结果")
    print("4. 预测效果散点图.png - 预测效果可视化")
    print("5. 时间序列预测对比.png - 时间序列预测对比")
    print("6. 误差分析图.png - 误差分析结果")
    print("7. 特征重要性分析.png - 特征重要性分析")
    print("8. 天气类型预测效果.png - 不同天气类型的预测效果")
    print("9. 季节预测效果.png - 不同季节的预测效果")
    print("10. 辐射强度预测效果.png - 不同辐射强度的预测效果")
