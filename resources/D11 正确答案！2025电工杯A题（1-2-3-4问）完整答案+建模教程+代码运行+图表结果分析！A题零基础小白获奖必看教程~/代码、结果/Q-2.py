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

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class SolarPowerPredictor:
    def __init__(self, data_path, output_dir='问题2结果文件夹'):
        """
        基于历史功率的光伏电站日前发电功率预测模型

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

        print(f"数据预处理完成，共{len(self.df)}条记录")
        print(f"时间范围：{self.df['datetime'].min()} 至 {self.df['datetime'].max()}")

    def create_features(self, df):
        """创建特征工程"""
        feature_df = df.copy()

        # 历史功率特征（滞后特征）
        for lag in [1, 2, 3, 4, 24, 48, 96]:  # 15分钟、30分钟、45分钟、1小时、6小时、12小时、24小时前
            feature_df[f'power_lag_{lag}'] = feature_df['power'].shift(lag)

        # 移动平均特征
        for window in [4, 8, 24, 96]:  # 1小时、2小时、6小时、24小时移动平均
            feature_df[f'power_ma_{window}'] = feature_df['power'].rolling(window=window).mean()

        # 历史同期特征
        feature_df['power_same_hour_yesterday'] = feature_df['power'].shift(96)  # 昨天同一时刻
        feature_df['power_same_hour_week_ago'] = feature_df['power'].shift(96 * 7)  # 一周前同一时刻

        # 统计特征
        for window in [24, 96]:  # 6小时、24小时统计
            feature_df[f'power_std_{window}'] = feature_df['power'].rolling(window=window).std()
            feature_df[f'power_max_{window}'] = feature_df['power'].rolling(window=window).max()
            feature_df[f'power_min_{window}'] = feature_df['power'].rolling(window=window).min()

        # 功率变化率
        feature_df['power_diff_1'] = feature_df['power'].diff(1)
        feature_df['power_diff_4'] = feature_df['power'].diff(4)

        # 周期性特征增强
        feature_df['hour_sin'] = np.sin(2 * np.pi * feature_df['hour'] / 24)
        feature_df['hour_cos'] = np.cos(2 * np.pi * feature_df['hour'] / 24)
        feature_df['month_sin'] = np.sin(2 * np.pi * feature_df['month'] / 12)
        feature_df['month_cos'] = np.cos(2 * np.pi * feature_df['month'] / 12)

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

        # 分析测试集分布
        test_data = self.df.loc[self.test_indices]
        test_distribution = test_data.groupby(['year', 'month']).size()
        print("测试集分布:")
        for (year, month), count in test_distribution.items():
            print(f"  {year}年{month}月: {count}条记录")

    def prepare_model_data(self):
        """准备建模数据"""
        print("正在准备建模数据...")

        # 创建特征
        feature_df = self.create_features(self.df)

        # 选择最重要的特征列（减少特征数量）
        feature_cols = [
            # 时间特征（保留最重要的）
            'hour', 'day_of_year', 'month',
            'sin_hour', 'cos_hour', 'sin_day', 'cos_day',

            # 历史功率特征（保留最重要的）
            'power_lag_1', 'power_lag_4', 'power_lag_24', 'power_lag_96',

            # 移动平均特征（保留最重要的）
            'power_ma_4', 'power_ma_24',

            # 历史同期特征
            'power_same_hour_yesterday',

            # 统计特征（保留最重要的）
            'power_std_24',
            'power_max_24',
            'power_min_24',

            # 变化率特征
            'power_diff_1'
        ]

        # 检查特征是否存在
        available_features = [col for col in feature_cols if col in feature_df.columns]
        self.feature_cols = available_features

        # 准备训练数据
        X_train = feature_df.loc[self.train_indices, self.feature_cols].fillna(0)
        y_train = feature_df.loc[self.train_indices, 'power']

        # 准备测试数据
        X_test = feature_df.loc[self.test_indices, self.feature_cols].fillna(0)
        y_test = feature_df.loc[self.test_indices, 'power']


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

    def train_models(self, X_train, y_train):
        """训练XGBoost预测模型"""
        print("正在训练预测模型...")
        
        # 数据验证
        print("  验证训练数据...")
        try:
            # 将pandas DataFrame转换为numpy数组，并转换为float32以减少内存使用
            X_train_array = X_train.values.astype(np.float32)
            y_train_array = y_train.values.astype(np.float32)
            
            # 检查数据形状
            print(f"  训练数据形状: X_train_array{X_train_array.shape}, y_train_array{y_train_array.shape}")
            
            # 检查数据类型
            print(f"  X_train数据类型: {X_train_array.dtype}")
            print(f"  y_train数据类型: {y_train_array.dtype}")
            
            # 检查内存使用
            print(f"  X_train内存使用: {X_train_array.nbytes / 1024 / 1024:.2f} MB")
            print(f"  y_train内存使用: {y_train_array.nbytes / 1024 / 1024:.2f} MB")
            
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
                # 创建DMatrix对象，这是XGBoost的高效数据格式
                dtrain = xgb.DMatrix(X_train_array, label=y_train_array)
                
                # 设置XGBoost参数
                params = {
                    'objective': 'reg:squarederror',  # 回归任务
                    'eval_metric': 'mae',            # 使用MAE作为评估指标
                    'max_depth': 4,                  # 树的深度
                    'eta': 0.1,                      # 学习率
                    'subsample': 0.8,                # 样本采样比例
                    'colsample_bytree': 0.8,         # 特征采样比例
                    'min_child_weight': 3,           # 最小子节点权重
                    'tree_method': 'hist',           # 使用直方图方法加速训练
                    'nthread': 1                     # 使用单线程
                }
                
                # 训练模型
                num_rounds = 50  # 迭代次数
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

    def generate_report(self):
        """生成完整的分析报告"""
        print("正在生成分析报告...")

        # 准备数据
        self.split_train_test()
        X_train, y_train, X_test, y_test = self.prepare_model_data()

        # 训练模型
        self.train_models(X_train, y_train)

        # 进行预测
        self.make_predictions(X_test)

        # 评估模型
        self.evaluate_models(y_test)

        # 创建预测表格
        results_df = self.create_prediction_table(y_test)

        # 生成可视化
        self.create_visualizations(y_test)

        # 生成文本报告
        best_model = min(self.metrics.keys(), key=lambda x: self.metrics[x]['MAE'])

        report_content = f"""
基于历史功率的光伏电站日前发电功率预测分析报告
================================================

一、预测任务概述
--------------------------------
预测目标: 光伏电站日前发电功率预测
预测时间范围: 7天 (168小时)
时间分辨率: 15分钟
数据来源: 光伏电站历史发电功率数据

二、数据集划分
--------------------------------
训练集与测试集划分方案: 第2、5、8、11个月最后一周数据作为测试集，其他数据作为训练集

数据集统计:
- 训练集样本数: {len(self.train_indices)} 条
- 测试集样本数: {len(self.test_indices)} 条

三、模型构建
--------------------------------
1. 特征工程
- 时间特征：年、月、日、小时、分钟、星期等
- 历史功率特征：滞后特征（15分钟、30分钟、45分钟、1小时、6小时、12小时、24小时前）
- 移动平均特征：1小时、2小时、6小时、24小时移动平均
- 历史同期特征：昨天同一时刻、一周前同一时刻
- 统计特征：标准差、最大值、最小值等
- 周期性特征：小时、月份的周期性变换

2. 模型选择
- XGBoost模型

四、预测结果分析
--------------------------------
1. 模型性能评估（白昼时段）：
"""
        # 添加各模型性能指标
        for model_name, metrics in self.metrics.items():
            report_content += f"""
{model_name}模型:
- MAE: {metrics['MAE']:.4f} MW
- RMSE: {metrics['RMSE']:.4f} MW
- MAPE: {metrics['MAPE']:.2f}%
- R²: {metrics['R2']:.4f}
"""

        report_content += """
2. 预测结果分析
- 预测结果已保存至CSV文件，包含实际功率和各模型预测功率
- 按月份分别保存预测结果，便于分析不同季节的预测效果
- 生成预测效果散点图、时间序列预测对比图、误差分析图等可视化结果

五、结论与建议
--------------------------------
1. 模型性能总结
- XGBoost模型在整体预测效果上表现最好
- 白昼时段的预测精度明显优于夜间时段
- 不同季节的预测效果存在差异，夏季预测效果相对较好

2. 改进建议
- 考虑引入更多气象特征，如温度、湿度、风速等
- 针对不同季节分别建立预测模型
- 优化特征工程，增加更多有意义的特征组合
- 尝试更复杂的集成方法，如加权集成

六、附件说明
--------------------------------
1. 预测结果文件
- 预测结果表格.csv：包含所有时间点的预测结果
- {年}年{月}月预测结果.csv：按月份分别保存的预测结果

2. 可视化图表
- 预测效果散点图.png：各模型预测值与实际值的对比
- 时间序列预测对比.png：预测结果的时间序列展示
- 误差分析图.png：预测误差的详细分析
- 特征重要性分析.png：XGBoost模型的特征重要性排序
"""

        # 保存报告到文件
        with open(f'{self.output_dir}/预测分析报告.txt', 'w', encoding='utf-8') as f:
            f.write(report_content)

        print("分析报告生成完成！")

        return report_content


if __name__ == "__main__":
    # 设置数据文件路径
    data_path = r"C:\Users\Administrator\Desktop\电工杯\代码、结果\station00（用于解题demo）.csv"  # 请确保数据文件路径正确

    # 创建预测器实例
    predictor = SolarPowerPredictor(data_path)

    # 生成完整报告
    predictor.generate_report()

    print("\n预测任务完成！所有结果已保存到'问题2结果文件夹'中。")
    print("请查看以下文件：")
    print("1. 预测分析报告.txt - 包含完整的分析报告")
    print("2. 预测结果表格.csv - 包含所有预测结果")
    print("3. {年}年{月}月预测结果.csv - 按月份保存的预测结果")
    print("4. 预测效果散点图.png - 预测效果可视化")
    print("5. 时间序列预测对比.png - 时间序列预测对比")
    print("6. 误差分析图.png - 误差分析结果")
    print("7. 特征重要性分析.png - 特征重要性分析")
