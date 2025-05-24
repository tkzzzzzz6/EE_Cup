import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class SolarPowerAnalysis:
    def __init__(self, data_path, output_dir='问题1结果文件夹'):
        """
        初始化光伏电站发电特性分析类

        Parameters:
        data_path: 数据文件路径
        output_dir: 输出结果文件夹
        """
        self.data_path = data_path
        self.output_dir = output_dir

        # 创建输出文件夹
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 光伏电站实际参数
        self.station_latitude = 38.04778  # 纬度（度）
        self.station_longitude = 114.95139  # 经度（度）
        self.station_capacity = 26.0  # 装机容量（MW）
        self.panel_power = 255  # 单块组件功率（Wp）
        self.panel_count = 6600  # 组件总数
        self.panel_efficiency = 0.156  # 光伏板效率（255W/1640mm/990mm=15.6%）
        self.system_losses = 0.10  # 系统损失（考虑逆变器效率等）
        self.panel_tilt = 33  # 光伏板倾斜角度（度）
        self.panel_azimuth = 180  # 光伏板方位角（正南为180度）

        # 光伏电站配置信息
        self.modules_per_string = 20  # 每串组件数
        self.strings_per_inverter = 96  # 每台逆变器串数
        self.inverter_power = 500  # 逆变器功率（kW）

        # 加载数据
        self.load_data()

    def load_data(self):
        """加载并预处理数据"""
        print("正在加载数据...")

        # 读取CSV数据
        self.df = pd.read_csv(self.data_path)

        # 转换时间格式
        self.df['datetime'] = pd.to_datetime(self.df['date_time'])

        # 提取时间特征
        self.df['year'] = self.df['datetime'].dt.year
        self.df['month'] = self.df['datetime'].dt.month
        self.df['day'] = self.df['datetime'].dt.day
        self.df['hour'] = self.df['datetime'].dt.hour
        self.df['minute'] = self.df['datetime'].dt.minute
        self.df['day_of_year'] = self.df['datetime'].dt.dayofyear
        self.df['season'] = self.df['month'].apply(self.get_season)

        # 计算太阳时间
        self.df['solar_time'] = self.df['hour'] + self.df['minute'] / 60

        print(f"数据加载完成，共{len(self.df)}条记录")
        print(f"时间范围：{self.df['datetime'].min()} 至 {self.df['datetime'].max()}")

    def get_season(self, month):
        """根据月份确定季节"""
        if month in [12, 1, 2]:
            return '冬季'
        elif month in [3, 4, 5]:
            return '春季'
        elif month in [6, 7, 8]:
            return '夏季'
        else:
            return '秋季'

    def solar_position(self, day_of_year, solar_time):
        """
        计算太阳位置角度

        Parameters:
        day_of_year: 一年中的第几天
        solar_time: 太阳时间（小时）

        Returns:
        elevation: 太阳高度角（弧度）
        azimuth: 太阳方位角（弧度）
        """
        # 太阳赤纬角
        declination = np.radians(23.45) * np.sin(np.radians(360 * (284 + day_of_year) / 365))

        # 时角
        hour_angle = np.radians(15 * (solar_time - 12))

        # 纬度转弧度
        lat_rad = np.radians(self.station_latitude)

        # 太阳高度角
        elevation = np.arcsin(np.sin(declination) * np.sin(lat_rad) +
                              np.cos(declination) * np.cos(lat_rad) * np.cos(hour_angle))

        # 太阳方位角
        azimuth = np.arctan2(np.sin(hour_angle),
                             np.cos(hour_angle) * np.sin(lat_rad) -
                             np.tan(declination) * np.cos(lat_rad))

        return elevation, azimuth

    def calculate_theoretical_power(self):
        """计算理论可发功率"""
        print("正在计算理论可发功率...")

        # 计算太阳位置
        elevations, azimuths = self.solar_position(self.df['day_of_year'], self.df['solar_time'])

        # 太阳高度角和方位角（度）
        self.df['sun_elevation'] = np.degrees(elevations)
        self.df['sun_azimuth'] = np.degrees(azimuths)

        # 计算倾斜面上的太阳入射角
        # 考虑光伏板倾斜角和方位角
        panel_tilt_rad = np.radians(self.panel_tilt)
        panel_azimuth_rad = np.radians(self.panel_azimuth)

        # 太阳入射角余弦
        cos_incidence = (np.sin(elevations) * np.cos(panel_tilt_rad) +
                         np.cos(elevations) * np.sin(panel_tilt_rad) *
                         np.cos(azimuths - panel_azimuth_rad))

        # 确保入射角余弦不为负值
        cos_incidence = np.maximum(cos_incidence, 0)

        # 计算大气质量系数
        self.df['air_mass'] = np.where(
            self.df['sun_elevation'] > 0,
            1 / np.maximum(np.sin(elevations), 0.01),
            0
        )

        # 使用实际的全球水平辐照度数据或理论计算
        if 'nwp_globalirrad' in self.df.columns and self.df['nwp_globalirrad'].sum() > 0:
            # 使用NWP辐照度数据
            ghi = self.df['nwp_globalirrad']  # 全球水平辐照度

            # 将水平辐照度转换为倾斜面辐照度（简化模型）
            self.df['tilted_irradiance'] = np.where(
                self.df['sun_elevation'] > 0,
                ghi * cos_incidence / np.sin(elevations),
                0
            )
        else:
            # 使用理论辐照度计算
            direct_normal_irradiance = 900  # 标准直射辐照度（W/m²）

            # 考虑大气衰减的理论辐照度
            self.df['tilted_irradiance'] = np.where(
                self.df['sun_elevation'] > 0,
                direct_normal_irradiance * cos_incidence * np.power(0.7, self.df['air_mass']),
                0
            )

        # 计算温度修正系数（多晶硅温度系数约-0.45%/°C）
        temp_coefficient = -0.0045  # 每摄氏度功率损失
        standard_temp = 25  # 标准测试条件温度

        if 'nwp_temperature' in self.df.columns:
            temp_factor = 1 + temp_coefficient * (self.df['nwp_temperature'] - standard_temp)
        else:
            temp_factor = 1  # 如果没有温度数据，不考虑温度影响

        # 计算理论可发功率
        # 光伏板总面积
        panel_area_per_module = 1.64 * 0.99  # 1640mm × 990mm
        total_panel_area = panel_area_per_module * self.panel_count  # 总面积（m²）

        # 理论功率计算
        self.df['theoretical_power'] = (
                self.df['tilted_irradiance'] * total_panel_area * self.panel_efficiency *
                temp_factor * (1 - self.system_losses) / 1000000  # 转换为MW
        )

        # 限制最大功率不超过装机容量
        self.df['theoretical_power'] = np.minimum(self.df['theoretical_power'], self.station_capacity)

        print("理论可发功率计算完成")
        print(f"计算基准：装机容量{self.station_capacity}MW, 组件数量{self.panel_count}块")

    def analyze_long_term_characteristics(self):
        """分析长周期（季节性）特性"""
        print("正在分析长周期特性...")

        # 按月统计
        monthly_stats = self.df.groupby('month').agg({
            'power': ['mean', 'max', 'std'],
            'theoretical_power': ['mean', 'max', 'std'],
            'nwp_globalirrad': 'mean',
            'nwp_temperature': 'mean'
        }).round(3)

        monthly_stats.columns = ['Actual Power_Mean', 'Actual Power_Max', 'Actual Power_Std',
                                 'Theoretical Power_Mean', 'Theoretical Power_Max', 'Theoretical Power_Std',
                                 'Average Irradiance', 'Average Temperature']

        # 按季节统计
        seasonal_stats = self.df.groupby('season').agg({
            'power': ['mean', 'max', 'std'],
            'theoretical_power': ['mean', 'max', 'std'],
            'nwp_globalirrad': 'mean',
            'nwp_temperature': 'mean'
        }).round(3)

        seasonal_stats.columns = ['Actual Power_Mean', 'Actual Power_Max', 'Actual Power_Std',
                                  'Theoretical Power_Mean', 'Theoretical Power_Max', 'Theoretical Power_Std',
                                  'Average Irradiance', 'Average Temperature']

        # 计算发电效率
        self.df['power_efficiency'] = np.where(
            self.df['theoretical_power'] > 0,
            self.df['power'] / self.df['theoretical_power'],
            0
        )

        efficiency_stats = self.df.groupby(['month', 'season']).agg({
            'power_efficiency': ['mean', 'std']
        }).round(3)

        return monthly_stats, seasonal_stats, efficiency_stats

    def analyze_short_term_characteristics(self):
        """分析短周期（日内波动）特性"""
        print("正在分析短周期特性...")

        # 按小时统计
        hourly_stats = self.df.groupby('hour').agg({
            'power': ['mean', 'max', 'std'],
            'theoretical_power': ['mean', 'max', 'std'],
            'nwp_globalirrad': 'mean'
        }).round(3)

        hourly_stats.columns = ['Actual Power_Mean', 'Actual Power_Max', 'Actual Power_Std',
                                'Theoretical Power_Mean', 'Theoretical Power_Max', 'Theoretical Power_Std',
                                'Average Irradiance']

        # 分析不同季节的日内变化
        seasonal_hourly = self.df.groupby(['season', 'hour']).agg({
            'power': 'mean',
            'theoretical_power': 'mean'
        }).round(3)

        # 计算日内波动特征
        daily_stats = self.df.groupby(self.df['datetime'].dt.date).agg({
            'power': ['mean', 'max', 'min', 'std'],
            'theoretical_power': ['mean', 'max', 'min', 'std']
        }).round(3)

        # 计算日内波动系数
        daily_stats['power_volatility'] = daily_stats[('power', 'std')] / daily_stats[('power', 'mean')]
        daily_stats['theoretical_volatility'] = daily_stats[('theoretical_power', 'std')] / daily_stats[
            ('theoretical_power', 'mean')]

        return hourly_stats, seasonal_hourly, daily_stats

    def analyze_power_deviation(self):
        """分析实际功率与理论功率的偏差"""
        print("正在分析功率偏差...")

        # 计算绝对偏差和相对偏差
        self.df['power_deviation'] = self.df['power'] - self.df['theoretical_power']
        self.df['power_relative_deviation'] = np.where(
            self.df['theoretical_power'] > 0,
            (self.df['power'] - self.df['theoretical_power']) / self.df['theoretical_power'] * 100,
            0
        )

        # 统计偏差特征
        deviation_stats = {
            'Average Absolute Deviation': self.df['power_deviation'].mean(),
            'Absolute Deviation Std': self.df['power_deviation'].std(),
            'Average Relative Deviation(%)': self.df['power_relative_deviation'].mean(),
            'Relative Deviation Std(%)': self.df['power_relative_deviation'].std(),
            'Positive Deviation Ratio(%)': (self.df['power_deviation'] > 0).mean() * 100,
            'Negative Deviation Ratio(%)': (self.df['power_deviation'] < 0).mean() * 100
        }

        # 不同条件下的偏差分析
        condition_deviation = self.df.groupby(['season', 'hour']).agg({
            'power_deviation': ['mean', 'std'],
            'power_relative_deviation': ['mean', 'std']
        }).round(3)

        return deviation_stats, condition_deviation

    def create_visualizations(self):
        """创建可视化图表"""
        print("正在生成可视化图表...")

        # 设置图表样式
        plt.style.use('default')

        # 1. 长周期特性 - 月度发电量变化
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 月度平均功率
        monthly_power = self.df.groupby('month')['power'].mean()
        monthly_theoretical = self.df.groupby('month')['theoretical_power'].mean()

        axes[0, 0].plot(monthly_power.index, monthly_power.values, 'b-o', label='Actual Power', linewidth=2)
        axes[0, 0].plot(monthly_theoretical.index, monthly_theoretical.values, 'r--s', label='Theoretical Power', linewidth=2)
        axes[0, 0].set_title('Monthly Average Power Generation', fontsize=14)
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('Power (MW)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 季节性对比
        seasonal_power = self.df.groupby('season')['power'].mean()
        seasonal_theoretical = self.df.groupby('season')['theoretical_power'].mean()

        x = np.arange(len(seasonal_power))
        width = 0.35

        axes[0, 1].bar(x - width / 2, seasonal_power.values, width, label='Actual Power')
        axes[0, 1].bar(x + width / 2, seasonal_theoretical.values, width, label='Theoretical Power')
        axes[0, 1].set_title('Seasonal Power Comparison', fontsize=14)
        axes[0, 1].set_xlabel('Season')
        axes[0, 1].set_ylabel('Power (MW)')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(seasonal_power.index)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 2. 短周期特性 - 日内变化
        hourly_power = self.df.groupby('hour')['power'].mean()
        hourly_theoretical = self.df.groupby('hour')['theoretical_power'].mean()

        axes[1, 0].plot(hourly_power.index, hourly_power.values, 'b-o', label='Actual Power', linewidth=2)
        axes[1, 0].plot(hourly_theoretical.index, hourly_theoretical.values, 'r--s', label='Theoretical Power', linewidth=2)
        axes[1, 0].set_title('Diurnal Average Power Generation', fontsize=14)
        axes[1, 0].set_xlabel('Hour')
        axes[1, 0].set_ylabel('Power (MW)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 功率偏差分布
        axes[1, 1].hist(self.df['power_deviation'], bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].set_title('Power Deviation Distribution', fontsize=14)
        axes[1, 1].set_xlabel('Power Deviation (MW)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/长短周期特性分析.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. 详细的季节-小时热力图
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 实际功率热力图
        pivot_actual = self.df.pivot_table(values='power', index='hour', columns='season', aggfunc='mean')
        sns.heatmap(pivot_actual, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[0])
        axes[0].set_title('Actual Power Season-Hour Heatmap (MW)', fontsize=14)
        axes[0].set_xlabel('Season')
        axes[0].set_ylabel('Hour')

        # 理论功率热力图
        pivot_theoretical = self.df.pivot_table(values='theoretical_power', index='hour', columns='season',
                                                aggfunc='mean')
        sns.heatmap(pivot_theoretical, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[1])
        axes[1].set_title('Theoretical Power Season-Hour Heatmap (MW)', fontsize=14)
        axes[1].set_xlabel('Season')
        axes[1].set_ylabel('Hour')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/季节小时热力图.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 4. 功率效率分析
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 发电效率月度变化
        monthly_efficiency = self.df.groupby('month')['power_efficiency'].mean()
        axes[0, 0].plot(monthly_efficiency.index, monthly_efficiency.values, 'g-o', linewidth=2)
        axes[0, 0].set_title('Monthly Power Generation Efficiency', fontsize=14)
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('Power Generation Efficiency')
        axes[0, 0].grid(True, alpha=0.3)

        # 发电效率箱线图
        self.df.boxplot(column='power_efficiency', by='season', ax=axes[0, 1])
        axes[0, 1].set_title('Power Generation Efficiency Distribution by Season', fontsize=14)
        axes[0, 1].set_xlabel('Season')
        axes[0, 1].set_ylabel('Power Generation Efficiency')

        # 相对偏差分析
        axes[1, 0].scatter(self.df['theoretical_power'], self.df['power_relative_deviation'], alpha=0.6)
        axes[1, 0].set_title('Theoretical Power vs Relative Deviation', fontsize=14)
        axes[1, 0].set_xlabel('Theoretical Power (MW)')
        axes[1, 0].set_ylabel('Relative Deviation (%)')
        axes[1, 0].grid(True, alpha=0.3)

        # 温度对功率的影响分析
        if 'nwp_temperature' in self.df.columns and self.df['nwp_temperature'].sum() > 0:
            temp_data = self.df['nwp_temperature']
        elif 'lmd_temperature' in self.df.columns and self.df['lmd_temperature'].sum() > 0:
            temp_data = self.df['lmd_temperature']
        else:
            temp_data = pd.Series([25] * len(self.df))  # 默认温度

        axes[1, 1].scatter(temp_data, self.df['power'], alpha=0.6, s=10)

        # 计算温度与功率的相关性
        if len(temp_data.dropna()) > 0 and temp_data.std() > 0:
            correlation = temp_data.corr(self.df['power'])
            axes[1, 1].set_title(f'Temperature vs Actual Power (Correlation: {correlation:.3f})', fontsize=14)
        else:
            axes[1, 1].set_title('Temperature vs Actual Power', fontsize=14)

        axes[1, 1].set_xlabel('Temperature (°C)')
        axes[1, 1].set_ylabel('Actual Power (MW)')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/功率效率分析.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("可视化图表生成完成")

    def generate_report(self):
        """生成分析报告"""
        print("正在生成分析报告...")

        # 计算理论可发功率
        self.calculate_theoretical_power()

        # 进行各项分析
        monthly_stats, seasonal_stats, efficiency_stats = self.analyze_long_term_characteristics()
        hourly_stats, seasonal_hourly, daily_stats = self.analyze_short_term_characteristics()
        deviation_stats, condition_deviation = self.analyze_power_deviation()

        # 生成可视化
        self.create_visualizations()

        # 生成文本报告
        report_content = f"""
光伏电站发电特性分析报告
================================

一、数据概况
--------------------------------
数据时间范围: {self.df['datetime'].min()} 至 {self.df['datetime'].max()}
数据总量: {len(self.df)} 条记录

光伏电站基本信息:
- 装机容量: {self.station_capacity} MW
- 电站位置: 纬度{self.station_latitude}°, 经度{self.station_longitude}°
- 组件类型: LW255(29)P 多晶硅
- 组件功率: {self.panel_power}W × {self.panel_count}块 = {self.panel_power * self.panel_count / 1000000:.1f}MW
- 组件尺寸: 1640mm × 990mm
- 倾斜角度: {self.panel_tilt}° (正南朝向)
- 逆变器配置: {self.inverter_power}kW × {int(self.panel_count / self.modules_per_string / self.strings_per_inverter)}台
- 系统配置: {self.modules_per_string}块/串, {self.strings_per_inverter}串/逆变器

二、长周期（季节性）特性分析
--------------------------------
2.1 月度发电特性统计
{monthly_stats.to_string()}

2.2 季节性发电特性统计
{seasonal_stats.to_string()}

2.3 主要发现：
- 发电功率具有明显的季节性变化规律
- 夏季发电功率最高，平均值为 {seasonal_stats.loc['夏季', 'Actual Power_Mean']:.2f} MW
- 冬季发电功率最低，平均值为 {seasonal_stats.loc['冬季', 'Actual Power_Mean']:.2f} MW
- 季节性变化主要受太阳高度角和日照时间影响

三、短周期（日内波动）特性分析
--------------------------------
3.1 小时发电特性统计
{hourly_stats.to_string()}

3.2 主要发现：
- 发电功率呈现明显的日内变化规律，符合太阳辐射变化特征
- 发电高峰时段为 {hourly_stats['Actual Power_Mean'].idxmax()} 时，峰值功率为 {hourly_stats['Actual Power_Mean'].max():.2f} MW
- 夜间发电功率为0，日出日落时段功率快速变化
- 日内波动系数平均值为 {daily_stats['power_volatility'].mean():.3f}

四、实际功率与理论功率偏差分析
--------------------------------
4.1 偏差统计特征
"""

        for key, value in deviation_stats.items():
            report_content += f"{key}: {value:.3f}\n"

        report_content += f"""

4.2 主要发现：
- 实际功率普遍低于理论功率，平均相对偏差为 {deviation_stats['Average Relative Deviation(%)']:.2f}%
- 偏差主要由以下因素造成：
  * 气象条件影响（云量、湿度、风速等）
  * 设备效率损失和系统损耗
  * 光伏板表面污染和老化
  * 温度对光伏板效率的影响

五、影响因素分析
--------------------------------
5.1 气象因素影响
- 辐照度是影响发电功率的主要因素，倾斜面辐照度直接影响发电量
- 温度对光伏板效率有负面影响，多晶硅组件温度系数约-0.45%/°C
- 湿度和风速对散热和清洁有一定影响
- 云量变化导致辐照度短期波动，影响发电稳定性

5.2 系统配置影响
- 光伏板倾斜角({self.panel_tilt}°)和方位角(正南)影响太阳能接收效率
- 逆变器效率和系统损耗约{self.system_losses * 100:.0f}%
- 组串配置({self.modules_per_string}块/串)影响系统匹配性
- 设备老化和维护状态影响实际发电效率

5.3 地理位置影响
- 纬度({self.station_latitude}°)决定太阳高度角的年变化规律
- 经度({self.station_longitude}°)影响当地太阳时间

六、结论与建议
--------------------------------
6.1 主要结论
1. 光伏电站发电功率具有明显的长周期（季节性）和短周期（日内）变化特征
2. 实际发电功率受多种因素影响，与理论功率存在系统性偏差
3. 气象条件是影响发电功率的关键因素，特别是太阳辐照度
4. 发电效率在不同季节和时段存在显著差异

6.2 优化建议
1. 建立精确的气象-功率关联模型，提高预测精度
2. 考虑温度修正因子，改进功率预测算法
3. 定期维护清洁光伏板，减少系统损耗
4. 优化运维策略，提高发电效率

分析完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        # 保存报告
        with open(f'{self.output_dir}/光伏电站发电特性分析报告.txt', 'w', encoding='utf-8') as f:
            f.write(report_content)

        # 保存详细数据
        monthly_stats.to_csv(f'{self.output_dir}/月度统计数据.csv', encoding='utf-8')
        seasonal_stats.to_csv(f'{self.output_dir}/季节统计数据.csv', encoding='utf-8')
        hourly_stats.to_csv(f'{self.output_dir}/小时统计数据.csv', encoding='utf-8')

        print("分析报告生成完成！")
        print(f"结果已保存到文件夹: {self.output_dir}")


def main():
    """主函数"""
    # 数据文件路径
    data_path = r"C:\Users\Administrator\Desktop\电工杯\代码、结果\station00（用于解题demo）.csv"

    # 创建分析对象
    analyzer = SolarPowerAnalysis(data_path)

    # 生成完整分析报告
    analyzer.generate_report()


if __name__ == "__main__":
    main()
