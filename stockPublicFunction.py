import requests
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np
import json
import os
import sys
from bs4 import BeautifulSoup
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.interpolate import CubicSpline
import matplotlib as mpl
font_path = 'msyh.ttc'

from matplotlib.font_manager import FontProperties

font_properties = FontProperties(fname=font_path)
mpl.rcParams['font.family'] = font_properties.get_name()
mpl.rcParams['axes.unicode_minus'] = False  # Ensure minus signs are displayed correctly

os.environ['PYTHONUNBUFFERED'] = '1'
# sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
# sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)

# 常量
URL_TEMPLATE = "https://statementdog.com/api/v2/fundamentals/{stock_code}/2014/2024/cf?qbu=true&qf=analysis"

# 定義請求頭
HEADERS = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7,zh-CN;q=0.6",
    "cache-control": "no-cache",
    "cookie": "statementdog_device_id=aVNiSlJJNXpJL0prMk1ybitSenFyZVY1MkdROFh0SzVIeHMzSGc0MmF4VmM1V3d1dVJya2h0c1dUQWUrOXdOci0tTUN3aXlSTGhsQ0ptVHJrUmRTUlRTUT09--cb40a658096ab1ae069e0e788ee96fad6c769d54; easy_ab=88e3e53f-9979-4305-8ee6-01dce716965f; _ga=GA1.1.1382045611.1724032914; upgrade_browser=1; g_state={\"i_l\":0}; g_csrf_token=37de11532f4b258c; AMP_0ab77a441f=JTdCJTIyZGV2aWNlSWQlMjIlM0ElMjI1NmIwZWMyYS1kYTJmLTRkMjgtYmY3My0xZDZkN2ZiNjNlZjclMjIlMkMlMjJ1c2VySWQlMjIlM0EyMjA2OCUyQyUyMnNlc3Npb25JZCUyMiUzQTE3MjQwMzgyNTg1ODQlMkMlMjJvcHRPdXQlMjIlM0FmYWxzZSUyQyUyMmxhc3RFdmVudFRpbWUlMjIlM0ExNzI0MDM4MjgzNzQ2JTJDJTIybGFzdEV2ZW50SWQlMjIlM0E2MSUyQyUyMnBhZ2VDb3VudGVyJTIyJTNBMCU3RA==; _ga_K9Y9Y589MM=GS1.1.1724038235.2.1.1724038302.60.0.0; search_stock_referer=https%3A%2F%2Fstatementdog.com%2Fportfolios; _statementdog_session_v2=vtMAdvQujtMr%2BsggeSeES5%2BEwDxq8RDQy%2F4qBSPd%2Fx%2FCtC6vEuYzA0LDupGh2EC7qzlojtrdZp8I2ZxG%2BIRqJNaqZxCdMfbLRngFaE2uZ9NchVpO2Sdm2wabpfZFYn3rlo5q%2FJvRSqmt%2BXRv9vJ2Ov7nRBDlykVSpgolWMud491J8G8T3M3n4R5BgPKiZvNoeYO5Cr1TsXnVU9oA9hZ8a8vK4vfSOvdfSzZ2oB%2BxsASFiu0CLvFUkCWBpcYmhjXYCmYK11kjZiuTDIiBGBGuh%2FYYxTjviOMZWLzrVtAPQUPB%2Bx%2BV4jScRVx0RcCeUz44iWSYHB6oGN%2B41cm57hcLV%2FvOxYtqU81KZuq9sPQ0P9yo9DM5%2Fr%2BQyOPfi9hvCdNMQfW5MN4DjYYEV4riosl%2BqSY4sD%2F2A5gIo3%2BwLTXS8JGeQhGVkWHnjjXVype8uTltaUVMPB9wXPAizte8KGwcSHLwl2FaMCOBWxfE1FDQE7%2Fel%2BbaSX9uHJbx%2BMdEW3L71QWHVxyk0sMwgtIxfeRPYwJI77Cmj8OooJL4okUH9DYPuBp3LzC8qrYmla0%2Fz3gIOss%3D--UM7YcRCc%2F1IlStw2--ZEGRU3Mm0aZXH63JnVzfYQ%3D%3D",
    "pragma": "no-cache",
    "referer": "https://statementdog.com/analysis/2002/monthly-revenue-growth-rate",
    "sec-ch-ua": "\"Not)A;Brand\";v=\"99\", \"Google Chrome\";v=\"127\", \"Chromium\";v=\"127\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"Windows\"",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "x-csrf-token": "m5_fWNYl2lS5m4iIczHtKCm2wHDZI9S9mTlVt1l3aTjb3mg9DK9yfbtypwyRp60MEKT4KWzyuSq1r__ShI-Ddw"
}

def fetch_stock_data(NUM_DATA_POINTS ,FETCH_LATEST_CLOSE_PRICE_ONLINE, stock_code):
    """从本地文件获取股票数据"""
    file_path = f'stockData/{stock_code}.json'
#    volume_path= f'stockData/{stock_code}_m_vol.json'

    if not os.path.exists(file_path) :
        raise ValueError(f"文件 {file_path} 不存在。请确保文件路径和股票代码正确。")

#    if not os.path.exists(volume_path) :
#        raise ValueError(f"文件 {volume_path} 不存在。请确保文件路径和股票代码正确。")


    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

#    with open(volume_path, 'r', encoding='utf-8') as file:
#        volume_data = json.load(file)


    monthly_data = data.get("monthly", {})
    quarterly_data = data.get("quarterly", {})

    # 读取原始数据
    def extract_data(key):
        return [parse_float(item[1]) for item in monthly_data.get(key, {}).get("data", []) if item[1] != '無']
    def extract_data_quarterly(key):
        return [parse_float(item[1]) for item in quarterly_data.get(key, {}).get("data", []) if item[1] != '無']

    #季度數據
    epst4q = extract_data_quarterly("EPST4Q")
    epst4q_velocity = calculate_sign_changes(epst4q)

    # 计算每个月的 EPS 與 velocity
    epst4q_interpolated = []
    for eps in epst4q:
        epst4q_interpolated.extend([eps / 3] * 3)  # 每个季度的 EPS 平均分配到三个月

    #get_volume_3m_avf
    volume_m = m_volume_data[-NUM_DATA_POINTS:]
    extended_volume_data  = m_volume_data[-NUM_DATA_POINTS-2:]
    # 将数据转换为 pandas 的 DataFrame
    df = pd.DataFrame(extended_volume_data , columns=['volume'])
    # 使用 pandas 计算3m的移动平均
    df['3_m_MA'] = df['volume'].rolling(window=3, min_periods=3).mean()
    # 移除前两个 NaN 值
    volume_m_avg = df['3_m_MA'].dropna().values.tolist()

    


    PB = extract_data("PB")
    revenue_per_share = extract_data("RevenuePerShare")
    revenue_per_share_yoy = extract_data("RevenuePerShareYOY")
    price_data = extract_data("Price")
    revenue_t3m_avg = extract_data("RevenueT3MAvg")
    revenue_t3m_yoy = extract_data("RevenueT3MYOY")
    majority_shareholders_share_ratio = extract_data("MajorityShareholdersShareRatio")
    total_shareholders_count = extract_data("TotalShareholdersCount")

    # 获取最新股价
    price_file_path = os.path.join('stockData', 'latest_price.json')
    with open(price_file_path, 'r', encoding='utf-8') as file:
        latest_price_data = json.load(file)

    latest_close_price = next(
        (item['price'] for item in latest_price_data if item['stock_code'] == stock_code),
        None
    )

    if latest_close_price is None:
        raise ValueError(f"Stock code {stock_code} not found in latest_price.json")

    # 计算有效数据长度
    def calculate_valid_length(data_list):
        return len([x for x in data_list if x is not None])

    valid_length = min(NUM_DATA_POINTS, calculate_valid_length(price_data))

    # # 从最后开始提取有效长度的数据
    def get_last_valid_data(lst):
        valid_data = [x for x in lst if x is not None]
        return valid_data[-valid_length:]

    epst4q = get_last_valid_data(epst4q_interpolated_last)
    PB = get_last_valid_data(PB)
    revenue_per_share = get_last_valid_data(revenue_per_share)
    revenue_per_share_yoy = get_last_valid_data(revenue_per_share_yoy)
    price_data = get_last_valid_data(price_data)
    revenue_t3m_avg = get_last_valid_data(revenue_t3m_avg)
    revenue_t3m_yoy = get_last_valid_data(revenue_t3m_yoy)
    majority_shareholders_share_ratio = get_last_valid_data(majority_shareholders_share_ratio)
    total_shareholders_count = get_last_valid_data(total_shareholders_count)

    # 填充不足的部分
    epst4q = pad_list(epst4q, valid_length)
    PB = pad_list(PB, valid_length)
    revenue_per_share = pad_list(revenue_per_share, valid_length)
    revenue_per_share_yoy = pad_list(revenue_per_share_yoy, valid_length)
    price_data = pad_list(price_data, valid_length)
    revenue_t3m_avg = pad_list(revenue_t3m_avg, valid_length)
    revenue_t3m_yoy = pad_list(revenue_t3m_yoy, valid_length)
    majority_shareholders_share_ratio = pad_list(majority_shareholders_share_ratio, valid_length)
    total_shareholders_count = pad_list(total_shareholders_count, valid_length)

    return (revenue_per_share_yoy,
            price_data,
            revenue_per_share,
            PB,
            revenue_t3m_avg,
            revenue_t3m_yoy,
            majority_shareholders_share_ratio,
            total_shareholders_count,epst4q, volume_m,volume_m_avg,
            latest_close_price)


def linear_interpolate_sign(data, target_length):
    """
    对符号数据进行线性插值处理，并保留符号数据的特性。
    """
    # 创建插值的原始索引
    original_indices = np.arange(len(data))

    # 创建目标索引
    target_indices = np.linspace(0, len(data) - 1, target_length)

    # 进行线性插值
    interpolated_data = np.interp(target_indices, original_indices, data)

    # 将插值后的数据取符号，确保仍然是 -1 或 1
    interpolated_sign_data = np.sign(interpolated_data)

    return interpolated_sign_data

def extract_sign(data):
    """
    将正数转换为1，负数转换为-1。

    参数:
    - data: 包含数值的列表或数组。

    返回:
    - sign_data: 与输入数据相同长度的列表，正数为1，负数为-1。
    """
    sign_data = []
    for x in data:
        if x is None:
            sign_data.append(None)  # 或者使用其他适合的填充值
        elif x > 0:
            sign_data.append(1)
        else:
            sign_data.append(-1)

    return sign_data

def getLatestPrice():
    """從網頁獲取最新的股價"""
    headers = {
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'accept-language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7,zh-CN;q=0.6',
        'cache-control': 'no-cache',
        'cookie': 'fastivalName_Mall_20240402=closeday_fastivalName_Mall_20240402; bottomADName_20240402=closeday_bottomADName_20240402; _ga=GA1.2.1346814139.1722849008; _gcl_au=1.1.1681944876.1722849008; ASP.NET_SessionId=f4y0yefm0uhzdi550rgceb0k; _gid=GA1.2.929478691.1724133370; fastivalName_Mall_20240402=closeday_fastivalName_Mall_20240402; bottomADName_20240402=closeday_bottomADName_20240402; __gsas=ID=31ddb8e553c103a7:T=1724133412:RT=1724133412:S=ALNI_MYByiOJC8zDAXukb3TM5gPJMU1pXg; _ga_S0YRRCXLNT=GS1.2.1724145464.3.1.1724145535.60.0.0',
        'pragma': 'no-cache',
        'priority': 'u=0, i',
        'referer': 'https://histock.tw/stock/rank.aspx?m=0&d=1&p=all',
        'sec-ch-ua': '"Not)A;Brand";v="99", "Google Chrome";v="127", "Chromium";v="127"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'document',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'same-origin',
        'sec-fetch-user': '?1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36',
    }
    url = 'https://histock.tw/stock/rank.aspx?m=0&d=0&p=all'
    response = requests.get(url, headers=headers)
    html = response.text

    soup = BeautifulSoup(html, 'html.parser')

    stocks = []
    for row in soup.select('tr'):
        stock_link = row.select_one('a[href^="/stock/"]')
        if stock_link:
            stock_code = stock_link['href'].split('/')[2]
            stock_name = stock_link.text.strip()

            price_span = row.select_one('span[id^="CPHB1_gv_lbDeal_"]')
            if price_span:
                price_text = price_span.text.strip()
                try:
                    price = float(price_text.replace(',', ''))
                except ValueError:
                    continue

                stocks.append({
                    'stock_name': stock_name,
                    'stock_code': stock_code,
                    'price': price
                })

    output_dir = 'stockData'
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, 'latest_price.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stocks, f, ensure_ascii=False, indent=4)

    print(f"數據已保存到 {output_file}")

def parse_float(value):
    """將值解析為浮點數"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def pad_list(lst, length):
    """Pad the list to the specified length with None values if it is shorter, or truncate if it is longer."""
    if len(lst) < length:
        return lst + [None] * (length - len(lst))
    return lst[:length]

def interpolate_quarterly_to_monthly(quarterly_data, num_last_values):
    """将季度数据插值到每月数据，并从插值结果中提取最后特定数量的数据点"""
    if len(quarterly_data) < 2:
        return [None] * (num_last_values * 3), [None] * num_last_values

    # 计算插值的月数，取提取数据量的3倍
    num_months = len(quarterly_data) * 3

    # 创建季度时间轴，季度数据点的位置
    quarterly_indices = np.arange(len(quarterly_data)) * 3
    # 创建对应的月度时间轴
    monthly_indices = np.arange(num_months)

    # 使用三次样条插值
    cs = CubicSpline(quarterly_indices, quarterly_data, bc_type='natural')

    # 计算月度数据
    monthly_data = cs(monthly_indices)

    # 获取最后特定数量的数据点
    last_values = monthly_data[-num_last_values-2:-2]

    return monthly_data, last_values

def pad_data(data, length):
    """将数据填充到指定长度"""
    if len(data) < length:
        return data + [None] * (length - len(data))
    return data[:length]

def normalize_and_standardize_data(X):
    """
    对输入数据X进行极端值处理、标准化和归一化。
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # # 处理极端值：向量化处理
    # median = np.median(X, axis=0)
    # iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
    #
    # # 设置上下限
    # lower_bound = median - 3 * iqr
    # upper_bound = median + 3 * iqr
    #
    # # 向量化处理极端值
    # X_clipped = np.clip(X, lower_bound, upper_bound)


    # 标准化
    standard_scaler = StandardScaler()
    X_standardized = standard_scaler.fit_transform(X)

    # 归一化到 [0, 1] 范围
    #min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    #X_normalized = min_max_scaler.fit_transform(X_standardized)

    return X_standardized, standard_scaler

def linear_interpolation(data, target_length):
    """
    对数据进行线性插值，确保数据长度为 target_length。
    
    参数：
    data -- 输入的数据数组，形状为 (原始长度, 特征维度)
    target_length -- 目标长度，即插值后的数据长度
    
    返回：
    插值后的数据数组，形状为 (目标长度, 特征维度)
    """
    original_length = data.shape[0]
    
    if original_length >= target_length:
        return data[:target_length]
    
    original_indices = np.arange(original_length)
    target_indices = np.linspace(0, original_length - 1, target_length)
    
    df = pd.DataFrame(data, columns=['value'])
    df.index = original_indices
    
    df_interpolated = df.reindex(target_indices).interpolate(method='linear').bfill().ffill()
    
    return df_interpolated.values

def plot_dtw_error(X, y, dtw_distance, dtw_path):
    """
    绘制 DTW 路径和误差。
    """
    plt.figure(figsize=(12, 6))

    # 绘制对齐后的时间序列
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(X)), X, label='Standardized Data')
    plt.plot(np.arange(len(y)), y, label='Standardized Price Data', linestyle='--')
    plt.title('Aligned Time Series')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()

    # 绘制 DTW 路径
    plt.subplot(2, 1, 2)
    plt.plot([p[0] for p in dtw_path], [p[1] for p in dtw_path], marker='o', linestyle='-', color='r')
    plt.title('DTW Path')
    plt.xlabel('Index in X')
    plt.ylabel('Index in Y')

    plt.suptitle(f'DTW Distance: {dtw_distance:.2f}')
    plt.show()

def spline_interpolation(data, factor=1):
    """对数据进行样条插值"""
    n = len(data)
    x = np.arange(n)
    cs = CubicSpline(x, data)

    # 新的插值点
    x_new = np.linspace(0, n - 1, factor * n - (factor - 1))
    interpolated_data = cs(x_new)

    return interpolated_data

def create_lag_features(data, lags=1):
    """创建滞后特征"""
    df = pd.DataFrame(data)
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df[0].shift(lag)
    df = df.dropna()
    return df.values

def calculate_sign_changes(data):
    """计算符号变化，前面插入一个0，并对 None 值进行前向填充"""
    # 前向填充 None 值
    filled_data = []
    last_valid = None
    for value in data:
        if value is None:
            if last_valid is not None:
                filled_data.append(last_valid)
            else:
                filled_data.append(0)  # 根据需求选择填充值，这里用0填充
        else:
            filled_data.append(value)
            last_valid = value

    """计算每个相邻数据点的差值，并在结果前面添加0"""
    differences = [0]  # 添加0作为第一个元素
    for i in range(1, len(filled_data)):
        diff = filled_data[i] - filled_data[i - 1]
        differences.append(diff)


    return differences

def plot_stock_analysis(model , stock_name, stock_code, aligned_price, predicted_price , plot=False):
    """Plot actual vs. predicted stock prices and save the plot as an image file."""

    plt.figure(figsize=(10, 6))

    # Plot actual stock price
    plt.plot(aligned_price, label='实际股价', color='blue')

    # Plot predicted stock price
    plt.plot(predicted_price, label='预测股价', color='orange')

    # Customize plot
    plt.title(f'{stock_name} ({stock_code}) - 实际股价与预测股价对比')
    plt.xlabel('时间')
    plt.ylabel('股价')
    plt.legend()
    plt.grid(True)

    # Create directory if it doesn't exist
    save_dir = f'docs/{model}/'
    os.makedirs(save_dir, exist_ok=True)

    # Save the plot as an image file
    save_path = f'{save_dir}/{stock_code}.png'
    plt.savefig(save_path)
    if plot:
        # Show plot
        plt.show()
    # Close the figure to free up memory
    plt.close()
