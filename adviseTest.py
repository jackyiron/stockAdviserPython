from skopt import gp_minimize
from joblib import parallel_backend
from skopt.space import Real, Integer
import requests
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, ConstantKernel as C
from scipy.interpolate import interp1d
import numpy as np
import json
import os
import sys
from bs4 import BeautifulSoup
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed


os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)

# 常量
URL_TEMPLATE = "https://statementdog.com/api/v2/fundamentals/{stock_code}/2014/2024/cf?qbu=true&qf=analysis"
NUM_DATA_POINTS = 80  # 控制要使用的數據點數量
FETCH_LATEST_CLOSE_PRICE_ONLINE = False  # 設置為 True 以從線上獲取最新股價，False 則使用本地文件數據

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

def interpolate_quarterly_to_monthly(quarterly_data, num_months):
    """将季度数据插值到每月数据"""
    if len(quarterly_data) < 2:
        return [None] * num_months

    # 创建季度时间轴
    quarterly_indices = np.arange(len(quarterly_data)) * 3
    # 创建对应的月度时间轴
    monthly_indices = np.arange(num_months)

    # 插值函数
    interp_func = interp1d(quarterly_indices, quarterly_data, kind='linear', fill_value='extrapolate')
    return interp_func(monthly_indices)

def pad_list(data, length, pad_value=0.0):
    """填充列表到指定长度"""
    # 过滤掉 None 值
    filtered_data = [x for x in data if x is not None]
    if len(filtered_data) < length:
        # 使用默认值填充不足部分
        return filtered_data + [pad_value] * (length - len(filtered_data))
    return filtered_data

def pad_data(data, length):
    """将数据填充到指定长度"""
    if len(data) < length:
        return data + [None] * (length - len(data))
    return data[:length]

import os
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed


def parse_float(value):
    try:
        return float(value)
    except ValueError:
        return None

def pad_list(data, length):
    """将列表填充到指定长度"""
    return (data + [None] * length)[-length:]

import os
import json
import numpy as np

def parse_float(value):
    try:
        return float(value)
    except ValueError:
        return None

def pad_list(data_list, max_length):
    return (data_list[-max_length:] if len(data_list) > max_length else
            [None] * (max_length - len(data_list)) + data_list)

def fetch_stock_data(stock_code):
    """从本地文件获取股票数据"""
    file_path = f'stockData/{stock_code}.json'

    if not os.path.exists(file_path):
        raise ValueError(f"文件 {file_path} 不存在。请确保文件路径和股票代码正确。")

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    monthly_data = data.get("monthly", {})
    quarterly_data = data.get("quarterly", {})

    # 读取原始数据
    PB = [parse_float(item[1]) for item in monthly_data.get("PB", {}).get("data", []) if
          parse_float(item[1]) is not None]
    revenue_per_share = [parse_float(item[1]) for item in monthly_data.get("RevenuePerShare", {}).get("data", []) if
                         parse_float(item[1]) is not None]
    revenue_per_share_yoy = [parse_float(item[1]) for item in monthly_data.get("RevenuePerShareYOY", {}).get("data", [])
                             if parse_float(item[1]) is not None]
    price_data = [parse_float(item[1]) for item in monthly_data.get("Price", {}).get("data", []) if
                  parse_float(item[1]) is not None]

    # 获取其他数据
    revenue_t3m_avg = [parse_float(item[1]) for item in monthly_data.get("RevenueT3MAvg", {}).get("data", []) if
                       parse_float(item[1]) is not None]
    revenue_t3m_yoy = [parse_float(item[1]) for item in monthly_data.get("RevenueT3MYOY", {}).get("data", []) if
                       parse_float(item[1]) is not None]
    majority_shareholders_share_ratio = [parse_float(item[1]) for item in
                                         monthly_data.get("MajorityShareholdersShareRatio", {}).get("data", []) if
                                         parse_float(item[1]) is not None]
    total_shareholders_count = [parse_float(item[1]) for item in
                                monthly_data.get("TotalShareholdersCount", {}).get("data", []) if
                                parse_float(item[1]) is not None]

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

    # 确定最大长度
    max_length = min(NUM_DATA_POINTS, len(price_data))

    # 从最后开始提取最大长度的数据
    PB = PB[-max_length:]
    revenue_per_share = revenue_per_share[-max_length:]
    revenue_per_share_yoy = revenue_per_share_yoy[-max_length:]
    price_data = price_data[-max_length:]
    revenue_t3m_avg = revenue_t3m_avg[-max_length:]
    revenue_t3m_yoy = revenue_t3m_yoy[-max_length:]
    majority_shareholders_share_ratio = majority_shareholders_share_ratio[-max_length:]
    total_shareholders_count = total_shareholders_count[-max_length:]

    # 填充不足的部分
    PB = pad_list(PB, max_length)
    revenue_per_share = pad_list(revenue_per_share, max_length)
    revenue_per_share_yoy = pad_list(revenue_per_share_yoy, max_length)
    price_data = pad_list(price_data, max_length)
    revenue_t3m_avg = pad_list(revenue_t3m_avg, max_length)
    revenue_t3m_yoy = pad_list(revenue_t3m_yoy, max_length)
    majority_shareholders_share_ratio = pad_list(majority_shareholders_share_ratio, max_length)
    total_shareholders_count = pad_list(total_shareholders_count, max_length)


    return (revenue_per_share_yoy,
            price_data,
            revenue_per_share,
            PB,
            revenue_t3m_avg,
            revenue_t3m_yoy,
            majority_shareholders_share_ratio,
            total_shareholders_count,
            latest_close_price)

def analyze_stock(stock_name, stock_code, stock_type, revenue_per_share_yoy, price_data, revenue_per_share,
                  PB, revenue_t3m_avg, revenue_t3m_yoy, majority_shareholders_share_ratio, total_shareholders_count,
                  latest_close_price):
    """分析股票数据"""

    # 创建有效数据列表
    valid_data = [
        (yoy, price, revenue, pb, t3m_avg, t3m_yoy, majority, total_share_count)
        for
        yoy, price, revenue, pb, t3m_avg, t3m_yoy, majority, total_share_count
        in zip(
            revenue_per_share_yoy, price_data, revenue_per_share, PB,
            revenue_t3m_avg, revenue_t3m_yoy, majority_shareholders_share_ratio, total_shareholders_count
        )
        if None not in (
            yoy, price, revenue, pb, t3m_avg, t3m_yoy, majority, total_share_count
        )
    ]

    if not valid_data:
        return None

    # 解包有效数据
    valid_yoy_values, valid_price_data, valid_revenue_per_share, valid_PB, valid_t3m_avg, valid_t3m_yoy, valid_majority, valid_total_share_count = zip(
        *valid_data)

    # 检查特征数量
    if len(valid_yoy_values) == 0:
        return None

    X = np.column_stack((
        valid_yoy_values,
        valid_revenue_per_share,
        valid_PB,
        valid_t3m_avg,
        valid_t3m_yoy,
        valid_majority,
        valid_total_share_count
    ))

    y = np.array(valid_price_data)

    # 标准化数据
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # 定义目标函数
    def objective(length_scale, alpha):
        kernel = RationalQuadratic(length_scale=length_scale, alpha=alpha)
        gp = GaussianProcessRegressor(kernel=kernel, optimizer=None)
        gp.fit(X_train, y_train)
        y_pred = gp.predict(X_test)
        return mean_squared_error(y_test, y_pred)

    # 定义超参数搜索空间
    length_scale_values = np.logspace(-2, 2, num=10)  # 1e-2 到 1e2 的范围
    alpha_values = np.logspace(-1, 5, num=10)  # 1e-1 到 1e5 的范围

    # 使用 joblib 进行并行计算
    def parallel_objective(params):
        length_scale, alpha = params
        return objective(length_scale, alpha)

    param_grid = [(length_scale, alpha) for length_scale in length_scale_values for alpha in alpha_values]
    results = Parallel(n_jobs=-1)(delayed(parallel_objective)(params) for params in param_grid)

    # 找到最佳参数
    best_index = np.argmin(results)
    best_length_scale, best_alpha = param_grid[best_index]

    # 使用最佳参数训练最终模型
    best_kernel = RationalQuadratic(length_scale=best_length_scale, alpha=best_alpha)
    gp_final = GaussianProcessRegressor(kernel=best_kernel, optimizer=None)
    gp_final.fit(X_train, y_train)

    # 预测和评估
    y_pred_final = gp_final.predict(X_test)
    final_mse = mean_squared_error(y_test, y_pred_final)

    # 使用最新数据进行预测
    current_features = scaler_X.transform([[valid_yoy_values[-1], valid_revenue_per_share[-1], valid_PB[-1],
                                            valid_t3m_avg[-1], valid_t3m_yoy[-1], valid_majority[-1],
                                            valid_total_share_count[-1]]])
    estimated_price_scaled = gp_final.predict(current_features)
    estimated_price = scaler_y.inverse_transform(estimated_price_scaled.reshape(-1, 1)).ravel()[0]

    # 计算价格差异
    price_difference = estimated_price - latest_close_price
    price_diff_percentage = price_difference / latest_close_price * 100

    if abs(price_diff_percentage) > 30:
        color = 'darkred' if latest_close_price > estimated_price else 'lightseagreen'
        action = '强力卖出' if latest_close_price > estimated_price else '强力买入'
    elif 15 <= abs(price_diff_percentage) <= 30:
        color = 'red' if latest_close_price > estimated_price else 'green'
        action = '卖出' if latest_close_price > estimated_price else '买入'
    else:
        color = 'black'
        action = ''

    return (f'<span style="color: {color};">{stock_name} {stock_code} ({stock_type}) - '
            f'实际股价: {latest_close_price:.2f}, 推算股价: {estimated_price:.2f} ({price_diff_percentage:.2f}%) {action} MSE: {final_mse:.2f} </span><br>')



def main():
    with open('stockList.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split(' ')
        if len(parts) != 3:
            continue

        stock_code = parts[0]
        stock_name = parts[1]
        stock_type = parts[2]

        try:
            (revenue_per_share_yoy, price_data, revenue_per_share, PB,
             revenue_t3m_avg, revenue_t3m_yoy, majority_shareholders_share_ratio,
             total_shareholders_count, latest_close_price) = fetch_stock_data(stock_code)

            result = analyze_stock(stock_name, stock_code, stock_type, revenue_per_share_yoy, price_data,
                                   revenue_per_share, PB, revenue_t3m_avg, revenue_t3m_yoy,
                                   majority_shareholders_share_ratio, total_shareholders_count,
                                   latest_close_price)
            if result:
                print(result)
        except ValueError as e:
            print(f"Error processing stock {stock_code}: {e}")

    # 打印实际使用的数据点数量
    if 'price_data' in locals():
        num_data_points_used = len(price_data)
        print(f"本次使用了 {num_data_points_used} 个数据点分析")


if __name__ == "__main__":
    main()
