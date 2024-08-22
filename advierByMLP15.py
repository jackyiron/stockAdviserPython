from sklearn.impute import SimpleImputer

from stockPublicFunction import *
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt


def analyze_stock(stock_name, stock_code, stock_type, revenue_per_share_yoy, price_data, revenue_per_share,
                  PB, revenue_t3m_avg, revenue_t3m_yoy, majority_shareholders_share_ratio, total_shareholders_count,
                  latest_close_price):
    """分析股票数据"""

    # 创建有效数据列表
    valid_data = [
        (revenue, price)
        for revenue, price in zip(revenue_t3m_yoy, price_data)
        if None not in (revenue, price) and not (np.isnan(revenue) or np.isnan(price))
    ]

    if not valid_data:
        return None

    # 解包有效数据
    valid_revenue, valid_price = zip(*valid_data)

    # 准备时间序列数据
    price_series = np.array(valid_price).reshape(-1, 1)
    revenue_series = np.array(valid_revenue).reshape(-1, 1)


    # 正规化与归一化数据
    revenue_normalized, _, scaler_X = normalize_and_standardize_data(revenue_series)
    price_normalized, min_max_scaler_y, scaler_y = normalize_and_standardize_data(price_series)

    # 使用 fastdtw 对齐时间序列
    distance, path = fastdtw(revenue_normalized, price_normalized, dist=euclidean)

    # 根据 DTW 路径对齐数据
    aligned_X = np.array([revenue_normalized[i] for i, _ in path])
    aligned_y = np.array([price_normalized[j] for _, j in path]).flatten()

    # 确保对齐后的时间序列长度一致
    min_length = min(len(aligned_X), len(aligned_y))
    aligned_X = aligned_X[:min_length]
    aligned_y = aligned_y[:min_length]

    # 训练集和测试集划分
    X_train, X_test, y_train, y_test = train_test_split(aligned_X, aligned_y, test_size=0.2, random_state=42)

    # 定义和训练神经网络回归模型
    mlp = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', alpha=0.0001, max_iter=3000,
                       random_state=42)

    # 训练模型
    mlp.fit(X_train, y_train)

    # 预测和评估
    y_pred_final = mlp.predict(X_test)
    final_mse = mean_squared_error(y_test, y_pred_final)

    # 使用最新数据进行预测
    current_feature = np.array([[revenue_t3m_yoy[-1]]])
    current_feature_scaled = scaler_X.transform(current_feature)
    estimated_price_scaled = mlp.predict(current_feature_scaled)
    estimated_price = scaler_y.inverse_transform(estimated_price_scaled.reshape(-1, 1)).ravel()[0]

    # 计算价格差异
    price_difference = estimated_price - latest_close_price
    price_diff_percentage = price_difference / latest_close_price * 100


    if abs(price_diff_percentage) > 60:
        color = 'darkred' if latest_close_price > estimated_price else 'lightseagreen'
        action = '强力卖出' if latest_close_price > estimated_price else '强力买入'
    elif 30 <= abs(price_diff_percentage) <= 60:
        color = 'red' if latest_close_price > estimated_price else 'green'
        action = '卖出' if latest_close_price > estimated_price else '买入'
    else:
        color = 'black'
        action = ''

    result_message =  (f'<span style="color: {color};">{stock_name} {stock_code} ({stock_type}) - '
            f'实际股价: {latest_close_price:.2f}, 推算股价: {estimated_price:.2f} ({price_diff_percentage:.2f}%) {action} '
            f'MSE: {final_mse:.2f} </span><br>')

    # 调用绘图函数
    # plot_dtw_error(aligned_X, aligned_y, distance, path)

    return result_message

def main():
    NUM_DATA_POINTS = 40  # 控制要使用的數據點數量
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
             total_shareholders_count, latest_close_price) = fetch_stock_data(NUM_DATA_POINTS, stock_code)

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