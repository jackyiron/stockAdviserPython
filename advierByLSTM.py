import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

def prepare_lstm_data(data, time_steps):
    """将数据准备为适用于 LSTM 的格式"""
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

def analyze_stock(stock_name, stock_code, stock_type, revenue_per_share_yoy, price_data, revenue_per_share,
                  PB, revenue_t3m_avg, revenue_t3m_yoy, majority_shareholders_share_ratio, total_shareholders_count,
                  latest_close_price):
    """分析股票数据使用 LSTM 模型"""

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

    # 使用 SimpleImputer 填充 NaN 值
    imputer = SimpleImputer(strategy='mean')
    revenue_series = imputer.fit_transform(revenue_series)
    price_series = imputer.fit_transform(price_series)

    # 正规化数据
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    revenue_normalized = scaler_X.fit_transform(revenue_series)
    price_normalized = scaler_y.fit_transform(price_series)

    # 准备 LSTM 数据
    TIME_STEPS = 10  # 使用的时间步长，可以根据需要调整
    X, y = prepare_lstm_data(price_normalized, TIME_STEPS)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 构建 LSTM 模型
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(TIME_STEPS, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 训练模型
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

    # 预测和评估
    y_pred_final = model.predict(X_test)
    final_mse = mean_squared_error(y_test, y_pred_final)

    # 使用最新数据进行预测
    current_feature = np.array([[revenue_t3m_yoy[-1]]])
    current_feature_scaled = scaler_X.transform(current_feature)
    last_sequence = np.array(price_normalized[-TIME_STEPS:]).reshape(1, TIME_STEPS, 1)
    estimated_price_scaled = model.predict(last_sequence)
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

    result_message = (f'<span style="color: {color};">{stock_name} {stock_code} ({stock_type}) - '
                      f'实际股价: {latest_close_price:.2f}, 推算股价: {estimated_price:.2f} ({price_diff_percentage:.2f}%) {action} '
                      f'MSE: {final_mse:.2f} </span><br>')

    return result_message

def main():
    NUM_DATA_POINTS = 40  # 控制要使用的數據點數量
    output_file_name = 'lstm.html'  # 输出文件名

    # 打开输出文件准备写入
    with open(f'docs/{output_file_name}', 'w', encoding='utf-8') as file:
        file.write('<html><head><title>Stock Analysis Results</title></head><body>\n')
        file.write('<h1>Stock Analysis Results</h1>\n')

        with open('stockList.txt', 'r', encoding='utf-8') as file_list:
            lines = file_list.readlines()

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
                    # 输出到终端
                    print(result)
                    # 写入到 HTML 文件
                    file.write(result)

            except ValueError as e:
                error_message = f"<p>Error processing stock {stock_code}: {e}</p>"
                # 输出到终端
                print(error_message)
                # 写入到 HTML 文件
                file.write(error_message)

        file.write('</body></html>\n')

    # 打印实际使用的数据点数量
    if 'price_data' in locals():
        num_data_points_used = len(price_data)
        print(f"本次使用了 {num_data_points_used} 个数据点分析")

from stockPublicFunction import *

if __name__ == "__main__":
    main()