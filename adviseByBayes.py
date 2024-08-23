from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def analyze_stock(stock_name, stock_code, stock_type, revenue_per_share_yoy, price_data, revenue_per_share,
                  PB, revenue_t3m_avg, revenue_t3m_yoy, majority_shareholders_share_ratio, total_shareholders_count,
                  latest_close_price):
    """分析股票数据使用贝叶斯回归模型"""

    valid_data = [
        (revenue, price, rev_per_share)
        for revenue, price, rev_per_share in zip(revenue_t3m_yoy, price_data, revenue_per_share)
        if None not in (revenue, price, rev_per_share) and not (np.isnan(revenue) or np.isnan(price) or np.isnan(rev_per_share))
    ]

    if not valid_data:
        return None

    # 解包有效数据
    valid_revenue, valid_price, valid_rev_per_share = zip(*valid_data)

    # 准备时间序列数据
    price_series = np.array(valid_price).reshape(-1, 1)
    revenue_series = np.array(valid_revenue).reshape(-1, 1)
    rev_per_share_series = np.array(valid_rev_per_share).reshape(-1, 1)

    # 设置权重：对负的营收赋予更高的负权重
    revenue_weights = np.where(revenue_series < 0, 2.0, 1.0)

    # 正规化与归一化数据，加入权重参数
    revenue_normalized, _, scaler_X1 = normalize_and_standardize_data_weight(revenue_series, weights=revenue_weights)
    rev_per_share_normalized, _, scaler_X2 = normalize_and_standardize_data(rev_per_share_series)
    price_normalized, min_max_scaler_y, scaler_y = normalize_and_standardize_data(price_series)

    # 合并对齐后的数据作为模型输入
    X_combined = np.hstack((revenue_normalized.reshape(-1, 1), rev_per_share_normalized.reshape(-1, 1)))

    # 训练集和测试集划分
    X_train, X_test, y_train, y_test = train_test_split(X_combined, price_normalized.flatten(), test_size=0.2, random_state=42)

    # 定义贝叶斯回归模型
    model = BayesianRidge()

    # 训练模型
    model.fit(X_train, y_train)

    # 预测和评估
    y_pred_final = model.predict(X_test)
    final_mse = mean_squared_error(y_test, y_pred_final)

    # 使用最新数据进行预测
    current_feature = np.array([[revenue_t3m_yoy[-1], revenue_per_share[-1]]])
    current_feature_scaled = np.hstack((
        scaler_X1.transform(current_feature[:, 0].reshape(-1, 1)),
        scaler_X2.transform(current_feature[:, 1].reshape(-1, 1))
    ))
    estimated_price_scaled = model.predict(current_feature_scaled)
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


from stockPublicFunction import *



def main():
    NUM_DATA_POINTS = 40  # 控制要使用的数据点数量
    FETCH_LATEST_CLOSE_PRICE_ONLINE = True  # 設置為 True 以從線上獲取最新股價，False 則使用本地文>件數據
    output_file_name = 'bayes.html'  # 输出文件名

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
                 total_shareholders_count, latest_close_price) = fetch_stock_data(NUM_DATA_POINTS, FETCH_LATEST_CLOSE_PRICE_ONLINE, stock_code)

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
