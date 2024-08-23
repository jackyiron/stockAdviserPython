from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np
import os
from stockPublicFunction import *
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np

def analyze_stock(stock_name, stock_code, stock_type, revenue_per_share_yoy, price_data, revenue_per_share,
                  PB, revenue_t3m_avg, revenue_t3m_yoy, majority_shareholders_share_ratio, total_shareholders_count,
                  epst4q, latest_close_price):
    """分析股票数据"""
    # 提取 revenue_t3m_yoy 和 epst4q 的符号信息
    revenue_t3m_yoy_sign = extract_sign(revenue_t3m_yoy)
    epst4q_sign = extract_sign(epst4q)

    # 创建有效数据列表
    valid_data = [
        (revenue, price, epst4q_value, sign)
        for revenue, price, epst4q_value, sign in
        zip(revenue_t3m_yoy, price_data, epst4q, revenue_t3m_yoy_sign)
        if None not in (revenue, price, epst4q_value, sign) and not (
                    np.isnan(revenue) or np.isnan(price) or np.isnan(epst4q_value))
    ]

    if not valid_data:
        return None

    # 解包有效数据
    valid_revenue, valid_price, valid_epst4q, valid_sign = zip(*valid_data)

    # 对数据进行样条插值
    interpolated_revenue = spline_interpolation(np.array(valid_revenue))
    interpolated_price = spline_interpolation(np.array(valid_price))
    interpolated_epst4q = spline_interpolation(np.array(valid_epst4q))

    # 使用 linear_interpolate_sign 插值符号数据
    interpolated_sign = linear_interpolate_sign(np.array(valid_sign), len(interpolated_price))

    # 准备时间序列数据
    price_series = interpolated_price.reshape(-1, 1)
    revenue_series = interpolated_revenue.reshape(-1, 1)
    epst4q_series = interpolated_epst4q.reshape(-1, 1)
    sign_series = interpolated_sign.reshape(-1, 1)

    # 设置权重：对负的营收可赋予更高的负权重
    revenue_weights = np.where(revenue_series < 0, 2.0, 1.0)

    # 正规化与归一化数据，加入权重参数
    revenue_normalized, _, scaler_X1 = normalize_and_standardize_data_weight(revenue_series, weights=revenue_weights)
    epst4q_normalized, _, scaler_X2 = normalize_and_standardize_data(epst4q_series)
    sign_normalized, _, scaler_X3 = normalize_and_standardize_data(sign_series)
    price_normalized, min_max_scaler_y, scaler_y = normalize_and_standardize_data(price_series)

    # 分别使用 fastdtw 对齐时间序列
    _, revenue_path = fastdtw(revenue_normalized, price_normalized, dist=euclidean)
    _, epst4q_path = fastdtw(epst4q_normalized, price_normalized, dist=euclidean)

    # 根据 DTW 路径对齐数据
    aligned_revenue = np.array([revenue_normalized[i] for i, _ in revenue_path])
    aligned_epst4q = np.array([epst4q_normalized[i] for i, _ in epst4q_path])
    aligned_sign = interpolated_sign  # 已经插值完成，无需DTW对齐
    aligned_y = np.array([price_normalized[j] for _, j in revenue_path]).flatten()

    # 插值对齐后的数据
    target_length = len(aligned_y)
    aligned_revenue_interpolated = linear_interpolation(aligned_revenue, target_length)
    aligned_epst4q_interpolated = linear_interpolation(aligned_epst4q, target_length)
    aligned_sign_interpolated = linear_interpolation(aligned_sign, target_length)

    # 合并对齐后的数据作为模型输入
    X_combined = np.hstack((
        aligned_revenue_interpolated.reshape(-1, 1),
        aligned_epst4q_interpolated.reshape(-1, 1),
        aligned_sign_interpolated.reshape(-1, 1)
    ))

    # 训练集和测试集划分
    X_train, X_test, y_train, y_test = train_test_split(X_combined, aligned_y, test_size=0.2, random_state=42)

    # 直接使用 SVR 模型
    svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)  # 设置 C 和 epsilon 参数
    svr.fit(X_train, y_train)

    # 预测和评估
    y_pred_final = svr.predict(X_test)
    final_mse = mean_squared_error(y_test, y_pred_final)

    # 使用最新数据进行预测
    current_feature = np.array([[revenue_t3m_yoy[-1], epst4q[-1], revenue_t3m_yoy_sign[-1]]])
    current_feature_scaled = np.hstack((
        scaler_X1.transform(current_feature[:, 0].reshape(-1, 1)),
        scaler_X2.transform(current_feature[:, 1].reshape(-1, 1)),
        scaler_X3.transform(current_feature[:, 2].reshape(-1, 1))
    ))
    estimated_price_scaled = svr.predict(current_feature_scaled)
    estimated_price = scaler_y.inverse_transform(estimated_price_scaled.reshape(-1, 1)).ravel()[0]

    # 计算价格差异
    price_difference = estimated_price - latest_close_price
    price_diff_percentage = price_difference / latest_close_price * 100

    # 根据价格差异和 EPST4Q 的值确定颜色和操作
    if price_diff_percentage > 60:
        if epst4q[-1] > 0:
            color = 'lightseagreen'
            action = '强力买入'
        else:
            color = 'darkred'
            action = '强力卖出'
    elif 30 <= price_diff_percentage <= 60:
        if epst4q[-1] > 0:
            color = 'green'
            action = '买入'
        else:
            color = 'red'
            action = '卖出'
    else:
        color = 'black'
        action = ''

    result_message = (f'<span style="color: {color};">{stock_name} {stock_code} ({stock_type}) - '
                      f'实际股价: {latest_close_price:.2f}, 推算股价: {estimated_price:.2f} ({price_diff_percentage:.2f}%) {action} '
                      f'MSE: {final_mse:.2f} </span><br>')


    result_message = (f'<span style="color: {color};">{stock_name} {stock_code} ({stock_type}) - '
                      f'实际股价: {latest_close_price:.2f}, 推算股价: {estimated_price:.2f} ({price_diff_percentage:.2f}%) {action} '
                      f'MSE: {final_mse:.2f} </span><br>')

    return result_message


def main():
    NUM_DATA_POINTS = 40  # 控制要使用的数据点数量
    FETCH_LATEST_CLOSE_PRICE_ONLINE = False  # 設置為 True 以從線上獲取最新股價，False 則使用本地文>件數據
    output_file_name = 'svm.html'  # 输出文件名
    results = []  # 收集结果以便于同时写入文件和屏幕显示

    # 确保输出目录存在
    if not os.path.exists('docs'):
        os.makedirs('docs')

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
             total_shareholders_count, epst4q ,latest_close_price) = fetch_stock_data(NUM_DATA_POINTS, FETCH_LATEST_CLOSE_PRICE_ONLINE,  stock_code)

            result = analyze_stock(stock_name, stock_code, stock_type, revenue_per_share_yoy, price_data,
                                   revenue_per_share, PB, revenue_t3m_avg, revenue_t3m_yoy,
                                   majority_shareholders_share_ratio, total_shareholders_count,epst4q,
                                   latest_close_price)

            if result:
                print(result)
                results.append(result)

        except ValueError as e:
            error_message = f"<p>处理股票 {stock_code} 时出错: {e}</p>"
            # 收集错误信息
            results.append(error_message)

    # 写入 HTML 文件
    with open(f'docs/{output_file_name}', 'w', encoding='utf-8') as file:
        file.write('<html><head><title>股票分析结果</title></head><body>\n')
        file.write('<h1>股票分析结果</h1>\n')
        for result in results:
            file.write(result)
        file.write('</body></html>\n')


    # 打印实际使用的数据点数量
    if 'price_data' in locals():
        num_data_points_used = len(price_data)
        print(f"本次使用了 {num_data_points_used} 个数据点分析")

from stockPublicFunction import *

if __name__ == "__main__":
    main()
