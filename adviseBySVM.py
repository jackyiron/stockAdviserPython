
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np
import os
from stockPublicFunction import *
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib as mpl
# Specify the path to your Chinese font
font_path = 'msyh.ttc'
MODEL='svm'
from matplotlib.font_manager import FontProperties

MODEL='bayes'

def analyze_stock(stock_name, stock_code, stock_type, revenue_per_share_yoy, price_data, revenue_per_share,
                  PB, revenue_t3m_avg, revenue_t3m_yoy, majority_shareholders_share_ratio, total_shareholders_count,
                  epst4q, volume_m, volume_m_avg, latest_close_price):
    """分析股票数据"""
    # 提取 revenue_t3m_yoy, epst4q 的符号信息
    revenue_t3m_yoy_sign = calculate_sign_changes(revenue_t3m_yoy)
    epst4q_velocity = calculate_sign_changes(epst4q)

    # 创建有效数据列表
    valid_data = [
        (revenue_per_share_value, revenue_per_share_yoy_value, revenue, price, epst4q_value, epst4q_velocity_value,
         majority_shareholders_value, revenue_t3m_avg_value, pb_value, volume_value, sign)
        for revenue_per_share_value, revenue_per_share_yoy_value, revenue, price, epst4q_value, epst4q_velocity_value,
            majority_shareholders_value, revenue_t3m_avg_value, pb_value, volume_value, sign in
        zip(revenue_per_share, revenue_per_share_yoy, revenue_t3m_yoy, price_data, epst4q, epst4q_velocity,
            majority_shareholders_share_ratio, revenue_t3m_avg, PB, volume_m_avg, revenue_t3m_yoy_sign)
        if None not in (revenue_per_share_value, revenue_per_share_yoy_value, revenue, price, epst4q_value,
                        epst4q_velocity_value, majority_shareholders_value, revenue_t3m_avg_value, pb_value, volume_value, sign)
        and not (np.isnan(revenue_per_share_value) or np.isnan(revenue_per_share_yoy_value) or np.isnan(revenue)
                 or np.isnan(price) or np.isnan(epst4q_value) or np.isnan(epst4q_velocity_value)
                 or np.isnan(majority_shareholders_value) or np.isnan(revenue_t3m_avg_value)
                 or np.isnan(pb_value) or np.isnan(volume_value))
    ]

    if not valid_data:
        return None

    # 解包有效数据
    vaild_revenue_per_share, vaild_revenue_per_share_yoy, valid_revenue, valid_price, valid_epst4q, valid_epst4q_velocity, \
    valid_majority_shareholders, valid_revenue_t3m_avg, valid_pb, valid_volume, valid_sign = zip(*valid_data)

    # 对数据进行样条插值
    interpolated_revenue_per_share = spline_interpolation(np.array(vaild_revenue_per_share))
    interpolated_revenue_per_share_yoy = spline_interpolation(np.array(vaild_revenue_per_share_yoy))
    interpolated_revenue = spline_interpolation(np.array(valid_revenue))
    interpolated_price = spline_interpolation(np.array(valid_price))
    interpolated_epst4q = spline_interpolation(np.array(valid_epst4q))
    interpolated_epst4q_velocity = spline_interpolation(np.array(valid_epst4q_velocity))
    interpolated_majority_shareholders = spline_interpolation(np.array(valid_majority_shareholders))
    interpolated_revenue_t3m_avg = spline_interpolation(np.array(valid_revenue_t3m_avg))
    interpolated_pb = spline_interpolation(np.array(valid_pb))
    interpolated_volume = spline_interpolation(np.array(valid_volume))
    interpolated_sign = spline_interpolation(np.array(valid_sign))

    # 准备时间序列数据
    revenue_per_share_series = interpolated_revenue_per_share.reshape(-1, 1)
    revenue_per_share_yoy_series = interpolated_revenue_per_share_yoy.reshape(-1, 1)
    price_series = interpolated_price.reshape(-1, 1)
    revenue_series = interpolated_revenue.reshape(-1, 1)
    epst4q_series = interpolated_epst4q.reshape(-1, 1)
    epst4q_velocity_series = interpolated_epst4q_velocity.reshape(-1, 1)
    majority_shareholders_series = interpolated_majority_shareholders.reshape(-1, 1)
    revenue_t3m_avg_series = interpolated_revenue_t3m_avg.reshape(-1, 1)
    pb_series = interpolated_pb.reshape(-1, 1)
    volume_series = interpolated_volume.reshape(-1, 1)
    sign_series = interpolated_sign.reshape(-1, 1)

    # 正规化与归一化数据
    revenue_per_share_normalized, scaler_X1 = normalize_and_standardize_data(revenue_per_share_series)
    revenue_per_share_yoy_normalized, scaler_X2 = normalize_and_standardize_data(revenue_per_share_yoy_series)
    revenue_normalized, scaler_X3 = normalize_and_standardize_data(revenue_series)
    epst4q_normalized, scaler_X4 = normalize_and_standardize_data(epst4q_series)
    epst4q_velocity_normalized, scaler_X5 = normalize_and_standardize_data(epst4q_velocity_series)
    majority_shareholders_normalized, scaler_X6 = normalize_and_standardize_data(majority_shareholders_series)
    revenue_t3m_avg_normalized, scaler_X7 = normalize_and_standardize_data(revenue_t3m_avg_series)
    pb_normalized, scaler_X8 = normalize_and_standardize_data(pb_series)
    volume_normalized, scaler_X9 = normalize_and_standardize_data(volume_series)
    sign_normalized, scaler_X10 = normalize_and_standardize_data(sign_series)
    price_normalized, scaler_y = normalize_and_standardize_data(price_series)

    # 合并数据
    X_combined = np.hstack((
        revenue_per_share_normalized.reshape(-1, 1),
        revenue_per_share_yoy_normalized.reshape(-1, 1),
        revenue_normalized.reshape(-1, 1),
        epst4q_normalized.reshape(-1, 1),
        epst4q_velocity_normalized.reshape(-1, 1),
        majority_shareholders_normalized.reshape(-1, 1),
        revenue_t3m_avg_normalized.reshape(-1, 1),
        pb_normalized.reshape(-1, 1),
        volume_normalized.reshape(-1, 1),
        sign_normalized.reshape(-1, 1)
    ))

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_combined, price_normalized.flatten(), test_size=0.2, random_state=42)

    # 使用 SVM 模型
    svm_model = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=1.0, epsilon=0.1))

    svm_model.fit(X_train, y_train)

    # 预测和评估
    y_pred_final = svm_model.predict(X_test)
    final_mse = mean_squared_error(y_test, y_pred_final)

    # 使用最新数据进行预测
    current_feature = np.array([[revenue_per_share[-1], revenue_per_share_yoy[-1], revenue_t3m_yoy[-1], epst4q[-1],
                                 epst4q_velocity[-1], majority_shareholders_share_ratio[-1], revenue_t3m_avg[-1], PB[-1], volume_m[-1], revenue_t3m_yoy_sign[-1]]])
    current_feature_scaled = np.hstack((
        scaler_X1.transform(current_feature[:, 0].reshape(-1, 1)),
        scaler_X2.transform(current_feature[:, 1].reshape(-1, 1)),
        scaler_X3.transform(current_feature[:, 2].reshape(-1, 1)),
        scaler_X4.transform(current_feature[:, 3].reshape(-1, 1)),
        scaler_X5.transform(current_feature[:, 4].reshape(-1, 1)),
        scaler_X6.transform(current_feature[:, 5].reshape(-1, 1)),
        scaler_X7.transform(current_feature[:, 6].reshape(-1, 1)),
        scaler_X8.transform(current_feature[:, 7].reshape(-1, 1)),
        scaler_X9.transform(current_feature[:, 8].reshape(-1, 1)),
        scaler_X10.transform(current_feature[:, 9].reshape(-1, 1))
    ))
    estimated_price_scaled = svm_model.predict(current_feature_scaled)
    estimated_price = scaler_y.inverse_transform(estimated_price_scaled.reshape(-1, 1)).ravel()[0]

    # 计算价格差异
    price_difference = estimated_price - latest_close_price
    price_diff_percentage = price_difference / latest_close_price * 100

    # 根据价格差异和 EPST4Q 的值确定颜色和操作
    if price_diff_percentage > 50:
        color = 'lightseagreen'
        action = '强力买入'
    elif price_diff_percentage < -50:
        color = 'darkred'
        action = '强力卖出'
    elif 20 <= price_diff_percentage <= 50:
        color = 'green'
        action = '买入'
    elif -50 <= price_diff_percentage <= -20:
        color = 'red'
        action = '卖出'
    else:
        color = 'black'
        action = ''

    # Predict full data set for plotting
    combined_features_all = np.hstack((
        revenue_per_share_normalized.reshape(-1, 1),
        revenue_per_share_yoy_normalized.reshape(-1, 1),
        revenue_normalized.reshape(-1, 1),
        epst4q_normalized.reshape(-1, 1),
        epst4q_velocity_normalized.reshape(-1, 1),
        majority_shareholders_normalized.reshape(-1, 1),
        revenue_t3m_avg_normalized.reshape(-1, 1),
        pb_normalized.reshape(-1, 1),
        volume_normalized.reshape(-1, 1),
        sign_normalized.reshape(-1, 1)
    ))

    predicted_price = svm_model.predict(combined_features_all)
    predicted_price = scaler_y.inverse_transform(predicted_price.reshape(-1, 1)).ravel()

    # 绘图
    # Plot and save the results
    plot_stock_analysis('bayes' , stock_name, stock_code, interpolated_price, predicted_price, False)

    # 返回结果信息
    result_message = (f'<span style="color: {color};">{stock_name} {stock_code} ({stock_type}) - '
                      f'实际股价: {latest_close_price:.2f}, 推算股价: {estimated_price:.2f} ({price_diff_percentage:.2f}%) {action} '
                      f'MSE: {final_mse:.2f} </span><br>')

    return result_message


def main():
    NUM_DATA_POINTS = 60  # 控制要使用的数据点数量
    FETCH_LATEST_CLOSE_PRICE_ONLINE = False  # 設置為 True 以從線上獲取最新股價，False 則使用本地文>件數據
    results = []  # 收集结果以便于同时写入文件和屏幕显示

    if FETCH_LATEST_CLOSE_PRICE_ONLINE:
        getLatestPrice()

    # 确保输出目录存在
    if not os.path.exists(f'docs/{MODEL}'):
        os.makedirs(f'docs/{MODEL}')

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
             total_shareholders_count, epst4q ,volume_m , volume_m_avg ,latest_close_price) = fetch_stock_data(NUM_DATA_POINTS, FETCH_LATEST_CLOSE_PRICE_ONLINE,  stock_code)

            result = analyze_stock(stock_name, stock_code, stock_type, revenue_per_share_yoy, price_data,
                                   revenue_per_share, PB, revenue_t3m_avg, revenue_t3m_yoy,
                                   majority_shareholders_share_ratio, total_shareholders_count,epst4q, volume_m, volume_m_avg,
                                   latest_close_price)

            if result:
                print(result)
                results.append(result)

        except ValueError as e:
            error_message = f"<p>处理股票 {stock_code} 时出错: {e}</p>"
            # 收集错误信息
            results.append(error_message)

    # 写入 HTML 文件
    with open(f'docs/{MODEL}/index.html', 'w', encoding='utf-8') as file:
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
