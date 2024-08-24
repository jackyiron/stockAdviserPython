from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from stockPublicFunction import *


def analyze_stock(stock_name, stock_code, stock_type, revenue_per_share_yoy, price_data, revenue_per_share,
                  PB, revenue_t3m_avg, revenue_t3m_yoy, majority_shareholders_share_ratio, total_shareholders_count,
                  epst4q, latest_close_price):
    """分析股票数据"""
    # 提取 revenue_t3m_yoy 的符号信息
    revenue_t3m_yoy_sign = calculate_sign_changes(revenue_t3m_yoy)

    # 创建有效数据列表
    valid_data = [
        (price, epst4q_value, sign)
        for price, epst4q_value, sign in
        zip(price_data, epst4q, revenue_t3m_yoy_sign)
        if None not in (price, epst4q_value, sign) and not (np.isnan(price) or np.isnan(epst4q_value))
    ]

    if not valid_data:
        return None

    # 解包有效数据
    valid_price, valid_epst4q, valid_sign = zip(*valid_data)

    # 对数据进行样条插值
    interpolated_price = spline_interpolation(np.array(valid_price))
    interpolated_epst4q = spline_interpolation(np.array(valid_epst4q))

    # 使用 linear_interpolate_sign 插值符号数据
    interpolated_sign = linear_interpolate_sign(np.array(valid_sign), len(interpolated_price))

    # 准备时间序列数据
    price_series = interpolated_price.reshape(-1, 1)
    epst4q_series = interpolated_epst4q.reshape(-1, 1)
    sign_series = interpolated_sign.reshape(-1, 1)

    # 设置权重：对负的营收可赋予更高的负权重
    price_weights = np.ones(price_series.shape[0])  # 权重设置为1，不进行加权

    # 正规化与归一化数据，加入权重参数
    price_normalized, min_max_scaler_y, scaler_y = normalize_and_standardize_data(price_series)
    epst4q_normalized, _, scaler_X1 = normalize_and_standardize_data(epst4q_series)
    sign_normalized, _, scaler_X2 = normalize_and_standardize_data(sign_series)

    # 对齐时间序列
    _, epst4q_path = fastdtw(epst4q_normalized, price_normalized, dist=euclidean)

    # 根据 DTW 路径对齐数据
    aligned_epst4q = np.array([epst4q_normalized[i] for i, _ in epst4q_path])
    aligned_sign = interpolated_sign  # 已经插值完成，无需DTW对齐
    aligned_y = np.array([price_normalized[j] for _, j in epst4q_path]).flatten()

    # 插值对齐后的数据
    target_length = len(aligned_y)
    aligned_epst4q_interpolated = linear_interpolation(aligned_epst4q, target_length)
    aligned_sign_interpolated = linear_interpolation(aligned_sign, target_length)

    # 合并对齐后的数据作为模型输入
    X_combined = np.hstack((
        aligned_epst4q_interpolated.reshape(-1, 1),
        aligned_sign_interpolated.reshape(-1, 1)
    ))

    # 训练集和测试集划分
    X_train, X_test, y_train, y_test = train_test_split(X_combined, aligned_y, test_size=0.2, random_state=42)

    # 使用 GridSearchCV 进行 alpha 参数优化
    ridge = Ridge()
    parameters = {'alpha': [0.1, 1.0, 10.0, 100.0, 200.0]}
    grid_search = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(X_train, y_train)

    # 获取最佳模型
    model = grid_search.best_estimator_

    # 预测和评估
    y_pred_final = model.predict(X_test)
    final_mse = mean_squared_error(y_test, y_pred_final)

    # 使用最新数据进行预测
    current_feature = np.array([[epst4q[-1], revenue_t3m_yoy_sign[-1]]])
    current_feature_scaled = np.hstack((
        scaler_X1.transform(current_feature[:, 0].reshape(-1, 1)),
        scaler_X2.transform(current_feature[:, 1].reshape(-1, 1))
    ))
    estimated_price_scaled = model.predict(current_feature_scaled)
    estimated_price = scaler_y.inverse_transform(estimated_price_scaled.reshape(-1, 1)).ravel()[0]

    # 计算价格差异
    price_difference = estimated_price - latest_close_price
    price_diff_percentage = price_difference / latest_close_price * 100

    # 根据价格差异和 EPST4Q 的值确定颜色和操作
    if price_diff_percentage > 50 and epst4q[-1] > 0:
        color = 'lightseagreen'
        action = '强力买入'
    elif price_diff_percentage < -50 and epst4q[-1] < 0:
        color = 'darkred'
        action = '强力卖出'
    elif 20 <= price_diff_percentage <= 50 and epst4q[-1] > 0:
        color = 'green'
        action = '买入'
    elif -50 <= price_diff_percentage <= -20 and epst4q[-1] < 0:
        color = 'red'
        action = '卖出'
    else:
        color = 'black'
        action = ''

    result_message = (f'<span style="color: {color};">{stock_name} {stock_code} ({stock_type}) - '
                      f'实际股价: {latest_close_price:.2f}, 推算股价: {estimated_price:.2f} ({price_diff_percentage:.2f}%) {action} '
                      f'MSE: {final_mse:.2f} </span><br>')

    return result_message



def main():
    NUM_DATA_POINTS = 40  # 控制要使用的数据点数量
    FETCH_LATEST_CLOSE_PRICE_ONLINE = False  # 設置為 True 以從線上獲取最新股價，False 則使用本地文>件數據
    output_file_name = 'ridge.html'  # 输出文件名
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
