from sklearn.neural_network import MLPRegressor
from matplotlib.font_manager import FontProperties
import matplotlib as mpl
from sklearn.linear_model import Ridge
# Specify the path to your Chinese font
font_path = 'msyh.ttc'
font_properties = FontProperties(fname=font_path)
mpl.rcParams['font.family'] = font_properties.get_name()
mpl.rcParams['axes.unicode_minus'] = False
MODEL='ridge'

from sklearn.neural_network import MLPRegressor
def analyze_stock(stock_name, stock_code, stock_type, revenue_per_share_yoy, price_data, revenue_per_share,
                  PB, revenue_t3m_avg, revenue_t3m_yoy, majority_shareholders_share_ratio, total_shareholders_count,
                  epst4q, volume_m, volume_m_avg, volume_ratio, latest_close_price):
    """分析股票数据"""

    # 提取 revenue_t3m_yoy, epst4q 的符号信息
    revenue_t3m_yoy_sign = calculate_sign_changes(revenue_t3m_yoy)
    epst4q_velocity = calculate_sign_changes(epst4q)

    # 创建有效数据列表
    valid_data = [
        (revenue_per_share_value, revenue_per_share_yoy_value, revenue, price, epst4q_value, epst4q_velocity_value,
         majority_shareholders_value, revenue_t3m_avg_value, pb_value, volume_value, volume_ratio_value, sign)
        for revenue_per_share_value, revenue_per_share_yoy_value, revenue, price, epst4q_value, epst4q_velocity_value,
            majority_shareholders_value, revenue_t3m_avg_value, pb_value, volume_value, volume_ratio_value, sign in
        zip(revenue_per_share, revenue_per_share_yoy, revenue_t3m_yoy, price_data, epst4q, epst4q_velocity,
            majority_shareholders_share_ratio, revenue_t3m_avg, PB, volume_m_avg, volume_ratio, revenue_t3m_yoy_sign)
        if None not in (revenue_per_share_value, revenue_per_share_yoy_value, revenue, price, epst4q_value,
                        epst4q_velocity_value, majority_shareholders_value, revenue_t3m_avg_value, pb_value, volume_value, volume_ratio_value, sign)
        and not (np.isnan(revenue_per_share_value) or np.isnan(revenue_per_share_yoy_value) or np.isnan(revenue)
                 or np.isnan(price) or np.isnan(epst4q_value) or np.isnan(epst4q_velocity_value)
                 or np.isnan(majority_shareholders_value) or np.isnan(revenue_t3m_avg_value)
                 or np.isnan(pb_value) or np.isnan(volume_value) or np.isnan(volume_ratio_value))
    ]

    if not valid_data:
        return None

    # 解包有效数据
    vaild_revenue_per_share, vaild_revenue_per_share_yoy, valid_revenue, valid_price, valid_epst4q, valid_epst4q_velocity, \
    valid_majority_shareholders, valid_revenue_t3m_avg, valid_pb, valid_volume, valid_vr_value, valid_sign = zip(*valid_data)

    # 将数据集打包后统一进行插值
    interpolated_data = interpolate_multiple_data(
        revenue_per_share=vaild_revenue_per_share,
        revenue_per_share_yoy=vaild_revenue_per_share_yoy,
        revenue=valid_revenue,
        epst4q=valid_epst4q,
        epst4q_velocity=valid_epst4q_velocity,
        majority_shareholders=valid_majority_shareholders,
        revenue_t3m_avg=valid_revenue_t3m_avg,
        pb=valid_pb,
        volume=valid_volume,
        vr_value=valid_vr_value,
        sign=valid_sign
    )


    # 正规化与归一化数据
    interpolated_price = spline_interpolation(valid_price)
    price_normalized, scaler_y = normalize_and_standardize_data(interpolated_price)

    # 合并数据
    # 对 interpolated_data 字典中的每个键值进行 reshape，然后堆叠成一个二维数组
    X_combined = np.hstack([value.reshape(-1, 1) for value in interpolated_data.values()])

    # 对输入特征进行标准化
    X_combined_normalize , x_scaler = normalize_and_standardize_data(X_combined)

    assert X_combined.shape[1] == 11, "X_combined 数据的特征数量应为 11"
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_combined_normalize, price_normalized.flatten(), test_size=0.2, random_state=42)

    # 初始化Ridge模型，指定正则化参数alpha
    ridge_model = Ridge(alpha=1.0)

    # 训练模型
    ridge_model.fit(X_train, y_train)

    ridge_model.fit(X_train, y_train)

    # 预测和评估
    y_pred_final = ridge_model.predict(X_test)
    final_mse = mean_squared_error(y_test, y_pred_final)

    # 预测和评估
    y_pred_final = ridge_model.predict(X_test)
    assert y_pred_final.shape == y_test.shape, "预测结果与测试集标签的形状不匹配"

    estimated_price_scaled = ridge_model.predict(X_combined_normalize)
    estimated_price = scaler_y.inverse_transform(estimated_price_scaled.reshape(-1, 1)).ravel()
    estimated_price_last = estimated_price[-1]

    assert estimated_price.shape == interpolated_price.shape, "反归一化后的预测值形状不正确"
    # 计算价格差异
    price_difference = estimated_price_last - latest_close_price
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

    interpolated_price = np.append((interpolated_price),latest_close_price)
    df = pd.DataFrame(estimated_price, columns=['Value'])
    # 計算 5 日移動平均值
    df['3_day_MA'] = df['Value'].rolling(window=3).mean()

    # 绘图
    # Plot and save the results
    plot_stock_analysis(MODEL , stock_name, stock_code, interpolated_price, estimated_price ,False)

    # 返回结果信息
    result_message = (f'<span style="color: {color};">{stock_name} {stock_code} ({stock_type}) - '
                      f'实际股价: {latest_close_price:.2f}, 推算股价: {estimated_price_last:.2f} ({price_diff_percentage:.2f}%) {action} '
                      f'MSE: {final_mse:.2f} </span><br>')

    return result_message


def main():
    NUM_DATA_POINTS = 120  # 控制要使用的数据点数量
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
             total_shareholders_count, epst4q ,volume_m , volume_m_avg ,volume_ratio ,latest_close_price) = fetch_stock_data(NUM_DATA_POINTS, FETCH_LATEST_CLOSE_PRICE_ONLINE,  stock_code)


            result = analyze_stock(stock_name, stock_code, stock_type, revenue_per_share_yoy, price_data,
                                   revenue_per_share, PB, revenue_t3m_avg, revenue_t3m_yoy,
                                   majority_shareholders_share_ratio, total_shareholders_count,epst4q, volume_m,
                                   volume_m_avg, volume_ratio,
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
