from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, Exponentiation, ConstantKernel as C

def create_gaussian_process(kernel_name='RationalQuadratic'):
    # 创建指定的内核
    if kernel_name == 'RBF':
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    elif kernel_name == 'Matern':
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
    elif kernel_name == 'RationalQuadratic':
        kernel = C(1.0, (1e-3, 1e3)) * RationalQuadratic(length_scale=1.0, alpha=1.0)
    elif kernel_name == 'Exponential':
        kernel = C(1.0, (1e-3, 1e3)) * Exponentiation(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    else:
        raise ValueError(f"Unsupported kernel name: {kernel_name}")

    # 创建高斯过程回归模型
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, optimizer='fmin_l_bfgs_b')
    return gp

# 示例使用
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

    # 创建高斯过程回归模型
    gp = create_gaussian_process(kernel_name='RBF')  # 或 'M

def main():
    NUM_DATA_POINTS = 40  # 控制要使用的数据点数量
    output_file_name = 'gauss.html'  # 输出文件名

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
