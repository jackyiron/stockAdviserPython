import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_json_data(file_path):
    """读取JSON文件并提取数据"""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def preprocess_data(data):
    """预处理数据并使用插值统一长度"""
    monthly_data = data.get("monthly", {})
    all_data = {}

    for key, value in monthly_data.items():
        if 'data' in value:
            # 提取数据并去除无效值
            data_list = [item[1] for item in value['data'] if item[1] not in ["無", None]]
            if data_list:
                all_data[key] = pd.Series([float(i) for i in data_list])

    # 将所有数据列转换为 DataFrame
    df = pd.DataFrame(all_data)

    # 确定统一的长度
    max_length = df.apply(len).max()

    # 对所有列进行插值，使数据列长度一致
    for column in df.columns:
        df[column] = df[column].reindex(range(max_length)).interpolate(method='linear').bfill().ffill()

    return df

def create_dataset(df, target_column):
    """创建数据集"""
    if df.empty:
        raise ValueError("数据框为空，请检查数据预处理步骤。")

    if target_column not in df.columns:
        raise ValueError(f"目标列 {target_column} 不在数据中。请检查数据集。")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"特征矩阵 X 和目标变量 y 行数不一致：X={X.shape[0]}, y={y.shape[0]}")

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def build_model(input_dim, layers_config):
    """构建神经网络模型"""
    model = Sequential()
    for units, activation in layers_config:
        model.add(Dense(units, activation=activation, input_dim=input_dim))
    model.add(Dense(1, activation='linear'))  # 输出层，根据具体问题调整激活函数
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def save_model(model, file_path):
    """保存训练好的模型"""
    model.save(file_path)

def load_model(filepath):
    """加载保存的模型"""
    try:
        model = tf.keras.models.load_model(filepath)
        return model
    except OSError:
        print("无法加载模型。请确保模型文件存在。")
        return None

def predict_stock_price(model, X):
    """使用模型进行股价预测"""
    predictions = model.predict(X)
    return predictions

def calculate_fair_price(predictions, adjustment_factor=1.0):
    """根据预测结果和调整因子计算合理价格"""
    fair_price = np.mean(predictions) * adjustment_factor
    return fair_price

def main():
    stock_code = '1101'  # 示例股票代码
    file_path = f'stockData/{stock_code}.json'  # JSON 文件路径
    model_path = 'saved_model.h5'  # 模型保存路径

    data = load_json_data(file_path)
    df = preprocess_data(data)

    target_column = 'GordonReturnRateBy4Y'  # 使用实际存在的目标列
    X_train, X_test, y_train, y_test = create_dataset(df, target_column)

    # 模型参数配置
    layers_config = [(64, 'relu'), (32, 'relu')]  # 层配置: (units, activation)

    # 训练模型
    model = build_model(X_train.shape[1], layers_config)
    model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.1)

    # 保存模型
    save_model(model, model_path)

    # 进行预测
    X_scaled = StandardScaler().fit_transform(df.drop(columns=[target_column]))
    predictions = predict_stock_price(model, X_scaled)
    fair_price = calculate_fair_price(predictions, adjustment_factor=1.0)

    print(f"股票代码 {stock_code} 的预测合理价格: {fair_price}")

if __name__ == "__main__":
    main()
