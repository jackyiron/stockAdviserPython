import requests
import json
import os

# 目标URL
url = "https://www.twse.com.tw/rwd/zh/afterTrading/FMSRFK"

# 请求头
headers = {
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7,zh-CN;q=0.6",
    "Connection": "keep-alive",
    "Cookie": "_ga_LTMT28749H=GS1.1.1723996122.1.0.1723996122.0.0.0; _ga=GA1.1.451585127.1723996114; JSESSIONID=33D2447ED3FA25F31723D327A4885349; _ga_J2HVMN6FVP=GS1.1.1724571217.1.1.1724575158.38.0.0",
    "Host": "www.twse.com.tw",
    "Referer": "https://www.twse.com.tw/zh/trading/historical/fmsrfk.html",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest",
    "sec-ch-ua": "\"Google Chrome\";v=\"123\", \"Not:A-Brand\";v=\"8\", \"Chromium\";v=\"123\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"Windows\""
}

# 读取股票代码，指定文件编码
with open('stockList.txt', 'r', encoding='utf-8') as file:
    stock_codes = [line.strip().split()[0] for line in file]

# 请求参数的基本结构
params = {
    "response": "json",
    "_": "1724575155361"
}

# 创建存储数据的目录（如果不存在）
os.makedirs('stockData', exist_ok=True)

# 遍历所有股票代码
for stock_no in stock_codes:
    # 临时存储该股票的合并数据
    combined_data = []

    # 获取从2004年到2024年的数据
    for year in range(2004, 2025):
        # 格式化日期
        date_str = f"{year}0101"  # 使用每年的1月1日作为日期

        params["date"] = date_str
        params["stockNo"] = stock_no
        response = requests.get(url, headers=headers, params=params)

        # 检查响应状态码
        if response.status_code == 200:
            # 解析JSON内容
            response_json = response.json()
            data = response_json.get('data', [])

            if data:
                # 将数据添加到列表中
                combined_data.extend(data)
                print(f"Data for stock code {stock_no} on {date_str} has been collected.")
            else:
                print(f"No data found for stock code: {stock_no} on {date_str}")
        else:
            print(f"Request failed for stock code: {stock_no} on {date_str}, Status code: {response.status_code}")

    # 保存合并后的数据为 JSON 文件
    file_path = os.path.join('stockData', f'{stock_no}_m_vol.json')
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump({
            "stockNo": stock_no,
            "data": combined_data,
            "note": "年度,月份,最高價,最低價,加權(A/B)平均價,成交筆數,成交金額(A),成交股數(B),週轉率(%)"
        }, f, ensure_ascii=False, indent=4)

    print(f"Combined data for stock code {stock_no} has been saved to {file_path}")
