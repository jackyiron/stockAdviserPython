import requests
import os
import json

# 定義 URL 模板
url_template = 'https://statementdog.com/api/v2/fundamentals/{stock_code}/2004/2024/cf?qbu=true&qf=analysis'

# 定義 headers
headers = {
    'accept': 'application/json, text/plain, */*',
    'accept-language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7,zh-CN;q=0.6',
    'baggage': 'sentry-environment=production,sentry-public_key=78ceb91faa50443fbc1127429dec792a,sentry-trace_id=dd75357973c546818063c9afbb37a559,sentry-sample_rate=0.3,sentry-sampled=false',
    'cookie': 'statementdog_device_id=UHhoSjBYZVF6VlliTjkyOXg0RDJGU2ZINDljdG1DOXUxM0RkYzZvRWRrQ0JWKzJ2aENsL05jRlhYVVR1R244NS0tSjhOOHZUcDc2Rm9aTDdCa0ZWR0M4dz09--de44e6822a6ad682d35fb9f362f917d323ffb10a; easy_ab=89f392a9-d6c1-430a-9cf5-b4c1693741ae; _fbp=fb.1.1723996138562.1220722206314028; _ga=GA1.1.1927792681.1723996139; g_state={"i_l":0}; upgrade_browser=1; AMP_0ab77a441f=JTdCJTIyZGV2aWNlSWQlMjIlM0ElMjIwMTk3ZGMyMi0zZDkyLTRhNDgtODU4Mi04ODcwNDgzM2U4MzElMjIlMkMlMjJ1c2VySWQlMjIlM0EyMjA2OCUyQyUyMnNlc3Npb25JZCUyMiUzQTE3MjQ0OTE3MjI1ODYlMkMlMjJvcHRPdXQlMjIlM0FmYWxzZSUyQyUyMmxhc3RFdmVudFRpbWUlMjIlM0ExNzI0NDkxNzI5MzkzJTJDJTIybGFzdEV2ZW50SWQlMjIlM0EyOTE1JTJDJTIycGFnZUNvdW50ZXIlMjIlM0EwJTdE; _ga_K9Y9Y589MM=GS1.1.1724494118.29.0.1724494118.60.0.0; search_stock_referer=https%3A%2F%2Fstatementdog.com%2F; _statementdog_session_v2=yq6HHs9zonClb994hhAnQYY1C9x9HTRliiMEyQMGEAWhe4bqmrkPd%2Fync04Sfh4akon%2B%2B0ApuefbXdngzmtidCnjYgXRTHqmze66O1QcT3RjoSfcUipZ3QfZbsaNViSPCYPWcoL5%2BDfdDqXuHwA%2BJvFvgD0IPC1C29nsE4x3XczIQHIo%2FRBdB%2FZnA94Z%2BwKwNPZuo2T8SeF6sSL0uGcQHhLqK1L1vtlTsO1jYwaCvx6vcBgIuuf2XsVksWcfL%2FQZjlB5D9xqsABPa6bpnDcPpV%2FNeDwLabREzRkP%2BoI48M4NGZKHXpQHljVHZWPYaoNwyBMtnSJgKwBRjTaP5LGAR4KrB%2BhUkvxddGxBv7ry8E1941WjJoPYWl1ek%2BTwS8Md2ej0mTwhlbw6%2FheiOEgE7OvjNoGb6N5kOvHeE1tfW%2FtzDQzf0K5lOB6c--ajGHvwcLoPvgEFLp--UX0IdxbdnybK3uhh2%2F1HEA%3D%3D',
    'if-none-match': 'W/"a341c139af6b1bcabedcfc4ecd988ff3"',
    'referer': 'https://statementdog.com/analysis/2330',
    'sec-ch-ua': '"Google Chrome";v="123", "Not:A-Brand";v="8", "Chromium";v="123"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'sentry-trace': 'dd75357973c546818063c9afbb37a559-af4fc905c13c882b-0',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
    'x-csrf-token': '10Iz8bqSgufD6BefUmhwsG92V6ggqC8eM3LQME57aXyHHqreUfwjeGfCJiLDJNRihHIfDu3MnDW199pQoXGmSg'
}

# stockList.txt 文件路徑
stock_list_file = 'stockList.txt'

# 設置輸出目錄
output_dir = 'stockData'
os.makedirs(output_dir, exist_ok=True)

def main():
    # 打開 stockList.txt 文件並讀取內容
    with open(stock_list_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 遍歷每一行
    for line in lines:
        parts = line.strip().split(' ')
        if len(parts) != 3:
            continue

        stock_code = parts[0]
        stock_name = parts[1]
        stock_type = parts[2]

        # 替換 URL 模板中的 {stock_code}
        url = url_template.format(stock_code=stock_code)

        # 發送 GET 請求
        response = requests.get(url, headers=headers)

        # 獲取響應內容
        data = response.json()

        # 設置輸出文件路徑
        output_file = os.path.join(output_dir, f'{stock_code}.json')

        # 將數據保存到文件中
        with open(output_file, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)

        print(f"Saved data for {stock_name} ({stock_code}) to {output_file}")

if __name__ == '__main__':
    main()
