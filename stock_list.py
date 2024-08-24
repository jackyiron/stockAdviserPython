import pandas as pd
import requests
import io

# 加入 headers
headers = {
    "user-agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.157 Safari/537.36"
}

# 對網站進行 requests，並加入指定的 headers 一同請求
html_data = requests.get("https://isin.twse.com.tw/isin/C_public.jsp?strMode=2", headers=headers)

# 使用 io.StringIO 包裝 HTML 內容
html_content = io.StringIO(html_data.text)

# 使用 pandas 的 read_html 處理表格式
x = pd.read_html(html_content)
# list 取出 list 裡面的第一個元素，就是我們的 DataFrame
x = x[0]
# pandas 的好用函數 iloc 切片，我們指定 DataFrame 的欄位為第一列
x.columns = x.iloc[0, :]
# 欄位雖然變成了正確的，但本來的那一列仍然存在，我們把它拿掉
x = x.iloc[1:, :]
# 使用 split 方法，以兩個空白切割字串，並取切割完後第一個，儲存至新增的代號欄位
x['代號'] = x['有價證券代號及名稱'].apply(lambda x: x.split()[0])
# 使用 split 方法，以兩個空白切割字串，並取切割完後第一個，儲存至新增的股票名稱欄位
x['股票名稱'] = x['有價證券代號及名稱'].apply(lambda x: x.split()[-1])
# 善用 to_datetime 函數，並指定日期格式，將無法轉成 datetime 的資料化為 Nan
x['上市日'] = pd.to_datetime(x['上市日'], format='%Y/%m/%d', errors='coerce')
# 把上市日的 Nan 去掉即可
x = x.dropna(subset=['上市日'])
# Drop 掉不要的欄位
x = x.drop(['有價證券代號及名稱', '國際證券辨識號碼(ISIN Code)', 'CFICode', '備註'], axis=1)
# 更換剩餘的欄位順序
x = x[['代號', '股票名稱', '上市日', '市場別', '產業別']]
# Drop 掉產業別是空的欄位
x = x.dropna(subset=['產業別'])
# pandas 的 str.isdigit() 函數，確認是不是為數字
x = x[x["代號"].str.isdigit()]

# 只保留代號、股票名稱、產業別三個欄位
x = x[['代號', '股票名稱', '產業別']]

# 儲存成 txt 文件，使用空白作為分隔符
x.to_csv('stockList.txt', sep=' ', index=False, header=False)
