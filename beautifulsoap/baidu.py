import requests
from bs4 import BeautifulSoup
import pymysql

# 爬取百度热搜
url = "https://top.baidu.com/board?tab=realtime"
headers = {"User-Agent": "Mozilla/5.0"}
resp = requests.get(url, headers=headers)
soup = BeautifulSoup(resp.text, "html.parser")

# 提取前10个热搜标题
items = soup.select(".c-single-text-ellipsis")[:10]
hot_list = [item.text.strip() for item in items]

print("百度热搜Top10：")
for i, title in enumerate(hot_list, 1):
    print(f"{i}. {title}")

# 存入数据库
db = pymysql.connect(host="localhost", user="root", password="123456", database="school", charset="utf8")
cursor = db.cursor()

cursor.execute("DROP TABLE IF EXISTS hotsearch")
cursor.execute("""
    CREATE TABLE hotsearch (
        id INT AUTO_INCREMENT PRIMARY KEY,
        title VARCHAR(255)
    )
""")

for title in hot_list:
    cursor.execute("INSERT INTO hotsearch (title) VALUES (%s)", (title,))

db.commit()
db.close()
