import requests
from bs4 import BeautifulSoup
import pymysql

def get_douban_top100():
    movies = []
    headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/114.0.0.0 Safari/537.36"
    ),
    "Referer": "https://movie.douban.com/top250",   
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",   
    "Connection": "keep-alive"
}

    for start in range(0, 100, 25):
        url = f"https://movie.douban.com/top250?start={start}"
        resp = requests.get(url, headers=headers)
        soup = BeautifulSoup(resp.text, "html.parser")
        items = soup.select(".item")
        for item in items:
            title = item.select_one(".title").text
            rating = item.select_one(".rating_num").text
            info = item.select_one(".bd p").text.strip().split("\n")[0]
            movies.append((title, rating, info))
    return movies

movies = get_douban_top100()

# 存入数据库
db = pymysql.connect(host="localhost", user="root", password="123456", database="school", charset="utf8")
cursor = db.cursor()

cursor.execute("DROP TABLE IF EXISTS douban_top100")
cursor.execute("""
    CREATE TABLE douban_top100 (
        id INT AUTO_INCREMENT PRIMARY KEY,
        title VARCHAR(255),
        rating FLOAT,
        info VARCHAR(255)
    )
""")

for movie in movies:
    cursor.execute("INSERT INTO douban_top100 (title, rating, info) VALUES (%s, %s, %s)", movie)

db.commit()
db.close()

print("成功保存豆瓣Top100电影到数据库！")