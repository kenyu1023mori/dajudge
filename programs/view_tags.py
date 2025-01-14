import sqlite3

# データベースに接続
db_path = "/home/group4/data/tags.db"
conn = sqlite3.connect(db_path)
c = conn.cursor()

# テーブルの内容を表示
c.execute("SELECT * FROM tags")
rows = c.fetchall()

for row in rows:
    print(row)

conn.close()
