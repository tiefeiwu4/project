import pymysql
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

def get_conn():
    """获取数据库连接"""
    try:
        conn = pymysql.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', 3306)),
            user=os.getenv('DB_USER', 'root'),
            password=os.getenv('DB_PASSWORD', '123456'),
            database=os.getenv('DB_NAME', 'school'),
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        print("数据库连接成功")
        return conn
    except Exception as e:
        print(f"数据库连接错误: {e}")
        raise e

# 初始化数据库表
def init_db():
    """初始化数据库表"""
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            # 创建用户表
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    password VARCHAR(100) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            print("用户表创建成功或已存在")
            
            # 创建电影表（如果不存在）
            cur.execute("""
                CREATE TABLE IF NOT EXISTS douban_top100 (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    title VARCHAR(255) NOT NULL,
                    rating DECIMAL(3,1) NOT NULL,
                    info TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            print("电影表创建成功或已存在")
            
            conn.commit()
        conn.close()
        print("数据库初始化成功")
    except Exception as e:
        print(f"数据库初始化错误: {e}")

# 测试数据库连接和查询
def test_db():
    """测试数据库功能"""
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            # 查看所有表
            cur.execute("SHOW TABLES")
            tables = cur.fetchall()
            print("数据库中的表:", tables)
            
            # 查看用户表结构
            cur.execute("DESCRIBE users")
            user_table_structure = cur.fetchall()
            print("用户表结构:", user_table_structure)
            
            # 查看现有用户
            cur.execute("SELECT * FROM users")
            existing_users = cur.fetchall()
            print("现有用户:", existing_users)
            
        conn.close()
    except Exception as e:
        print(f"数据库测试错误: {e}")

if __name__ == "__main__":
    init_db()
    test_db()