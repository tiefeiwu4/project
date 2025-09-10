from flask import Blueprint, request, jsonify
from db import get_conn

auth_bp = Blueprint("auth", __name__)

@auth_bp.post("/register")
def register():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "请求数据为空"}), 400
            
        username = data.get("username")
        password = data.get("password")
        
        print(f"注册请求: username={username}")  # 安全起见，不打印密码
        
        if not username or not password:
            return jsonify({"error": "用户名和密码不能为空"}), 400
        
        conn = get_conn()
        
        with conn.cursor() as cur:
            # 检查用户是否存在
            cur.execute("SELECT id FROM users WHERE username=%s", (username,))
            existing_user = cur.fetchone()
            
            if existing_user:
                return jsonify({"error": "用户已存在"}), 400
            
            # 插入新用户
            cur.execute("INSERT INTO users (username, password) VALUES (%s, %s)", 
                       (username, password))
            conn.commit()
        
        conn.close()
        return jsonify({"message": "注册成功"})
        
    except Exception as e:
        print(f"注册错误: {e}")
        return jsonify({"error": "注册失败，请稍后重试"}), 500

@auth_bp.post("/login")
def login():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "请求数据为空"}), 400
            
        username = data.get("username")
        password = data.get("password")
        
        print(f"登录请求: username={username}")
        
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT id, password FROM users WHERE username=%s", (username,))
            user = cur.fetchone()
        
        if user and user["password"] == password:
            return jsonify({
                "message": "登录成功", 
                "user_id": user["id"],
                "username": username
            })
        else:
            return jsonify({"error": "用户名或密码错误"}), 401
            
    except Exception as e:
        print(f"登录错误: {e}")
        return jsonify({"error": "登录失败，请稍后重试"}), 500