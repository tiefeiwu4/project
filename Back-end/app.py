from flask import Flask, jsonify, request
from flask_cors import CORS
from db import get_conn
from auth import auth_bp
import base64
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)
CORS(app, supports_credentials=True) 

app.register_blueprint(auth_bp, url_prefix="/api")

#################################

from recognizer import DigitRecognizer

recognizer = DigitRecognizer()

# 只保留一个ping路由定义
@app.route('/api/ping', methods=['GET'])
def ping():
    """健康检查接口"""
    return jsonify({'status': 'success', 'message': 'pong'})

@app.route('/api/recognize', methods=['POST'])
def recognize_digit():
    """手写数字识别接口"""
    try:
        # 获取上传的图片
        if 'image' not in request.files and 'image_data' not in request.json:
            return jsonify({'error': '没有上传图片'}), 400
        
        image_data = None
        
        # 处理文件上传
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': '没有选择文件'}), 400
            image_data = file.read()
        
        # 处理base64编码的图片
        elif 'image_data' in request.json:
            image_data = base64.b64decode(request.json['image_data'].split(',')[1])
        
        if not image_data:
            return jsonify({'error': '图片数据无效'}), 400
        
        # 进行预测
        result = recognizer.predict(image_data)
        
        logger.info(f"预测结果: {result}")
        
        return jsonify({
            'status': 'success',
            'result': result
        })
        
    except Exception as e:
        logger.error(f"识别错误: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """获取模型信息接口"""
    try:
        info = {
            'model_loaded': recognizer.model is not None,
            'weights_loaded': recognizer.weights is not None,
            'input_size': 784,
            'output_classes': 10
        }
        return jsonify({'status': 'success', 'info': info})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

#################################

@app.get("/api/movies")
def get_movies():
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("""SELECT id, title, rating, info FROM douban_top100 ORDER BY rating DESC""")
            rows = cur.fetchall()
        conn.close()
        return jsonify(rows)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/api/echo")
def echo_param():
    param = request.args.get('param', '')
    return jsonify({'message': f'参数是{param}'})

@app.post("/api/test")
def test_post():
    try:
        data = request.get_json()
        body_param = data.get('bodyParam', '') if data else ''
        query_param = request.args.get('queryParam', '')
        return jsonify({
            'body_param': f'body中的参数是{body_param}',
            'query_param': f'param中的参数是{query_param}'
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/api/charts/movies")
def chart_movies():
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT title, rating FROM douban_top100 ORDER BY rating DESC LIMIT 10")
            rows = cur.fetchall()
        conn.close()
        titles = [row['title'] for row in rows]
        ratings = [float(row['rating']) for row in rows]
        return jsonify({
            'title': '电影评分排行',
            'xAxis': titles,
            'series': ratings
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)