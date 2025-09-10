import io
import logging
import numpy as np
import cv2
import os
import pickle
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

# -----------------------------
# 手写数字识别器类
# -----------------------------
class DigitRecognizer:
    """手写数字识别器"""
    
    def __init__(
        self, 
        model_path='C:\\Users\\tiefeiwu\\Desktop\\back&frontend\\RecogNum\\model_weight\\model_structure.pkl', 
        weights_path='C:\\Users\\tiefeiwu\\Desktop\\back&frontend\\RecogNum\\model_weight\\model_weights.npz'
    ):
        self.model = None
        self.weights = None
        self.model_info = None
        self.load_model(model_path, weights_path)
    
    def load_model(self, model_path, weights_path):
        """加载模型和权重"""
        try:
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model_info = pickle.load(f)
                logging.info(f"模型结构加载成功: {self.model_info}")
            else:
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            if os.path.exists(weights_path):
                self.weights = np.load(weights_path)
                logging.info("模型权重加载成功")
            else:
                raise FileNotFoundError(f"权重文件不存在: {weights_path}")
                
        except Exception as e:
            logging.error(f"加载模型失败: {e}")
            raise
    
    def preprocess_image(self, image_bytes):
        """预处理上传的图片"""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            
            if image.mode != 'L':
                image = image.convert('L')
            image = image.resize((28, 28), Image.Resampling.LANCZOS)
            img_array = np.array(image)
   
            if np.mean(img_array) > 127:
                img_array = 255 - img_array              
            img_array = img_array.astype(np.float32) / 255.0
            img_array = img_array.reshape(1, -1)
            
            return img_array
            
        except Exception as e:
            logging.error(f"图片预处理失败: {e}")
            raise
    
    def predict(self, image_bytes):
        """预测手写数字"""
        try:
            processed_image = self.preprocess_image(image_bytes)
            
            if self.weights is not None:
                return self._model_predict(processed_image)
            else:
                return self._random_predict(processed_image)
            
        except Exception as e:
            logging.error(f"预测失败: {e}")
            raise
    
    def _model_predict(self, image_array):
        """使用加载的模型进行预测"""
        try:
            W1 = self.weights['W1']
            b1 = self.weights['b1']
            W2 = self.weights['W2']
            b2 = self.weights['b2']
            
            z1 = np.dot(image_array, W1) + b1
            a1 = np.maximum(0, z1)
            z2 = np.dot(a1, W2) + b2
            exp_z = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
            probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            
            predicted_class = int(np.argmax(probs, axis=1)[0])
            confidence = float(np.max(probs, axis=1)[0])
            
            all_probs = {str(i): float(probs[0][i]) for i in range(10)}
            
            return {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "probabilities": all_probs
            }
            
        except Exception as e:
            logging.error(f"模型预测失败，使用随机预测: {e}")
            return self._random_predict(image_array)
    
    def _random_predict(self, image_array):
        """随机预测（用于测试）"""
        predicted_class = np.random.randint(0, 10)
        confidence = np.random.uniform(0.7, 0.95)
        probs = np.random.dirichlet(np.ones(10))
        all_probs = {str(i): float(probs[i]) for i in range(10)}
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": all_probs
        }

# -----------------------------
# Flask API
# -----------------------------
app = Flask(__name__)
CORS(app)

recognizer = DigitRecognizer()

@app.route("/api/recognize", methods=["POST"])
def recognize_digit():
    try:
        if "image" not in request.files:
            return jsonify({"status": "error", "error": "没有上传图片"}), 400
        
        file = request.files["image"]
        image_bytes = file.read()
        
        result = recognizer.predict(image_bytes)
        
        return jsonify({
            "status": "success",
            "predicted_class": result["predicted_class"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"]
        })
    
    except Exception as e:
        logging.exception("识别失败")
        return jsonify({"status": "error", "error": str(e)}), 500

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host="0.0.0.0", port=5000, debug=True)