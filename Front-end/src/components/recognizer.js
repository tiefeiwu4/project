import React, { useState, useRef } from 'react';
import { Card, Button, Upload, message, Spin, Progress, Row, Col } from 'antd';
import { UploadOutlined, ReloadOutlined } from '@ant-design/icons';

const DigitRecognizer = () => {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const fileInputRef = useRef(null);

  const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:5000";

  // 处理文件上传
  const handleUpload = async (rawFile) => {
    setLoading(true);
    setResult(null);

    try {
      if (!rawFile || !(rawFile instanceof File)) {
        message.error('无效的文件');
        return false;
      }

      const validTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/bmp'];
      if (!validTypes.includes(rawFile.type)) {
        message.error('请上传图片文件 (JPEG, PNG, GIF, BMP)');
        return false;
      }

      const maxSize = 5 * 1024 * 1024;
      if (rawFile.size > maxSize) {
        message.error('文件大小不能超过 5MB');
        return false;
      }

      const formData = new FormData();
      formData.append('image', rawFile);

      const url = `${API_BASE_URL}/api/recognize`;
      console.log('发送请求到:', url);

      const response = await fetch(url, { method: 'POST', body: formData });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`服务器错误: ${response.status} - ${errorText}`);
      }

      const data = await response.json();
      console.log("后端返回 JSON:", JSON.stringify(data, null, 2));

      if (data && data.status === 'success' && data.result) {
        const r = data.result;
        setResult({
          predicted_class: r.predicted_class,
          confidence: r.confidence,
          probabilities: r.probabilities
        });
        message.success('识别成功！');
      } else {
        message.error(data.error || '识别失败');
        console.error('未知的响应格式:', data);
      }
    } catch (error) {
      console.error('识别错误详情:', error);
      if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
        message.error('无法连接到服务器，请检查后端服务是否启动');
      } else {
        message.error(error.message || '识别失败，请稍后重试');
      }
    } finally {
      setLoading(false);
    }

    return false;
  };

  // 监听文件选择
  const handleFileChange = ({ file }) => {
    if (file.status === 'removed') {
      setImagePreview(null);
      setResult(null);
      return;
    }

    const rawFile = file.originFileObj || file;

    if (!(rawFile instanceof File)) {
      message.error('文件格式不正确');
      return;
    }

    // 预览
    const reader = new FileReader();
    reader.onload = (e) => setImagePreview(e.target.result);
    reader.readAsDataURL(rawFile);

    // 上传并识别
    handleUpload(rawFile);
  };

  const clearResults = () => {
    setResult(null);
    setImagePreview(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <Card 
      title="手写数字识别" 
      style={{ width: 600, margin: '50px auto' }}
      extra={
        <Button icon={<ReloadOutlined />} onClick={clearResults} disabled={loading}>
          清空
        </Button>
      }
    >
      <div style={{ textAlign: 'center', marginBottom: 20 }}>
        <Upload
          ref={fileInputRef}
          accept=".png,.jpg,.jpeg,.bmp,.gif"
          beforeUpload={() => false}
          onChange={handleFileChange}
          showUploadList={false}
        >
          <Button icon={<UploadOutlined />} loading={loading}>
            选择手写数字图片
          </Button>
        </Upload>
        <div style={{ fontSize: 12, color: '#666', marginTop: 8 }}>
          支持格式: PNG, JPG, JPEG, BMP, GIF
        </div>
      </div>

      {loading && (
        <div style={{ textAlign: 'center', margin: '20px 0' }}>
          <Spin size="large" />
          <div style={{ marginTop: 10 }}>识别中...</div>
        </div>
      )}

      <Row gutter={16}>
        <Col span={12}>
          {imagePreview && (
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontWeight: 'bold', marginBottom: 10 }}>上传的图片</div>
              <img 
                src={imagePreview} 
                alt="手写数字" 
                style={{ 
                  maxWidth: '100%', 
                  maxHeight: 200, 
                  border: '1px solid #ddd',
                  borderRadius: 4 
                }} 
              />
            </div>
          )}
        </Col>
        
        <Col span={12}>
          {result && (
            <div>
              <div style={{ fontWeight: 'bold', marginBottom: 10 }}>识别结果</div>
              <div style={{ fontSize: 24, fontWeight: 'bold', color: '#1890ff' }}>
                数字: {result?.predicted_class ?? "未识别"}
              </div>
              {result.confidence !== undefined && (
                <div style={{ marginTop: 5, color: '#666' }}>
                  置信度: {(result.confidence * 100).toFixed(2)}%
                </div>
              )}
              {result.probabilities && Object.keys(result.probabilities).length > 0 && (
                <div style={{ marginTop: 15 }}>
                  <div style={{ fontWeight: 'bold', marginBottom: 5 }}>概率分布:</div>
                  {Object.entries(result.probabilities).map(([digit, prob]) => (
                    <div key={digit} style={{ marginBottom: 3 }}>
                      <span>数字 {digit}: </span>
                      <Progress 
                        percent={parseFloat((prob * 100).toFixed(1))} 
                        size="small" 
                        style={{ marginLeft: 10, width: 100 }}
                        showInfo={false}
                      />
                      <span style={{ marginLeft: 10 }}>{(prob * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </Col>
      </Row>
    </Card>
  );
};

export default DigitRecognizer;