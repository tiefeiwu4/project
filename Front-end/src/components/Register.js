import React, { useState } from "react";
import { Card, Input, Button, message } from "antd";
const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:5000";
export default function Register({ onRegister, onSwitch }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);

  const handleRegister = async () => {
    if (!username || !password) {
      message.error("用户名和密码不能为空");
      return;
    }

    setLoading(true);
    try {
      const res = await fetch("/api/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password })
      });
      
      const data = await res.json();
      if (res.ok) {
        message.success("注册成功");
        onRegister(); // 调用父组件的注册成功处理函数
      } else {
        message.error(data.error || "注册失败");
      }
    } catch (error) {
      message.error("注册请求失败");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card title="注册" style={{ width: 300, margin: "100px auto" }}>
      <Input 
        placeholder="用户名" 
        value={username} 
        onChange={e => setUsername(e.target.value)} 
        style={{ marginBottom: 10 }} 
      />
      <Input.Password 
        placeholder="密码" 
        value={password} 
        onChange={e => setPassword(e.target.value)} 
        style={{ marginBottom: 10 }} 
        onPressEnter={handleRegister}
      />
      <Button 
        type="primary" 
        onClick={handleRegister} 
        loading={loading}
        block 
        style={{ marginBottom: 10 }}
      >
        注册
      </Button>
      <Button type="link" onClick={onSwitch} block>
        已有账号？去登录
      </Button>
    </Card>
  );
}