import React, { useState } from "react";
import { Card, Input, Button, message } from "antd";
const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:5000";
export default function Login({ onLogin, onSwitch }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);

  const handleLogin = async () => {
    if (!username || !password) {
      message.error("用户名和密码不能为空");
      return;
    }

    setLoading(true);
    try {
      const res = await fetch("/api/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password })
      });
      
      const data = await res.json();
      if (res.ok) {
        message.success("登录成功");
        onLogin(data.user_id);
      } else {
        message.error(data.error || "登录失败");
      }
    } catch (error) {
      message.error("登录请求失败");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card title="登录" style={{ width: 300, margin: "100px auto" }}>
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
        onPressEnter={handleLogin}
      />
      <Button 
        type="primary" 
        onClick={handleLogin} 
        loading={loading}
        block 
        style={{ marginBottom: 10 }}
      >
        登录
      </Button>
      <Button type="link" onClick={onSwitch} block>
        没有账号？去注册
      </Button>
    </Card>
  );
} 