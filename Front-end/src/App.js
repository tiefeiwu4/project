import React, { useState, useEffect } from "react";
import { Layout, Menu, Card, Input, Button, message, Table, Tag, Spin } from "antd";
import { UserOutlined, BarChartOutlined, DatabaseOutlined, VideoCameraOutlined, LoginOutlined, LogoutOutlined, ScanOutlined } from "@ant-design/icons";
import ReactECharts from 'echarts-for-react';
import { BrowserRouter as Router, Routes, Route, useNavigate, Navigate } from "react-router-dom";
import Login from "./components/Login";
import Register from "./components/Register";
import Recognizer from "./components/recognizer"; // 导入识别组件

const { Header, Sider, Content } = Layout;
const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:5000";

function MovieApp({ userId, onLogout }) {
  const [currentView, setCurrentView] = useState("data");
  const [chartData, setChartData] = useState(null);
  const [movies, setMovies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [getParam, setGetParam] = useState("");
  const [postBody, setPostBody] = useState("");
  const [postQuery, setPostQuery] = useState("");

  useEffect(() => {
    if (currentView === "data") {
      loadMovies();
    } else if (currentView === "chart") {
      loadChart();
    }
  }, [currentView]);

  const loadMovies = () => {
    setLoading(true);
    fetch("/api/movies")
      .then(r => r.json())
      .then(movieData => {
        setMovies(movieData);
      })
      .catch((e) => {
        console.error("Fetch error:", e);
        message.error('加载电影数据失败');
      })
      .finally(() => setLoading(false));
  };

  const loadChart = async () => {
    setLoading(true);
    try {
      const response = await fetch(`/api/charts/movies`);
      const data = await response.json();
      setChartData(data);
    } catch (error) {
      message.error('加载图表数据失败');
    } finally {
      setLoading(false);
    }
  };

  const handleGetRequest = async () => {
    try {
      const response = await fetch(`/api/echo?param=${getParam}`);
      const data = await response.json();
      message.success(data.message);
    } catch (error) {
      message.error('GET请求失败');
    }
  };

  const handlePostRequest = async () => {
    try {
      const response = await fetch(`/api/test?queryParam=${postQuery}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ bodyParam: postBody }),
      });
      const data = await response.json();
      message.success(`${data.body_param}, ${data.query_param}`);
    } catch (error) {
      message.error('POST请求失败');
    }
  };

  const getChartOption = () => {
    if (!chartData) return {};
    
    return {
      title: { text: chartData.title, left: 'center' },
      xAxis: { type: 'category', data: chartData.xAxis },
      yAxis: { type: 'value', name: '评分' },
      series: [{ data: chartData.series, type: 'bar' }],
      tooltip: { trigger: 'axis' }
    };
  };

  const movieColumns = [
    { title: '电影名称', dataIndex: 'title', key: 'title', ellipsis: true },
    { title: '评分', dataIndex: 'rating', key: 'rating', render: (rating) => (
        <Tag color={rating >= 9 ? 'red' : rating >= 8 ? 'orange' : 'green'}>{rating}</Tag>
      ), sorter: (a, b) => b.rating - a.rating },
    { title: '详细信息', dataIndex: 'info', key: 'info', ellipsis: true }
  ];

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ color: 'white', fontSize: '20px', fontWeight: 'bold', display: 'flex', alignItems: 'center' }}>
        <VideoCameraOutlined style={{ marginRight: 12 }} />
        电影数据可视化平台
        <div style={{ marginLeft: "auto", display: 'flex', alignItems: 'center' }}>
          <span style={{ marginRight: 16 }}>用户ID: {userId}</span>
          <Button icon={<LogoutOutlined />} onClick={onLogout}>
            退出登录
          </Button>
        </div>
      </Header>
      <Layout>
        <Sider width={200} style={{ background: '#fff' }}>
          <Menu
            mode="inline"
            selectedKeys={[currentView]}
            style={{ height: '100%' }}
            items={[
              { key: 'data', icon: <DatabaseOutlined />, label: '电影数据列表', onClick: () => setCurrentView('data') },
              { key: 'chart', icon: <BarChartOutlined />, label: '评分排行图表', onClick: () => { setCurrentView('chart'); } },
              { key: 'test', icon: <UserOutlined />, label: '接口测试', onClick: () => setCurrentView('test') },
              { key: 'recognize', icon: <ScanOutlined />, label: '手写数字识别', onClick: () => setCurrentView('recognize') }
            ]}
          />
        </Sider>
        <Content style={{ padding: '24px', background: '#f0f2f5' }}>
          {loading ? (
            <div style={{ textAlign: 'center', padding: '50px' }}>
              <Spin size="large" />
            </div>
          ) : (
            <>
              {currentView === 'data' && (
                <Card title="豆瓣电影 Top100">
                  <Table columns={movieColumns} dataSource={movies.map(m => ({ ...m, key: m.id }))} />
                </Card>
              )}
              {currentView === 'chart' && chartData && (
                <Card title={chartData.title}>
                  <ReactECharts option={getChartOption()} style={{ height: '500px' }} />
                </Card>
              )}
              {currentView === 'test' && (
                <div>
                  <Card title="GET请求测试" style={{ marginBottom: 16 }}>
                    <Input value={getParam} onChange={e => setGetParam(e.target.value)} placeholder="输入参数" style={{ marginBottom: 8 }} />
                    <Button onClick={handleGetRequest}>发送GET</Button>
                  </Card>
                  <Card title="POST请求测试">
                    <Input value={postBody} onChange={e => setPostBody(e.target.value)} placeholder="Body参数" style={{ marginBottom: 8 }} />
                    <Input value={postQuery} onChange={e => setPostQuery(e.target.value)} placeholder="Query参数" style={{ marginBottom: 8 }} />
                    <Button onClick={handlePostRequest}>发送POST</Button>
                  </Card>
                </div>
              )}
              {currentView === 'recognize' && (
                <Recognizer />
              )}
            </>
          )}
        </Content>
      </Layout>
    </Layout>
  );
}

// LoginPage, RegisterPage, App 组件保持不变...

function LoginPage({ onLogin, isLoggedIn }) {
  const navigate = useNavigate();

  if (isLoggedIn) {
    return <Navigate to="/" replace />;
  }

  return (
    <Login 
      onLogin={onLogin} 
      onSwitch={() => navigate('/register')} 
    />
  );
}

function RegisterPage({ onRegister, isLoggedIn }) {
  const navigate = useNavigate();

  if (isLoggedIn) {
    return <Navigate to="/" replace />;
  }

  return (
    <Register 
      onRegister={onRegister} 
      onSwitch={() => navigate('/login')} 
    />
  );
}

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [userId, setUserId] = useState(null);
  const [loading, setLoading] = useState(true);

  // 检查本地存储中的登录状态
  useEffect(() => {
    const savedLoginState = localStorage.getItem('isLoggedIn');
    const savedUserId = localStorage.getItem('userId');
    
    if (savedLoginState === 'true' && savedUserId) {
      setIsLoggedIn(true);
      setUserId(savedUserId);
    }
    setLoading(false);
  }, []);

  const handleLogin = (userId) => {
    setIsLoggedIn(true);
    setUserId(userId);
    localStorage.setItem('isLoggedIn', 'true');
    localStorage.setItem('userId', userId);
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    setUserId(null);
    localStorage.removeItem('isLoggedIn');
    localStorage.removeItem('userId');
  };

  const handleRegister = () => {
    message.success("注册成功，请登录");
  };

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <Spin size="large" />
      </div>
    );
  }

  return (
    <Router>
      <Routes>
        <Route 
          path="/" 
          element={
            isLoggedIn ? 
            <MovieApp userId={userId} onLogout={handleLogout} /> : 
            <Navigate to="/login" replace />
          } 
        />
        <Route 
          path="/login" 
          element={
            <LoginPage onLogin={handleLogin} isLoggedIn={isLoggedIn} />
          } 
        />
        <Route 
          path="/register" 
          element={
            <RegisterPage onRegister={handleRegister} isLoggedIn={isLoggedIn} />
          } 
        />
      </Routes>
    </Router>
  );
}

export default App;