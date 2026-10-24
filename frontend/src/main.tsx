import { Component, ReactNode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles.css";

type ErrorBoundaryState = { error: Error | null };

class AppErrorBoundary extends Component<{ children: ReactNode }, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  render() {
    if (this.state.error) {
      return <main className="runtime-error"><div><p className="eyebrow">前端运行错误</p><h1>研究工作台无法正常显示。</h1><p>{this.state.error.message}</p><button onClick={() => window.location.reload()}>重新加载页面</button><small>如果登录后出现此问题，请尝试使用隐私窗口，并关闭浏览器自动翻译和扩展后重试。</small></div></main>;
    }
    return this.props.children;
  }
}

createRoot(document.getElementById("root")!).render(
  <AppErrorBoundary>
    <App />
  </AppErrorBoundary>,
);
