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
      return <main className="runtime-error"><div><p className="eyebrow">Frontend runtime error</p><h1>The research desk could not render.</h1><p>{this.state.error.message}</p><button onClick={() => window.location.reload()}>Reload preview</button><small>If this appears after login, retry in an InPrivate window with browser translation/extensions disabled.</small></div></main>;
    }
    return this.props.children;
  }
}

createRoot(document.getElementById("root")!).render(
  <AppErrorBoundary>
    <App />
  </AppErrorBoundary>,
);
