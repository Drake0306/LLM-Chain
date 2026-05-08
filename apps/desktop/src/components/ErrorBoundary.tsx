import { Component } from "react";
import type { ErrorInfo, ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

/** Top-level error boundary so a thrown render error in any screen falls
 * back to a recoverable panel instead of an unrecoverable white page. The
 * Tauri WebView doesn't reload on a runtime crash; without this, the only
 * way out is to kill and re-launch the app.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Surface to the dev console — production users see the panel below.
    console.error("ErrorBoundary caught:", error, info.componentStack);
  }

  reset = () => {
    this.setState({ error: null });
  };

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="p-6 space-y-4 max-w-2xl">
        <h1 className="text-2xl font-semibold text-red-700">Something broke.</h1>
        <p className="text-sm text-zinc-600 leading-relaxed">
          The UI hit an unexpected error and stopped rendering this screen.
          The sidecar is unaffected; in-flight runs keep going. You can
          dismiss this panel and try again — if it keeps happening, please
          file an issue with the stack trace below.
        </p>
        <pre className="text-xs bg-red-50 border border-red-200 rounded p-3 whitespace-pre-wrap overflow-auto max-h-64">
          {this.state.error.stack ?? this.state.error.message}
        </pre>
        <button
          type="button"
          onClick={this.reset}
          className="rounded-md bg-blue-600 text-white px-4 py-2 text-sm font-medium"
        >
          Dismiss
        </button>
      </div>
    );
  }
}
