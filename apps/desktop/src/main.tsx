import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";

import App from "./App";
import "./index.css";
import { SelectionProvider } from "./state/selection";

// StrictMode is intentionally NOT wrapped here. React 18 StrictMode
// double-invokes effects in dev (mount → cleanup → mount) to surface bugs.
// That trips up the SSE-driven training stream: the first useEffect opens an
// EventSource, the cleanup closes it, and our streamRun setup never opens a
// second one — so the executor never iterates and the run sits at "pending".
// Production builds strip StrictMode entirely; this only matters for
// `npm run tauri dev`. Re-enable once the run/SSE flow is event-store-backed
// (planned for v1.3) so multiple SSE clients can subscribe to the same run.
ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <BrowserRouter>
    <SelectionProvider>
      <App />
    </SelectionProvider>
  </BrowserRouter>,
);
