import { Link, NavLink, Route, Routes } from "react-router-dom";

import { Dashboard } from "./screens/Dashboard";
import { DatasetPicker } from "./screens/DatasetPicker";
import { ModelPicker } from "./screens/ModelPicker";
import { RunDetail, RunsList } from "./screens/Runs";
import { Train } from "./screens/Train";

const NAV = [
  { to: "/", label: "Dashboard", end: true },
  { to: "/models", label: "Model" },
  { to: "/dataset", label: "Dataset" },
  { to: "/train", label: "Train" },
  { to: "/runs", label: "Runs" },
];

export default function App() {
  return (
    <div className="flex h-full bg-zinc-50 text-zinc-900">
      <aside className="w-56 shrink-0 border-r border-zinc-200 bg-white p-4 space-y-6">
        <Link to="/" className="block">
          <div className="font-semibold text-lg">LLM-Chain</div>
          <div className="text-xs text-zinc-500">Train your own LLM, locally.</div>
        </Link>
        <nav className="space-y-1 text-sm">
          {NAV.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.end}
              className={({ isActive }) =>
                `block px-3 py-2 rounded-md ${
                  isActive ? "bg-blue-50 text-blue-800" : "text-zinc-700 hover:bg-zinc-100"
                }`
              }
            >
              {n.label}
            </NavLink>
          ))}
        </nav>
      </aside>
      <main className="flex-1 overflow-auto">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/models" element={<ModelPicker />} />
          <Route path="/dataset" element={<DatasetPicker />} />
          <Route path="/train" element={<Train />} />
          <Route path="/runs" element={<RunsList />} />
          <Route path="/runs/:runId" element={<RunDetail />} />
        </Routes>
      </main>
    </div>
  );
}
