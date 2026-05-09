import { Link, NavLink, Route, Routes } from "react-router-dom";

import { SystemStats } from "./components/SystemStats";
import { ChatMulti } from "./screens/ChatMulti";
import { Compare } from "./screens/Compare";
import { ComparePrompts } from "./screens/ComparePrompts";
import { Dashboard } from "./screens/Dashboard";
import { DatasetCurated } from "./screens/DatasetCurated";
import { DatasetPicker } from "./screens/DatasetPicker";
import { DatasetSynth } from "./screens/DatasetSynth";
import { DatasetWorkshop } from "./screens/DatasetWorkshop";
import { EvalScreen } from "./screens/Eval";
import { Library } from "./screens/Library";
import { ModelPicker } from "./screens/ModelPicker";
import { Playground } from "./screens/Playground";
import { Recipes } from "./screens/Recipes";
import { RunDetail, RunsList } from "./screens/Runs";
import { Settings } from "./screens/Settings";
import { Train } from "./screens/Train";

const NAV = [
  { to: "/", label: "Dashboard", end: true },
  { to: "/models", label: "Model" },
  { to: "/dataset", label: "Dataset" },
  { to: "/train", label: "Train" },
  { to: "/runs", label: "Runs" },
  { to: "/library", label: "Library" },
  { to: "/settings", label: "Settings" },
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
      <main className="flex-1 flex flex-col overflow-hidden">
        <header className="flex items-center justify-end h-10 px-4 border-b border-zinc-200 bg-white shrink-0">
          <SystemStats />
        </header>
        <div className="flex-1 overflow-auto">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/models" element={<ModelPicker />} />
            <Route path="/dataset" element={<DatasetPicker />} />
            <Route path="/dataset/curated" element={<DatasetCurated />} />
            <Route path="/dataset/workshop" element={<DatasetWorkshop />} />
            <Route path="/dataset/synth" element={<DatasetSynth />} />
            <Route path="/train" element={<Train />} />
            <Route path="/train/recipes" element={<Recipes />} />
            <Route path="/runs" element={<RunsList />} />
            <Route path="/runs/compare" element={<Compare />} />
            <Route path="/compare/prompts" element={<ComparePrompts />} />
            <Route path="/chat/multi" element={<ChatMulti />} />
            <Route path="/runs/:runId/play" element={<Playground />} />
            <Route path="/runs/:runId/eval" element={<EvalScreen />} />
            <Route path="/runs/:runId" element={<RunDetail />} />
            <Route path="/library" element={<Library />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </div>
      </main>
    </div>
  );
}
