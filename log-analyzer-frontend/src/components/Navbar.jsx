import React from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { logout } from "../services/api";
import {
  Activity,
  BrainCircuit,
  Home,
  LogOut,
  Settings,
  Workflow,
  ArrowUpRight,
} from "lucide-react";

function Navbar() {
  const navigate = useNavigate();
  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  return (
    <nav className="sticky top-0 z-50 border-b border-black/5 bg-white/80 backdrop-blur-xl">
      <div className="max-w-[96rem] mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-[auto_1fr_auto] items-center h-20 gap-4">
          <div className="flex items-center gap-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-full bg-google-text text-white">
              <Activity size={18} strokeWidth={2.4} />
            </div>
          </div>

          <div className="justify-self-center flex items-center gap-1 sm:gap-5">
            <NavLink
              to="/dashboard"
              className={({ isActive }) =>
                `relative px-2 sm:px-1 py-2 text-xs sm:text-sm font-medium transition-colors flex items-center ${
                  isActive
                    ? "text-google-text after:absolute after:inset-x-1 after:-bottom-[19px] after:h-0.5 after:bg-google-text"
                    : "text-google-muted hover:text-google-text"
                }`
              }
            >
              <Home size={15} className="sm:mr-1.5" />
              <span className="hidden sm:inline">Dashboard</span>
            </NavLink>
            <NavLink
              to="/models"
              className={({ isActive }) =>
                `relative px-2 sm:px-1 py-2 text-xs sm:text-sm font-medium transition-colors flex items-center ${
                  isActive
                    ? "text-google-text after:absolute after:inset-x-1 after:-bottom-[19px] after:h-0.5 after:bg-google-text"
                    : "text-google-muted hover:text-google-text"
                }`
              }
            >
              <BrainCircuit size={15} className="sm:mr-1.5" />
              <span className="hidden sm:inline">Models</span>
            </NavLink>
            <NavLink
              to="/training"
              className={({ isActive }) =>
                `relative px-2 sm:px-1 py-2 text-xs sm:text-sm font-medium transition-colors flex items-center ${
                  isActive
                    ? "text-google-text after:absolute after:inset-x-1 after:-bottom-[19px] after:h-0.5 after:bg-google-text"
                    : "text-google-muted hover:text-google-text"
                }`
              }
            >
              <Workflow size={15} className="sm:mr-1.5" />
              <span className="hidden sm:inline">Training</span>
            </NavLink>
            <NavLink
              to="/settings"
              className={({ isActive }) =>
                `relative px-2 sm:px-1 py-2 text-xs sm:text-sm font-medium transition-colors flex items-center ${
                  isActive
                    ? "text-google-text after:absolute after:inset-x-1 after:-bottom-[19px] after:h-0.5 after:bg-google-text"
                    : "text-google-muted hover:text-google-text"
                }`
              }
            >
              <Settings size={15} className="sm:mr-1.5" />
              <span className="hidden sm:inline">Settings</span>
            </NavLink>
          </div>
          <div className="justify-self-end">
            <button
              onClick={handleLogout}
              className="group px-2 sm:px-3 py-2 text-google-text text-xs sm:text-sm font-semibold transition-colors flex items-center hover:opacity-60"
            >
              <LogOut size={15} className="sm:hidden" />
              <span className="hidden sm:inline">Sign out</span>
              <ArrowUpRight size={14} className="hidden sm:block ml-1 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;
