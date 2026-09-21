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
} from "lucide-react";

function Navbar() {
  const navigate = useNavigate();
  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  return (
    <nav className="box-bg border-b border-gray-800 backdrop-blur-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-20">
          <div className="flex items-center">
            <Activity className="text-sky-500 mr-2" size={28} />
          </div>

          <div className="flex items-center gap-1 sm:gap-2">
            <NavLink
              to="/dashboard"
              className={({ isActive }) =>
                `px-2 sm:px-3 py-2 rounded-md text-xs sm:text-sm font-medium transition-colors flex items-center ${
                  isActive
                    ? "text-white bg-gray-900"
                    : "text-gray-500 hover:text-white"
                }`
              }
            >
              <Home size={16} className="sm:mr-1" />
              <span className="hidden sm:inline">Dashboard</span>
            </NavLink>
            <NavLink
              to="/models"
              className={({ isActive }) =>
                `px-2 sm:px-3 py-2 rounded-md text-xs sm:text-sm font-medium transition-colors flex items-center ${
                  isActive
                    ? "text-white bg-gray-900"
                    : "text-gray-500 hover:text-white"
                }`
              }
            >
              <BrainCircuit size={16} className="sm:mr-1" />
              <span className="hidden sm:inline">Models</span>
            </NavLink>
            <NavLink
              to="/training"
              className={({ isActive }) =>
                `px-2 sm:px-3 py-2 rounded-md text-xs sm:text-sm font-medium transition-colors flex items-center ${
                  isActive
                    ? "text-white bg-gray-900"
                    : "text-gray-500 hover:text-white"
                }`
              }
            >
              <Workflow size={16} className="sm:mr-1" />
              <span className="hidden sm:inline">Training</span>
            </NavLink>
            <NavLink
              to="/settings"
              className={({ isActive }) =>
                `px-2 sm:px-3 py-2 rounded-md text-xs sm:text-sm font-medium transition-colors flex items-center ${
                  isActive
                    ? "text-white bg-gray-900"
                    : "text-gray-500 hover:text-white"
                }`
              }
            >
              <Settings size={16} className="sm:mr-1" />
              <span className="hidden sm:inline">Settings</span>
            </NavLink>
            <button
              onClick={handleLogout}
              className="text-gray-500 hover:text-red-600 px-2 sm:px-3 py-2 rounded-md text-xs sm:text-sm font-medium transition-colors flex items-center"
            >
              <LogOut size={16} className="sm:mr-1" />
              <span className="hidden sm:inline">Logout</span>
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;
