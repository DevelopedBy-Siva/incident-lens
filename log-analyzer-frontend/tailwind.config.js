/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: [
          '"Inter"',
          '"SF Pro Display"',
          '"SF Pro Text"',
          '"Helvetica Neue"',
          "sans-serif",
        ],
      },
      colors: {
        google: {
          bg: "#f3f3f5",
          surface: "#fffefe",
          subtle: "#f7f7f8",
          hover: "#f1f1f3",
          chip: "#ededf0",
          border: "#dedee3",
          text: "#111114",
          muted: "#747782",
          blue: "#17171b",
          "blue-dark": "#000000",
          red: "#cf3f4b",
          yellow: "#a66b08",
          green: "#217a50",
        },
      },
      boxShadow: {
        google: "0 18px 55px rgba(24, 22, 35, 0.08)",
      },
    },
  },
  plugins: [],
};
