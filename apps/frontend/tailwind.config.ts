import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Segoe UI Variable", "Avenir Next", "Trebuchet MS", "sans-serif"]
      },
      boxShadow: {
        panel: "0 8px 30px rgba(15, 23, 42, 0.35)"
      }
    }
  },
  plugins: []
};

export default config;