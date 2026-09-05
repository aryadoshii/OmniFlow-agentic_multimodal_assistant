import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
//
// Dev-only API proxy: the frontend always calls `${VITE_API_BASE_URL ?? '/api'}/...`
// (see src/api/client.ts). In development, VITE_API_BASE_URL is unset, so
// requests go to same-origin `/api/*`; this proxy forwards them to the
// FastAPI backend (default http://127.0.0.1:8000, matching omniflow.config
// .Settings' HOST/PORT defaults) and strips the `/api` prefix, since the
// backend's own routes (/query, /ingest, /health) are mounted at root. No
// production URL is hardcoded anywhere -- a real deployment sets
// VITE_API_BASE_URL to the backend's origin at build time and this proxy
// is simply unused.
const BACKEND_ORIGIN = process.env.OMNIFLOW_BACKEND_ORIGIN ?? 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: BACKEND_ORIGIN,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
