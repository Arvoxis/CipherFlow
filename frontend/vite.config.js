import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // Proxying keeps every fetch in the app a relative /api/... path, so the UI does not
    // care which port the Python service is on and nothing depends on CORS.
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
})
