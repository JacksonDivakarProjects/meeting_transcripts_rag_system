import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: '0.0.0.0',   // Required for Docker — Vite only binds localhost by default
    proxy: {
      '/query': {
        target: 'http://fastapi:8000',   // Docker service name
        changeOrigin: true,
      },
      '/health': {
        target: 'http://fastapi:8000',
        changeOrigin: true,
      },
    },
  },
})
