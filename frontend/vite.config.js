import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: '0.0.0.0', // Allow external connections (Docker)
    proxy: {
      '/query': {
        // Changed from localhost to 'fastapi' (the docker service name)
        target: 'http://fastapi:8000', 
        changeOrigin: true,
      }
    }
  }
})