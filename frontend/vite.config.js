import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig(({ mode }) => {
  const rootDir = path.resolve(__dirname, '..')   // parent folder
  const env = loadEnv(mode, rootDir, '')          // load all vars

  console.log("LOADED ENV:", env)

  return {
    plugins: [react()],
    define: {
      ...Object.fromEntries(
        Object.entries(env).map(([k, v]) => [
          `import.meta.env.${k}`,
          JSON.stringify(v)
        ])
      )
    }
  }
})