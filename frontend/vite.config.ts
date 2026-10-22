import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Ariadne's API defaults to :8100 because :8000 is commonly taken by other local apps.
// Override with VITE_API_TARGET=http://localhost:XXXX
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  return {
    plugins: [react(), tailwindcss()],
    server: {
      port: 5173,
      host: true,
      proxy: {
        '/api': {
          target: env.VITE_API_TARGET || 'http://localhost:8100',
          changeOrigin: true,
          secure: false,
        },
      },
    },
  }
})
