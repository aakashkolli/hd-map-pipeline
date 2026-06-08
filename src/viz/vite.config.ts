import { defineConfig } from 'vite';

// In Docker the API container is reachable at http://api:8000 (internal network).
// Locally it runs on localhost:8000 (started by `npm run dev` via concurrently).
const apiTarget = process.env.VITE_API_TARGET ?? 'http://localhost:8000';

export default defineConfig({
  server: {
    host: '0.0.0.0',
    port: 5173,
    fs: {
      allow: ['../..'],
    },
    proxy: {
      '/api': {
        target: apiTarget,
        changeOrigin: true,
      },
    },
  },
});
