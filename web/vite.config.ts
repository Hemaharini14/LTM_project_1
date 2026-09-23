import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Builds into the Flask app's static directory so the existing server can serve
// the showcase directly - no second process in production, and the Flask app
// (model, DB, APIs, all product pages) is untouched.
export default defineConfig({
  plugins: [react()],
  base: '/static/showcase/',
  build: {
    outDir: '../static/showcase',
    emptyOutDir: true,
    rollupOptions: {
      output: {
        // Split the 3D stack out so the shell and loading screen paint before
        // ~1MB of WebGL code has to parse. Rollup 4 wants the function form.
        manualChunks(id: string) {
          if (id.includes('node_modules/three')) return 'three';
          if (id.includes('@react-three')) return 'r3f';
          if (id.includes('framer-motion') || id.includes('gsap')) return 'motion';
        },
      },
    },
  },
  server: {
    // Dev-only: proxy the real app so CTAs and APIs work while developing.
    proxy: {
      '/api': 'http://127.0.0.1:5000',
      '/flight-delay': 'http://127.0.0.1:5000',
      '/budget-trip': 'http://127.0.0.1:5000',
      '/login': 'http://127.0.0.1:5000',
      '/signup': 'http://127.0.0.1:5000',
    },
  },
})
