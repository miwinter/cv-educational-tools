import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [{
    name: 'development-entry',
    configureServer(server) {
      server.middlewares.use((req, _res, next) => {
        // The repository index redirects to the checked-in static build.
        // During development, serve the editable source entry instead.
        if (req.url === '/' || req.url === '/index.html') req.url = '/app.html';
        next();
      });
    },
  }],
  build: { rollupOptions: { input: 'app.html' } },
});
