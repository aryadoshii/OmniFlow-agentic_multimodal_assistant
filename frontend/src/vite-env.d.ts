/// <reference types="vite/client" />

interface ImportMetaEnv {
  /**
   * Base URL for the OmniFlow backend API, e.g. "https://api.example.com".
   * Unset in development: requests go to same-origin "/api/*", proxied to
   * the local backend by vite.config.ts. Set at build time for production
   * (never hardcoded in source).
   */
  readonly VITE_API_BASE_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
