/**
 * Generates a client-side-only identifier (React keys, session_id
 * correlation label). Never sent anywhere that requires cryptographic
 * randomness -- this is just a stable, locally-unique tag.
 *
 * `crypto.randomUUID()` is only defined in "secure contexts" (HTTPS, or
 * http://localhost) per spec -- calling it on a plain-HTTP, non-localhost
 * deployment throws `TypeError: crypto.randomUUID is not a function`.
 * App.tsx calls this during initial render (`useState(() => generateId())`),
 * so an unguarded call there would crash the entire app before it ever
 * renders anything, on exactly the kind of deployment this project's own
 * frontend README doesn't rule out (VITE_API_BASE_URL pointing at a
 * separately-hosted backend). Falls back to a Math.random()-based id,
 * identical in spirit to lib/files.ts's existing staged-file id scheme.
 */
export function generateId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  return `id-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`
}
