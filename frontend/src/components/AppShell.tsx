import type { ReactNode } from 'react'
import './AppShell.css'

interface AppShellProps {
  children: ReactNode
}

/** Top-level page chrome: header/branding + a centered content area. */
export function AppShell({ children }: AppShellProps) {
  return (
    <div className="app-shell">
      <header className="app-shell__header">
        <div className="app-shell__header-inner">
          <h1 className="app-shell__title">OmniFlow</h1>
          <p className="app-shell__subtitle">Agentic multimodal AI assistant</p>
        </div>
      </header>
      <main className="app-shell__main">{children}</main>
    </div>
  )
}
