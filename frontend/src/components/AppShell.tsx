import type { ReactNode } from 'react'
import type { ConversationSummary } from '../api/historyTypes'
import { Sidebar } from './Sidebar'
import './AppShell.css'

interface AppShellProps {
  children: ReactNode
  conversations: ConversationSummary[]
  activeConversationId: string | null
  isHistoryLoading: boolean
  onSelectConversation: (id: string) => void
  onNewConversation: () => void
  onDeleteConversation: (id: string) => void
  isSidebarOpen: boolean
  onCloseSidebar: () => void
}

/** Top-level page chrome: a fixed sidebar (drawer on mobile) + main content area. */
export function AppShell({
  children,
  conversations,
  activeConversationId,
  isHistoryLoading,
  onSelectConversation,
  onNewConversation,
  onDeleteConversation,
  isSidebarOpen,
  onCloseSidebar,
}: AppShellProps) {
  return (
    <div className="app-shell">
      <Sidebar
        conversations={conversations}
        activeConversationId={activeConversationId}
        isLoading={isHistoryLoading}
        onSelectConversation={onSelectConversation}
        onNewConversation={onNewConversation}
        onDeleteConversation={onDeleteConversation}
        isOpen={isSidebarOpen}
        onClose={onCloseSidebar}
      />
      <main className="app-shell__main">
        <div className="app-shell__main-inner">{children}</div>
      </main>
    </div>
  )
}
