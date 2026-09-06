import { Menu } from 'lucide-react'
import './Hero.css'

interface HeroProps {
  onOpenSidebar: () => void
}

export function Hero({ onOpenSidebar }: HeroProps) {
  return (
    <section className="hero" aria-label="About OmniFlow">
      <button type="button" className="hero__menu-btn" onClick={onOpenSidebar} aria-label="Open conversation history">
        <Menu size={20} />
      </button>
      <span className="hero__badge">Agentic · Multimodal · Intelligent</span>
      <h1 className="hero__title">OmniFlow</h1>
      <p className="hero__subtitle">One workspace for understanding anything.</p>
      <p className="hero__description">
        OmniFlow is an agentic multimodal AI assistant that understands documents, images, audio, and text — then
        reasons across them to answer complex questions.
      </p>
    </section>
  )
}
