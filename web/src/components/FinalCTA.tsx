import { Plane } from 'lucide-react';
import { Reveal, MagneticButton } from './ui';

export function FinalCTA() {
  return (
    <section id="cta" className="relative flex min-h-screen flex-col justify-between py-28">
      <div className="mx-auto flex w-full max-w-[1500px] flex-1 items-center px-6 sm:px-10">
        <div className="w-full text-center">
          <Reveal>
            <h2 className="display text-[clamp(2.8rem,10vw,9rem)] text-white">
              TRAVEL WITH<br /><span className="text-cyan-glow text-glow">INTELLIGENCE.</span>
            </h2>
          </Reveal>

          <Reveal delay={0.12}>
            <div className="mx-auto mt-10 flex max-w-md flex-col gap-1.5 font-mono text-[11px] uppercase tracking-[0.2em] text-white/40">
              <span>Predict delays.</span>
              <span>Understand your flight.</span>
              <span>Plan your journey.</span>
            </div>
          </Reveal>

          <Reveal delay={0.22}>
            <div className="mt-12 flex flex-wrap justify-center gap-3">
              <MagneticButton href="/flight-delay">Check A Flight</MagneticButton>
              <MagneticButton href="/budget-trip" variant="ghost">Plan A Trip</MagneticButton>
            </div>
          </Reveal>
        </div>
      </div>

      <footer className="mx-auto w-full max-w-[1500px] px-6 sm:px-10">
        <div className="rule" />
        <div className="flex flex-col items-start justify-between gap-6 py-9 sm:flex-row sm:items-center">
          <div className="flex items-center gap-2.5">
            <Plane className="h-4 w-4 rotate-45 text-cyan" strokeWidth={1.5} />
            <div>
              <p className="font-display text-sm font-semibold tracking-[0.2em] text-white">SmartRoute<span className="text-cyan">AI</span></p>
              <p className="font-mono text-[9px] uppercase tracking-[0.2em] text-white/30">
                AI-Powered Flight Intelligence
              </p>
            </div>
          </div>

          <div className="flex flex-wrap gap-x-8 gap-y-2">
            {[
              ['Flight Intelligence', '/flight-delay'],
              ['Trip Planner', '/budget-trip'],
              ['Sign In', '/login'],
            ].map(([label, href]) => (
              <a key={href} href={href}
                 className="font-mono text-[10px] uppercase tracking-[0.18em] text-white/40 transition-colors hover:text-cyan">
                {label}
              </a>
            ))}
          </div>
        </div>
      </footer>
    </section>
  );
}
