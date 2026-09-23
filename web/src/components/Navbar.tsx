import { useEffect, useState } from 'react';
import { Plane, ArrowUpRight } from 'lucide-react';
import { MagneticButton } from './ui';

const LINKS = [
  { label: 'Flight Intelligence', href: '#prediction' },
  { label: 'Trip Planner', href: '#planner' },
  { label: 'How It Works', href: '#causes' },
];

export function Navbar({ progress }: { progress: number }) {
  const [solid, setSolid] = useState(false);
  useEffect(() => {
    const onScroll = () => setSolid(window.scrollY > 60);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return (
    <>
      <header
        className={`fixed inset-x-0 top-0 z-50 transition-[background-color,backdrop-filter,border-color] duration-500
          ${solid ? 'border-b border-white/[0.06] bg-void/70 backdrop-blur-xl' : 'border-b border-transparent bg-transparent'}`}
      >
        <nav className="mx-auto flex max-w-[1500px] items-center justify-between px-6 py-5 sm:px-10">
          <a href="#hero" className="flex items-center gap-2.5">
            <Plane className="h-4 w-4 rotate-45 text-cyan" strokeWidth={1.5} />
            <span className="font-display text-sm font-semibold tracking-[0.2em] text-white">AEROVA</span>
          </a>

          <div className="hidden items-center gap-9 md:flex">
            {LINKS.map((l) => (
              <a key={l.href} href={l.href}
                 className="group relative font-mono text-[10px] uppercase tracking-[0.18em] text-white/50 transition-colors hover:text-white">
                {l.label}
                <span className="absolute -bottom-1.5 left-0 h-px w-0 bg-cyan transition-[width] duration-300 group-hover:w-full" />
              </a>
            ))}
          </div>

          <MagneticButton href="/login" className="!px-5 !py-2.5 !text-[10px]">
            Get Started <ArrowUpRight className="h-3 w-3" strokeWidth={2} />
          </MagneticButton>
        </nav>
      </header>

      {/* scroll progress hairline */}
      <div className="fixed inset-x-0 top-0 z-[60] h-px bg-transparent">
        <div className="h-full bg-gradient-to-r from-cyan/0 via-cyan to-cyan-glow" style={{ width: `${progress * 100}%` }} />
      </div>
    </>
  );
}
