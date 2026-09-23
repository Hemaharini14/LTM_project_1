import { useRef, type ReactNode, type MouseEvent } from 'react';
import { motion, useInView } from 'framer-motion';

/** Magnetic button: follows the cursor slightly, then springs back. */
export function MagneticButton({ children, href, variant = 'primary', className = '' }: {
  children: ReactNode; href?: string; variant?: 'primary' | 'ghost'; className?: string;
}) {
  const ref = useRef<HTMLAnchorElement>(null);
  const onMove = (e: MouseEvent) => {
    const el = ref.current; if (!el) return;
    const r = el.getBoundingClientRect();
    // capped at ~6px so it reads as responsiveness, not a toy
    el.style.transform = `translate(${((e.clientX - r.left) / r.width - 0.5) * 12}px, ${((e.clientY - r.top) / r.height - 0.5) * 8}px)`;
  };
  const reset = () => { if (ref.current) ref.current.style.transform = 'translate(0,0)'; };

  const base = 'group relative inline-flex items-center justify-center gap-2 px-7 py-3.5 rounded-full font-mono text-[11px] uppercase tracking-[0.2em] transition-[background,border-color,box-shadow] duration-300 will-change-transform';
  const styles = variant === 'primary'
    ? 'bg-cyan/10 border border-cyan/40 text-cyan-glow hover:bg-cyan/20 hover:border-cyan/70 hover:shadow-[0_0_30px_-4px_rgba(34,211,238,0.5)]'
    : 'border border-white/15 text-white/70 hover:text-white hover:border-white/35 hover:bg-white/5';

  return (
    <a ref={ref} href={href} onMouseMove={onMove} onMouseLeave={reset}
       className={`${base} ${styles} ${className}`} style={{ transition: 'transform .35s cubic-bezier(.22,1,.36,1)' }}>
      {children}
    </a>
  );
}

/** Fades + lifts children once, when they first enter view. */
export function Reveal({ children, delay = 0, y = 26, className = '' }: {
  children: ReactNode; delay?: number; y?: number; className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: '-12% 0px -12% 0px' });
  return (
    <motion.div
      ref={ref} className={className}
      initial={{ opacity: 0, y }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.9, delay, ease: [0.22, 1, 0.36, 1] }}
    >
      {children}
    </motion.div>
  );
}

/** Counts up to a value when scrolled into view. */
export function AnimatedNumber({ value, suffix = '', decimals = 0 }: { value: number; suffix?: string; decimals?: number }) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true });
  return (
    <span ref={ref}>
      <motion.span
        initial={{ opacity: 0 }} animate={inView ? { opacity: 1 } : {}}
      >
        {inView ? <Counter to={value} decimals={decimals} /> : (0).toFixed(decimals)}
      </motion.span>{suffix}
    </span>
  );
}

function Counter({ to, decimals }: { to: number; decimals: number }) {
  const ref = useRef<HTMLSpanElement>(null);
  useRef(() => {});
  // simple rAF count-up; avoids pulling a spring lib in for one number
  if (typeof window !== 'undefined') {
    requestAnimationFrame(() => {
      const el = ref.current; if (!el) return;
      const start = performance.now(), dur = 1400;
      const tick = (now: number) => {
        const t = Math.min(1, (now - start) / dur);
        const eased = 1 - Math.pow(1 - t, 3);
        el.textContent = (to * eased).toFixed(decimals);
        if (t < 1) requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    });
  }
  return <span ref={ref}>{(0).toFixed(decimals)}</span>;
}

export function SectionLabel({ index, children }: { index: string; children: ReactNode }) {
  return (
    <div className="flex items-center gap-4">
      <span className="font-mono text-[10px] text-cyan/50">{index}</span>
      <span className="h-px w-10 bg-cyan/30" />
      <span className="label">{children}</span>
    </div>
  );
}
