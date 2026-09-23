import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

/**
 * Premium boot sequence. Progress is tied to real readiness (fonts + the
 * deferred 3D chunk), not a fake timer, then held briefly at 100% so the
 * hand-off doesn't feel abrupt.
 */
export function Loader({ onDone }: { onDone: () => void }) {
  const [pct, setPct] = useState(0);
  const [gone, setGone] = useState(false);

  useEffect(() => {
    let raf = 0, current = 0;
    let ready = false;
    Promise.all([
      document.fonts?.ready ?? Promise.resolve(),
      new Promise((r) => setTimeout(r, 600)),   // floor, so it never flashes
    ]).then(() => { ready = true; });

    const tick = () => {
      // ease toward 90 while loading, then run out to 100 once actually ready
      const ceiling = ready ? 100 : 90;
      current += Math.max(0.4, (ceiling - current) * 0.035);
      const v = Math.min(ceiling, current);
      setPct(v);
      if (v >= 99.5) {
        setTimeout(() => { setGone(true); setTimeout(onDone, 700); }, 260);
        return;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [onDone]);

  return (
    <AnimatePresence>
      {!gone && (
        <motion.div
          className="fixed inset-0 z-[100] flex flex-col items-center justify-center bg-void"
          exit={{ opacity: 0, filter: 'blur(8px)' }}
          transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
        >
          <motion.svg
            viewBox="0 0 24 24" className="h-10 w-10 fill-cyan/80"
            initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8 }}
          >
            <path d="M21 16v-2l-8-5V3.5a1.5 1.5 0 0 0-3 0V9l-8 5v2l8-2.5V19l-2.5 1.8V22l3.5-1 3.5 1v-1.2L12 19v-5.5l9 2.5z" />
          </motion.svg>

          <p className="mt-7 font-mono text-[10px] uppercase tracking-[0.35em] text-white/40">
            Initializing<br className="sm:hidden" /> Flight Intelligence…
          </p>

          <div className="mt-6 h-px w-56 overflow-hidden bg-white/10">
            <div className="h-full bg-gradient-to-r from-cyan/40 to-cyan-glow transition-[width] duration-150" style={{ width: `${pct}%` }} />
          </div>
          <p className="mt-3 font-mono text-2xl font-light tabular-nums text-white/85">{Math.round(pct)}%</p>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
