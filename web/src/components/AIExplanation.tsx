import { useEffect, useRef, useState } from 'react';
import { useInView } from 'framer-motion';
import { Sparkles } from 'lucide-react';
import { Reveal, SectionLabel } from './ui';
import { AI_EXPLANATION, AI_HIGHLIGHTS } from '../data/flightData';

/**
 * The explanation types itself in, then the load-bearing phrases light up.
 * Highlighting happens after the text settles rather than during, so the
 * sentence reads once as prose before it reads as data.
 */
function TypedExplanation({ text }: { text: string }) {
  const ref = useRef<HTMLParagraphElement>(null);
  const inView = useInView(ref, { once: true, margin: '-20% 0px' });
  const [shown, setShown] = useState(0);
  const [lit, setLit] = useState(false);

  useEffect(() => {
    if (!inView) return;
    let raf = 0;
    const start = performance.now();
    const dur = 2200;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / dur);
      setShown(Math.floor(t * text.length));
      if (t < 1) raf = requestAnimationFrame(tick);
      else setTimeout(() => setLit(true), 220);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [inView, text]);

  // Split on the highlight phrases so they can be styled once typing finishes.
  const pattern = new RegExp(`(${AI_HIGHLIGHTS.join('|')})`, 'g');
  const visible = text.slice(0, shown);
  const parts = visible.split(pattern);

  return (
    <p ref={ref} className="font-display text-[clamp(1.05rem,2.1vw,1.55rem)] font-light leading-[1.65] text-white/80">
      {parts.map((part, i) =>
        AI_HIGHLIGHTS.includes(part) ? (
          <span
            key={i}
            className={`transition-all duration-700 ${
              lit ? 'text-cyan-glow [text-shadow:0_0_24px_rgba(34,211,238,.45)]' : 'text-white/80'
            }`}
          >
            {part}
          </span>
        ) : (
          <span key={i}>{part}</span>
        ),
      )}
      {shown < text.length && <span className="ml-0.5 inline-block h-[1.1em] w-[2px] translate-y-[3px] animate-pulse bg-cyan" />}
    </p>
  );
}

export function AIExplanation() {
  return (
    <section id="explanation" className="relative flex min-h-screen items-center py-28">
      <div className="mx-auto w-full max-w-[1500px] px-6 sm:px-10">
        <div className="grid gap-14 lg:grid-cols-[0.9fr_1.1fr] lg:items-center">
          <div>
            <Reveal><SectionLabel index="04">Explainable AI</SectionLabel></Reveal>
            <Reveal delay={0.08}>
              <h2 className="display mt-7 text-[clamp(2.2rem,5.6vw,4.4rem)] text-white">
                DO NOT JUST GET<br />A PREDICTION.<br />
                <span className="text-cyan-glow">GET THE REASON.</span>
              </h2>
            </Reveal>
          </div>

          <Reveal delay={0.16}>
            <div className="glass glass-hair relative overflow-hidden p-9">
              {/* slow scanline, the only "AI" flourish - restrained on purpose */}
              <div className="pointer-events-none absolute inset-0 opacity-[0.06]">
                <div className="h-24 w-full animate-scan bg-gradient-to-b from-transparent via-cyan to-transparent" />
              </div>

              <div className="flex items-center gap-2.5">
                <Sparkles className="h-3.5 w-3.5 text-cyan" strokeWidth={1.5} />
                <span className="label">AI Explanation</span>
              </div>

              <div className="mt-7">
                <TypedExplanation text={AI_EXPLANATION} />
              </div>

              <div className="mt-8 rule" />
              <div className="mt-5 flex flex-wrap gap-x-8 gap-y-3">
                {[
                  ['Model', 'DelayNet v2'],
                  ['Held-out AUC', '0.873'],
                  ['Grounding', 'Real flight records'],
                ].map(([k, v]) => (
                  <div key={k}>
                    <p className="font-mono text-[9px] uppercase tracking-[0.2em] text-white/30">{k}</p>
                    <p className="mt-1 font-display text-sm text-white/85">{v}</p>
                  </div>
                ))}
              </div>
            </div>
          </Reveal>
        </div>
      </div>
    </section>
  );
}
