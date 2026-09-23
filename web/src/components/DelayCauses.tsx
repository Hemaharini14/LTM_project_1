import { useState } from 'react';
import { motion } from 'framer-motion';
import { Reveal, SectionLabel } from './ui';
import { PREVIOUS_SECTOR } from '../data/flightData';
import { useLiveFlight } from '../data/useLiveFlight';

/**
 * Causes orbit the aircraft with connection lines drawn back to centre.
 * Share-of-delay-minutes drives node emphasis and bar width, so the dominant
 * cause is visually dominant rather than merely listed first.
 */
export function DelayCauses() {
  const [active, setActive] = useState(0);
  const { causes: DELAY_CAUSES, live } = useLiveFlight();

  return (
    <section id="causes" className="relative flex min-h-screen items-center py-28">
      <div className="mx-auto w-full max-w-[1500px] px-6 sm:px-10">
        <Reveal><SectionLabel index="02">Root Cause Analysis</SectionLabel></Reveal>
        <Reveal delay={0.08}>
          <h2 className="display mt-7 max-w-3xl text-[clamp(2.2rem,6vw,4.8rem)] text-white">
            WHY IS YOUR<br /><span className="text-white/35">FLIGHT DELAYED?</span>
          </h2>
        </Reveal>

        <div className="mt-16 grid gap-12 lg:grid-cols-[1fr_400px]">
          <Reveal delay={0.14}>
            <div className="relative mx-auto aspect-square w-full max-w-[520px]">
              <svg viewBox="0 0 400 400" className="absolute inset-0 h-full w-full">
                {[70, 120, 170].map((r) => (
                  <circle key={r} cx="200" cy="200" r={r} fill="none" stroke="rgba(34,211,238,.09)" strokeWidth="1" />
                ))}
                {DELAY_CAUSES.map((c, i) => {
                  const rad = (c.angle * Math.PI) / 180;
                  const x = 200 + Math.cos(rad) * 170;
                  const y = 200 + Math.sin(rad) * 170;
                  return (
                    <line
                      key={c.id} x1="200" y1="200" x2={x} y2={y}
                      stroke={i === active ? '#67e8f9' : 'rgba(34,211,238,.2)'}
                      strokeWidth={i === active ? 1.6 : 1}
                      strokeDasharray="3 5"
                      className="transition-all duration-500"
                    >
                      <animate attributeName="stroke-dashoffset" from="16" to="0" dur="1.4s" repeatCount="indefinite" />
                    </line>
                  );
                })}
                <circle cx="200" cy="200" r="7" fill="#67e8f9" />
                <circle cx="200" cy="200" r="16" fill="none" stroke="#22d3ee" strokeWidth="1" opacity=".4" />
              </svg>

              {DELAY_CAUSES.map((c, i) => {
                const rad = (c.angle * Math.PI) / 180;
                return (
                  <button
                    key={c.id}
                    onMouseEnter={() => setActive(i)}
                    onFocus={() => setActive(i)}
                    className="absolute -translate-x-1/2 -translate-y-1/2 whitespace-nowrap rounded-full border px-3 py-1.5 font-mono text-[9px] uppercase tracking-[0.15em] transition-all duration-300"
                    style={{
                      left: `${50 + Math.cos(rad) * 42.5}%`,
                      top: `${50 + Math.sin(rad) * 42.5}%`,
                      borderColor: i === active ? 'rgba(103,232,249,.6)' : 'rgba(255,255,255,.1)',
                      background: i === active ? 'rgba(34,211,238,.12)' : 'rgba(255,255,255,.03)',
                      color: i === active ? '#67e8f9' : 'rgba(255,255,255,.45)',
                      backdropFilter: 'blur(8px)',
                    }}
                  >
                    {c.label}
                  </button>
                );
              })}
            </div>
          </Reveal>

          <Reveal delay={0.2}>
            <motion.div
              key={active}
              initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
              className="glass glass-hair p-8"
            >
              <span className="label">{live ? 'Dominant Factor · Real Data' : 'Dominant Factor'}</span>
              <h3 className="display mt-3 text-2xl text-cyan-glow">{DELAY_CAUSES[active].label}</h3>

              <div className="mt-6 h-1 w-full overflow-hidden rounded-full bg-white/[0.06]">
                <motion.div
                  className="h-full rounded-full bg-gradient-to-r from-cyan-dim to-cyan-glow"
                  initial={{ width: 0 }}
                  animate={{ width: `${DELAY_CAUSES[active].share * 100}%` }}
                  transition={{ duration: 1, ease: [0.22, 1, 0.36, 1] }}
                />
              </div>
              <p className="mt-2 font-mono text-[10px] text-white/40">
                {(DELAY_CAUSES[active].share * 100).toFixed(0)}% of total delay minutes
              </p>

              {active === 0 && (
                <div className="mt-7 divide-y divide-white/[0.05] border-t border-white/[0.05] pt-1">
                  {[
                    ['Previous flight', PREVIOUS_SECTOR.code],
                    ['Arrival', `${PREVIOUS_SECTOR.arrivedLateMin} MIN LATE`],
                    ['Impact', PREVIOUS_SECTOR.impact],
                    ['Downstream delay', '18 MIN'],
                  ].map(([k, v]) => (
                    <div key={k} className="flex items-baseline justify-between py-2.5">
                      <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-white/35">{k}</span>
                      <span className="font-display text-sm text-white/90">{v}</span>
                    </div>
                  ))}
                </div>
              )}
            </motion.div>
          </Reveal>
        </div>
      </div>
    </section>
  );
}
