import { ArrowRight } from 'lucide-react';
import { Reveal, SectionLabel } from './ui';
import { ROTATION_LEGS, PREVIOUS_SECTOR, FEATURED_FLIGHT } from '../data/flightData';

/**
 * The cascade story. The 3D layer draws the real route map behind this section;
 * here the causal chain is made legible - one late inbound eats the turnaround,
 * and the next departure inherits whatever is left.
 */
export function AircraftRotation() {
  const CHAIN = ['31 MIN LATE', 'TURNAROUND', '18 MIN DELAY'];

  return (
    <section id="rotation" className="relative flex min-h-screen items-center py-28">
      <div className="mx-auto w-full max-w-[1500px] px-6 sm:px-10">
        <Reveal><SectionLabel index="03">Aircraft Rotation</SectionLabel></Reveal>
        <Reveal delay={0.08}>
          <h2 className="display mt-7 max-w-3xl text-[clamp(2.2rem,6vw,4.8rem)] text-white">
            ONE AIRCRAFT.<br /><span className="text-white/35">FIVE CITIES.</span>
          </h2>
        </Reveal>
        <Reveal delay={0.14}>
          <p className="mt-7 max-w-lg text-[15px] leading-relaxed text-white/55">
            Delays travel with the aircraft. We trace the tail number across its day to see what
            your flight is actually inheriting.
          </p>
        </Reveal>

        <Reveal delay={0.2}>
          <div className="mt-14 flex flex-wrap items-center gap-x-3 gap-y-4">
            {ROTATION_LEGS.map((l, i) => (
              <div key={l.code} className="flex items-center gap-3">
                <div className="flex flex-col">
                  <span
                    className={`font-display text-lg font-semibold tracking-tight ${
                      i === ROTATION_LEGS.length - 1 ? 'text-cyan-glow' : 'text-white/80'
                    }`}
                  >
                    {l.code}
                  </span>
                  <span className="font-mono text-[9px] uppercase tracking-[0.2em] text-white/30">{l.city}</span>
                </div>
                {i < ROTATION_LEGS.length - 1 && <ArrowRight className="h-3.5 w-3.5 text-cyan/40" strokeWidth={1.5} />}
              </div>
            ))}
          </div>
        </Reveal>

        <div className="mt-14 grid gap-5 md:grid-cols-3">
          <Reveal delay={0.24}>
            <div className="glass h-full p-7">
              <span className="label">Previous Flight</span>
              <p className="display mt-3 text-2xl text-white">{PREVIOUS_SECTOR.code}</p>
              <p className="mt-1 font-mono text-[11px] text-white/40">
                {PREVIOUS_SECTOR.from} &rarr; {PREVIOUS_SECTOR.to}
              </p>
              <div className="mt-6 rule" />
              <p className="mt-4 font-display text-3xl font-semibold text-amber-300">
                {PREVIOUS_SECTOR.arrivedLateMin} MIN
              </p>
              <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-white/35">Arrived late</p>
            </div>
          </Reveal>

          <Reveal delay={0.3}>
            <div className="glass glass-hair flex h-full flex-col items-center justify-center p-7 text-center">
              <span className="label">Turnaround</span>
              <div className="my-5 flex flex-col items-center gap-2">
                {CHAIN.map((s, i) => (
                  <div key={s} className="flex flex-col items-center gap-2">
                    <span
                      className={`font-mono text-[11px] tracking-[0.15em] ${
                        i === 2 ? 'text-cyan-glow' : 'text-white/55'
                      }`}
                    >
                      {s}
                    </span>
                    {i < CHAIN.length - 1 && <span className="h-5 w-px bg-gradient-to-b from-cyan/60 to-transparent" />}
                  </div>
                ))}
              </div>
              <p className="font-mono text-[9px] leading-relaxed text-white/30">
                Recovered time is capped by the scheduled ground window
              </p>
            </div>
          </Reveal>

          <Reveal delay={0.36}>
            <div className="glass h-full p-7">
              <span className="label">Next Flight</span>
              <p className="display mt-3 text-2xl text-white">{FEATURED_FLIGHT.code}</p>
              <p className="mt-1 font-mono text-[11px] text-white/40">
                {FEATURED_FLIGHT.from} &rarr; {FEATURED_FLIGHT.to}
              </p>
              <div className="mt-6 rule" />
              <p className="mt-4 font-display text-3xl font-semibold text-cyan-glow">HIGH</p>
              <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-white/35">Inherited risk</p>
            </div>
          </Reveal>
        </div>
      </div>
    </section>
  );
}
