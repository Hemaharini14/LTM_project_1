import { Plane, Building2, Landmark, UtensilsCrossed } from 'lucide-react';
import { Reveal, SectionLabel } from './ui';
import { ITINERARY } from '../data/tripData';

const ICONS = { flight: Plane, hotel: Building2, sight: Landmark, food: UtensilsCrossed } as const;

/**
 * Cinematic timeline. Items reveal in sequence down a lit spine so the day
 * reads as a schedule unfolding, not a bulleted list.
 */
export function Itinerary() {
  return (
    <section id="itinerary" className="relative flex min-h-screen items-center py-28">
      <div className="mx-auto w-full max-w-[1500px] px-6 sm:px-10">
        <Reveal><SectionLabel index="06">Smart Itinerary</SectionLabel></Reveal>
        <Reveal delay={0.08}>
          <h2 className="display mt-7 max-w-3xl text-[clamp(2.2rem,6vw,4.8rem)] text-white">
            THE JOURNEY STARTS<br /><span className="text-white/35">BEFORE TAKEOFF.</span>
          </h2>
        </Reveal>

        <div className="mt-16 grid gap-12 lg:grid-cols-2">
          {ITINERARY.map((day, di) => (
            <div key={day.day}>
              <Reveal delay={di * 0.1}>
                <div className="flex items-baseline gap-4">
                  <span className="display text-3xl text-cyan-glow">{day.day}</span>
                  <span className="h-px flex-1 bg-gradient-to-r from-cyan/30 to-transparent" />
                </div>
              </Reveal>

              <div className="relative mt-8 pl-8">
                {/* lit spine */}
                <div className="absolute left-[7px] top-2 h-[calc(100%-1rem)] w-px bg-gradient-to-b from-cyan/50 via-cyan/20 to-transparent" />

                {day.items.map((item, i) => {
                  const Icon = ICONS[item.kind as keyof typeof ICONS] ?? Landmark;
                  return (
                    <Reveal key={item.time} delay={di * 0.1 + i * 0.09} y={16}>
                      <div className="group relative pb-9">
                        <span className="absolute -left-8 top-1.5 flex h-[15px] w-[15px] items-center justify-center rounded-full border border-cyan/40 bg-void">
                          <span className="h-1.5 w-1.5 rounded-full bg-cyan transition-transform duration-300 group-hover:scale-150" />
                        </span>
                        <div className="flex items-baseline gap-4">
                          <span className="font-mono text-xs tabular-nums text-cyan/70">{item.time}</span>
                          <div className="flex items-center gap-2">
                            <Icon className="h-3.5 w-3.5 text-white/35" strokeWidth={1.5} />
                            <span className="font-display text-base text-white/85">{item.label}</span>
                          </div>
                        </div>
                      </div>
                    </Reveal>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
