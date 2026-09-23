import { MapPin, Calendar, Users, Wallet, ArrowRight } from 'lucide-react';
import { Reveal, SectionLabel, MagneticButton } from './ui';
import { TRIP_QUERY, TRIP_RESULT } from '../data/tripData';

/**
 * The hand-off from aircraft to journey. The form is a real link into the
 * working Flask trip planner - its values prefill the query string so the
 * showcase leads directly into the product rather than dead-ending.
 */
const FIELDS = [
  { icon: MapPin, label: 'From', value: TRIP_QUERY.from },
  { icon: MapPin, label: 'To', value: TRIP_QUERY.to },
  { icon: Calendar, label: 'Date', value: TRIP_QUERY.date },
  { icon: Users, label: 'Travelers', value: String(TRIP_QUERY.travelers) },
  { icon: Wallet, label: 'Budget', value: `₹${TRIP_QUERY.budget.toLocaleString('en-IN')}` },
];

export function TripPlanner() {
  const href = `/budget-trip?origin_place=${encodeURIComponent(TRIP_QUERY.from)}&destination_place=${encodeURIComponent(TRIP_QUERY.to)}`;

  return (
    <section id="planner" className="relative flex min-h-screen items-center py-28">
      <div className="mx-auto w-full max-w-[1500px] px-6 sm:px-10">
        <div className="max-w-2xl">
          <Reveal><SectionLabel index="05">Smart Trip Planner</SectionLabel></Reveal>
          <Reveal delay={0.08}>
            <h2 className="display mt-7 text-[clamp(2.4rem,7vw,5.6rem)] text-white">
              FROM FLIGHT<br /><span className="text-white/35">TO JOURNEY.</span>
            </h2>
          </Reveal>
          <Reveal delay={0.14}>
            <p className="mt-7 max-w-md text-[15px] leading-relaxed text-white/55">
              Plan the entire trip, not just the flight.
            </p>
          </Reveal>
        </div>

        <Reveal delay={0.2}>
          <div className="glass glass-hair mt-14 max-w-4xl p-3">
            <div className="grid gap-px overflow-hidden rounded-xl bg-white/[0.04] sm:grid-cols-2 lg:grid-cols-5">
              {FIELDS.map((f) => (
                <div key={f.label} className="group bg-void/60 px-5 py-5 transition-colors hover:bg-white/[0.03]">
                  <div className="flex items-center gap-1.5">
                    <f.icon className="h-3 w-3 text-cyan/60" strokeWidth={1.5} />
                    <span className="font-mono text-[9px] uppercase tracking-[0.2em] text-white/35">{f.label}</span>
                  </div>
                  <p className="mt-2 font-display text-base font-medium text-white/90">{f.value}</p>
                </div>
              ))}
            </div>
            <div className="flex justify-end p-4">
              <MagneticButton href={href}>
                Plan My Trip <ArrowRight className="h-3 w-3" strokeWidth={2} />
              </MagneticButton>
            </div>
          </div>
        </Reveal>

        {/* generated result cards */}
        <div className="mt-16 grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
          {[
            { label: 'Flight', main: TRIP_RESULT.flight.code, sub: `${TRIP_RESULT.flight.depart} → ${TRIP_RESULT.flight.arrive}` },
            { label: 'Hotel', main: `${TRIP_RESULT.hotel.nights} nights`, sub: `₹${TRIP_RESULT.hotel.cost.toLocaleString('en-IN')}` },
            { label: 'Attractions', main: `${TRIP_RESULT.attractions.length} stops`, sub: TRIP_RESULT.attractions.join(' · ') },
            { label: 'Total', main: `₹${TRIP_RESULT.total.toLocaleString('en-IN')}`, sub: `Under ₹${TRIP_QUERY.budget.toLocaleString('en-IN')}`, accent: true },
          ].map((c, i) => (
            <Reveal key={c.label} delay={0.24 + i * 0.06}>
              <div className={`glass h-full p-6 ${c.accent ? 'glass-hair' : ''}`}>
                <span className="label">{c.label}</span>
                <p className={`display mt-3 text-xl ${c.accent ? 'text-cyan-glow' : 'text-white'}`}>{c.main}</p>
                <p className="mt-1.5 font-mono text-[10px] leading-relaxed text-white/40">{c.sub}</p>
              </div>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}
