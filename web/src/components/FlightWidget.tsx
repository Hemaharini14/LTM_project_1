import { motion, AnimatePresence } from 'framer-motion';
import { activeSection } from '../animations/scrollAnimations';
import { FEATURED_FLIGHT } from '../data/flightData';

/**
 * Floating live-status widget. It tracks the story beat, so it reads like an
 * instrument responding to the scene rather than static decoration.
 */
const STATES: Record<string, { tone: 'ok' | 'warn' | 'info'; title: string; value: string; sub: string }> = {
  hero:        { tone: 'info', title: 'Live Flight Intelligence', value: FEATURED_FLIGHT.code, sub: `${FEATURED_FLIGHT.from} → ${FEATURED_FLIGHT.to}` },
  prediction:  { tone: 'warn', title: 'Delay Risk', value: '18 MIN', sub: '72% probability' },
  causes:      { tone: 'warn', title: 'Primary Cause', value: 'LATE AIRCRAFT', sub: '44% of delay minutes' },
  rotation:    { tone: 'warn', title: 'Inbound Sector', value: '31 MIN LATE', sub: 'AI 471 · BOM → MAA' },
  explanation: { tone: 'info', title: 'AI Explanation', value: 'GENERATED', sub: 'Grounded in real data' },
  planner:     { tone: 'ok',   title: 'Trip Planner', value: 'READY', sub: 'Chennai → Delhi' },
  trip:        { tone: 'ok',   title: 'Itinerary Built', value: '₹18,740', sub: 'Within ₹25,000 budget' },
  itinerary:   { tone: 'ok',   title: 'Schedule', value: '2 DAYS', sub: '7 planned stops' },
  cta:         { tone: 'ok',   title: 'Ready For Departure', value: 'SmartRouteAI', sub: 'Travel with intelligence' },
};

const TONE = {
  ok:   { dot: 'bg-emerald-400', text: 'text-emerald-300' },
  warn: { dot: 'bg-amber-400', text: 'text-amber-300' },
  info: { dot: 'bg-cyan', text: 'text-cyan-glow' },
};

export function FlightWidget({ progress }: { progress: number }) {
  const key = activeSection(progress);
  const s = STATES[key] ?? STATES.hero;
  const tone = TONE[s.tone];

  return (
    <div className="pointer-events-none fixed bottom-7 left-6 z-40 hidden [@media(min-width:1536px)_and_(min-height:820px)]:block">
      <div className="glass glass-hair w-[236px] px-5 py-4">
        <div className="flex items-center gap-2">
          <span className="relative flex h-1.5 w-1.5">
            <span className={`absolute inline-flex h-full w-full animate-ping rounded-full ${tone.dot} opacity-60`} />
            <span className={`relative inline-flex h-1.5 w-1.5 rounded-full ${tone.dot}`} />
          </span>
          <span className="font-mono text-[9px] uppercase tracking-[0.2em] text-white/45">{s.title}</span>
        </div>

        <AnimatePresence mode="wait">
          <motion.div
            key={key}
            initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
          >
            <p className={`mt-2.5 font-display text-xl font-semibold tracking-tight ${tone.text}`}>{s.value}</p>
            <p className="mt-0.5 font-mono text-[10px] text-white/40">{s.sub}</p>
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
