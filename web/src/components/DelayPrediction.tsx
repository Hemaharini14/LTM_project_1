import { Reveal, SectionLabel, AnimatedNumber } from './ui';
import { useLiveFlight } from '../data/useLiveFlight';

/** Radial probability ring - aviation instrument, not a dashboard donut. */
function ProbabilityRing({ value }: { value: number }) {
  const r = 78, c = 2 * Math.PI * r;
  return (
    <div className="relative h-[210px] w-[210px]">
      <svg viewBox="0 0 200 200" className="h-full w-full -rotate-90">
        <circle cx="100" cy="100" r={r} fill="none" stroke="rgba(255,255,255,.07)" strokeWidth="1.5" />
        {/* tick ring */}
        {Array.from({ length: 60 }).map((_, i) => (
          <line key={i} x1="100" y1="14" x2="100" y2={i % 5 === 0 ? 22 : 18}
                stroke="rgba(34,211,238,.25)" strokeWidth="1"
                transform={`rotate(${i * 6} 100 100)`} />
        ))}
        <circle
          cx="100" cy="100" r={r} fill="none" stroke="url(#ring)" strokeWidth="2.5" strokeLinecap="round"
          strokeDasharray={c} strokeDashoffset={c * (1 - value)}
          style={{ transition: 'stroke-dashoffset 1.8s cubic-bezier(.22,1,.36,1)' }}
        />
        <defs>
          <linearGradient id="ring" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#0e7490" /><stop offset="100%" stopColor="#67e8f9" />
          </linearGradient>
        </defs>
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="font-display text-5xl font-semibold text-white text-glow">
          <AnimatedNumber value={value * 100} />%
        </span>
        <span className="label mt-1">Delay Probability</span>
      </div>
    </div>
  );
}

function Row({ k, v, accent }: { k: string; v: string; accent?: boolean }) {
  return (
    <div className="flex items-baseline justify-between py-2.5">
      <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-white/35">{k}</span>
      <span className={`font-display text-sm font-medium ${accent ? 'text-cyan-glow' : 'text-white/90'}`}>{v}</span>
    </div>
  );
}

export function DelayPrediction() {
  const F = useLiveFlight();
  return (
    <section id="prediction" className="relative flex min-h-screen items-center py-28">
      <div className="mx-auto w-full max-w-[1500px] px-6 sm:px-10">
        <div className="grid items-center gap-14 lg:grid-cols-2">
          <div>
            <Reveal><SectionLabel index="01">Know Before You Go</SectionLabel></Reveal>
            <Reveal delay={0.08}>
              <h2 className="display mt-7 text-[clamp(2.4rem,6.5vw,5.2rem)] text-white">
                FLIGHT DELAY<br /><span className="text-white/35">PREDICTION</span>
              </h2>
            </Reveal>
            <Reveal delay={0.16}>
              <p className="mt-7 max-w-sm text-[15px] leading-relaxed text-white/55">
                Turn complex aviation data into an understandable delay forecast.
              </p>
            </Reveal>
          </div>

          <Reveal delay={0.2} className="lg:justify-self-end">
            <div className="glass glass-hair w-full max-w-[400px] p-8">
              <div className="flex items-center justify-between">
                <span className="label">{F.live ? 'Live Model Output' : 'Flight AI Analysis'}</span>
                <span className="rounded-full border border-amber-400/30 bg-amber-400/10 px-2.5 py-1 font-mono text-[9px] uppercase tracking-widest text-amber-300">
                  {F.riskLabel} Risk
                </span>
              </div>

              <div className="mt-7 flex justify-center"><ProbabilityRing value={F.delayProbability} /></div>

              {F.live && (
                <p className="mt-5 font-mono text-[9px] leading-relaxed text-white/30">
                  Live from the trained model{F.modelAuc ? ` · held-out AUC ${F.modelAuc}` : ''}
                  {F.sampleSize ? ` · duration from ${F.sampleSize.toLocaleString()} real delayed flights` : ''}
                </p>
              )}
              <div className="mt-7 rule" />
              <div className="mt-1 divide-y divide-white/[0.05]">
                <Row k="Flight" v={F.code} />
                <Row k="Route" v={`${F.from} → ${F.to}`} />
                <Row k="Departure" v={F.departure} />
                <Row k="Predicted delay" v={`${F.medianDelayMin} MIN`} accent />
              </div>
            </div>
          </Reveal>
        </div>
      </div>
    </section>
  );
}
