import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react';
import { useScrollProgress } from './animations/scrollAnimations';
import { Loader } from './components/Loader';
import { Navbar } from './components/Navbar';
import { FlightWidget } from './components/FlightWidget';
import { Hero } from './components/Hero';
import { DelayPrediction } from './components/DelayPrediction';
import { DelayCauses } from './components/DelayCauses';
import { AircraftRotation } from './components/AircraftRotation';
import { AIExplanation } from './components/AIExplanation';
import { TripPlanner } from './components/TripPlanner';
import { Itinerary } from './components/Itinerary';
import { FinalCTA } from './components/FinalCTA';

// The WebGL stack is ~1MB parsed; deferring it lets the shell and the loading
// screen paint immediately instead of waiting on Three.
const AircraftScene = lazy(() =>
  import('./components/AircraftScene').then((m) => ({ default: m.AircraftScene })),
);

/**
 * Decide the render budget once, from what the device actually reports, rather
 * than from viewport width - a narrow window on a workstation should still get
 * the full scene. Drives geometry detail, particle counts, shadows and DPR.
 */
function useQuality() {
  return useMemo(() => {
    if (typeof window === 'undefined') return 1;
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduced) return 0.3;
    const cores = navigator.hardwareConcurrency ?? 4;
    const mem = (navigator as Navigator & { deviceMemory?: number }).deviceMemory ?? 4;
    const coarse = window.matchMedia('(pointer: coarse)').matches;
    if (coarse && (cores <= 4 || mem <= 4)) return 0.35;   // low-end phone
    if (coarse) return 0.55;                                // decent phone/tablet
    if (cores <= 4 || mem <= 4) return 0.6;                 // light laptop
    return 1;
  }, []);
}

export default function App() {
  const [booted, setBooted] = useState(false);
  const quality = useQuality();
  const progress = useScrollProgress();

  // The 3D layer reads scroll from a ref so React never re-renders on scroll -
  // the whole story updates inside one useFrame instead.
  const progressRef = useRef(0);
  progressRef.current = progress;

  useEffect(() => {
    document.body.style.overflow = booted ? '' : 'hidden';
  }, [booted]);

  return (
    <>
      <Loader onDone={() => setBooted(true)} />

      <Suspense fallback={null}>
        <AircraftScene progressRef={progressRef} quality={quality} />
      </Suspense>

      {/* Vignette + horizon wash: sits between the canvas and the copy so text
          always has contrast regardless of what the 3D scene is doing. */}
      <div className="pointer-events-none fixed inset-0 -z-[5]"
           style={{ background: 'radial-gradient(120% 80% at 50% 0%, transparent 35%, rgba(3,7,13,.75) 100%)' }} />

      <Navbar progress={progress} />
      <FlightWidget progress={progress} />

      <main className="relative z-10">
        <Hero />
        <DelayPrediction />
        <DelayCauses />
        <AircraftRotation />
        <AIExplanation />
        <TripPlanner />
        <Itinerary />
        <FinalCTA />
      </main>
    </>
  );
}
