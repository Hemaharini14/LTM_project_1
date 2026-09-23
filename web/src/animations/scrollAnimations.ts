/**
 * Scroll is the single source of truth for the whole story.
 *
 * One normalised progress value (0..1 across the page) drives the camera, the
 * aircraft and the globe. DOM sections read the same value, so 3D and copy can
 * never drift out of sync - which is the usual failure mode when each section
 * animates itself independently.
 */
import { useEffect, useState } from 'react';

export const SECTIONS = [
  'hero', 'prediction', 'causes', 'rotation', 'explanation', 'planner', 'trip', 'itinerary', 'cta',
] as const;
export type SectionId = (typeof SECTIONS)[number];

/** Where each beat sits along the page, as a scroll fraction. */
export const BEATS: Record<SectionId, number> = {
  hero: 0.0, prediction: 0.12, causes: 0.25, rotation: 0.38,
  explanation: 0.51, planner: 0.63, trip: 0.75, itinerary: 0.86, cta: 0.96,
};

export const clamp = (v: number, a = 0, b = 1) => Math.min(b, Math.max(a, v));
export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
/** Frame-rate independent smoothing - critical so motion feels identical at 60 and 120Hz. */
export const damp = (cur: number, target: number, lambda: number, dt: number) =>
  lerp(cur, target, 1 - Math.exp(-lambda * dt));

/** 0..1 ramp between two scroll positions, smoothstepped. */
export function range(p: number, start: number, end: number) {
  const t = clamp((p - start) / (end - start));
  return t * t * (3 - 2 * t);
}

export function useScrollProgress() {
  const [progress, setProgress] = useState(0);
  useEffect(() => {
    let raf = 0;
    const read = () => {
      const max = document.documentElement.scrollHeight - window.innerHeight;
      setProgress(max > 0 ? clamp(window.scrollY / max) : 0);
      raf = 0;
    };
    // rAF-coalesced: scroll fires far more often than we can usefully render.
    const onScroll = () => { if (!raf) raf = requestAnimationFrame(read); };
    read();
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll);
    return () => {
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onScroll);
      if (raf) cancelAnimationFrame(raf);
    };
  }, []);
  return progress;
}

/** Which beat the story is currently on - drives the live status widget. */
export function activeSection(p: number): SectionId {
  let current: SectionId = 'hero';
  for (const id of SECTIONS) if (p >= BEATS[id] - 0.05) current = id;
  return current;
}
