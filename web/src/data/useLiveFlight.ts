import { useEffect, useState } from 'react';
import { FEATURED_FLIGHT, DELAY_CAUSES } from './flightData';

/**
 * Pulls a real prediction from the trained model behind the Flask app.
 *
 * The showcase ships with static demo figures so it renders instantly and still
 * works if the API is down or the page is served standalone. This hook upgrades
 * those to genuine model output once it arrives - same numbers the product
 * itself would show for that route.
 *
 * `live` says which you are looking at, so the UI can label it honestly rather
 * than implying every figure came from the model.
 */
export type LiveFlight = {
  live: boolean;
  code: string;
  from: string;
  to: string;
  departure: string;
  delayProbability: number;
  riskLabel: string;
  medianDelayMin: number;
  p90DelayMin: number | null;
  topCause: string | null;
  causes: { id: string; label: string; share: number; angle: number }[];
  sampleSize: number | null;
  modelAuc: number | null;
};

const FALLBACK: LiveFlight = {
  live: false,
  code: FEATURED_FLIGHT.code,
  from: FEATURED_FLIGHT.from,
  to: FEATURED_FLIGHT.to,
  departure: FEATURED_FLIGHT.departure,
  delayProbability: FEATURED_FLIGHT.delayProbability,
  riskLabel: FEATURED_FLIGHT.riskLabel,
  medianDelayMin: FEATURED_FLIGHT.predictedDelayMin,
  p90DelayMin: null,
  topCause: null,
  causes: DELAY_CAUSES.map((c) => ({ ...c })),
  sampleSize: null,
  modelAuc: null,
};

type ApiResponse = {
  available: boolean;
  carrier?: string;
  from?: string;
  to?: string;
  delay_probability?: number;
  risk_label?: string;
  median_delay_min?: number | null;
  p90_delay_min?: number | null;
  top_cause?: string | null;
  causes?: Record<string, number>;
  sample_size?: number | null;
  model_auc?: number | null;
};

/** Keep the radial layout stable while swapping in real shares. */
function mergeCauses(apiCauses: Record<string, number>) {
  const byLabel = new Map(
    Object.entries(apiCauses).map(([k, v]) => [k.toLowerCase(), v]),
  );
  const matched = DELAY_CAUSES.map((c) => {
    // API labels differ slightly from the display copy ("air traffic / airport"
    // vs "AIRPORT CONGESTION"), so match on the distinguishing word.
    const key = [...byLabel.keys()].find(
      (k) =>
        k.includes(c.label.toLowerCase().split(' ')[0]) ||
        (c.id === 'congestion' && k.includes('air traffic')) ||
        (c.id === 'late-aircraft' && k.includes('late arriving')),
    );
    return { ...c, share: key ? byLabel.get(key)! : c.share };
  });
  const total = matched.reduce((s, c) => s + c.share, 0) || 1;
  return matched
    .map((c) => ({ ...c, share: c.share / total }))
    .sort((a, b) => b.share - a.share)
    // re-spread around the circle so the dominant cause leads
    .map((c, i, arr) => ({ ...c, angle: -18 + (360 / arr.length) * i }));
}

export function useLiveFlight(): LiveFlight {
  const [flight, setFlight] = useState<LiveFlight>(FALLBACK);

  useEffect(() => {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 8000);

    fetch(`/api/showcase-flight?carrier=AI&from=${FALLBACK.from}&to=${FALLBACK.to}&hour=10`, {
      signal: ctrl.signal,
    })
      .then((r) => (r.ok ? (r.json() as Promise<ApiResponse>) : null))
      .then((d) => {
        if (!d?.available || typeof d.delay_probability !== 'number') return;
        setFlight({
          ...FALLBACK,
          live: true,
          code: `${d.carrier ?? 'AI'} 472`,
          from: d.from ?? FALLBACK.from,
          to: d.to ?? FALLBACK.to,
          delayProbability: d.delay_probability,
          riskLabel: d.risk_label ?? FALLBACK.riskLabel,
          medianDelayMin: d.median_delay_min ?? FALLBACK.medianDelayMin,
          p90DelayMin: d.p90_delay_min ?? null,
          topCause: d.top_cause ?? null,
          causes: d.causes && Object.keys(d.causes).length ? mergeCauses(d.causes) : FALLBACK.causes,
          sampleSize: d.sample_size ?? null,
          modelAuc: d.model_auc ?? null,
        });
      })
      .catch(() => {
        /* offline or standalone: the demo figures already on screen stand */
      })
      .finally(() => clearTimeout(timer));

    return () => {
      clearTimeout(timer);
      ctrl.abort();
    };
  }, []);

  return flight;
}
