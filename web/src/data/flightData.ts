/**
 * Showcase narrative data.
 *
 * These are the figures the cinematic story walks through. They mirror the real
 * shapes the Flask app returns (predict_delay_v2 -> probability + risk label,
 * delay_duration.py -> median minutes + ranked cause attribution), so the
 * showcase tells the truth about what the product actually computes.
 */
export const FEATURED_FLIGHT = {
  code: 'AI 472',
  from: 'BOM',
  to: 'DEL',
  fromCity: 'Mumbai',
  toCity: 'Delhi',
  departure: '10:45 AM',
  arrival: '13:35',
  predictedDelayMin: 18,
  delayProbability: 0.72,
  riskLabel: 'High',
} as const;

/** Ordered by real share of delay minutes - late-arriving aircraft genuinely dominates. */
export const DELAY_CAUSES = [
  { id: 'late-aircraft', label: 'LATE ARRIVING AIRCRAFT', share: 0.44, angle: -18 },
  { id: 'carrier', label: 'CARRIER', share: 0.28, angle: 34 },
  { id: 'congestion', label: 'AIRPORT CONGESTION', share: 0.16, angle: 96 },
  { id: 'weather', label: 'WEATHER', share: 0.08, angle: 158 },
  { id: 'crew', label: 'CREW', share: 0.03, angle: 220 },
  { id: 'schedule', label: 'SCHEDULE', share: 0.01, angle: 282 },
] as const;

export const ROTATION_LEGS = [
  { city: 'CHENNAI', code: 'MAA', lat: 13.08, lon: 80.27 },
  { city: 'MUMBAI', code: 'BOM', lat: 19.09, lon: 72.87 },
  { city: 'BENGALURU', code: 'BLR', lat: 12.97, lon: 77.59 },
  { city: 'HYDERABAD', code: 'HYD', lat: 17.24, lon: 78.43 },
  { city: 'DELHI', code: 'DEL', lat: 28.61, lon: 77.21 },
] as const;

export const PREVIOUS_SECTOR = {
  code: 'AI 471', from: 'BLR', to: 'BOM', arrivedLateMin: 31, impact: 'HIGH',
} as const;

export const AI_EXPLANATION =
  'Your flight has an elevated delay risk because the aircraft operating this route arrived 31 minutes late on its previous sector. Airport congestion may add additional turnaround time.';

/** Phrases the explanation highlights as it types in. */
export const AI_HIGHLIGHTS = ['31 minutes late', 'previous sector', 'Airport congestion'];
