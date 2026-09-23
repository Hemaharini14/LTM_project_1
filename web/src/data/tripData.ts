export const TRIP_QUERY = {
  from: 'Chennai', to: 'Delhi', date: '12 OCT', travelers: 2, budget: 25000,
} as const;

export const TRIP_RESULT = {
  flight: { code: 'AI 472', depart: '10:45', arrive: '13:35' },
  hotel: { nights: 3, cost: 6400 },
  attractions: ['India Gate', 'Red Fort', 'Qutub Minar'],
  total: 18740,
} as const;

export const ITINERARY = [
  {
    day: 'DAY 01',
    items: [
      { time: '10:45', label: 'Departure', kind: 'flight' },
      { time: '13:35', label: 'Landing', kind: 'flight' },
      { time: '15:00', label: 'Hotel Check-in', kind: 'hotel' },
      { time: '17:30', label: 'India Gate', kind: 'sight' },
    ],
  },
  {
    day: 'DAY 02',
    items: [
      { time: '09:00', label: 'Red Fort', kind: 'sight' },
      { time: '13:00', label: 'Lunch', kind: 'food' },
      { time: '15:00', label: 'Qutub Minar', kind: 'sight' },
    ],
  },
] as const;
