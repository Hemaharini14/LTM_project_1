/*
 * Trip planner UI. Talks to POST /api/plan-trip and renders the itinerary.
 * No framework: the page has one form and four cards, which is not enough
 * state to justify a build step.
 */
const $ = (id) => document.getElementById(id);

const rupee = new Intl.NumberFormat('en-IN', {
  style: 'currency', currency: 'INR', maximumFractionDigits: 0,
});
const MONTHS = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'];

/* ------------------------------------------------------------- date field */
// A native date input is the only accessible way to pick a date, but it can't
// be styled to "12 OCT". So it sits invisibly over a plain text display and
// the two are kept in sync.
function syncDate() {
  const [y, m, d] = $('date').value.split('-').map(Number);
  if (!y) return;
  $('date_display').value = `${d} ${MONTHS[m - 1]}`;
}
$('date').addEventListener('change', syncDate);
syncDate();

/* ----------------------------------------------------------- budget field */
const budgetEl = $('budget_display');
const budgetDigits = () => Number(budgetEl.value.replace(/[^\d]/g, '')) || 0;

budgetEl.addEventListener('focus', () => { budgetEl.value = String(budgetDigits() || ''); });
budgetEl.addEventListener('blur', () => {
  const n = budgetDigits();
  budgetEl.value = n ? rupee.format(n) : '';
});
budgetEl.addEventListener('input', () => {
  // keep it numeric while typing, format on blur
  budgetEl.value = budgetEl.value.replace(/[^\d,₹\s]/g, '');
});

/* -------------------------------------------------------------- rendering */
function showError(lines) {
  const box = $('error');
  box.innerHTML = Array.isArray(lines) ? lines.map((l) => `• ${l}`).join('<br>') : lines;
  box.hidden = false;
  $('results').hidden = true;
}

function stagger() {
  // cards rise one after another, badge last
  const cards = document.querySelectorAll('#results .card');
  cards.forEach((c, i) => {
    c.classList.remove('in');
    void c.offsetWidth;                       // restart the animation
    c.style.animationDelay = `${i * 100}ms`;
    c.classList.add('in');
  });
  const badge = $('badge');
  badge.classList.remove('in');
  void badge.offsetWidth;
  badge.style.animationDelay = `${cards.length * 100}ms`;
  badge.classList.add('in');
}

function render(data) {
  $('error').hidden = true;
  $('results').hidden = false;

  const f = data.flight;
  $('r-flight').textContent = f ? f.number : '—';
  $('r-flight-sub').textContent = f ? `${f.depart} → ${f.arrive}` : 'No flight found';
  $('r-rerouted').hidden = !(f && f.was_rerouted);

  const h = data.hotel;
  $('r-hotel').textContent = h ? `${h.nights} night${h.nights === 1 ? '' : 's'}` : '—';
  $('r-hotel-sub').textContent = h ? rupee.format(h.price_total) : 'No stay planned';

  const a = data.attractions || [];
  $('r-sights').textContent = `${a.length} stop${a.length === 1 ? '' : 's'}`;
  $('r-sights-sub').textContent = a.length ? a.map((x) => x.name).join(' · ') : 'None within budget';

  $('r-total').textContent = rupee.format(data.total);
  const totalSub = $('r-total-sub');
  totalSub.textContent = `${data.under_budget ? 'Under' : 'Over'} ${rupee.format(data.budget)}`;
  totalSub.classList.toggle('over', !data.under_budget);

  const badge = $('badge');
  badge.hidden = false;
  badge.dataset.status = data.status;
  $('badge-status').textContent = data.status.replace('_', ' ');
  $('badge-route').textContent = data.route_label;

  stagger();
}

/* ---------------------------------------------------------------- submit */
$('trip-form').addEventListener('submit', async (e) => {
  e.preventDefault();

  const payload = {
    from_city: $('from_city').value.trim(),
    to_city: $('to_city').value.trim(),
    date: $('date').value,
    travelers: Number($('travelers').value),
    budget: budgetDigits(),
  };

  // Client-side first so obvious mistakes never cost a round trip.
  const problems = [];
  if (!payload.from_city || !payload.to_city) problems.push('Enter both a origin and a destination.');
  else if (payload.from_city.toLowerCase() === payload.to_city.toLowerCase())
    problems.push('Origin and destination must be different.');
  if (!payload.date) problems.push('Pick a travel date.');
  if (!(payload.travelers >= 1)) problems.push('At least one traveller is required.');
  if (!(payload.budget > 0)) problems.push('Enter a budget greater than zero.');
  if (problems.length) return showError(problems);

  const btn = $('submit');
  btn.disabled = true;
  btn.classList.add('loading');
  btn.querySelector('.cta-text').textContent = 'Planning…';

  try {
    const res = await fetch('/api/plan-trip', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (res.status === 422) return showError(data.details || [data.error || 'Invalid request.']);
    if (!res.ok) return showError('The planner is unavailable right now. Please try again.');
    render(data);
  } catch {
    showError('Could not reach the planner. Check that the server is running.');
  } finally {
    btn.disabled = false;
    btn.classList.remove('loading');
    btn.querySelector('.cta-text').textContent = 'Plan my trip';
  }
});
