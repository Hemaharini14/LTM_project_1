/* Live flight search. Talks only to this app's own /api/flights/* endpoints - the
   Aviationstack key never reaches the browser. All server text is inserted with
   textContent, never as markup. */
(function () {
  var form = document.getElementById('fs-form');
  var statusEl = document.getElementById('fs-status');
  var resultsEl = document.getElementById('fs-results');
  var modal = document.getElementById('fs-modal');
  var modalBody = document.getElementById('fs-modal-body');
  if (!form) return;

  var dateInput = document.getElementById('fs-date');
  var today = new Date();
  var iso = today.getFullYear() + '-' + String(today.getMonth() + 1).padStart(2, '0') + '-' + String(today.getDate()).padStart(2, '0');
  dateInput.min = iso;
  dateInput.value = iso;

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = text;
    return n;
  }
  function clock(s) { return s && s.length >= 16 ? s.substr(11, 5) : '—'; }
  function dur(min) {
    if (!min) return null;
    return Math.floor(min / 60) + 'h ' + String(min % 60).padStart(2, '0') + 'm';
  }
  function statusBadge(st) {
    var s = (st || '').toLowerCase();
    var map = { scheduled: ['good', '🟢', 'Scheduled'], active: ['good', '🟢', 'In the air'],
                landed: ['neutral', '🛬', 'Landed'], cancelled: ['bad', '🔴', 'Cancelled'],
                incident: ['bad', '🔴', 'Incident'], diverted: ['warn', '🟠', 'Diverted'] };
    var m = map[s] || ['neutral', '⚪', st ? st : 'Status unavailable'];
    return el('span', 'fs-badge ' + m[0], m[1] + ' Flight status: ' + m[2]);
  }
  function setStatus(msg, kind, spinner) {
    statusEl.className = 'fs-status' + (kind ? ' ' + kind : '');
    statusEl.textContent = '';
    if (spinner) statusEl.appendChild(el('span', 'fs-spinner'));
    if (msg) statusEl.appendChild(document.createTextNode(msg));
  }
  function showNotice(meta) {
    if (meta && meta.notice) statusEl.appendChild(el('div', 'fs-notice', '⏱ ' + meta.notice));
  }

  function predictionBlock(p, detailed) {
    var box = el('div', 'fs-pred');
    if (!p || p.status !== 'ok') {
      var head = el('div', 'fs-pred-head');
      head.appendChild(el('strong', null, 'Delay prediction'));
      box.appendChild(head);
      box.appendChild(el('div', 'fs-unavail', (p && p.reason) || 'Prediction unavailable.'));
      return box;
    }
    var pct = Math.round(p.delay_probability * 100);
    var delayed = p.predicted === 'DELAYED';
    var color = delayed ? 'var(--danger)' : (p.risk_tier === 'Moderate' ? 'var(--warn)' : 'var(--safe)');
    var head2 = el('div', 'fs-pred-head');
    head2.appendChild(el('strong', null, '⚠️ Delay probability: ' + pct + '%'));
    head2.appendChild(el('span', 'fs-badge ' + (delayed ? 'bad' : 'good'),
      (delayed ? '🔴 Predicted: DELAYED' : '🟢 Predicted: ON TIME')));
    box.appendChild(head2);
    var bar = el('div', 'fs-bar');
    var fill = el('span'); fill.style.width = Math.max(2, pct) + '%'; fill.style.background = color;
    bar.appendChild(fill);
    if (p.threshold !== null && p.threshold !== undefined) {
      var tick = el('i', 'fs-tick'); tick.style.left = Math.round(p.threshold * 100) + '%';
      tick.title = 'Decision threshold ' + Math.round(p.threshold * 100) + '%';
      bar.appendChild(tick);
    }
    box.appendChild(bar);
    box.appendChild(el('div', 'fs-bar-label',
      'Risk tier: ' + p.risk_tier + (p.threshold !== null ? ' · decision threshold ' + Math.round(p.threshold * 100) + '%' : '')));
    var t = p.typical_delay_if_delayed;
    if (t) {
      box.appendChild(el('div', 'fs-fine',
        'If delayed, flights like this typically run about ' + t.median_min + ' min late (median of ' +
        t.sample_size.toLocaleString() + ' historical delayed flights: ' + t.basis + '). This is a historical statistic, not a forecast for this flight.'));
    }
    if (detailed) box.appendChild(el('div', 'fs-fine', 'Model: ' + p.model));
    return box;
  }

  function card(f) {
    var c = el('article', 'fs-card');
    var top = el('div', 'fs-card-top');
    top.appendChild(el('span', 'fs-airline', f.airline_name || f.airline_iata || 'Airline unavailable'));
    top.appendChild(el('span', 'fs-code', f.flight_iata || ((f.airline_iata || '') + (f.flight_number || ''))));
    c.appendChild(top);

    var route = el('div', 'fs-route');
    var l = el('div', 'fs-end');
    l.appendChild(el('span', 'fs-iata', f.origin_iata || '—'));
    l.appendChild(el('span', 'fs-time', clock(f.scheduled_departure)));
    l.appendChild(el('span', 'fs-city', f.origin_airport || ''));
    var r = el('div', 'fs-end right');
    r.appendChild(el('span', 'fs-iata', f.destination_iata || '—'));
    r.appendChild(el('span', 'fs-time', clock(f.scheduled_arrival)));
    r.appendChild(el('span', 'fs-city', f.destination_airport || ''));
    route.appendChild(l); route.appendChild(el('div', 'fs-line')); route.appendChild(r);
    c.appendChild(route);
    var d = dur(f.duration_min);
    if (d) c.appendChild(el('div', 'fs-meta', d + ' scheduled'));

    var row = el('div', 'fs-row');
    row.appendChild(statusBadge(f.status));
    if (f.departure_delay_min) row.appendChild(el('span', 'fs-badge warn', 'Airline-reported departure delay: ' + f.departure_delay_min + ' min'));
    c.appendChild(row);

    c.appendChild(predictionBlock(f.prediction, false));
    c.appendChild(el('div', 'fs-fine', 'Baggage information unavailable'));

    var act = el('div', 'fs-actions');
    var btn = el('button', 'btn btn-secondary btn-sm', 'View Details');
    btn.type = 'button';
    btn.disabled = !f.id;
    btn.addEventListener('click', function () { openDetails(f); });
    act.appendChild(btn);
    c.appendChild(act);
    return c;
  }

  function errorText(body, fallback) {
    return (body && body.error && body.error.message) || fallback;
  }

  form.addEventListener('submit', function (e) {
    e.preventDefault();
    var params = new URLSearchParams();
    new FormData(form).forEach(function (v, k) { if (String(v).trim()) params.set(k, String(v).trim()); });
    resultsEl.textContent = '';
    setStatus('Searching live flights…', null, true);
    var submit = document.getElementById('fs-submit'); submit.disabled = true;
    fetch('/api/flights/search?' + params.toString(), { headers: { 'Accept': 'application/json' } })
      .then(function (r) {
        if (r.status === 401 || r.redirected) { window.location = '/login?next=/flight-search'; return null; }
        return r.json().then(function (b) { return { ok: r.ok, body: b }; });
      })
      .then(function (res) {
        submit.disabled = false;
        if (!res) return;
        if (!res.ok) { setStatus(errorText(res.body, 'Search failed. Please try again.'), 'error'); return; }
        var n = res.body.count;
        setStatus(n + ' flight' + (n === 1 ? '' : 's') + ' found' +
          (res.body.meta.mode === 'schedule' ? ' (airline schedule — no live status yet)' : ''));
        showNotice(res.body.meta);
        res.body.flights.forEach(function (f) { resultsEl.appendChild(card(f)); });
      })
      .catch(function () {
        submit.disabled = false;
        setStatus('Could not reach the server. Check your connection and try again.', 'error');
      });
  });

  // ---------------------------------------------------------------- details
  function kv(dl, k, v) {
    dl.appendChild(el('dt', null, k));
    dl.appendChild(el('dd', null, (v === null || v === undefined || v === '') ? 'Not provided by the API' : String(v)));
  }
  function wxLine(w) {
    if (!w) return null;
    return Math.round(w.temp_f) + '°F, wind ' + w.wind_speed + ' mph, visibility ' + w.visibility + ' mi, precip ' + w.precip_in +
      ' in, pressure ' + w.pressure + ' inHg';
  }

  function closeModal() { modal.hidden = true; modalBody.textContent = ''; }
  document.getElementById('fs-modal-close').addEventListener('click', closeModal);
  modal.addEventListener('click', function (e) { if (e.target === modal) closeModal(); });
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape' && !modal.hidden) closeModal(); });

  function openDetails(summary) {
    modal.hidden = false;
    modalBody.textContent = '';
    modalBody.appendChild(el('p', null, 'Loading details…'));
    fetch('/api/flights/' + encodeURIComponent(summary.id) + '?include_price=1', { headers: { 'Accept': 'application/json' } })
      .then(function (r) { return r.json().then(function (b) { return { ok: r.ok, body: b }; }); })
      .then(function (res) {
        modalBody.textContent = '';
        if (!res.ok) { modalBody.appendChild(el('p', 'fs-unavail', errorText(res.body, 'Could not load details.'))); return; }
        render(res.body.flight, res.body.meta);
      })
      .catch(function () {
        modalBody.textContent = '';
        modalBody.appendChild(el('p', 'fs-unavail', 'Could not reach the server.'));
      });
  }

  function render(f, meta) {
    var h = el('h2', null, (f.airline_name || f.airline_iata || 'Flight') + ' ' + (f.flight_iata || ''));
    h.id = 'fs-modal-title';
    modalBody.appendChild(h);
    modalBody.appendChild(el('div', 'fs-city', (f.origin_iata || '') + ' → ' + (f.destination_iata || '')));
    if (meta && meta.notice) modalBody.appendChild(el('div', 'fs-notice', '⏱ ' + meta.notice));

    modalBody.appendChild(el('h3', null, 'Flight'));
    var dl = el('dl', 'fs-kv');
    kv(dl, 'Flight', f.flight_iata);
    kv(dl, 'Airline', f.airline_name);
    kv(dl, 'Route', (f.origin_iata || '?') + ' → ' + (f.destination_iata || '?'));
    kv(dl, 'Scheduled departure', f.scheduled_departure ? f.scheduled_departure.substr(0, 16).replace('T', ' ') : null);
    kv(dl, 'Scheduled arrival', f.scheduled_arrival ? f.scheduled_arrival.substr(0, 16).replace('T', ' ') : null);
    kv(dl, 'Estimated departure', f.estimated_departure ? f.estimated_departure.substr(0, 16).replace('T', ' ') : null);
    kv(dl, 'Actual departure', f.actual_departure ? f.actual_departure.substr(0, 16).replace('T', ' ') : null);
    kv(dl, 'Status', f.status);
    kv(dl, 'Aircraft', f.aircraft);
    kv(dl, 'Departure terminal / gate', [f.terminal_departure, f.gate_departure].filter(Boolean).join(' / ') || null);
    kv(dl, 'Arrival terminal / gate', [f.terminal_arrival, f.gate_arrival].filter(Boolean).join(' / ') || null);
    kv(dl, 'Baggage', f.baggage_belt ? 'Belt ' + f.baggage_belt + ' (allowance unavailable)' : 'Baggage information unavailable');
    modalBody.appendChild(dl);

    modalBody.appendChild(el('h3', null, 'Fare'));
    if (f.fare) {
      var fd = el('dl', 'fs-kv');
      kv(fd, 'Cheapest on this route', f.fare.display);
      kv(fd, 'Cheapest carrier', f.fare.airline);
      modalBody.appendChild(fd);
      modalBody.appendChild(el('div', 'fs-fine', f.fare.source + '. It is a route-level market fare and may not be this exact flight.'));
    } else {
      modalBody.appendChild(el('div', 'fs-unavail', 'No fare available (price lookup unconfigured, out of quota, or no result).'));
    }

    var p = f.prediction;
    modalBody.appendChild(el('h3', null, 'Weather'));
    var wd = el('dl', 'fs-kv');
    kv(wd, 'Departure (' + (f.origin_iata || '') + ')', p && p.weather ? wxLine(p.weather.origin) : null);
    kv(wd, 'Arrival (' + (f.destination_iata || '') + ')', p && p.weather ? wxLine(p.weather.destination) : null);
    modalBody.appendChild(wd);
    modalBody.appendChild(el('div', 'fs-fine', 'Forecast from Open-Meteo for the scheduled hour.'));

    modalBody.appendChild(el('h3', null, 'ML prediction'));
    modalBody.appendChild(predictionBlock(p, true));
    if (p && p.status === 'ok') {
      var inp = p.inputs, id = el('dl', 'fs-kv');
      modalBody.appendChild(el('h3', null, 'What the model was given'));
      kv(id, 'Airline / route', inp.airline + ' · ' + inp.origin + ' → ' + inp.destination);
      kv(id, 'Day / month', inp.weekday + ', ' + inp.month);
      kv(id, 'Scheduled hour', inp.scheduled_hour + ':00 local');
      kv(id, 'Public holiday', inp.holiday ? 'Yes' : 'No');
      kv(id, 'Scheduled duration', inp.scheduled_duration_min + ' min (' + inp.duration_source + ')');
      kv(id, 'Previous-leg delay', inp.previous_leg_delay);
      modalBody.appendChild(id);
      modalBody.appendChild(el('div', 'fs-fine',
        'These are the inputs, not feature importances; the model does not publish per-flight attributions, so none are claimed.'));
    }
  }
})();
