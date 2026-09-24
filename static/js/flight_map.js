/*
 * Live traffic map for the flight-delay result.
 *
 * Draws the corridor between the two airports and the aircraft actually in that
 * airspace right now, from /api/live-traffic (which holds the OpenSky
 * credentials server-side - nothing here ever sees them).
 *
 * This is the one live question the receiver network can answer about a flight
 * that has not departed. It cannot say which airframe is assigned to it, so
 * nothing here is fed to the model - the map is context around the prediction,
 * not an input to it, and the caption says so.
 */
(function () {
  var el = document.getElementById("flight-map");
  if (!el || typeof L === "undefined") return;

  var note = document.getElementById("flight-map-note");
  var origin = el.dataset.origin;
  var dest = el.dataset.dest;
  if (!origin || !dest) return;

  var map = L.map(el, { scrollWheelZoom: false, attributionControl: true });
  // Keyless CARTO basemap: no API key reaches the browser, and the dark palette
  // matches the page instead of fighting it.
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    attribution:
      '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> ' +
      '&copy; <a href="https://carto.com/attributions">CARTO</a> · aircraft ' +
      '<a href="https://opensky-network.org">OpenSky Network</a>',
    subdomains: "abcd",
    maxZoom: 11,
  }).addTo(map);

  var planeLayer = L.layerGroup().addTo(map);
  var fitted = false;

  function airportMarker(p, label) {
    return L.circleMarker([p.lat, p.lon], {
      radius: 7,
      color: "#22d3ee",
      weight: 2,
      fillColor: "#0b2733",
      fillOpacity: 1,
    }).bindTooltip(label + " " + p.iata, { direction: "top" });
  }

  function planeIcon(ac) {
    // Rotate the glyph to the aircraft's true track so the map reads as traffic
    // with direction, not a scatter of dots.
    var colour = ac.is_yours ? "#fbbf24" : ac.on_ground ? "#7c8fa3" : "#22d3ee";
    var size = ac.is_yours ? 26 : 18;
    return L.divIcon({
      className: "plane-icon",
      iconSize: [size, size],
      iconAnchor: [size / 2, size / 2],
      html:
        '<svg viewBox="0 0 24 24" width="' + size + '" height="' + size + '" ' +
        'style="transform:rotate(' + (ac.heading || 0) + 'deg)" fill="' + colour + '">' +
        '<path d="M12 2c-.6 0-1 .9-1 2v5L2 14v2l9-2.5V19l-2.5 1.5V22L12 21l3.5 1.5v-1.5L13 19v-5.5L22 16v-2l-9-5V4c0-1.1-.4-2-1-2z"/>' +
        "</svg>",
    });
  }

  function describe(ac) {
    var bits = [];
    if (ac.altitude_ft !== null) bits.push(ac.altitude_ft.toLocaleString() + " ft");
    if (ac.speed_kt !== null) bits.push(ac.speed_kt + " kt");
    if (ac.on_ground) bits.push("on the ground");
    return (
      "<strong>" + (ac.callsign || ac.icao24) + "</strong>" +
      (ac.is_yours ? " — your flight" : "") +
      (bits.length ? "<br>" + bits.join(" · ") : "")
    );
  }

  function refresh() {
    var url =
      "/api/live-traffic?origin=" + encodeURIComponent(origin) +
      "&dest=" + encodeURIComponent(dest) +
      "&carrier=" + encodeURIComponent(el.dataset.carrier || "") +
      "&flight=" + encodeURIComponent(el.dataset.flight || "");

    fetch(url)
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (!data.route) {
          el.style.display = "none";
          if (note) note.textContent = data.note || "No map available for this route.";
          return;
        }

        if (!fitted) {
          var o = data.route.origin, d = data.route.destination;
          airportMarker(o, "Departing").addTo(map);
          airportMarker(d, "Arriving").addTo(map);
          L.polyline([[o.lat, o.lon], [d.lat, d.lon]], {
            color: "#22d3ee", weight: 2, opacity: 0.45, dashArray: "6 8",
          }).addTo(map);
          map.fitBounds(
            L.latLngBounds([[o.lat, o.lon], [d.lat, d.lon]]).pad(0.35)
          );
          fitted = true;
        }

        planeLayer.clearLayers();
        var yours = null;
        data.aircraft.forEach(function (ac) {
          var m = L.marker([ac.lat, ac.lon], { icon: planeIcon(ac) })
            .bindPopup(describe(ac));
          planeLayer.addLayer(m);
          if (ac.is_yours) yours = ac;
        });

        if (note) {
          var when = new Date().toLocaleTimeString([], {
            hour: "2-digit", minute: "2-digit",
          });
          note.innerHTML =
            "<strong>" + data.count + "</strong> aircraft over this corridor at " + when +
            " — real transponder positions from the OpenSky Network." +
            (yours
              ? " Your flight <strong>" + yours.callsign + "</strong> is airborne and shown in amber."
              : " The network reports aircraft already flying, so it cannot show the airframe " +
                "assigned to a flight that has not departed. This is context around the " +
                "prediction, not an input to it.");
        }
      })
      .catch(function () {
        if (note) note.textContent = "Live traffic is unavailable right now.";
      });
  }

  refresh();

  // Positions move, but every fetch spends from a shared daily credit budget -
  // so refresh slowly, and not at all while the tab is in the background.
  setInterval(function () {
    if (!document.hidden) refresh();
  }, 30000);
})();
