/*
 * City picker for the trip form's location fields.
 *
 * Attaches to any <input data-place-autocomplete>. Typing queries /api/places,
 * which is restricted to real cities - so the traveller picks from a list of
 * actual places instead of submitting free text that might resolve to a shop.
 * ("newyork" used to match a hairdresser in Tokyo.)
 *
 * The input keeps working as a plain text field if JS or the network fails;
 * the form still submits whatever was typed.
 */
(function () {
  "use strict";

  var DEBOUNCE_MS = 250;
  var MIN_CHARS = 2;

  function attach(input) {
    var wrap = document.createElement("div");
    wrap.className = "ac-wrap";
    input.parentNode.insertBefore(wrap, input);
    wrap.appendChild(input);

    var list = document.createElement("ul");
    list.className = "ac-list";
    list.hidden = true;
    wrap.appendChild(list);

    var timer = null;
    var items = [];
    var active = -1;
    // Guards against a slow earlier request overwriting a newer one's results.
    var seq = 0;

    function close() {
      list.hidden = true;
      active = -1;
    }

    function choose(i) {
      if (i < 0 || i >= items.length) return;
      input.value = items[i].label;
      close();
    }

    function render() {
      list.innerHTML = "";
      if (!items.length) { close(); return; }
      items.forEach(function (item, i) {
        var li = document.createElement("li");
        li.textContent = item.label;
        li.className = i === active ? "ac-item ac-active" : "ac-item";
        // mousedown, not click: blur would hide the list before click fires
        li.addEventListener("mousedown", function (e) { e.preventDefault(); choose(i); });
        list.appendChild(li);
      });
      list.hidden = false;
    }

    function search() {
      var q = input.value.trim();
      if (q.length < MIN_CHARS) { items = []; close(); return; }
      var mine = ++seq;
      fetch("/api/places?q=" + encodeURIComponent(q))
        .then(function (r) { return r.ok ? r.json() : { results: [] }; })
        .then(function (data) {
          if (mine !== seq) return;      // a newer keystroke already won
          items = data.results || [];
          active = -1;
          render();
        })
        .catch(function () { /* leave the field usable as plain text */ });
    }

    input.setAttribute("autocomplete", "off");

    input.addEventListener("input", function () {
      clearTimeout(timer);
      timer = setTimeout(search, DEBOUNCE_MS);
    });

    input.addEventListener("keydown", function (e) {
      if (list.hidden || !items.length) return;
      if (e.key === "ArrowDown") {
        e.preventDefault(); active = (active + 1) % items.length; render();
      } else if (e.key === "ArrowUp") {
        e.preventDefault(); active = (active - 1 + items.length) % items.length; render();
      } else if (e.key === "Enter" && active >= 0) {
        e.preventDefault(); choose(active);      // don't submit the form mid-pick
      } else if (e.key === "Escape") {
        close();
      }
    });

    input.addEventListener("blur", function () { setTimeout(close, 120); });
  }

  document.querySelectorAll("input[data-place-autocomplete]").forEach(attach);
})();
