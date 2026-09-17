// Auto-uppercase airport/carrier code inputs as the user types.
document.querySelectorAll('input[name="carrier_code"], input[name="origin_airport"], input[name="destination_airport"]')
  .forEach((el) => {
    el.addEventListener('input', () => {
      el.value = el.value.toUpperCase();
    });
  });

// Scroll the recommendation/result panel into view once it renders.
const resultPanel = document.querySelector('.recommendation-panel, .result-panel');
if (resultPanel && window.location.search.includes('scroll')) {
  resultPanel.scrollIntoView({ behavior: 'smooth' });
}

// Budget trip planner: click a suggested-hotel card to mark it as your pick (visual only -
// these have no verified price, so nothing to recalculate).
document.querySelectorAll('.hotel-suggestion-card').forEach((card) => {
  card.addEventListener('click', () => {
    document.querySelectorAll('.hotel-suggestion-card').forEach((c) => c.classList.remove('selected'));
    card.classList.add('selected');
  });
});
