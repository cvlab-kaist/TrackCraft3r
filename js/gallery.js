// Click-to-play gallery for TrackCraft3R demos.
// Each thumbnail has data-src (mp4) and data-label. Clicking swaps the main player.

document.addEventListener('DOMContentLoaded', () => {
  const player = document.getElementById('main-player');
  const video = player.querySelector('video');
  const overlay = player.querySelector('.overlay-info');
  const thumbs = document.querySelectorAll('.thumb');

  const setLoading = (on) => {
    player.classList.toggle('loading', on);
  };

  const switchTo = (btn, autoplay = true) => {
    const src = btn.dataset.src;
    const label = btn.dataset.label || '';
    if (!src) return;

    thumbs.forEach(t => t.classList.remove('active'));
    btn.classList.add('active');

    setLoading(true);
    video.oncanplay = () => setLoading(false);
    video.onerror   = () => setLoading(false);

    // briefly delay so the loader can fade in even on cached files
    setTimeout(() => {
      video.poster = btn.dataset.poster || '';
      video.src = src;
      overlay.textContent = label;
      video.load();
      if (autoplay) {
        const p = video.play();
        if (p && p.catch) p.catch(() => {/* autoplay may be blocked; user can click */});
      }
    }, 80);
  };

  thumbs.forEach(btn => btn.addEventListener('click', () => switchTo(btn)));

  // Initialize with the first thumbnail (don't autoplay until user has interacted
  // to keep the page lightweight; the poster shows in the player).
  if (thumbs.length) {
    const first = thumbs[0];
    first.classList.add('active');
    overlay.textContent = first.dataset.label || '';
    video.poster = first.dataset.poster || '';
    video.src = first.dataset.src;
    video.load();
    // attempt autoplay (muted): most browsers allow muted autoplay
    const p = video.play();
    if (p && p.catch) p.catch(() => { /* swallow */ });
  }
});
