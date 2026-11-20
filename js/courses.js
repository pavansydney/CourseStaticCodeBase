(function () {
    'use strict';

    const trackIds = ['courses', 'track-ai-eng', 'track-dsa-sd', 'track-os-net', 'track-programming-languages'];

    function showTrack(id, doScroll = true) {
        if (!trackIds.includes(id)) {
            document.getElementById('courseSearchStatus').textContent = 'That track is not available. Choose one of the five tracks below.';
            return;
        }
        trackIds.forEach(trackId => { document.getElementById(trackId).hidden = trackId !== id; });
        document.querySelectorAll('.track-card, .track-switch-link').forEach(button => {
            const active = button.dataset.track === id;
            button.classList.toggle('active', active);
            button.setAttribute('aria-pressed', String(active));
            button.setAttribute('aria-controls', button.dataset.track);
        });
        if (doScroll) {
            const url = new URL(window.location.href);
            url.searchParams.set('track', id);
            url.searchParams.delete('panel');
            url.hash = '';
            history.replaceState(null, '', url);
            scrollToSection(id);
        }
    }

    window.showTrack = showTrack;

    document.addEventListener('DOMContentLoaded', () => {
        const params = new URLSearchParams(window.location.search);
        const hash = window.location.hash.slice(1);
        const requested = params.get('track') || (trackIds.includes(hash) ? hash : '');
        const selected = trackIds.includes(requested) ? requested : 'courses';
        showTrack(selected, false);
        const requestedPanel = params.get('panel');
        if (requestedPanel) {
            const panel = document.getElementById(requestedPanel);
            const tab = Array.from(document.querySelectorAll('.section-tab')).find(button => button.dataset.target === requestedPanel);
            if (panel && panel.closest('section')?.id === selected && tab) {
                tab.click();
            } else {
                document.getElementById('courseSearchStatus').textContent = 'That topic is not available in this track. Choose a topic below.';
            }
        }
        if (requested && !trackIds.includes(requested)) {
            document.getElementById('courseSearchStatus').textContent = 'That track is not available. Showing Machine Learning; you can choose another track below.';
        }

        const input = document.getElementById('moduleSearch');
        const results = document.getElementById('courseSearchResults');
        const status = document.getElementById('courseSearchStatus');
        const cards = Array.from(document.querySelectorAll('.module-card'));

        input.addEventListener('input', () => {
            const query = input.value.trim().toLocaleLowerCase();
            results.replaceChildren();
            results.hidden = !query;
            status.textContent = '';
            if (!query) return;
            const matches = cards.filter(card => card.textContent.toLocaleLowerCase().includes(query));
            status.textContent = matches.length ? matches.length + ' matching module' + (matches.length === 1 ? '' : 's') + ' across all tracks.' : 'No modules found. Try a topic such as Python, networks, or regression.';
            matches.forEach(card => {
                const result = document.createElement('button');
                result.type = 'button';
                result.className = 'course-search-result';
                const title = document.createElement('strong');
                title.textContent = card.querySelector('.module-open').firstChild.textContent;
                const context = document.createElement('span');
                context.textContent = card.closest('section').querySelector('.section-title').textContent.trim() + ' / ' + card.querySelector('.module-number').textContent.trim();
                result.append(title, context);
                result.addEventListener('click', () => {
                    showTrack(card.closest('section').id, false);
                    const panel = card.closest('.tab-panel');
                    document.querySelector('.section-tab[data-target="' + panel.id + '"]').click();
                    card.querySelector('.module-open').click();
                });
                results.appendChild(result);
            });
        });
    });
})();
