function prefersReducedMotion() {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

function scrollToSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) section.scrollIntoView({ behavior: prefersReducedMotion() ? 'instant' : 'smooth', block: 'start' });
}

function enhanceExternalLinks(root = document) {
    root.querySelectorAll('a[target="_blank"]').forEach(link => {
        link.rel = 'noopener noreferrer';
        if (!link.hasAttribute('aria-label')) {
            link.setAttribute('aria-label', link.textContent.trim() + ' (opens in a new tab)');
        }
    });
    root.querySelectorAll('i.fas, i.far, i.fab').forEach(icon => icon.setAttribute('aria-hidden', 'true'));
}

function setupTabs() {
    document.querySelectorAll('.section-tabs').forEach(tabBar => {
        const tabs = Array.from(tabBar.querySelectorAll('.section-tab[data-target]'));
        if (!tabs.length) return;
        const section = tabBar.closest('section');
        tabBar.setAttribute('role', 'tablist');
        tabBar.setAttribute('aria-label', section.querySelector('.section-title')?.textContent.trim() || 'Topics');

        function selectTab(selected) {
            tabs.forEach(tab => {
                const active = tab === selected;
                const panel = document.getElementById(tab.dataset.target);
                tab.classList.toggle('active', active);
                tab.setAttribute('aria-selected', String(active));
                tab.tabIndex = active ? 0 : -1;
                panel.classList.toggle('active', active);
                panel.hidden = !active;
            });
        }

        tabs.forEach((tab, index) => {
            const panel = document.getElementById(tab.dataset.target);
            tab.id = tab.dataset.target + '-tab';
            tab.type = 'button';
            tab.setAttribute('role', 'tab');
            tab.setAttribute('aria-controls', panel.id);
            panel.setAttribute('role', 'tabpanel');
            panel.setAttribute('aria-labelledby', tab.id);
            tab.addEventListener('click', () => selectTab(tab));
            tab.addEventListener('keydown', event => {
                let next;
                if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
                if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
                if (event.key === 'Home') next = 0;
                if (event.key === 'End') next = tabs.length - 1;
                if (next === undefined) return;
                event.preventDefault();
                selectTab(tabs[next]);
                tabs[next].focus();
            });
        });
        selectTab(tabs.find(tab => tab.classList.contains('active')) || tabs[0]);
    });
}

const CERT_PROVIDERS = ['all', 'azure', 'aws', 'gcp', 'salesforce', 'sap'];

function applyProviderFilter(provider, updateURL = false) {
    const selected = CERT_PROVIDERS.includes(provider) ? provider : 'all';
    document.querySelectorAll('.cert-filter .filter-btn[data-provider]').forEach(button => {
        const active = button.dataset.provider === selected;
        button.classList.toggle('active', active);
        button.setAttribute('aria-pressed', String(active));
    });
    const sections = Array.from(document.querySelectorAll('section[data-provider]'));
    sections.forEach(section => { section.hidden = selected !== 'all' && section.dataset.provider !== selected; });
    const status = document.getElementById('certificationStatus');
    if (status) {
        const count = sections.filter(section => !section.hidden).length;
        status.textContent = selected === 'all' ? 'Showing all 12 certification tracks across 5 providers.' : 'Showing ' + count + ' provider. Choose a certification below.';
    }
    if (updateURL) {
        const url = new URL(window.location.href);
        if (selected === 'all') url.searchParams.delete('provider');
        else url.searchParams.set('provider', selected);
        url.searchParams.delete('panel');
        url.searchParams.delete('section');
        history.replaceState(null, '', url);
    }
}

function applyCertQuery() {
    if (document.body.dataset.page !== 'certifications') return;
    const params = new URLSearchParams(window.location.search);
    const requestedProvider = params.get('provider');
    const aliases = { 'panel-dp-100': 'panel-ai-300', 'panel-ai-900': 'panel-ai-901', 'panel-ai-102': 'panel-ai-103' };
    const requestedPanel = params.get('panel');
    const panelId = aliases[requestedPanel] || requestedPanel;
    const panel = panelId ? document.getElementById(panelId) : null;
    const panelSection = panel?.matches('.tab-panel') ? panel.closest('section[data-provider]') : null;
    const provider = panelSection ? panelSection.dataset.provider : requestedProvider;
    applyProviderFilter(provider || 'all');
    if (panelSection) {
        const tab = Array.from(panelSection.querySelectorAll('.section-tab')).find(button => button.dataset.target === panelId);
        if (tab) tab.click();
    }
    const section = document.getElementById(params.get('section'));
    if (section?.matches('section[data-provider]') && !section.hidden) {
        requestAnimationFrame(() => scrollToSection(section.id));
    }
    if ((requestedProvider && !CERT_PROVIDERS.includes(requestedProvider)) || (requestedPanel && !panelSection)) {
        const status = document.getElementById('certificationStatus');
        status.textContent = 'That selection is not available. Browse the certification tracks below.';
    }
}

function goToCert(sectionId, tabTarget, provider) {
    const params = new URLSearchParams({ provider, panel: tabTarget, section: sectionId });
    window.location.href = 'certifications.html?' + params.toString();
}

document.addEventListener('DOMContentLoaded', () => {
    setupTabs();
    enhanceExternalLinks();
    if (document.body.dataset.page === 'certifications') {
        document.querySelectorAll('.cert-filter .filter-btn[data-provider]').forEach(button => {
            button.addEventListener('click', () => applyProviderFilter(button.dataset.provider, true));
        });
        applyCertQuery();
    }
});
