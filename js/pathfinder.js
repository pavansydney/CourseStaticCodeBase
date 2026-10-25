// "Find Your Path" recommender: pick ecosystem + profile, get a recommended certification
function setupPathFinder() {
    const ecoGroup = document.getElementById('path-eco');
    const levelGroup = document.getElementById('path-level');
    const result = document.getElementById('path-result');
    if (!ecoGroup || !levelGroup || !result) return;

    const state = { eco: null, level: null };

    // Recommendation map keyed by "ecosystem|profile"
    const recos = {
        'azure|beginner':       { name: 'AI-901 · Azure AI Fundamentals', why: 'The best starting point on Azure — no prior AI experience needed.', section: 'certifications', tab: 'panel-ai-901', provider: 'azure' },
        'azure|developer':      { name: 'AI-103 · Azure AI Engineer', why: 'Build AI apps and agents on Azure. Do AI-901 first if you are new to AI.', section: 'certifications', tab: 'panel-ai-103', provider: 'azure' },
        'azure|datascientist':  { name: 'AI-300 · ML Operations Engineer', why: 'Explore MLOps and GenAIOps with Azure Machine Learning and Microsoft Foundry.', section: 'certifications', tab: 'panel-ai-300', provider: 'azure' },
        'aws|beginner':         { name: 'AIF-C01 · AWS AI Practitioner', why: 'The foundational AWS AI cert — start here before the associate exams.', section: 'aws-certifications', tab: 'panel-aif', provider: 'aws' },
        'aws|developer':        { name: 'MLA-C01 · AWS ML Engineer', why: 'Operationalize ML on AWS. Try the Generative AI Developer cert next for app-building.', section: 'aws-certifications', tab: 'panel-mla', provider: 'aws' },
        'aws|datascientist':    { name: 'MLA-C01 · AWS ML Engineer', why: 'The current associate ML path on AWS (the MLS-C01 Specialty retired in 2026).', section: 'aws-certifications', tab: 'panel-mla', provider: 'aws' },
        'gcp|beginner':         { name: 'Generative AI Leader', why: 'A foundational, business-friendly intro to generative AI on Google Cloud.', section: 'gcp-certifications', tab: 'panel-genai-leader', provider: 'gcp' },
        'gcp|developer':        { name: 'Professional ML Engineer', why: 'Design and productionize ML on Google Cloud with Vertex AI.', section: 'gcp-certifications', tab: 'panel-gcp-mle', provider: 'gcp' },
        'gcp|datascientist':    { name: 'Professional ML Engineer', why: 'The core ML certification on Google Cloud for data-focused roles.', section: 'gcp-certifications', tab: 'panel-gcp-mle', provider: 'gcp' },
        'salesforce|beginner':      { name: 'Agentforce Specialist', why: 'Learn to build and deploy Salesforce Agentforce AI agents.', section: 'salesforce-certifications', tab: 'panel-agentforce', provider: 'salesforce' },
        'salesforce|developer':     { name: 'Agentforce Specialist', why: 'Build agents, prompts, and actions on the Salesforce Platform.', section: 'salesforce-certifications', tab: 'panel-agentforce', provider: 'salesforce' },
        'salesforce|datascientist': { name: 'Agentforce Specialist', why: 'Ground agents in Data Cloud and design effective prompts.', section: 'salesforce-certifications', tab: 'panel-agentforce', provider: 'salesforce' },
        'sap|beginner':      { name: 'C_AIG · SAP Generative AI Developer', why: 'Learn to build generative AI apps on SAP BTP with the Generative AI Hub and SAP AI Core.', section: 'sap-certifications', tab: 'panel-sap-caig', provider: 'sap' },
        'sap|developer':     { name: 'C_AIG · SAP Generative AI Developer', why: 'Build prompts, RAG, and orchestration on SAP BTP with SAP AI Core and HANA Cloud.', section: 'sap-certifications', tab: 'panel-sap-caig', provider: 'sap' },
        'sap|datascientist': { name: 'C_AIG · SAP Generative AI Developer', why: 'Ground models in enterprise data using embeddings and SAP HANA Cloud Vector Engine.', section: 'sap-certifications', tab: 'panel-sap-caig', provider: 'sap' },
        'unsure|beginner':      { name: 'AIF-C01 · AWS AI Practitioner', why: 'One foundational starting point. Compare it with Azure AI Fundamentals and choose the platform most relevant to your goals.', section: 'aws-certifications', tab: 'panel-aif', provider: 'aws' },
        'unsure|developer':     { name: 'MLA-C01 · AWS ML Engineer', why: 'A practical option for production ML. Prefer an Azure or Google Cloud track if that is what your team uses.', section: 'aws-certifications', tab: 'panel-mla', provider: 'aws' },
        'unsure|datascientist': { name: 'Google Cloud Professional ML Engineer', why: 'Explore an end-to-end ML engineering path. If you work with Microsoft tools, compare the Azure AI-300 MLOps track.', section: 'gcp-certifications', tab: 'panel-gcp-mle', provider: 'gcp' }
    };

    function selectIn(group, attr, value) {
        group.querySelectorAll('.path-opt').forEach(function (btn) {
            const active = btn.dataset[attr] === value;
            btn.classList.toggle('active', active);
            btn.setAttribute('aria-pressed', String(active));
        });
    }

    function render() {
        if (!state.eco || !state.level) return;
        const reco = recos[state.eco + '|' + state.level];
        if (!reco) return;
        const params = new URLSearchParams({ provider: reco.provider, panel: reco.tab, section: reco.section });
        result.hidden = false;
        result.innerHTML =
            '<div class="path-result-inner">' +
                '<span class="path-result-label">Recommended for you</span>' +
                '<h2 class="path-result-name">' + reco.name + '</h2>' +
                '<p class="path-result-why">' + reco.why + '</p>' +
                '<a class="btn btn-primary path-result-btn" href="certifications.html?' + params.toString() + '">Explore this certification &rarr;</a>' +
            '</div>';
        // Connect the recommendation to the roadmap below so the user sees
        // the matching provider's path without hunting for it.
        if (reco.provider) selectRoadmapProvider(reco.provider);
    }

    ecoGroup.querySelectorAll('.path-opt').forEach(function (btn) {
        btn.setAttribute('aria-pressed', 'false');
        btn.setAttribute('aria-controls', 'path-result');
        btn.addEventListener('click', function () {
            state.eco = btn.dataset.eco;
            selectIn(ecoGroup, 'eco', state.eco);
            render();
        });
    });
    levelGroup.querySelectorAll('.path-opt').forEach(function (btn) {
        btn.setAttribute('aria-pressed', 'false');
        btn.setAttribute('aria-controls', 'path-result');
        btn.addEventListener('click', function () {
            state.level = btn.dataset.level;
            selectIn(levelGroup, 'level', state.level);
            render();
        });
    });
}

// Roadmap provider tabs: show one provider's certification path at a time
function setupRoadmapTabs() {
    const tabs = document.getElementById('roadmapTabs');
    if (!tabs) return;
    const buttons = Array.from(tabs.querySelectorAll('.roadmap-tab'));
    buttons.forEach(function (tab, index) {
        const id = 'roadmap-' + tab.dataset.provider;
        const track = document.querySelector('.roadmap-track[data-provider="' + tab.dataset.provider + '"]');
        tab.id = id + '-tab';
        tab.setAttribute('role', 'tab');
        tab.setAttribute('aria-controls', id);
        track.id = id;
        track.setAttribute('role', 'tabpanel');
        track.setAttribute('aria-labelledby', tab.id);
        tab.addEventListener('click', function () {
            selectRoadmapProvider(tab.dataset.provider);
        });
        tab.addEventListener('keydown', function (event) {
            let next;
            if (event.key === 'ArrowRight') next = (index + 1) % buttons.length;
            if (event.key === 'ArrowLeft') next = (index - 1 + buttons.length) % buttons.length;
            if (event.key === 'Home') next = 0;
            if (event.key === 'End') next = buttons.length - 1;
            if (next === undefined) return;
            event.preventDefault();
            selectRoadmapProvider(buttons[next].dataset.provider);
            buttons[next].focus();
        });
    });
    selectRoadmapProvider('azure');
}

function selectRoadmapProvider(provider) {
    const tabs = document.getElementById('roadmapTabs');
    if (!tabs) return;
    tabs.querySelectorAll('.roadmap-tab').forEach(function (t) {
        const active = t.dataset.provider === provider;
        t.classList.toggle('is-active', active);
        t.setAttribute('aria-selected', String(active));
        t.tabIndex = active ? 0 : -1;
    });
    document.querySelectorAll('.roadmap-track').forEach(function (track) {
        const active = track.dataset.provider === provider;
        track.classList.toggle('is-active', active);
        track.hidden = !active;
    });
}

document.addEventListener('DOMContentLoaded', function () {
    setupPathFinder();
    setupRoadmapTabs();
});
