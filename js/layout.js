(function () {
    'use strict';

    const navMount = document.getElementById('site-nav');
    const footerMount = document.getElementById('site-footer');
    const current = document.body.dataset.page;
    const external = 'target="_blank" rel="noopener noreferrer"';

    if (navMount) {
        navMount.innerHTML = `
            <header class="site-header">
                <nav class="navbar" aria-label="Primary navigation">
                    <div class="container nav-inner">
                        <a href="index.html" class="nav-brand" aria-label="Cloudvana home">
                            <img class="nav-logo" src="assets/Cloudvana%20logo.jpg" alt="Cloudvana" width="144" height="32" fetchpriority="high">
                        </a>
                        <ul class="nav-menu" id="navMenu">
                            <li><a href="index.html" class="nav-link" data-nav="home">Home</a></li>
                            <li><a href="courses.html" class="nav-link" data-nav="courses">Courses</a></li>
                            <li class="nav-dropdown">
                                <div class="nav-dropdown-row">
                                    <a href="certifications.html" class="nav-link" data-nav="certifications">Certifications</a>
                                    <button class="nav-dropdown-toggle nav-chevron" type="button" aria-label="Certification providers" aria-expanded="false" aria-controls="certMenu">
                                        <i class="fas fa-chevron-down" aria-hidden="true"></i>
                                    </button>
                                </div>
                                <ul class="nav-dropdown-menu" id="certMenu" hidden>
                                    <li><a href="certifications.html?provider=azure">Microsoft Azure</a></li>
                                    <li><a href="certifications.html?provider=aws">Amazon Web Services</a></li>
                                    <li><a href="certifications.html?provider=gcp">Google Cloud</a></li>
                                    <li><a href="certifications.html?provider=salesforce">Salesforce</a></li>
                                    <li><a href="certifications.html?provider=sap">SAP</a></li>
                                </ul>
                            </li>
                            <li><a href="mentorship.html" class="nav-link" data-nav="mentorship">Mentorship</a></li>
                            <li class="nav-dropdown">
                                <button class="nav-link nav-dropdown-toggle" type="button" aria-expanded="false" aria-controls="resourceMenu">
                                    Resources <i class="fas fa-chevron-down" aria-hidden="true"></i>
                                </button>
                                <ul class="nav-dropdown-menu" id="resourceMenu" hidden>
                                    <li><a href="find-your-path.html" data-nav="find-your-path">Find your path</a></li>
                                    <li><a href="prerequisites.html" data-nav="prerequisites">Prerequisites</a></li>
                                    <li><a href="hiring-procedures.html" data-nav="hiring-procedures">Interview playbooks</a></li>
                                    <li><a href="about.html" data-nav="about">About Cloudvana</a></li>
                                </ul>
                            </li>
                        </ul>
                        <div class="nav-actions">
                            <a class="btn btn-primary nav-cta" href="find-your-path.html">Find my path <span aria-hidden="true">&rarr;</span></a>
                            <button class="theme-toggle" id="themeToggle" type="button" aria-label="Switch to dark theme"></button>
                            <button class="nav-toggle" id="navToggle" type="button" aria-label="Open navigation" aria-expanded="false" aria-controls="navMenu">
                                <span></span><span></span><span></span>
                            </button>
                        </div>
                    </div>
                </nav>
            </header>
            <div class="site-notice" id="storageNotice" hidden>
                <p id="storageNoticeText" role="status"></p>
                <button type="button" aria-label="Dismiss storage notice">&times;</button>
            </div>`;
    }

    if (footerMount) {
        footerMount.innerHTML = `
            <footer class="footer">
                <div class="container">
                    <div class="footer-content">
                        <div class="footer-section footer-intro">
                            <a class="footer-brand" href="index.html">Cloudvana<span aria-hidden="true">.</span></a>
                            <p>A clearer path from curious to capable. Free learning resources for AI, software engineering, and cloud certifications.</p>
                            <div class="social-links">
                                <a href="https://www.linkedin.com/in/chilakalapalli-p-141267119/" ${external}><i class="fab fa-linkedin" aria-hidden="true"></i> LinkedIn</a>
                                <a href="https://github.com/pavansydney" ${external}><i class="fab fa-github" aria-hidden="true"></i> GitHub</a>
                            </div>
                        </div>
                        <div class="footer-section">
                            <h2>Learn</h2>
                            <ul>
                                <li><a href="courses.html">Learning tracks</a></li>
                                <li><a href="certifications.html">Cloud certifications</a></li>
                                <li><a href="prerequisites.html">Prerequisites</a></li>
                                <li><a href="https://developers.google.com/machine-learning/crash-course" ${external}>Google ML Crash Course</a></li>
                            </ul>
                        </div>
                        <div class="footer-section">
                            <h2>Take your next step</h2>
                            <ul>
                                <li><a href="find-your-path.html">Find your path</a></li>
                                <li><a href="hiring-procedures.html">Interview playbooks</a></li>
                                <li><a href="mentorship.html">Mentorship</a></li>
                                <li><a href="about.html">About Cloudvana</a></li>
                                <li><a href="about.html#privacy">Privacy &amp; your data</a></li>
                            </ul>
                        </div>
                        <div class="footer-section">
                            <h2>Official credentials</h2>
                            <ul>
                                <li><a href="https://learn.microsoft.com/credentials/certifications/" ${external}>Microsoft Learn</a></li>
                                <li><a href="https://aws.amazon.com/certification/" ${external}>AWS Certification</a></li>
                                <li><a href="https://cloud.google.com/learn/certification" ${external}>Google Cloud</a></li>
                                <li><a href="https://trailhead.salesforce.com/credentials/agentforcespecialist" ${external}>Salesforce Trailhead</a></li>
                                <li><a href="https://learning.sap.com/certifications/sap-certified-associate-sap-generative-ai-developer" ${external}>SAP Learning</a></li>
                            </ul>
                        </div>
                    </div>
                    <div class="footer-bottom">
                        <p>&copy; ${new Date().getFullYear()} Cloudvana. Founded by <a href="https://www.linkedin.com/in/chilakalapalli-p-141267119/" ${external}>Pavan Kalyan</a>.</p>
                        <p>Independent educational resources. Not affiliated with certification providers. Official exam fees are separate.</p>
                    </div>
                </div>
            </footer>`;
    }

    if (current === 'not-found') {
        [navMount, footerMount].filter(Boolean).forEach(mount => {
            mount.querySelectorAll('a[href], img[src]').forEach(element => {
                const attribute = element.tagName === 'IMG' ? 'src' : 'href';
                const value = element.getAttribute(attribute);
                if (!value.startsWith('https:') && !value.startsWith('/')) element.setAttribute(attribute, '/' + value);
            });
        });
    }

    document.querySelectorAll('[data-nav]').forEach(link => {
        if (link.dataset.nav !== current) return;
        link.classList.add('active');
        link.setAttribute('aria-current', 'page');
        const dropdown = link.closest('.nav-dropdown');
        if (dropdown) dropdown.querySelector('.nav-dropdown-toggle').classList.add('active');
    });

    const menu = document.getElementById('navMenu');
    const navToggle = document.getElementById('navToggle');
    const mobile = window.matchMedia('(max-width: 960px)');
    const dropdowns = Array.from(document.querySelectorAll('.nav-dropdown'));

    function setDropdown(dropdown, open) {
        dropdown.classList.toggle('open', open);
        dropdown.querySelector('.nav-dropdown-toggle').setAttribute('aria-expanded', String(open));
        dropdown.querySelector('.nav-dropdown-menu').hidden = !open;
    }

    function closeDropdowns() {
        dropdowns.forEach(dropdown => setDropdown(dropdown, false));
    }

    function closeNavigation() {
        if (!menu || !navToggle) return;
        menu.hidden = mobile.matches;
        menu.classList.remove('open');
        navToggle.classList.remove('open');
        navToggle.setAttribute('aria-expanded', 'false');
        navToggle.setAttribute('aria-label', 'Open navigation');
        closeDropdowns();
    }

    if (menu && navToggle) {
        closeNavigation();
        mobile.addEventListener('change', closeNavigation);
        navToggle.addEventListener('click', () => {
            const open = navToggle.getAttribute('aria-expanded') !== 'true';
            menu.hidden = !open;
            menu.classList.toggle('open', open);
            navToggle.classList.toggle('open', open);
            navToggle.setAttribute('aria-expanded', String(open));
            navToggle.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
            if (!open) closeDropdowns();
        });
        menu.querySelectorAll('a').forEach(link => link.addEventListener('click', closeNavigation));
    }

    dropdowns.forEach(dropdown => {
        const toggle = dropdown.querySelector('.nav-dropdown-toggle');
        toggle.addEventListener('click', () => {
            const open = toggle.getAttribute('aria-expanded') !== 'true';
            closeDropdowns();
            setDropdown(dropdown, open);
        });
        toggle.addEventListener('keydown', event => {
            if (event.key !== 'ArrowDown') return;
            event.preventDefault();
            closeDropdowns();
            setDropdown(dropdown, true);
            dropdown.querySelector('.nav-dropdown-menu a').focus();
        });
        dropdown.addEventListener('focusout', event => {
            if (!dropdown.contains(event.relatedTarget)) setDropdown(dropdown, false);
        });
    });

    document.addEventListener('click', event => {
        if (!event.target.closest('.navbar')) closeNavigation();
        else if (!event.target.closest('.nav-dropdown')) closeDropdowns();
    });
    document.addEventListener('keydown', event => {
        if (event.key !== 'Escape') return;
        const openDropdown = dropdowns.find(dropdown => dropdown.classList.contains('open'));
        if (openDropdown) {
            setDropdown(openDropdown, false);
            openDropdown.querySelector('.nav-dropdown-toggle').focus();
        } else if (navToggle && navToggle.getAttribute('aria-expanded') === 'true') {
            closeNavigation();
            navToggle.focus();
        }
    });

    const themeButton = document.getElementById('themeToggle');
    function updateThemeButton() {
        const dark = document.documentElement.dataset.theme === 'dark';
        themeButton.innerHTML = `<i class="fas fa-${dark ? 'sun' : 'moon'}" aria-hidden="true"></i>`;
        themeButton.setAttribute('aria-label', dark ? 'Switch to light theme' : 'Switch to dark theme');
        themeButton.setAttribute('title', themeButton.getAttribute('aria-label'));
    }
    if (themeButton) {
        updateThemeButton();
        themeButton.addEventListener('click', () => {
            const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
            document.documentElement.dataset.theme = next;
            window.CloudvanaStorage.set('cloudvana-theme-v2', next);
            updateThemeButton();
        });
    }

    const notice = document.getElementById('storageNotice');
    function showStorageNotice() {
        if (!notice || !window.CloudvanaStorage.warning) return;
        document.getElementById('storageNoticeText').textContent = window.CloudvanaStorage.warning;
        notice.hidden = false;
    }
    if (notice) notice.querySelector('button').addEventListener('click', () => { notice.hidden = true; });
    window.addEventListener('cloudvana:storage-warning', showStorageNotice);
    showStorageNotice();
})();
