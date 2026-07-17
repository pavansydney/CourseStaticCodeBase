// Shared layout: injects the navbar and footer into every page so there is a
// single source of truth (no duplicated markup). Set the current page with
// <body data-page="home"> etc. Runs synchronously (scripts sit at end of body).
(function () {
    // Apply the saved (or system) theme as early as possible to reduce flicker.
    try {
        var savedTheme = localStorage.getItem('cloudvana-theme');
        if (!savedTheme) {
            savedTheme = (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches)
                ? 'dark'
                : 'light';
        }
        document.documentElement.setAttribute('data-theme', savedTheme);
    } catch (e) {
        /* localStorage unavailable — default to light */
    }

    var navHTML =
        '<nav class="navbar">' +
            '<div class="container">' +
                '<a href="index.html" class="nav-brand" aria-label="Cloudvana home">' +
                    '<img class="nav-logo" src="assets/Cloudvana%20logo.jpg" alt="Cloudvana">' +
                '</a>' +
                '<button class="nav-toggle" id="navToggle" aria-label="Toggle navigation menu" aria-expanded="false" aria-controls="navMenu">' +
                    '<span></span><span></span><span></span>' +
                '</button>' +
                '<ul class="nav-menu" id="navMenu">' +
                    '<li><a href="index.html" class="nav-link" data-nav="home">Home</a></li>' +
                    '<li><a href="courses.html" class="nav-link" data-nav="courses">Courses</a></li>' +
                    '<li class="nav-dropdown" data-nav="certifications">' +
                        '<a href="certifications.html" class="nav-dropdown-toggle">Certifications <i class="fas fa-chevron-down"></i></a>' +
                        '<ul class="nav-dropdown-menu">' +
                            '<li><a href="certifications.html?provider=azure" class="nav-link">Azure AI</a></li>' +
                            '<li><a href="certifications.html?provider=aws" class="nav-link">AWS AI</a></li>' +
                            '<li><a href="certifications.html?provider=gcp" class="nav-link">GCP AI</a></li>' +
                            '<li><a href="certifications.html?provider=salesforce" class="nav-link">Salesforce AI</a></li>' +
                        '</ul>' +
                    '</li>' +
                    '<li><a href="mentorship.html" class="nav-link" data-nav="mentorship">Mentorship</a></li>' +
                    '<li class="nav-dropdown" data-nav="resources">' +
                        '<a href="#" class="nav-dropdown-toggle">Resources <i class="fas fa-chevron-down"></i></a>' +
                        '<ul class="nav-dropdown-menu">' +
                            '<li><a href="prerequisites.html" class="nav-link" data-nav="prerequisites">Prerequisites</a></li>' +
                            '<li><a href="find-your-path.html" class="nav-link" data-nav="find-your-path">Find Your Path</a></li>' +
                            '<li><a href="about.html" class="nav-link" data-nav="about">About</a></li>' +
                        '</ul>' +
                    '</li>' +
                    '<li><button class="theme-toggle" id="themeToggle" type="button" aria-label="Toggle dark mode" title="Toggle light/dark theme"><i class="fas fa-moon"></i></button></li>' +
                '</ul>' +
            '</div>' +
        '</nav>';

    var footerHTML =
        '<footer class="footer">' +
            '<div class="container">' +
                '<div class="footer-content">' +
                    '<div class="footer-section">' +
                        '<h4>Cloudvana</h4>' +
                        '<p>Machine learning fundamentals and Azure, AWS, Google Cloud &amp; Salesforce AI certification prep for everyone</p>' +
                    '</div>' +
                    '<div class="footer-section">' +
                        '<h4>Links</h4>' +
                        '<ul>' +
                            '<li><a href="mentorship.html">Mentorship Program</a></li>' +
                            '<li><a href="https://developers.google.com/machine-learning/crash-course" target="_blank">Original ML Course</a></li>' +
                            '<li><a href="https://learn.microsoft.com/credentials/certifications/browse/?products=azure&subjects=artificial-intelligence" target="_blank">Azure AI Certifications</a></li>' +
                            '<li><a href="https://aws.amazon.com/certification/" target="_blank">AWS Certifications</a></li>' +
                            '<li><a href="https://cloud.google.com/learn/certification" target="_blank">Google Cloud Certifications</a></li>' +
                            '<li><a href="https://trailhead.salesforce.com/credentials/agentforcespecialist" target="_blank">Salesforce Agentforce Certification</a></li>' +
                        '</ul>' +
                    '</div>' +
                    '<div class="footer-section">' +
                        '<h4>Follow</h4>' +
                        '<div class="social-links">' +
                            '<a href="https://www.linkedin.com/in/chilakalapalli-p-141267119/" target="_blank" class="social-link"><i class="fab fa-linkedin"></i> LinkedIn</a>' +
                            '<a href="https://github.com/pavansydney" target="_blank" class="social-link"><i class="fab fa-github"></i> GitHub</a>' +
                        '</div>' +
                    '</div>' +
                '</div>' +
                '<div class="footer-bottom">' +
                    '<p>Founded by <a href="https://www.linkedin.com/in/chilakalapalli-p-141267119/" target="_blank">Chilakalapalli Pavan Kalyan</a></p>' +
                    '<p>&copy; 2024–2026 Cloudvana - Educational Content. Based on Google\'s ML Course, Microsoft Learn, AWS, Google Cloud &amp; Salesforce Trailhead.</p>' +
                '</div>' +
            '</div>' +
        '</footer>';

    var navMount = document.getElementById('site-nav');
    if (navMount) navMount.innerHTML = navHTML;

    var footMount = document.getElementById('site-footer');
    if (footMount) footMount.innerHTML = footerHTML;

    // Highlight the current page's nav item
    var current = (document.body && document.body.getAttribute('data-page')) || '';
    document.querySelectorAll('[data-nav]').forEach(function (el) {
        if (el.getAttribute('data-nav') !== current) return;
        if (el.classList.contains('nav-dropdown')) {
            var toggle = el.querySelector('.nav-dropdown-toggle');
            if (toggle) toggle.classList.add('active');
        } else {
            el.classList.add('active');
            // If this link lives inside a dropdown, mark its parent toggle active too.
            var parentDd = el.closest ? el.closest('.nav-dropdown') : null;
            if (parentDd) {
                var parentToggle = parentDd.querySelector('.nav-dropdown-toggle');
                if (parentToggle) parentToggle.classList.add('active');
            }
        }
    });

    // Wire up the light/dark theme toggle
    function currentTheme() {
        return document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
    }
    function setThemeIcon(btn) {
        if (!btn) return;
        btn.innerHTML = currentTheme() === 'dark'
            ? '<i class="fas fa-sun"></i>'
            : '<i class="fas fa-moon"></i>';
    }
    var themeBtn = document.getElementById('themeToggle');
    setThemeIcon(themeBtn);
    if (themeBtn) {
        themeBtn.addEventListener('click', function () {
            var next = currentTheme() === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', next);
            try {
                localStorage.setItem('cloudvana-theme', next);
            } catch (e) {
                /* preference just won't persist */
            }
            setThemeIcon(themeBtn);
        });
    }
})();
