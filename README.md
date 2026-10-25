---
title: Cloudvana
description: Development, testing, and maintenance for the Cloudvana learning website.
---

## Develop locally

Cloudvana is a static HTML, CSS, and JavaScript website. There is no production
build step. Vercel serves the existing page URLs directly.

Use Node.js 20 or newer:

```console
npm install
npm run dev
```

Open `http://127.0.0.1:4173`. Stop the preview with Ctrl+C. Development dependencies
are only used for testing, not shipped to visitors.

## Run checks

```console
npm test
npx playwright install chromium
npm run test:browser
```

The browser suite starts its own local preview when needed. It checks desktop
and mobile layouts, light and dark themes, automated accessibility rules,
keyboard navigation, learning progress, quizzes, recommendation links, search,
diagram loading and text fallbacks, and mentorship submission states. Every
published module is opened and checked against its displayed lesson count.
Form tests mock the submission service; they never submit applications to the
live endpoint. Development manifests, test files, and the local preview server
are excluded from static deployments.

Automated accessibility checks are a baseline, not a claim of full WCAG
conformance. Review keyboard and screen-reader behavior when changing controls.

## Maintain the website

* [Shared layout](js/layout.js) owns navigation and the footer.
* [Theme and storage](js/theme.js) applies the saved theme before rendering and
  reports unavailable browser storage.
* [Shared interactions](js/site.js) owns accessible topic tabs, provider filters,
  and certification deep links.
* [Learning runtime](script.js) renders modules, practice quizzes, lesson dialogs,
  and local completion records.
* [Machine learning content](js/ml-content.js) and
  [certification content](js/certification-content.js) load only on their relevant
  pages. Other course files extend the same `courseData` object.
* [Website styles](styles.css) shares design tokens across every page and both
  themes. It also supports reduced motion and small screens.

Keep homepage module and certification counts aligned with the published tracks.
Archived course data remains available in the content files but is not included
in the published module count. Check provider documentation before changing
exam names, retirement notices, or objectives.

## Mentorship submission contract

[The mentorship form](mentorship.html) uses the existing Google Apps Script
endpoint configured in [its page script](js/mentorship.js).

The service must accept a form-encoded POST and return a successful HTTP response
that the browser can read through CORS. An acknowledgement must be JSON with
`success: true`, `ok: true`, or `status: "success"`. Other response shapes, opaque
responses, network failures, and timeouts are shown as unconfirmed delivery.
The form retains entered data when delivery cannot be confirmed.

The submission service is external to this repository. Its deployment, response
headers, processing, and email delivery must be verified by its owner. Do not
restore `no-cors` requests or report success without a readable acknowledgement.

No payment is taken by this website. Free orientation and focused mentorship at
149 INR per session remain separate from free learning content and official
certification exam fees.

## Privacy and completion records

After completing a track's requirements, choose **Issue Certificate** or
**Claim Readiness Badge**, enter the name on the page, then choose **Create
certificate** or **Create readiness badge**. Issuance does not depend on browser
popups. Empty names are rejected inline, and cancelling does not create a record.

Theme, module progress, best quiz scores, and completion records are stored in
the current browser, not synced to an account. If storage fails, the website
shows a notice and supports the current page in memory.

Cloudvana completion records are self-reported learning milestones, not official
vendor credentials or guarantees of exam readiness. Public-facing details about
data use and provider independence are on the [About page](about.html#privacy).
