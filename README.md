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

Social sharing uses the 1200 by 630 raster image
[cloudvana-social.png](assets/cloudvana-social.png), rendered from the editable
[SVG source](assets/cloudvana-social.svg). Keep the image, alternative text, and
Open Graph dimensions aligned when changing it. The original logo remains the
site icon; filenames with encoded spaces are valid URLs, not a sharing failure.

The [public review status](about.html#content-standards) distinguishes automated
website checks from subject-matter review. Do not invent review dates, verified
exam alignment, affiliations, mentor credentials, or customer testimonials.
Brand permissions have not been established by the independence notice; review
each vendor's current terms before treating logo use as approved. Accounts,
marketing email collection, and commercial terms require separate owner
decisions and are not implemented by these disclosures.

* [Shared layout](js/layout.js) owns navigation and the footer.
* [Theme and storage](js/theme.js) applies the saved theme before rendering and
  reports unavailable browser storage.
* [Shared interactions](js/site.js) owns accessible topic tabs, provider filters,
  and certification deep links.
* [Learning runtime](script.js) renders modules, practice quizzes, lesson dialogs,
  and local completion records.
* [Course catalog](js/course-catalog.js) loads card metadata without downloading
  every lesson. Opening a module loads its course bundle once, on demand.
  [Certification content](js/certification-content.js) loads on its own page.
  Course files extend the same `courseData` object.
* [Website styles](styles.css) shares design tokens across every page and both
  themes. It also supports reduced motion and small screens.

Keep homepage module and certification counts aligned with the published tracks.
Archived course data remains available in the content files but is not included
in the published module count. Check provider documentation before changing
exam names, retirement notices, or objectives.

## Content quality contract

Every published module includes an authored `learningGuide` alongside its
lessons. It contains prerequisites and workspace setup, a worked example with
reasoning and expected results, a complete practice task with at least two test
cases, a hint and solution, an interview question with an example answer, and
observable readiness criteria. These guides are supplementary practice, not
extra lessons or independently graded assessments.

After editing course titles, descriptions, durations, lesson counts, or code
examples, run `npm run content:catalog`. The generated metadata must stay aligned
with its source bundles; tests reject a stale catalog. Catalog metadata is not a
second copy of lesson content. Failed lesson downloads are shown explicitly
with a retry action and do not alter saved progress.

The course page downloads metadata first and loads lesson bundles only when a
module is opened. The existing course and certification transfer caps remain in
place. The hiring page has a 140 KB decoded-script cap to accommodate concrete
tasks for both audiences across 15 companies; this includes shared scripts.

Keep examples specific to the module. State units, boundary behaviour,
assumptions, and limitations. A toy model or paper architecture exercise is not
evidence of production readiness. Cloud exercises should have a no-account
paper option; explain cost, permissions, and cleanup before live experiments.
Check vendor documentation before using real services or booking an exam.

The [student sequence](courses.html#student-route-title) starts with one
programming language, then ML fundamentals and a project, with role-specific
interview practice alongside it. Estimated reading times exclude practice.
Completion remains self-reported and does not guarantee an exam or job result.

Run `npm test` to check guide coverage across all 160 published modules,
assessment formatting, distinct tasks, and content invariants. The browser
suite opens every module and checks guide rendering, lesson counts, answer
disclosures, accessibility, and progress behaviour. These automated checks
protect structure and known examples; they do not replace subject-matter review
or validate every external API, vendor objective, or employment outcome.

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
