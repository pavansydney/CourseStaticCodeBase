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

The deployment is external to this repository. The replacement
[Apps Script source](apps-script/mentorship.gs) preserves the first ten sheet
columns, generates a server-side UUID reference, and sends branded HTML plus
plain-text email through Google MailApp. The sender is the Google account that
owns and executes the deployment, with display name `Cloudvana`. Student
acknowledgements use `cloud.vana@outlook.com` as Reply-To. Admin alerts also go
to that Outlook inbox, with the student's address as Reply-To. Outlook is not
the sender. No Microsoft application, OAuth library, client secret, or mailbox
password is needed. Copying or pushing repository changes does not deploy this
script.

The new response includes `reference: "CV-<32 uppercase hex characters>"` and
`emailStatus: "accepted"`, `"pending"`, or `"unconfirmed"`. A successful MailApp
call means submitted for sending, not proof of delivery. The page shows the reference and
distinguishes a saved request from an unconfirmed email. It remains compatible
with the old acknowledgement, without inventing a reference or claiming an
email was sent. Session dates and times are agreed manually, not booked by the
form.

### Configure Gmail sending and deploy the intake script

1. Back up your existing Apps Script and sheet. Use a copy of the sheet and a
   test deployment first. Do not paste credentials into source code or chat.
2. Replace the old Apps Script code with the replacement source. In
   **Project Settings > Script properties**, add `SPREADSHEET_ID`. Its value is
   the portion between `/d/` and `/edit` in the registration sheet's URL.
   Optionally set `SHEET_NAME`; its default is `Sheet1`. Keep script editor
   access restricted to trusted owners. Do not change the first ten columns.
3. Run `doGet` once from the Apps Script editor and grant the Google permissions
   requested for spreadsheet access and sending email. Run as the personal
   Google account you intend to use as sender. `doGet` is a health check, not a
   test email. No Outlook sign-in or authorization is required.
4. Deploy the web app to **execute as you**, with access appropriate for public
   website submissions (**Anyone**). Update the existing deployment to a new
   version to preserve its `/exec` URL. If you create a different deployment,
   update the endpoint in the page script.
5. Run the live checks below against the test deployment before updating
   production. The owner must grant Google permissions. Do not enable a public
   sender merely because mock tests passed. If you previously configured the
   abandoned Outlook integration, revoke that application's consent and remove
   its unused credentials, tokens, and OAuth library after backing up as needed.

See Google's [MailApp reference](https://developers.google.com/apps-script/reference/mail/mail-app)
for HTML, plain-text, display-name, and Reply-To options. This is a small-pilot
setup, not a bulk-email service.

### Check delivery and operate the pilot

* Submit a test request to an address you control, never a real student's
  address without their request. Check the browser reference matches column K,
  the student email subject/body, and the admin notification in Outlook.
* Inspect the recipient's actual From and Reply-To headers: From must be the
  deployment owner's Google address and Reply-To must be the Cloudvana Outlook
  address for the student email. Reply and confirm it arrives in Outlook.
  Check inbox/spam placement, HTML and plain-text views, and mobile rendering.
  Do not assume MailApp messages appear in Gmail's Sent folder; check the
  controlled recipient and intake statuses. Submission is not proof of delivery.
* Repeat the exact request within 24 hours. It must return the same reference
  without another sheet row or email. Duplicate matching uses the normalized
  complete payload, so changed requests are treated as new registrations.
* Test an invalid plan, missing consent, exhausted Google mail quota, and a
  mail-send failure. Failed email must leave the saved
  registration visible with an honest status, not ask the learner to pay or
  resubmit.
* Existing columns A-J retain their order. New columns K-O are Reference,
  Request fingerprint, Student email, Admin email, and Consent. If your sheet
  already has a header row, add those header labels manually; existing rows
  are not rewritten. Empty sheets receive the complete header row.
* Monitor the sheet and owner execution logs. `pending` can mean a request is
  still processing or the script was interrupted. `unconfirmed` requires
  manual review. No automatic resend occurs after an ambiguous failure:
  investigate the send and contact the learner using their reference if needed.
  A student-send failure also prevents the admin alert in that execution, so
  the sheet remains the source of truth.
* A script lock prevents concurrent duplicate inserts. Intake is capped at
  200 new registrations per rolling 24 hours, with at most two email submissions
  per registration. The honeypot and cap are basic safeguards, not bot
  verification; add verified anti-abuse controls before promoting widely.
  An unauthenticated form cannot prove the entered email belongs to its user.
  At capacity, requests are not saved and the page offers direct contact.
  The registration cap does not raise Google's sending quota.
* Google currently documents 100 email recipients per day for consumer Apps
  Script accounts. With one student email and one admin alert, that allows
  roughly 50 fully emailed registrations if no other scripts consume the
  account's quota. The script checks `MailApp.getRemainingDailyQuota()` before
  each send. Further requests can still be saved up to the 200-registration cap,
  but emails are marked `unconfirmed` when quota runs out. They are not queued
  for automatic delivery the next day; manual follow-up is required. If only
  one recipient remains, the student email is sent first and the admin alert
  may be unconfirmed. Verify current [Google quotas](https://developers.google.com/apps-script/guides/services/quotas)
  before relying on volume estimates.
* Keep the existing deployment available for rollback and do not commit
  secrets or student data.

The owner must verify deployment, CORS responses, processing, and email
delivery. Do not restore `no-cors` requests or report success without a readable
acknowledgement. Tests use mocked Google services; they do not
send real email or establish production readiness.

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
