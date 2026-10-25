const { test, expect } = require('@playwright/test');
const AxeBuilder = require('@axe-core/playwright').default;

const pages = ['index.html', 'courses.html', 'certifications.html', 'find-your-path.html', 'prerequisites.html', 'hiring-procedures.html', 'mentorship.html', 'about.html'];

for (const filename of pages) {
    test(filename + ' is responsive, accessible, and error-free in both themes', async ({ page }, testInfo) => {
        const errors = [];
        page.on('pageerror', error => errors.push(error.message));
        const response = await page.goto('/' + filename);
        expect(response.status()).toBe(200);
        await expect(page.locator('main')).toHaveCount(1);
        await expect(page.locator('h1')).toHaveCount(1);
        await expect(page.locator('.skip-link')).toHaveAttribute('href', '#main-content');
        await expect(page.locator('[aria-current="page"]')).toHaveCount(1);

        const duplicates = await page.locator('[id]').evaluateAll(elements => {
            const ids = elements.map(element => element.id);
            return ids.filter((id, index) => ids.indexOf(id) !== index);
        });
        expect(duplicates).toEqual([]);
        const moduleIds = await page.locator('.module-card').evaluateAll(cards => cards.map(card => card.dataset.moduleId));
        expect(new Set(moduleIds).size, 'Every module must track progress independently').toBe(moduleIds.length);
        const brokenImages = await page.locator('img').evaluateAll(images => images.filter(image => image.complete && !image.naturalWidth).map(image => image.getAttribute('src')));
        expect(brokenImages).toEqual([]);

        for (const width of [320, 390, 768, 1440]) {
            await page.setViewportSize({ width, height: 960 });
            const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
            expect(overflow, filename + ' at ' + width + 'px').toBeLessThanOrEqual(1);
        }
        await page.setViewportSize(testInfo.project.use.viewport);
        await page.evaluate(() => document.fonts.ready);

        for (const theme of ['light', 'dark']) {
            if (theme === 'dark') await page.getByRole('button', { name: 'Switch to dark theme' }).click();
            await expect(page.locator('html')).toHaveAttribute('data-theme', theme);
            const audit = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa', 'wcag22aa', 'best-practice']).analyze();
            const violations = audit.violations.map(item => ({ id: item.id, impact: item.impact, targets: item.nodes.map(node => node.target) }));
            if (violations.length) await testInfo.attach(filename + '-' + theme + '-accessibility', { body: JSON.stringify(violations, null, 2), contentType: 'application/json' });
            expect(violations, theme + ' accessibility').toEqual([]);
        }
        await page.reload();
        await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
        await expect(page.getByRole('button', { name: 'Switch to light theme' })).toBeVisible();
        await page.keyboard.press('Escape');
        expect(errors).toEqual([]);
    });
}

test('keyboard navigation and mobile disclosures preserve focus and state', async ({ page }) => {
    await page.goto('/');
    await page.keyboard.press('Tab');
    await expect(page.getByRole('link', { name: 'Skip to main content' })).toBeFocused();
    await page.keyboard.press('Enter');
    await expect(page.locator('main')).toBeFocused();
    const mobile = page.viewportSize().width <= 960;
    if (mobile) await page.getByRole('button', { name: 'Open navigation' }).click();
    const resources = page.locator('button[aria-controls="resourceMenu"]');
    await resources.focus();
    await page.keyboard.press('ArrowDown');
    await expect(resources).toHaveAttribute('aria-expanded', 'true');
    await expect(page.locator('#resourceMenu a').first()).toBeFocused();
    await page.keyboard.press('Escape');
    await expect(resources).toBeFocused();
    await expect(page.locator('#resourceMenu')).toBeHidden();
    if (mobile) {
        await page.keyboard.press('Escape');
        await expect(page.getByRole('button', { name: 'Open navigation' })).toBeFocused();
        await expect(page.locator('#navMenu')).toBeHidden();
    }
});

test('course tabs, search, native dialogs, and completion tracking work together', async ({ page }) => {
    await page.goto('/courses.html');
    await expect(page.locator('.module-card')).toHaveCount(77);
    const tabs = page.locator('#courses [role="tab"]');
    await tabs.first().focus();
    await page.keyboard.press('ArrowRight');
    await expect(tabs.nth(1)).toHaveAttribute('aria-selected', 'true');
    await expect(page.locator('#panel-data')).toBeVisible();
    await page.keyboard.press('Home');
    await expect(tabs.first()).toHaveAttribute('aria-selected', 'true');

    const opener = page.locator('#ml-models-grid .module-open').first();
    await opener.focus();
    await page.keyboard.press('Enter');
    const dialog = page.getByRole('dialog');
    await expect(dialog).toBeVisible();
    await expect(page.getByRole('button', { name: 'Close module', exact: true })).toBeFocused();
    const headings = dialog.locator('.content-header');
    await expect(headings.first()).toHaveAttribute('aria-expanded', 'true');
    await dialog.getByRole('button', { name: 'Collapse All' }).click();
    await expect(headings.first()).toHaveAttribute('aria-expanded', 'false');
    await headings.first().focus();
    await page.keyboard.press('Space');
    await expect(headings.first()).toHaveAttribute('aria-expanded', 'true');
    await dialog.getByRole('button', { name: 'Mark as complete' }).click();
    await expect(dialog.getByRole('button', { name: 'Completed', exact: true })).toHaveAttribute('aria-pressed', 'true');
    const dialogAudit = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa', 'wcag22aa']).analyze();
    expect(dialogAudit.violations.map(item => ({ id: item.id, targets: item.nodes.map(node => node.target) }))).toEqual([]);
    await page.keyboard.press('Escape');
    await expect(dialog).not.toBeVisible();
    await expect(opener).toBeFocused();
    await expect(page.locator('#ml-models-grid .module-card').first()).toHaveClass(/read/);

    await page.reload();
    await expect(page.locator('#ml-models-grid .module-card').first()).toHaveClass(/read/);
    await page.getByLabel('Find a module across all tracks').fill('  JAVA CORE  ');
    await expect(page.locator('.course-search-result')).toHaveCount(1);
    await page.locator('.course-search-result').click();
    await expect(dialog.getByRole('heading', { name: 'Java Core and Toolchain', exact: true })).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(page.locator('#track-programming-languages')).toBeVisible();
    await page.getByLabel('Find a module across all tracks').fill('no-match-topic-xy123');
    await expect(page.locator('#courseSearchStatus')).toContainText('No modules found');
});

test('direct course links select the right track', async ({ page }) => {
    await page.goto('/courses.html?track=track-ai-eng');
    await expect(page.locator('#track-ai-eng')).toBeVisible();
    await expect(page.locator('#courses')).toBeHidden();
    await expect(page.locator('.track-card[data-track="track-ai-eng"]')).toHaveAttribute('aria-pressed', 'true');
});

test('duplicate legacy GCP labels no longer share progress', async ({ page }) => {
    await page.goto('/certifications.html?provider=gcp&panel=panel-gcp-mle');
    const cards = page.locator('#gcp-mle-grid .module-card');
    await expect(cards).toHaveCount(8);
    await expect(cards.nth(6).locator('.module-number')).toContainText('Module 7');
    await expect(cards.nth(7).locator('.module-number')).toContainText('Module 8');
    await cards.nth(4).locator('.module-open').click();
    await page.getByRole('button', { name: 'Mark as complete' }).click();
    await page.keyboard.press('Escape');
    await expect(cards.nth(4)).toHaveClass(/read/);
    await expect(cards.nth(6)).not.toHaveClass(/read/);
});

test('all eighteen recommendations target existing matching certification panels', async ({ page, context }) => {
    await page.goto('/find-your-path.html');
    const certifications = await context.newPage();
    await certifications.goto('/certifications.html');
    for (const eco of ['azure', 'aws', 'gcp', 'salesforce', 'sap', 'unsure']) {
        for (const level of ['beginner', 'developer', 'datascientist']) {
            await page.locator('[data-eco="' + eco + '"]').click();
            await page.locator('[data-level="' + level + '"]').click();
            await expect(page.locator('#path-result')).toBeVisible();
            const href = await page.locator('.path-result-btn').getAttribute('href');
            const url = new URL(href, 'http://127.0.0.1:4173/');
            const panel = certifications.locator('#' + url.searchParams.get('panel'));
            await expect(panel).toHaveCount(1);
            const provider = await panel.evaluate(element => element.closest('section').dataset.provider);
            expect(provider).toBe(url.searchParams.get('provider'));
            await expect(page.locator('.roadmap-tab.is-active')).toHaveAttribute('data-provider', provider);
        }
    }
    await certifications.close();
    await page.locator('[data-eco="azure"]').click();
    await page.locator('[data-level="datascientist"]').click();
    await page.locator('.path-result-btn').click();
    await expect(page.locator('#panel-ai-300')).toBeVisible();
    await expect(page.locator('.cert-filter [data-provider="azure"]')).toHaveAttribute('aria-pressed', 'true');
});

test('invalid and legacy certification links remain usable', async ({ page }) => {
    await page.goto('/certifications.html?provider=bad&panel=%22%5D');
    await expect(page.locator('#certificationStatus')).toContainText('not available');
    await expect(page.locator('section[data-provider]:visible')).toHaveCount(5);
    await page.goto('/certifications.html?provider=azure&panel=panel-dp-100');
    await expect(page.locator('#panel-ai-300')).toBeVisible();
    await page.locator('.cert-filter [data-provider="aws"]').click();
    await expect(page.locator('section[data-provider]:visible')).toHaveCount(1);
    await expect(page.locator('#aws-certifications')).toBeVisible();
    await expect(page).toHaveURL(/provider=aws/);
});

test('practice quiz scores only persist after completion and can be retried', async ({ page }) => {
    await page.goto('/certifications.html?provider=azure');
    const quiz = page.locator('#panel-ai-901 .quiz-section');
    const questions = quiz.locator('.quiz-q');
    const first = questions.first();
    await first.locator('.quiz-opt').first().click();
    await expect(first.locator('.quiz-option-result')).toHaveCount(2);
    expect(await page.evaluate(() => localStorage.getItem('cloudvana-quiz-panel-ai-901'))).toBeNull();
    await quiz.getByRole('button', { name: 'Reset answers' }).click();
    for (const question of await questions.all()) {
        const answer = Number(await question.getAttribute('data-answer'));
        await question.locator('.quiz-opt').nth(answer).click();
    }
    const count = await questions.count();
    await expect(quiz.locator('.quiz-score')).toContainText('Score: ' + count + '/' + count);
    expect(await page.evaluate(() => localStorage.getItem('cloudvana-quiz-panel-ai-901'))).toBe(String(count));
    await quiz.getByRole('button', { name: 'Reset answers' }).click();
    await expect(quiz.locator('.quiz-q.answered')).toHaveCount(0);
    await expect(quiz.locator('.quiz-score')).toContainText('Best: ' + count + '/' + count);
});

test('company search, filters, reset, and disclosures remain keyboard usable', async ({ page }) => {
    await page.goto('/hiring-procedures.html');
    await expect(page.locator('.hiring-card')).toHaveCount(15);
    await page.locator('[data-audience="experienced"]').click();
    await expect(page.locator('[data-audience="experienced"]')).toHaveAttribute('aria-pressed', 'true');
    await page.locator('[data-group="consulting"]').click();
    await expect(page.locator('[data-group="consulting"]')).toHaveAttribute('aria-pressed', 'true');
    await page.locator('#companySearch').fill('nonexistent-company-xy123');
    await expect(page.locator('.hiring-card')).toHaveCount(0);
    await expect(page.locator('.hiring-empty')).toBeVisible();
    await page.getByRole('button', { name: 'Reset Filters' }).click();
    await expect(page.locator('.hiring-card')).toHaveCount(15);
    await expect(page.locator('#companySearch')).toBeFocused();
    await expect(page.locator('[data-audience="fresher"]')).toHaveAttribute('aria-pressed', 'true');
    await page.locator('#companySearch').fill('MSFT');
    await expect(page.locator('.hiring-card')).toHaveCount(1);
    const summary = page.locator('.hiring-disclosure summary').first();
    await summary.focus();
    await page.keyboard.press('Space');
    await expect(page.locator('.hiring-disclosure').first()).toHaveAttribute('open', '');
});

async function fillMentorship(page) {
    await page.goto('/mentorship.html');
    await page.locator('[data-mentorship-plan="focused-149"]').click();
    await expect(page.locator('select[name="plan"]')).toHaveValue('focused-149');
    await page.locator('input[name="name"]').fill('Test Learner');
    await page.locator('input[name="email"]').fill('learner@example.com');
    await page.locator('input[name="course"]').fill('Python');
    await page.locator('select[name="currentLevel"]').selectOption('beginner');
    await page.locator('select[name="weeklyTime"]').selectOption('5-8');
    await page.locator('select[name="supportNeed"]').selectOption('direction');
    await page.locator('textarea[name="goal"]').fill('Build a consistent study plan.');
    await page.locator('#mentorshipConsent').check();
}

test('mentorship confirms only acknowledged requests and sends the intended payload', async ({ page }) => {
    let submitted;
    await page.route('https://script.google.com/**', async route => {
        submitted = { method: route.request().method(), body: route.request().postData() };
        await route.fulfill({ status: 200, contentType: 'application/json', headers: { 'Access-Control-Allow-Origin': '*' }, body: '{"success":true}' });
    });
    await fillMentorship(page);
    await page.locator('#mentorshipForm button[type="submit"]').click();
    await expect(page.locator('#mentorshipFormMsg')).toHaveAttribute('data-state', 'success');
    await expect(page.locator('input[name="name"]')).toHaveValue('');
    expect(submitted.method).toBe('POST');
    const params = new URLSearchParams(submitted.body);
    expect(params.get('name')).toBe('Test Learner');
    expect(params.get('plan')).toBe('focused-149');
    expect(params.get('consent')).toBe('true');
    expect(params.get('sourcePage')).toBe('mentorship');
    await expect(page.locator('#mentorshipForm button[type="submit"]')).toBeEnabled();
});

for (const failure of ['unacknowledged', 'network', 'invalid-json']) {
    test('mentorship preserves entries on ' + failure + ' delivery', async ({ page }) => {
        await page.route('https://script.google.com/**', async route => {
            if (failure === 'network') await route.abort('failed');
            else await route.fulfill({ status: 200, contentType: 'application/json', headers: { 'Access-Control-Allow-Origin': '*' }, body: failure === 'invalid-json' ? '<html>Not an acknowledgement</html>' : '{"success":false}' });
        });
        await fillMentorship(page);
        await page.locator('#mentorshipForm button[type="submit"]').click();
        await expect(page.locator('#mentorshipFormMsg')).toHaveAttribute('data-state', 'error');
        await expect(page.locator('#mentorshipFormMsg')).toContainText('could not confirm delivery');
        await expect(page.locator('input[name="name"]')).toHaveValue('Test Learner');
        await expect(page.locator('#mentorshipForm button[type="submit"]')).toBeEnabled();
        await expect(page.locator('#mentorshipForm')).not.toHaveAttribute('aria-busy');
    });
}

test('blocked storage is visible and does not prevent learning', async ({ page }) => {
    await page.addInitScript(() => Object.defineProperty(window, 'localStorage', { get() { throw new DOMException('Storage is blocked', 'SecurityError'); } }));
    await page.goto('/courses.html');
    await expect(page.locator('#storageNotice')).toBeVisible();
    await page.locator('#ml-models-grid .module-open').first().click();
    await page.getByRole('button', { name: 'Mark as complete' }).click();
    await expect(page.getByRole('button', { name: 'Completed', exact: true })).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(page.locator('#ml-models-grid .module-card').first()).toHaveClass(/read/);
});

test('initial JavaScript transfer stays within page-specific budgets', async ({ page }) => {
    const budgets = { 'index.html': 35000, 'courses.html': 1000000, 'certifications.html': 750000, 'find-your-path.html': 45000, 'hiring-procedures.html': 115000, 'mentorship.html': 40000, 'about.html': 35000, 'prerequisites.html': 35000 };
    for (const [filename, budget] of Object.entries(budgets)) {
        await page.goto('/' + filename);
        const bytes = await page.evaluate(() => performance.getEntriesByType('resource').filter(entry => entry.initiatorType === 'script').reduce((sum, entry) => sum + entry.decodedBodySize, 0));
        expect(bytes, filename).toBeLessThan(budget);
    }
});

test('a missing nested page has a working recovery path', async ({ page }) => {
    const response = await page.goto('/missing/page.html');
    expect(response.status()).toBe(404);
    await expect(page.getByRole('heading', { level: 1 })).toHaveText('This page took a different path.');
    await page.keyboard.press('Tab');
    await page.keyboard.press('Enter');
    await expect(page.locator('main')).toBeFocused();
    await page.getByRole('link', { name: 'Back to home', exact: true }).click();
    await expect(page).toHaveURL(/index\.html$/);
});
