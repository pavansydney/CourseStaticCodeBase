const { test, expect } = require('@playwright/test');
const AxeBuilder = require('@axe-core/playwright').default;

test('every prerequisite topic and the roadmap support accessible keyboard switching', async ({ page }) => {
    await page.goto('/prerequisites.html');
    const tabs = page.locator('.prereq-tabs [role="tab"]');
    for (const tab of await tabs.all()) {
        await tab.click();
        const panelId = await tab.getAttribute('aria-controls');
        await expect(page.locator('#' + panelId)).toBeVisible();
        const audit = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa', 'best-practice']).analyze();
        expect(audit.violations.map(item => ({ id: item.id, targets: item.nodes.map(node => node.target) }))).toEqual([]);
    }
    await page.goto('/find-your-path.html');
    const roadmap = page.locator('#roadmapTabs [role="tab"]');
    await roadmap.first().focus();
    await page.keyboard.press('End');
    await expect(roadmap.last()).toHaveAttribute('aria-selected', 'true');
    await expect(page.locator('#roadmap-sap')).toBeVisible();
    await page.keyboard.press('ArrowRight');
    await expect(roadmap.first()).toHaveAttribute('aria-selected', 'true');
});

test('completion records require every module and the exact 70 percent quiz threshold', async ({ page }) => {
    await page.goto('/certifications.html?provider=azure');
    const summary = await page.locator('#panel-ai-901').evaluate(panel => {
        const ids = [...panel.querySelectorAll('.module-card')].map(card => card.dataset.moduleId);
        const read = Object.fromEntries(ids.map(id => [id, 1]));
        const pass = Math.ceil(panel.querySelectorAll('.quiz-q').length * 0.7);
        localStorage.setItem('cloudvana-read', JSON.stringify(read));
        localStorage.setItem('cloudvana-quiz-panel-ai-901', String(pass - 1));
        return { pass, read };
    });
    await page.reload();
    const issue = page.locator('#panel-ai-901 .cert-issue-btn');
    await expect(issue).toBeDisabled();
    await page.evaluate(pass => localStorage.setItem('cloudvana-quiz-panel-ai-901', String(pass)), summary.pass);
    await page.reload();
    await expect(issue).toBeEnabled();
    await issue.click();
    await page.locator('#panel-ai-901').getByRole('textbox', { name: 'Name on readiness badge', exact: true }).fill('<Test Learner>');
    await page.locator('#panel-ai-901').getByRole('button', { name: 'Create readiness badge', exact: true }).click();
    await expect(page.locator('#panel-ai-901 .cert-issued-recipient')).toHaveText('<Test Learner>');
    await expect(issue).toBeFocused();
    await expect(page.locator('#panel-ai-901 .cert-storage-note')).toContainText('self-reported');
    await page.reload();
    await expect(page.locator('#panel-ai-901 .cert-issued-recipient')).toHaveText('<Test Learner>');
    const stored = await page.evaluate(() => JSON.parse(localStorage.getItem('cloudvana-issued-certificates-v1'))['panel-ai-901']);
    expect(stored.learnerName).toBe('<Test Learner>');
    expect(stored.certificateId).toMatch(/^CV-AI901-/);
});

test('corrupted local data is reported without breaking the learning page', async ({ page }) => {
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.addInitScript(() => {
        localStorage.setItem('cloudvana-read', '{invalid-json');
        localStorage.setItem('cloudvana-issued-certificates-v1', '{"panel-ai-901":true}');
        localStorage.setItem('cloudvana-quiz-panel-ai-901', '-10');
    });
    await page.goto('/certifications.html?provider=azure');
    await expect(page.locator('#storageNotice')).toBeVisible();
    await expect(page.locator('#storageNoticeText')).toContainText('could not be read');
    await expect(page.locator('.module-card')).toHaveCount(83);
    const questions = await page.locator('#panel-ai-901 .quiz-q').count();
    await expect(page.locator('#panel-ai-901 .quiz-score')).toHaveText('Best: 0/' + questions);
    await page.locator('#panel-ai-901 .module-open').first().click();
    await expect(page.getByRole('dialog')).toBeVisible();
    expect(errors).toEqual([]);
});

test('without JavaScript, form controls stay disabled and prerequisite content stays readable', async ({ browser }, testInfo) => {
    const context = await browser.newContext({ javaScriptEnabled: false, viewport: testInfo.project.use.viewport });
    const page = await context.newPage();
    await page.goto('http://127.0.0.1:4173/mentorship.html');
    await expect(page.locator('#mentorshipForm button[type="submit"]')).toBeDisabled();
    await expect(page.locator('input[name="email"]')).toBeDisabled();
    await expect(page.locator('.noscript-banner')).toBeVisible();
    expect(new URL(page.url()).search).toBe('');
    await page.goto('http://127.0.0.1:4173/prerequisites.html');
    await expect(page.locator('.tab-panel:visible')).toHaveCount(5);
    await context.close();
});

async function addDiagram(page) {
    await page.goto('/courses.html');
    await page.locator('#ml-models-grid .module-open').first().click();
    await page.evaluate(() => {
        const holder = document.createElement('div');
        holder.id = 'diagram-test';
        holder.innerHTML = renderVisualDiagram('graph TD\nStart["Start"] --> Finish["Finish"]');
        document.getElementById('modal-body').prepend(holder);
        renderMermaidDiagrams(holder);
    });
}

test('diagrams load only when needed and expose a readable text alternative', async ({ page }) => {
    const requests = [];
    page.on('request', request => { if (request.url().includes('mermaid@')) requests.push(request.url()); });
    await page.goto('/courses.html');
    expect(requests).toEqual([]);
    await addDiagram(page);
    const diagram = page.locator('#diagram-test .learning-diagram-mermaid');
    await expect(diagram).toHaveAttribute('data-mermaid-rendered', 'true', { timeout: 30000 });
    await expect(diagram.locator('svg')).toHaveAttribute('role', 'img');
    await expect(diagram.getByText('Read diagram definition')).toBeVisible();
    await diagram.locator('summary').click();
    await expect(diagram.locator('.diagram-description pre')).toContainText('Start');
    expect(requests.length).toBe(1);
});

test('failed diagram requests show an explicit text fallback', async ({ page }) => {
    await page.route('https://cdn.jsdelivr.net/**/mermaid*', route => route.abort());
    await addDiagram(page);
    const diagram = page.locator('#diagram-test .learning-diagram-mermaid');
    await expect(diagram).toHaveAttribute('data-mermaid-rendered', 'error');
    await expect(diagram.locator('.diagram-status')).toContainText('text definition is shown below');
    await expect(diagram.locator('.mermaid-host pre')).toContainText('Start');
});

test('every published module renders all lessons without errors', async ({ page }) => {
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.route('https://cdn.jsdelivr.net/**/mermaid*', route => route.abort());
    let count = 0;
    for (const filename of ['courses.html', 'certifications.html']) {
        await page.goto('/' + filename);
        for (const card of await page.locator('.module-card').all()) {
            const expected = await card.evaluate(element => ({
                title: element.querySelector('.module-open').firstChild.textContent,
                lessons: parseInt(element.querySelector('.meta-item').textContent, 10)
            }));
            await card.evaluate(element => element.querySelector('.module-open').click());
            const dialog = page.getByRole('dialog');
            await expect(dialog.locator('.modal-title')).toHaveText(expected.title);
            await expect(dialog.locator('.content-header')).toHaveCount(expected.lessons);
            await page.evaluate(() => closeModal());
            await page.waitForFunction(() => document.body.style.overflow !== 'hidden');
            count++;
        }
    }
    expect(count).toBe(160);
    expect(errors).toEqual([]);
});

test('storage quota failures retain current-page progress and explain the limitation', async ({ page }) => {
    await page.addInitScript(() => {
        Storage.prototype.setItem = function () { throw new DOMException('Quota exceeded', 'QuotaExceededError'); };
    });
    await page.goto('/courses.html');
    await page.locator('#ml-models-grid .module-open').first().click();
    await page.getByRole('button', { name: 'Mark as complete' }).click();
    await expect(page.getByRole('button', { name: 'Completed', exact: true })).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(page.locator('#storageNoticeText')).toContainText('could not be saved');
    await expect(page.locator('#ml-models-grid .module-card').first()).toHaveClass(/read/);
});

test('sticky navigation, reduced-motion jumps, focus trapping, and code copying work together', async ({ page }) => {
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.addInitScript(() => {
        Object.defineProperty(navigator, 'clipboard', { value: { writeText: async text => { window.copiedCode = text; } } });
    });
    await page.route('https://cdn.jsdelivr.net/**/mermaid*', route => route.abort());
    await page.goto('/courses.html');
    await page.locator('.track-switch-link[data-track="track-ai-eng"]').click();
    const positions = await page.evaluate(() => ({
        header: document.querySelector('.navbar').getBoundingClientRect(),
        switcher: document.querySelector('.track-switcher').getBoundingClientRect(),
        title: document.querySelector('#track-ai-eng .section-title').getBoundingClientRect()
    }));
    expect(positions.header.top).toBe(0);
    expect(positions.title.top - positions.switcher.bottom).toBeGreaterThanOrEqual(8);
    expect(positions.title.top - positions.switcher.bottom).toBeLessThanOrEqual(80);
    await page.evaluate(() => window.scrollBy(0, 120));
    const stickyGap = await page.evaluate(() => document.querySelector('.track-switcher').getBoundingClientRect().top - document.querySelector('.navbar').getBoundingClientRect().bottom);
    expect(Math.abs(stickyGap)).toBeLessThanOrEqual(2);
    await page.locator('.track-switch-link[data-track="courses"]').click();
    await page.locator('#ml-models-grid .module-open').first().click();
    const dialog = page.getByRole('dialog');
    await dialog.getByRole('button', { name: 'Expand All' }).click();
    const copy = dialog.locator('.copy-btn').first();
    await expect(copy).toHaveAccessibleName('Copy code');
    await copy.click();
    await expect(copy).toHaveText('Copied');
    expect(await page.evaluate(() => window.copiedCode.length)).toBeGreaterThan(20);
    await page.getByRole('button', { name: 'Close module', exact: true }).focus();
    await page.keyboard.press('Shift+Tab');
    expect(await page.evaluate(() => !!document.activeElement.closest('dialog'))).toBe(true);
    await page.keyboard.press('Escape');
    await expect(dialog).not.toBeVisible();
    expect(await page.evaluate(() => document.body.style.overflow)).not.toBe('hidden');
});
