const { test, expect } = require('@playwright/test');
const AxeBuilder = require('@axe-core/playwright').default;

async function completeEngineeringTrack(page) {
    await page.goto('/courses.html?track=track-ai-eng');
    await page.locator('#track-ai-eng').evaluate(track => {
        const read = Object.fromEntries([...track.querySelectorAll('.module-card')].map(card => [card.dataset.moduleId, 1]));
        localStorage.setItem('cloudvana-read', JSON.stringify(read));
    });
    await page.reload();
    const section = page.locator('#track-ai-eng .cert-issue-section');
    await expect(section.locator('.cert-issue-btn')).toBeEnabled();
    return section;
}

test('AI Engineering issues a certificate without browser prompts', async ({ page }) => {
    await page.addInitScript(() => {
        window.certificatePromptCalls = 0;
        window.prompt = () => {
            window.certificatePromptCalls++;
            return null;
        };
    });
    const section = await completeEngineeringTrack(page);
    await section.getByRole('button', { name: 'Issue Certificate', exact: true }).click();
    expect(await page.evaluate(() => window.certificatePromptCalls)).toBe(0);
    const name = section.getByRole('textbox', { name: 'Name on certificate', exact: true });
    await expect(name).toBeVisible();
    await expect(name).toBeFocused();
    await name.fill('  Test AI Learner  ');
    const audit = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa', 'wcag22aa']).analyze();
    expect(audit.violations.map(item => ({ id: item.id, targets: item.nodes.map(node => node.target) }))).toEqual([]);
    await section.getByRole('button', { name: 'Create certificate', exact: true }).click();
    await expect(section.locator('.cert-issued-recipient')).toHaveText('Test AI Learner');
    await expect(section.locator('.cert-issued-track')).toContainText('AI Engineering');
    await expect(section.locator('.cert-issue-btn')).toBeFocused();
    const certificateId = await section.locator('.cert-issued-id').textContent();
    await page.reload();
    await expect(section.locator('.cert-issued-recipient')).toHaveText('Test AI Learner');
    await expect(section.locator('.cert-issued-id')).toHaveText(certificateId);
});

test('certificate names are validated inline and cancellation preserves the draft', async ({ page }) => {
    const section = await completeEngineeringTrack(page);
    const issue = section.locator('.cert-issue-btn');
    await issue.click();
    const name = section.getByRole('textbox', { name: 'Name on certificate', exact: true });
    await name.fill('   ');
    await section.getByRole('button', { name: 'Create certificate', exact: true }).click();
    await expect(name).toHaveAttribute('aria-invalid', 'true');
    await expect(name).toBeFocused();
    await expect(section.getByRole('alert')).toContainText('Enter your name');
    await expect(section.locator('.cert-issued-card')).toHaveCount(0);
    expect(await page.evaluate(() => localStorage.getItem('cloudvana-issued-certificates-v1'))).toBeNull();
    await name.fill('Draft Learner');
    await section.getByRole('button', { name: 'Cancel', exact: true }).click();
    await expect(section.locator('.cert-claim-form')).toBeHidden();
    await expect(issue).toBeFocused();
    expect(await page.evaluate(() => localStorage.getItem('cloudvana-learner-name-v1'))).toBeNull();
    await issue.click();
    await expect(name).toHaveValue('Draft Learner');
    await name.press('Enter');
    await expect(section.locator('.cert-issued-recipient')).toHaveText('Draft Learner');
});

test('certificate issuance explains when browser storage cannot persist the record', async ({ page }) => {
    const section = await completeEngineeringTrack(page);
    await page.evaluate(() => {
        Storage.prototype.setItem = function () { throw new DOMException('Quota exceeded', 'QuotaExceededError'); };
    });
    await section.locator('.cert-issue-btn').click();
    await section.getByRole('textbox', { name: 'Name on certificate', exact: true }).fill('Current Page Learner');
    await section.getByRole('button', { name: 'Create certificate', exact: true }).click();
    await expect(section.locator('.cert-issued-recipient')).toHaveText('Current Page Learner');
    await expect(section.locator('.cert-storage-note')).toContainText('only available on this page');
    await expect(page.locator('#storageNotice')).toBeVisible();
    expect(await page.evaluate(() => localStorage.getItem('cloudvana-issued-certificates-v1'))).toBeNull();
});
