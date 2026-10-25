const { defineConfig, devices } = require('@playwright/test');
const path = require('node:path');

module.exports = defineConfig({
    testDir: path.join(__dirname, 'e2e'),
    timeout: 90000,
    expect: { timeout: 10000 },
    fullyParallel: true,
    workers: 2,
    reporter: 'list',
    use: {
        baseURL: 'http://127.0.0.1:4173',
        actionTimeout: 15000,
        trace: 'retain-on-failure',
        screenshot: 'only-on-failure'
    },
    projects: [
        { name: 'desktop', use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 960 } } },
        { name: 'mobile', use: { ...devices['iPhone 13'], browserName: 'chromium' } }
    ],
    webServer: {
        command: 'node preview.cjs',
        url: 'http://127.0.0.1:4173',
        reuseExistingServer: !process.env.CI
    }
});
