const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.resolve(__dirname, '..');
const pages = ['index.html', 'courses.html', 'certifications.html', 'find-your-path.html', 'prerequisites.html', 'hiring-procedures.html', 'mentorship.html', 'about.html'];
const read = filename => fs.readFileSync(path.join(root, filename), 'utf8');
const contents = Object.fromEntries(pages.map(page => [page, read(page)]));
const metadata = (html, name) => {
    const tag = [...html.matchAll(/<meta\b[^>]*>/g)].map(match => match[0]).find(tag => tag.includes('name="' + name + '"') || tag.includes('property="' + name + '"'));
    return tag?.match(/content="([^"]*)"/)?.[1];
};

for (const page of pages) {
    test(page + ' has complete page landmarks and search metadata', () => {
        const html = contents[page];
        assert.equal((html.match(/<main\b/g) || []).length, 1);
        assert.equal((html.match(/<\/main>/g) || []).length, 1);
        assert.equal((html.match(/<h1\b/g) || []).length, 1);
        assert.match(html, /<html lang="en">/);
        assert.match(html, /class="skip-link" href="#main-content"/);
        assert.match(html, /<main id="main-content"/);
        assert.ok(metadata(html, 'description')?.length >= 70);
        const canonical = html.match(/<link rel="canonical" href="([^"]+)"/)?.[1];
        assert.equal(canonical, 'https://cloudvana.vercel.app/' + (page === 'index.html' ? '' : page));
        assert.equal(metadata(html, 'og:url'), canonical);
        assert.ok(metadata(html, 'og:title'));
        assert.ok(metadata(html, 'twitter:title'));
        assert.equal(metadata(html, 'theme-color'), '#087e8b');
        assert.ok(html.indexOf('src="js/theme.js"') < html.indexOf('href="styles.css"'));
        const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]);
        assert.equal(new Set(ids).size, ids.length, 'Duplicate element IDs');
    });
}

test('every static local link, image, stylesheet, and script points to a real file', () => {
    for (const [page, html] of Object.entries(contents)) {
        for (const [, value] of html.matchAll(/(?:href|src)="([^"]+)"/g)) {
            if (/^(https?:|mailto:|tel:|data:)/.test(value)) continue;
            const url = new URL(value, 'https://cloudvana.vercel.app/' + page);
            const filename = decodeURIComponent(url.pathname).replace(/^\//, '') || 'index.html';
            assert.ok(fs.existsSync(path.join(root, filename)), page + ' references missing file ' + filename);
            if (url.hash && filename.endsWith('.html')) {
                const target = read(filename);
                assert.ok(target.includes('id="' + decodeURIComponent(url.hash.slice(1)) + '"'), page + ' references missing anchor ' + value);
            }
        }
    }
});

test('JavaScript and structured data parse successfully', () => {
    const scripts = ['script.js', 'preview.cjs', 'playwright.config.cjs', ...fs.readdirSync(path.join(root, 'js')).filter(file => file.endsWith('.js')).map(file => path.join('js', file))];
    for (const filename of scripts) assert.doesNotThrow(() => new vm.Script(read(filename), { filename }));
    for (const html of Object.values(contents)) {
        for (const [, json] of html.matchAll(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/g)) assert.doesNotThrow(() => JSON.parse(json));
    }
});

function loadContent(files) {
    const source = 'const courseData = {};\n' + files.map(file => read(path.join('js', file))).join('\n') + '\nJSON.stringify(courseData)';
    return JSON.parse(vm.runInNewContext(source, {}, { timeout: 5000 }));
}

test('all course collections retain valid, complete modules and code examples', () => {
    const courses = loadContent(['ml-content.js', 'ai-courses.js', 'dsa-system-design.js', 'os-networking.js', 'programming-languages.js']);
    const certifications = loadContent(['certification-content.js']);
    const count = data => Object.values(data).reduce((sum, modules) => sum + modules.length, 0);
    assert.equal(count(courses), 77);
    const publishedCertifications = { ...certifications };
    delete publishedCertifications.azureDataScientist;
    delete publishedCertifications.awsMlSpecialty;
    assert.equal(count(publishedCertifications), 83);
    assert.equal(count(courses) + count(publishedCertifications), 160);
    assert.ok(certifications.azureDataScientist.length > 0, 'Preserve archived Azure course content');
    assert.ok(certifications.awsMlSpecialty.length > 0, 'Preserve archived AWS course content');
    for (const [key, modules] of Object.entries({ ...courses, ...certifications })) {
        assert.equal(new Set(modules.map(module => module.number + '|' + module.title)).size, modules.length, key + ' has duplicate modules');
        for (const module of modules) {
            assert.ok(module.title && module.description && module.number, key);
            assert.ok(module.detailedContent.length > 0, key + ': ' + module.title);
            for (const section of module.detailedContent) {
                assert.equal(typeof section.title, 'string', module.title);
                assert.equal(typeof section.content, 'string', module.title);
                if (section.code !== undefined) assert.equal(typeof section.code, 'string', module.title);
            }
        }
    }
});

test('general pages do not load learning bundles or eager diagram libraries', () => {
    for (const page of pages.filter(page => !['courses.html', 'certifications.html'].includes(page))) {
        assert.doesNotMatch(contents[page], /src="script\.js"/);
        assert.doesNotMatch(contents[page], /src="js\/(?:ml-content|certification-content)\.js"/);
    }
    assert.doesNotMatch(contents['courses.html'], /<script[^>]*src="https:\/\/cdn\.jsdelivr\.net/);
    assert.doesNotMatch(contents['courses.html'], /src="js\/certification-content\.js"/);
    assert.doesNotMatch(contents['certifications.html'], /src="js\/ml-content\.js"/);
    assert.ok(Buffer.byteLength(read('styles.css')) < 90000, 'Keep shared CSS below 90 KB');
});

test('every advertised practice-question count matches its actual quiz', () => {
    const source = read('script.js');
    const quizzes = JSON.parse(vm.runInNewContext(source.slice(0, source.indexOf('const CERTIFICATE_PASS_RATIO')) + '\nJSON.stringify(quizData)'));
    const html = contents['certifications.html'];
    const panels = [...html.matchAll(/\bid="(panel-[^"]+)"/g)];
    assert.equal(panels.length, 12);
    for (let index = 0; index < panels.length; index++) {
        const panelId = panels[index][1];
        const section = html.slice(panels[index].index, panels[index + 1]?.index ?? html.length);
        assert.equal(Number(section.match(/(\d+) practice questions/)?.[1]), quizzes[panelId].length, panelId);
        for (const question of quizzes[panelId]) {
            assert.ok(question.q && question.explain);
            assert.ok(question.options.length >= 2);
            assert.ok(Number.isInteger(question.answer) && question.answer >= 0 && question.answer < question.options.length);
        }
    }
});

test('the canonical sitemap includes every published page exactly once', () => {
    const urls = [...read('sitemap.xml').matchAll(/<loc>([^<]+)<\/loc>/g)].map(match => match[1]);
    assert.equal(urls.length, pages.length);
    assert.equal(new Set(urls).size, pages.length);
    for (const page of pages) assert.ok(urls.includes('https://cloudvana.vercel.app/' + (page === 'index.html' ? '' : page)));
    assert.deepEqual([...read('robots.txt').matchAll(/^Sitemap: (.+)$/gm)].map(match => match[1].trim()), ['https://cloudvana.vercel.app/sitemap.xml']);
    assert.doesNotMatch(read('404.html'), /content="index,/);
});

test('verification files and deployment configuration remain valid', () => {
    for (const filename of ['googlec9b99f54109ea588.html', 'googled32c03a03e0322e0.html']) assert.equal(read(filename).trim(), 'google-site-verification: ' + filename);
    const config = JSON.parse(read('vercel.json'));
    assert.ok(config.headers.some(rule => rule.headers.some(header => header.key === 'X-Content-Type-Options' && header.value === 'nosniff')));
    assert.ok(config.headers.some(rule => rule.headers.some(header => header.key === 'Content-Type' && header.value.includes('application/xml'))));
});

test('mentorship cannot leak form fields through a no-JavaScript GET submission', () => {
    assert.match(contents['mentorship.html'], /<fieldset class="form-fields" id="mentorshipFields" disabled>/);
    assert.doesNotMatch(read(path.join('js', 'mentorship.js')), /no-cors/);
    assert.match(contents['mentorship.html'], /149 INR/);
    assert.match(contents['mentorship.html'], /value="free-guidance"/);
    assert.match(contents['mentorship.html'], /value="focused-149"/);
});
