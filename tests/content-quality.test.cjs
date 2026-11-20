const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { spawnSync } = require('node:child_process');

const root = path.resolve(__dirname, '..');
const read = filename => fs.readFileSync(path.join(root, filename), 'utf8');
const bundles = ['ml-content', 'ai-courses', 'dsa-system-design', 'os-networking', 'programming-languages', 'certification-content'];
const courses = JSON.parse(vm.runInNewContext(
    'const courseData = {};\n' + bundles.map(name => read(path.join('js', name + '.js'))).join('\n') + '\nJSON.stringify(courseData)',
    {}, { timeout: 10000 }
));
const archived = new Set(['azureDataScientist', 'awsMlSpecialty']);
const published = Object.entries(courses).filter(([key]) => !archived.has(key)).flatMap(([key, modules]) => modules.map(module => ({ key, module })));

function text(value, context) {
    assert.equal(typeof value, 'string', context);
    assert.ok(value.trim().length >= 12, context + ' must be substantive');
    assert.doesNotMatch(value, /<\/(?:p|div|script|strong|ul|li)>|<(?:script|iframe)\b/i, context + ' must use plain text');
    assert.doesNotMatch(value, /\b(?:TODO|TBD|lorem ipsum)\b/i, context + ' contains placeholder content');
}

test('every published module has complete authored guidance, not generic expansions', () => {
    assert.equal(published.length, 160);
    for (const { key, module } of published) {
        const label = key + ': ' + module.title;
        const guide = module.learningGuide;
        assert.ok(guide, label + ' lacks a learning guide');
        text(guide.prerequisites, label + ' prerequisites');
        text(guide.setup, label + ' setup');
        text(guide.workedExample?.problem, label + ' worked problem');
        assert.ok(guide.workedExample.steps.length >= 2, label + ' needs worked reasoning');
        guide.workedExample.steps.forEach(step => text(step, label + ' worked step'));
        text(guide.workedExample.expected, label + ' expected result');
        text(guide.practice?.task, label + ' task');
        assert.ok(guide.practice.cases.length >= 2, label + ' needs normal and boundary/failure cases');
        for (const item of guide.practice.cases) {
            assert.equal(typeof item.input, 'string', label);
            assert.ok(item.input.trim(), label + ' empty case');
            assert.equal(typeof item.expected, 'string', label);
            assert.ok(item.expected.trim(), label + ' empty expected result');
        }
        text(guide.practice.hint, label + ' hint');
        text(guide.practice.solution, label + ' solution');
        text(guide.interview?.question, label + ' interview question');
        text(guide.interview.answer, label + ' interview answer');
        assert.ok(guide.readiness.length >= 2, label + ' needs observable readiness');
        guide.readiness.forEach(item => text(item, label + ' readiness'));
    }
    for (const field of [g => g.workedExample.problem, g => g.practice.task, g => g.practice.solution]) {
        const values = published.map(({ module }) => field(module.learningGuide));
        assert.equal(new Set(values).size, values.length, 'Modules must not reuse generic worked examples, tasks or solutions');
    }
});

test('the lightweight catalog stays aligned with every authored course module', () => {
    const result = spawnSync(process.execPath, [path.join(root, 'scripts', 'generate-course-catalog.cjs'), '--check'], { encoding: 'utf8' });
    assert.equal(result.status, 0, result.stderr || result.error?.message);
    const html = read('courses.html');
    assert.match(html, /src="js\/course-catalog\.js"/);
    for (const bundle of bundles.filter(name => name !== 'certification-content')) {
        assert.ok(!html.includes('src="js/' + bundle + '.js"'), bundle + ' must load only on demand');
    }
});

test('authored module quizzes have four complete choices when using A-D format', () => {
    let questions = 0;
    for (const { module } of published) {
        for (const section of module.detailedContent) {
            if (!/^(module quiz|mini assessment)$/i.test(section.title)) continue;
            for (const line of section.content.split('\n').filter(line => /[A-D]\)/.test(line))) {
                const choices = [...line.matchAll(/(?:^|\s)([A-D])\)\s*([\s\S]*?)(?=\s+[A-D]\)|$)/g)];
                assert.deepEqual(choices.map(choice => choice[1]), ['A', 'B', 'C', 'D'], module.title + ': ' + line);
                assert.ok(choices.every(choice => choice[2].trim()), module.title + ': empty option');
                questions++;
            }
        }
    }
    assert.ok(questions >= 20, 'Exercise the authored quiz format across multiple languages');
});

test('beginner ML examples explain updates, leakage and the limits of toy evaluation', () => {
    const intro = courses.mlModels[0];
    assert.doesNotMatch(intro.detailedContent[0].content, /Adapts to new data automatically/);
    assert.match(intro.detailedContent[0].content, /predict\(\).*does not retrain/);
    assert.match(intro.learningGuide.workedExample.expected, /275/);
    assert.match(courses.data[0].learningGuide.practice.solution, /30/);
    assert.match(courses.mlModels[3].learningGuide.practice.solution, /undefined/);
    assert.match(courses.realWorldML[2].learningGuide.practice.solution, /undefined/);
    assert.doesNotMatch(read(path.join('js', 'ml-content.js')), /# 7\. Ready for deployment!|Confidence: 95%/);
});

test('the renderer preserves uppercase options and escapes practice content', () => {
    const runtime = read('script.js');
    const formatSource = runtime.slice(runtime.indexOf('function formatSectionContent('), runtime.indexOf('function renderLearningGuide('));
    const guideSource = runtime.slice(runtime.indexOf('function renderLearningGuide('), runtime.indexOf('// Open module details in modal'));
    const context = vm.createContext({
        escapeHtml: value => String(value).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char])),
        renderVisualDiagram: () => ''
    });
    vm.runInContext(formatSource + '\n' + guideSource, context);
    const quiz = vm.runInContext('formatSectionContent("LEGB stands for: A) Local Enclosing Global Built-in B) Local Global Bytecode C) Lookup Execution Graph D) none", "Module Quiz")', context);
    assert.equal((quiz.match(/class="quiz-option"/g) || []).length, 4);
    assert.match(quiz, /Local Enclosing Global Built-in/);
    const lesson = vm.runInContext('formatSectionContent("Why this matters: Learn by tracing.\\nLearning Objective: Add two inputs.\\nCore Theory: Addition combines quantities.\\nWorked Example: 2 + 3 = 5.\\nPractice: Calculate 4 + 6.", "Addition")', context);
    assert.match(lesson, /Why it matters:<\/span> Learn by tracing/);
    assert.match(lesson, /Worked example:<\/span><p[^>]*>2 \+ 3 = 5/);
    assert.match(lesson, /Mini Exercise:<\/span> Calculate 4 \+ 6/);
    context.guide = structuredClone(courses.mlModels[0].learningGuide);
    context.guide.practice.task = '<script>alert("not content")</script>';
    const html = vm.runInContext('renderLearningGuide(guide)', context);
    assert.doesNotMatch(html, /<script>/);
    assert.match(html, /&lt;script&gt;/);
    assert.match(html, /<details class="guide-solution">/);
    assert.doesNotMatch(html, /<details class="guide-solution" open/);
});
