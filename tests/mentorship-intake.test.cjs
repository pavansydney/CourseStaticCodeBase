const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const crypto = require('node:crypto');

const source = readFileSync(path.join(__dirname, '..', 'apps-script', 'mentorship.gs'), 'utf8');
const contact = 'cloud.vana@outlook.com';
const valid = {
    name: 'Test Learner', email: 'learner@example.com', plan: 'focused-149', course: 'Python',
    currentLevel: 'beginner', weeklyTime: '5-8', supportNeed: 'direction',
    goal: 'Build a project.', sourcePage: 'mentorship', consent: 'true', company: ''
};

function harness(options = {}) {
    const rows = options.rows || [];
    const sends = [];
    const logs = [];
    let locked = false;
    let quota = options.remainingQuota ?? 10000;
    const sheet = {
        getMaxColumns: () => 26,
        getLastRow: () => rows.length,
        getDataRange: () => ({ getValues: () => rows.map(row => [...row]) }),
        appendRow: row => rows.push([...row]),
        getRange: (row, column) => ({
            setValue: value => {
                if (options.statusWriteFails) throw new Error('Status storage unavailable');
                rows[row - 1][column - 1] = value;
            }
        })
    };
    const properties = {
        getProperty: key => ({ SPREADSHEET_ID: 'test-sheet' })[key]
    };
    const sandbox = {
        console: { log: value => logs.push(value), error: value => logs.push(value) },
        PropertiesService: { getScriptProperties: () => properties },
        LockService: {
            getScriptLock: () => ({
                waitLock: () => { assert.equal(locked, false); locked = true; },
                releaseLock: () => { locked = false; }
            })
        },
        SpreadsheetApp: {
            openById: () => ({ getSheetByName: () => sheet }),
            flush: () => {}
        },
        Utilities: {
            getUuid: () => crypto.randomUUID(),
            DigestAlgorithm: { SHA_256: 'sha256' }, Charset: { UTF_8: 'utf8' },
            computeDigest: (_, value) => [...crypto.createHash('sha256').update(value).digest()]
        },
        MailApp: {
            getRemainingDailyQuota: () => quota,
            sendEmail: email => {
                assert.equal(locked, false, 'Network calls must not hold the sheet lock');
                sends.push(email);
                if (options.sendThrows) throw new Error('Ambiguous network failure');
                if (sends.length === options.failSend) throw new Error('Google mail service rejected the send');
                quota--;
            }
        },
        ContentService: {
            MimeType: { JSON: 'application/json' },
            createTextOutput: text => ({ setMimeType: () => JSON.parse(text) })
        }
    };
    vm.createContext(sandbox);
    vm.runInContext(source, sandbox);
    return {
        sandbox, rows, sends, logs,
        submit: (values = valid) => sandbox.doPost({
            parameter: values, postData: { type: 'application/x-www-form-urlencoded', contents: new URLSearchParams(values).toString() }
        })
    };
}

test('intake saves a reference, sends Google HTML/plain-text email with Outlook replies, and deduplicates retries', () => {
    const app = harness();
    const first = app.submit();
    assert.equal(first.ok, true);
    assert.match(first.reference, /^CV-[A-F0-9]{32}$/);
    assert.equal(first.emailStatus, 'accepted');
    assert.equal(app.rows.length, 2);
    assert.equal(app.rows[1][10], first.reference);
    assert.equal(app.rows[1][12], 'accepted');
    assert.equal(app.rows[1][13], 'accepted');
    assert.equal(app.rows[1][14], 'true');
    assert.equal(app.sends.length, 2);
    const student = app.sends[0];
    assert.equal(student.name, 'Cloudvana');
    assert.equal(student.from, undefined, 'MailApp must use the authorized Google sender, not spoof Outlook');
    assert.equal(student.replyTo, contact);
    assert.equal(student.to, valid.email);
    assert.match(student.htmlBody, /Earlier slots depend on availability/);
    assert.match(student.body, /Your slot is not confirmed yet/);
    assert.ok(student.body.includes(first.reference));
    assert.equal(app.sends[1].to, contact);
    assert.equal(app.sends[1].replyTo, valid.email);
    assert.ok(app.sends[1].body.includes(valid.goal));
    const second = app.submit();
    assert.equal(second.reference, first.reference);
    assert.equal(second.duplicate, true);
    assert.equal(app.sends.length, 2);
    assert.equal(app.rows.length, 2);
    assert.notEqual(app.submit({ ...valid, goal: 'A different request.' }).reference, first.reference);
});

test('invalid consent, enum choices, header injection, oversized values and bots cannot save or send', () => {
    for (const values of [
        { ...valid, consent: 'false' }, { ...valid, plan: 'unknown' },
        { ...valid, email: 'learner@example.com\r\nBcc: other@example.com' },
        { ...valid, name: ' ' }, { ...valid, course: 'x'.repeat(201) },
        { ...valid, company: 'bot' }, { ...valid, name: { unexpected: true } }
    ]) {
        const app = harness();
        assert.equal(app.submit(values).ok, false);
        assert.equal(app.rows.length, 0);
        assert.equal(app.sends.length, 0);
        assert.ok(app.logs.length);
    }
});

test('saved requests survive Google quota, service and ambiguous-send failures', () => {
    for (const options of [{ remainingQuota: 0 }, { failSend: 1 }, { sendThrows: true }]) {
        const app = harness(options);
        const result = app.submit();
        assert.equal(result.ok, true);
        assert.equal(result.emailStatus, 'unconfirmed');
        assert.match(result.reference, /^CV-/);
        assert.equal(app.rows[1][12], 'unconfirmed');
        assert.ok(app.logs.length);
        app.submit();
        assert.equal(app.rows.length, 2);
        assert.equal(app.sends.length, options.failSend || options.sendThrows ? 1 : 0, 'No automatic resend after an ambiguous failure');
    }
});

test('admin alert failure does not invalidate the accepted student email', () => {
    for (const options of [{ failSend: 2 }, { remainingQuota: 1 }]) {
        const app = harness(options);
        const result = app.submit();
        assert.equal(result.ok, true);
        assert.equal(result.emailStatus, 'accepted');
        assert.equal(app.rows[1][12], 'accepted');
        assert.equal(app.rows[1][13], 'unconfirmed');
        assert.ok(app.logs.length);
    }
});

test('a status-write failure leaves a saved reference and does not trigger a retry email', () => {
    const app = harness({ statusWriteFails: true });
    const result = app.submit();
    assert.equal(result.ok, true);
    assert.equal(result.emailStatus, 'unconfirmed');
    assert.equal(app.sends.length, 1);
    assert.equal(app.submit().reference, result.reference);
    assert.equal(app.sends.length, 1);
    assert.ok(app.logs.some(log => log.includes('Could not persist email status')));
});

test('JSON bodies with charset are supported, and sheet formulas and HTML are escaped', () => {
    const app = harness();
    const data = { ...valid, name: '=HYPERLINK("https://example.com")', course: '<img src=x onerror=alert(1)>', plan: 'free-guidance' };
    const result = app.sandbox.doPost({ postData: { type: 'application/json; charset=utf-8', contents: JSON.stringify(data) } });
    assert.equal(result.ok, true);
    assert.ok(app.rows[1][1].startsWith("'="));
    const html = app.sends[0].htmlBody;
    assert.ok(!html.includes(data.course));
    assert.ok(html.includes('&lt;img'));
    assert.match(html, /orientation request/);
    assert.doesNotMatch(html, /149 INR/);
});

test('the rolling daily cap rejects new requests but still acknowledges saved duplicates', () => {
    const app = harness();
    for (let index = 0; index < 200; index++) assert.equal(app.submit({ ...valid, email: 'learner' + index + '@example.com' }).ok, true);
    const atCapacity = app.submit();
    assert.equal(atCapacity.ok, false);
    assert.equal(atCapacity.code, 'daily-limit');
    assert.equal(app.rows.length, 201);
    assert.equal(app.submit({ ...valid, email: 'learner0@example.com' }).duplicate, true);
    assert.equal(app.sends.length, 400);
});
