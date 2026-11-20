var CONTACT_EMAIL = 'cloud.vana@outlook.com';
var INTAKE_DAILY_LIMIT = 200;
var INTAKE_HEADERS = ['Timestamp', 'Name', 'Email', 'Plan', 'Course', 'Current level', 'Weekly time',
  'Support need', 'Goal', 'Source page', 'Reference', 'Request fingerprint', 'Student email', 'Admin email', 'Consent'];
var INTAKE_OPTIONS = {
  plan: { 'free-guidance': 'Free orientation', 'focused-149': 'Focused mentorship - 149 INR per session' },
  currentLevel: { beginner: 'Beginner', midway: 'Midway', revision: 'Revision', 'career-transition': 'Career transition' },
  weeklyTime: { 'under-5': 'Under 5 hours', '5-8': '5 to 8 hours', '8-12': '8 to 12 hours', '12-plus': '12+ hours' },
  supportNeed: { direction: 'Study direction', 'weak-topics': 'Weak-topic correction', revision: 'Revision planning', accountability: 'Accountability' }
};

function validateIntake_(event) {
  if (!event || !event.postData || String(event.postData.contents || '').length > 12000) {
    throw new Error('Invalid request body.');
  }
  var raw = /^application\/json(?:;|$)/i.test(event.postData.type || '') ?
    JSON.parse(event.postData.contents) : event.parameter;
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) throw new Error('Invalid request.');
  if (String(raw.company || '').trim()) throw new Error('Request rejected.');
  if (raw.consent !== true && raw.consent !== 'true') throw new Error('Contact consent is required.');
  var limits = { name: 100, email: 254, plan: 30, course: 200, currentLevel: 30, weeklyTime: 20, supportNeed: 30, goal: 4000, sourcePage: 30 };
  var data = {};
  Object.keys(limits).forEach(function (key) {
    if (raw[key] != null && typeof raw[key] !== 'string') throw new Error('Invalid field: ' + key);
    var value = String(raw[key] || '').trim();
    if ((!value && key !== 'goal' && key !== 'sourcePage') || value.length > limits[key] ||
        (key === 'goal' ? /[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/ : /[\x00-\x1f\x7f]/).test(value)) {
      throw new Error('Invalid field: ' + key);
    }
    data[key] = value;
  });
  data.email = data.email.toLowerCase();
  if (!/^[^@\s<>",;:\\]+@[^@\s<>",;:\\]+\.[^@\s<>",;:\\]+$/.test(data.email)) throw new Error('Invalid email.');
  Object.keys(INTAKE_OPTIONS).forEach(function (key) {
    if (!Object.prototype.hasOwnProperty.call(INTAKE_OPTIONS[key], data[key])) throw new Error('Invalid field: ' + key);
  });
  data.consent = true;
  return data;
}

function intakeFingerprint_(data) {
  var bytes = Utilities.computeDigest(Utilities.DigestAlgorithm.SHA_256, JSON.stringify(data), Utilities.Charset.UTF_8);
  return bytes.map(function (byte) { return ('0' + ((byte + 256) % 256).toString(16)).slice(-2); }).join('');
}

function sheetText_(value) {
  return /^\s*[=+\-@]/.test(value) ? "'" + value : value;
}

function saveIntake_(data) {
  var properties = PropertiesService.getScriptProperties();
  var spreadsheetId = properties.getProperty('SPREADSHEET_ID');
  if (!spreadsheetId) throw new Error('SPREADSHEET_ID is not configured.');
  var sheet = SpreadsheetApp.openById(spreadsheetId).getSheetByName(properties.getProperty('SHEET_NAME') || 'Sheet1');
  if (!sheet) throw new Error('The intake sheet was not found.');
  var fingerprint = intakeFingerprint_(data);
  var now = new Date();
  var lock = LockService.getScriptLock();
  lock.waitLock(10000);
  try {
    if (sheet.getMaxColumns() < INTAKE_HEADERS.length) sheet.insertColumnsAfter(sheet.getMaxColumns(), INTAKE_HEADERS.length - sheet.getMaxColumns());
    if (!sheet.getLastRow()) sheet.appendRow(INTAKE_HEADERS);
    var rows = sheet.getDataRange().getValues();
    var todayCount = 0;
    for (var index = rows.length - 1; index >= 0; index--) {
      var age = now.getTime() - new Date(rows[index][0]).getTime();
      if (age >= 0 && age < 86400000) {
        todayCount++;
        if (rows[index][11] === fingerprint && /^CV-[A-F0-9]{32}$/.test(rows[index][10])) {
          return { sheet: sheet, row: index + 1, reference: rows[index][10], emailStatus: rows[index][12], duplicate: true };
        }
      }
    }
    if (todayCount >= INTAKE_DAILY_LIMIT) {
      var capacityError = new Error('The daily registration limit has been reached. Contact Cloudvana directly.');
      capacityError.code = 'daily-limit';
      throw capacityError;
    }
    var reference = 'CV-' + Utilities.getUuid().replace(/-/g, '').toUpperCase();
    var values = [now, data.name, data.email, data.plan, data.course, data.currentLevel, data.weeklyTime,
      data.supportNeed, data.goal, data.sourcePage, reference, fingerprint, 'pending', 'pending', 'true'];
    sheet.appendRow(values.map(function (value) { return typeof value === 'string' ? sheetText_(value) : value; }));
    SpreadsheetApp.flush();
    return { sheet: sheet, row: sheet.getLastRow(), reference: reference, emailStatus: 'pending', duplicate: false };
  } finally {
    lock.releaseLock();
  }
}

function escapeEmailHtml_(value) {
  return String(value).replace(/[&<>"']/g, function (character) {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character];
  });
}

function registrationEmail_(data, reference) {
  var plan = INTAKE_OPTIONS.plan[data.plan];
  var subject = 'Cloudvana registration received | ' + reference;
  var summary = 'Reference: ' + reference + '\nSupport: ' + plan + '\nCourse: ' + data.course;
  var next = data.plan === 'focused-149' ?
    'We will review your request and email you with an available session date and time. Your slot is not confirmed yet.' :
    'We will review your orientation request and email you about the next step. Any call date and time will be agreed with you first.';
  var timing = 'If your request is time-sensitive, reply to this email with your availability, time zone, and deadline. Earlier slots depend on availability and are not guaranteed.';
  var payment = 'No payment has been taken. Registration does not reserve a slot or guarantee an exam result or job offer.';
  var text = 'Hi ' + data.name + ',\n\nThanks for registering with Cloudvana. Your request has been received.\n\n' +
    summary + '\n\nWhat happens next?\n' + next + '\n\n' + timing + '\n\n' + payment +
    '\n\nYou can keep learning at https://cloudvana.vercel.app/courses.html\n\nCloudvana\n' + CONTACT_EMAIL +
    '\nYou received this acknowledgement because this email address was entered on our mentorship request form. If that was not you, reply to let us know; no action or payment is required.';
  var html = '<!doctype html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1"></head>' +
    '<body style="margin:0;padding:0;background:#f3f7fa;color:#233744;font-family:Arial,Helvetica,sans-serif">' +
    '<div style="display:none;max-height:0;overflow:hidden">Your request is received. Keep your reference; session details will follow after review.</div>' +
    '<table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:24px 12px">' +
    '<table role="presentation" width="600" cellpadding="0" cellspacing="0" style="width:100%;max-width:600px;background:#ffffff;border:1px solid #d5e0e7;border-radius:12px">' +
    '<tr><td style="padding:28px 24px;background:#12313d;color:#ffffff"><p style="margin:0;font-size:26px;font-weight:bold">Cloudvana</p>' +
    '<p style="margin:8px 0 0;font-size:14px;color:#cbece9">Learn. Build. Explain.</p></td></tr>' +
    '<tr><td style="padding:28px 24px;font-size:16px;line-height:1.6"><h1 style="margin:0 0 20px;font-size:26px;line-height:1.3">Thanks for registering.</h1>' +
    '<p>Hi ' + escapeEmailHtml_(data.name) + ',</p><p>We have received your mentorship request. Keep the reference below for any follow-up.</p>' +
    '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#edf6f6;border:1px solid #b9d8d8">' +
    '<tr><td style="padding:18px;font-size:14px;line-height:1.8"><strong>YOUR REQUEST REFERENCE</strong><br>' +
    '<span style="font-family:monospace;font-size:15px;word-break:break-all">' + escapeEmailHtml_(reference) + '</span><br>' +
    '<strong>Support:</strong> ' + escapeEmailHtml_(plan) + '<br><strong>Course:</strong> ' + escapeEmailHtml_(data.course) + '</td></tr></table>' +
    '<h2 style="margin:24px 0 8px;font-size:20px">What happens next?</h2><p>' + next + '</p>' +
    '<h2 style="margin:24px 0 8px;font-size:20px">Have a deadline?</h2><p>' + timing + '</p>' +
    '<p style="margin:24px 0"><a href="https://cloudvana.vercel.app/courses.html" style="display:inline-block;background:#087580;color:#ffffff;padding:12px 20px;text-decoration:none;border-radius:6px;font-weight:bold">Continue learning</a></p>' +
    '<p style="font-size:14px;color:#526674">' + payment + '</p></td></tr>' +
    '<tr><td style="padding:20px 24px;border-top:1px solid #d5e0e7;font-size:12px;line-height:1.6;color:#526674">' +
    '<strong>Cloudvana</strong> &middot; Independent learning and mentorship<br><a href="mailto:' + CONTACT_EMAIL + '" style="color:#076b76">' + CONTACT_EMAIL + '</a><br>' +
    'You received this acknowledgement because this address was entered on our mentorship form. If that was not you, reply to let us know; no action or payment is required.<br>' +
    '<a href="https://cloudvana.vercel.app/about.html#privacy" style="color:#076b76">Privacy information</a></td></tr></table></td></tr></table></body></html>';
  return { subject: subject, text: text, html: html };
}

function sendRegistrationEmail_(recipient, replyTo, email) {
  if (MailApp.getRemainingDailyQuota() < 1) throw new Error('Google email recipient quota is exhausted.');
  MailApp.sendEmail({
    to: recipient, name: 'Cloudvana', replyTo: replyTo,
    subject: email.subject, body: email.text, htmlBody: email.html
  });
}

function setIntakeEmailStatus_(record, column, status) {
  record.sheet.getRange(record.row, column).setValue(status);
  SpreadsheetApp.flush();
}

function doPost(event) {
  var record;
  try {
    var data = validateIntake_(event);
    record = saveIntake_(data);
    if (record.duplicate) {
      return json_({ ok: true, reference: record.reference, emailStatus: record.emailStatus, duplicate: true });
    }
    sendRegistrationEmail_(data.email, CONTACT_EMAIL, registrationEmail_(data, record.reference));
    setIntakeEmailStatus_(record, 13, 'accepted');
    record.emailStatus = 'accepted';
    try {
      var fields = ['name', 'email', 'plan', 'course', 'currentLevel', 'weeklyTime', 'supportNeed', 'goal', 'sourcePage'];
      var adminText = INTAKE_HEADERS.slice(1, 10).map(function (label, index) {
        return label + ': ' + data[fields[index]];
      }).join('\n') + '\nReference: ' + record.reference;
      sendRegistrationEmail_(CONTACT_EMAIL, data.email, {
        subject: 'New Cloudvana request | ' + record.reference,
        text: adminText, html: '<pre style="white-space:pre-wrap">' + escapeEmailHtml_(adminText) + '</pre>'
      });
      setIntakeEmailStatus_(record, 14, 'accepted');
    } catch (error) {
      console.error('Admin alert unconfirmed for ' + record.reference + ': ' + error.message);
      setIntakeEmailStatus_(record, 14, 'unconfirmed');
    }
    return json_({ ok: true, reference: record.reference, emailStatus: 'accepted' });
  } catch (error) {
    console.error('Mentorship processing failed' + (record ? ' for ' + record.reference : '') + ': ' + error.message);
    if (record) {
      // Do not resend automatically: a timed-out send might still have been accepted.
      try {
        setIntakeEmailStatus_(record, record.emailStatus === 'accepted' ? 14 : 13, 'unconfirmed');
      } catch (statusError) {
        console.error('Could not persist email status for ' + record.reference + ': ' + statusError.message);
      }
      return json_({ ok: true, reference: record.reference, emailStatus: record.emailStatus === 'accepted' ? 'accepted' : 'unconfirmed' });
    }
    return json_({ ok: false, code: error.code === 'daily-limit' ? 'daily-limit' : 'request-failed',
      error: 'The request could not be saved. Check your entries or contact Cloudvana.' });
  }
}

function doGet() {
  return json_({ ok: true, message: 'Cloudvana mentorship endpoint is live', version: 2 });
}

function json_(value) {
  return ContentService.createTextOutput(JSON.stringify(value)).setMimeType(ContentService.MimeType.JSON);
}
