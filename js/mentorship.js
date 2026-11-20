(function () {
    'use strict';

    const endpoint = 'https://script.google.com/macros/s/AKfycbxMrq29SzR5HNh9pWGJIDDy8KnmSvbJkC1s1b1Rqf_zHdTgXiYDdyO8FocCaSRMhCLm/exec';
    const form = document.getElementById('mentorshipForm');
    const message = document.getElementById('mentorshipFormMsg');
    const submitButton = form.querySelector('button[type="submit"]');
    let submitting = false;

    function setMessage(text, state) {
        message.textContent = text;
        message.dataset.state = state;
    }

    document.querySelectorAll('[data-mentorship-plan]').forEach(link => {
        link.addEventListener('click', event => {
            event.preventDefault();
            form.elements.namedItem('plan').value = link.dataset.mentorshipPlan;
            scrollToSection('apply');
            form.elements.namedItem('name').focus({ preventScroll: true });
        });
    });

    form.addEventListener('submit', async event => {
        event.preventDefault();
        if (submitting) return;
        if (!form.reportValidity()) return;

        const params = new URLSearchParams();
        const required = ['name', 'email', 'plan', 'course', 'currentLevel', 'weeklyTime', 'supportNeed'];
        for (const fieldName of required) {
            const field = form.elements.namedItem(fieldName);
            const value = field.value.trim();
            if (!value) {
                setMessage('Please complete all required fields. Entries cannot contain only spaces.', 'error');
                field.focus();
                return;
            }
            params.set(fieldName, value);
        }
        if (form.elements.namedItem('company').value.trim()) {
            setMessage('This request could not be sent. Please contact us using the LinkedIn link below.', 'error');
            return;
        }
        params.set('goal', form.elements.namedItem('goal').value.trim());
        params.set('company', '');
        params.set('consent', String(document.getElementById('mentorshipConsent').checked));
        params.set('sourcePage', 'mentorship');

        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 20000);
        submitting = true;
        submitButton.disabled = true;
        submitButton.textContent = 'Sending your request...';
        form.setAttribute('aria-busy', 'true');
        setMessage('Sending your request. Please keep this page open.', 'pending');

        try {
            const response = await fetch(endpoint, { method: 'POST', body: params, signal: controller.signal });
            if (!response.ok || response.type === 'opaque') throw new Error('The submission service did not provide a readable successful response.');
            const result = await response.json();
            if (result && result.ok === false && result.code === 'daily-limit') {
                setMessage('The registration service has reached its daily limit. Your request has not been saved. Your entries are still here; please contact cloud.vana@outlook.com or try again later.', 'error');
                return;
            }
            const acknowledged = result && !result.skipped && (result.success === true || result.ok === true || result.status === 'success');
            if (!acknowledged) throw new Error('The submission service did not acknowledge the request.');
            if (result.reference != null && (typeof result.reference !== 'string' || !/^CV-[A-F0-9]{32}$/.test(result.reference))) {
                throw new Error('The submission service returned an invalid reference.');
            }
            const reference = result.reference ? ' Your reference: ' + result.reference + '.' : '';
            const emailNote = result.reference ?
                (result.emailStatus === 'accepted' ?
                    ' A confirmation email has been submitted for sending; check your inbox and spam folder.' :
                    ' Your request is saved, but the confirmation email is not confirmed. Keep your reference and contact cloud.vana@outlook.com if needed; you do not need to register again.') :
                '';
            setMessage('Thanks for registering. Your request has been received.' + reference + emailNote +
                ' We will review your request and email you about next steps and any available session date and time. Your slot is not confirmed. No payment has been taken.',
                result.reference && result.emailStatus !== 'accepted' ? 'warning' : 'success');
            form.reset();
        } catch (error) {
            console.error('Mentorship submission could not be confirmed:', error);
            setMessage('We could not confirm delivery. Your entries are still here. You can retry, or contact Pavan on LinkedIn below. Please avoid repeated submissions if you have already received a confirmation email.', 'error');
        } finally {
            clearTimeout(timeout);
            submitting = false;
            submitButton.disabled = false;
            submitButton.textContent = 'Request mentorship review';
            form.removeAttribute('aria-busy');
        }
    });
    document.getElementById('mentorshipFields').disabled = false;
})();
