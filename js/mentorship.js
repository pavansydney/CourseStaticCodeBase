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
            const acknowledged = result && (result.success === true || result.ok === true || result.status === 'success');
            if (!acknowledged) throw new Error('The submission service did not acknowledge the request.');
            setMessage('Your request has been received. We will review it and contact you by email. No payment has been taken.', 'success');
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
