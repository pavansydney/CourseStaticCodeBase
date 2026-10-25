(function () {
    'use strict';

    const memory = new Map();
    let warning = '';
    let persistent = true;

    function report(message, error) {
        if (warning === message) return;
        warning = message;
        console.warn(message, error);
        window.dispatchEvent(new CustomEvent('cloudvana:storage-warning'));
    }

    const storage = {
        get warning() { return warning; },
        get persistent() { return persistent; },
        reportInvalidData(error) {
            report('Some saved learning data could not be read. You can still learn, but previously saved progress may be unavailable.', error);
        },
        get(key) {
            if (memory.has(key)) return memory.get(key);
            if (!persistent) return null;
            try {
                return localStorage.getItem(key);
            } catch (error) {
                persistent = false;
                report('Browser storage is unavailable. Your preferences and progress will only last while this page is open.', error);
                return null;
            }
        },
        set(key, value) {
            memory.set(key, value);
            if (!persistent) return false;
            try {
                localStorage.setItem(key, value);
                return true;
            } catch (error) {
                persistent = false;
                report('Changes could not be saved to this device. Your preferences and progress will only last while this page is open.', error);
                return false;
            }
        },
        readObject(key) {
            const raw = storage.get(key);
            if (raw === null) return {};
            try {
                const value = JSON.parse(raw);
                if (!value || typeof value !== 'object' || Array.isArray(value)) {
                    throw new TypeError('Expected a saved object for ' + key);
                }
                return value;
            } catch (error) {
                storage.reportInvalidData(error);
                return {};
            }
        }
    };

    window.CloudvanaStorage = storage;
    const saved = storage.get('cloudvana-theme-v2');
    document.documentElement.dataset.theme = saved === 'dark' ? 'dark' : 'light';
})();
