/* ============================================
   CreditScoringAI - Main JavaScript
   ============================================ */

document.addEventListener('DOMContentLoaded', function() {
    initDarkMode();
    initToast();
    initAnimatedCounters();
    initFormValidation();
});

/* ============ Dark Mode Toggle ============ */
function initDarkMode() {
    const toggle = document.getElementById('darkModeToggle');
    if (!toggle) return;

    // Load saved preference
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        document.documentElement.setAttribute('data-bs-theme', 'dark');
        toggle.innerHTML = '<i class="bi bi-sun"></i>';
    }

    toggle.addEventListener('click', function() {
        const current = document.documentElement.getAttribute('data-bs-theme');
        const newTheme = current === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-bs-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        this.innerHTML = newTheme === 'dark'
            ? '<i class="bi bi-sun"></i>'
            : '<i class="bi bi-moon-stars"></i>';

        // Show toast
        showToast(newTheme === 'dark' ? '🌙 Dark mode enabled' : '☀️ Light mode enabled', 'info');
    });
}

/* ============ Toast Notifications ============ */
function initToast() {
    window.showToast = function(message, type = 'success') {
        const container = document.getElementById('toastContainer');
        if (!container) return;
        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-bg-${type} border-0 show`;
        toast.role = 'alert';
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">${message}</div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;
        container.appendChild(toast);
        setTimeout(() => { toast.remove(); }, 4000);
    };
}

/* ============ Animated Counters ============ */
function initAnimatedCounters() {
    const counters = document.querySelectorAll('.counter');
    if (!counters.length) return;

    const options = {
        root: null,
        rootMargin: '0px',
        threshold: 0.5,
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const el = entry.target;
                const target = parseInt(el.getAttribute('data-target')) || 0;
                const suffix = el.getAttribute('data-suffix') || '';
                const speed = Math.max(30, 2000 / target);
                let current = 0;
                const increment = Math.max(1, Math.ceil(target / 50));

                const timer = setInterval(() => {
                    current += increment;
                    if (current >= target) {
                        clearInterval(timer);
                        el.textContent = target.toLocaleString() + suffix;
                    } else {
                        el.textContent = current.toLocaleString() + suffix;
                    }
                }, speed);

                observer.unobserve(el);
            }
        });
    }, options);

    counters.forEach(c => observer.observe(c));
}

/* ============ Form Validation ============ */
function initFormValidation() {
    const form = document.getElementById('predictionForm');
    if (!form) return;

    form.addEventListener('submit', function(e) {
        const inputs = this.querySelectorAll('input[required], select[required]');
        let valid = true;

        inputs.forEach(input => {
            if (!input.value) {
                input.classList.add('is-invalid');
                valid = false;
            } else {
                input.classList.remove('is-invalid');
            }

            // Numeric validation
            if (input.type === 'number' && input.value) {
                const min = parseFloat(input.min);
                const max = parseFloat(input.max);
                const val = parseFloat(input.value);
                if ((!isNaN(min) && val < min) || (!isNaN(max) && val > max)) {
                    input.classList.add('is-invalid');
                    valid = false;
                }
            }
        });

        if (!valid) {
            e.preventDefault();
            showToast('⚠️ Please fix the highlighted fields', 'warning');
        }
    });
}

/* ============ Auto-close alerts after 5 seconds ============ */
setTimeout(() => {
    document.querySelectorAll('.alert-dismissible').forEach(a => {
        const bsAlert = new bootstrap.Alert(a);
        bsAlert.close();
    });
}, 5000);
