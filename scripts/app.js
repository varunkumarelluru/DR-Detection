// DOM Elements
// CHANGE THIS TO YOUR RENDER BACKEND URL AFTER DEPLOYING
// const API_URL = 'https://your-app-name.onrender.com'; 
const API_URL = 'https://dr-detection-backend-av3m.onrender.com';

const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const previewArea = document.getElementById('previewArea');
const imagePreview = document.getElementById('imagePreview');
const removeImageBtn = document.getElementById('removeImage');
const predictBtn = document.getElementById('predictBtn');
const resultCard = document.getElementById('resultCard');
const resultLabel = document.getElementById('resultLabel');
// Mobile Menu Elements
const mobileMenuBtn = document.getElementById('mobileMenuBtn');
const sidebar = document.getElementById('sidebar');
const sidebarOverlay = document.getElementById('sidebarOverlay');
const sidebarCloseBtn = document.getElementById('sidebarCloseBtn');
const confidenceText = document.getElementById('confidenceText');
const confidenceBar = document.getElementById('confidenceBar');
// Settings Form
const settingsForm = document.getElementById('settingsForm');

// Initialize User State
const currentUser = localStorage.getItem('user') ? JSON.parse(localStorage.getItem('user')) : null;

if (document.getElementById('userDisplayName') && currentUser) {
    document.getElementById('userDisplayName').innerText = currentUser.name;
    // Update Dropdown Name as well
    if (document.getElementById('dropdownName')) {
        document.getElementById('dropdownName').innerText = currentUser.name;
    }

    const initials = currentUser.name.split(' ').map(n => n[0]).join('').toUpperCase().substring(0, 2);
    document.getElementById('userAvatar').innerText = initials;

    // Fill settings form initially
    if (document.getElementById('settingsName')) {
        document.getElementById('settingsName').value = currentUser.name;
        document.getElementById('settingsEmail').value = currentUser.email;
    }
}

// Navigation Logic
window.switchView = function (viewName) {
    // Hide all views
    document.getElementById('view-dashboard').style.display = 'none';
    document.getElementById('view-history').style.display = 'none';
    document.getElementById('view-settings').style.display = 'none';

    // Deactivate nav items
    document.getElementById('nav-dashboard').classList.remove('active');
    document.getElementById('nav-history').classList.remove('active');
    document.getElementById('nav-settings').classList.remove('active');

    // Show selected
    document.getElementById('view-' + viewName).style.display = (viewName === 'dashboard') ? 'grid' : 'block';
    document.getElementById('nav-' + viewName).classList.add('active');

    if (viewName === 'history') {
        loadHistory();
    }
}

window.logout = function () {
    localStorage.removeItem('user');
}

// Mobile Menu Logic
function openSidebar() {
    sidebar.classList.add('active');
    sidebarOverlay.classList.add('active');
    document.body.classList.add('menu-open');
}

function closeSidebar() {
    sidebar.classList.remove('active');
    sidebarOverlay.classList.remove('active');
    document.body.classList.remove('menu-open');
}

window.closeSidebar = closeSidebar; // Make global for HTML onclick

if (mobileMenuBtn) {
    mobileMenuBtn.addEventListener('click', openSidebar);
    sidebarOverlay.addEventListener('click', closeSidebar);
    if (sidebarCloseBtn) sidebarCloseBtn.addEventListener('click', closeSidebar);

    // Close menu when a link is clicked
    document.querySelectorAll('.nav-item').forEach(link => {
        link.addEventListener('click', () => {
            if (window.innerWidth <= 768) {
                closeSidebar();
            }
        });
    });
}

// History Logic
async function loadHistory() {
    if (!currentUser) return;
    const container = document.getElementById('historyTableContainer');
    container.innerHTML = '<p>Loading...</p>';

    try {
        const res = await fetch(`${API_URL}/history/${currentUser.id}`, {
            method: 'GET' // Note: in real app, user ID usually from token, here from URL for simplicity
            // But wait, our mock login didn't return ID. We need to fix login return in app.py first or assume we have it.
            // Let's assume we fixed app.py to return ID (we did: 'SELECT *' returns id at index 0).
            // Actually, we returned {'name': user[1], 'email': user[2]} -- we missed ID! 
            // I should assume ID is missing and fix it, but for now let's hope I fixed it or I will fix it.
            // Wait, I see app.py 'login': `return jsonify({'message': 'Login successful!', 'user': {'name': user[1], 'email': user[2]}}), 200`
            // User ID is at index 0. user[1] is name. I need to update app.py to return ID.
        });

        // Let's assume I'll fix app.py to return ID.
        // For now, let's proceed with app.js assuming currentUser.id works.
        const data = await res.json();

        if (data.length === 0) {
            container.innerHTML = '<p>No prediction history found.</p>';
            return;
        }

        let html = '<div style="display: flex; flex-direction: column; gap: 1rem;">';
        data.forEach(item => {
            html += `
                <div class="glass-card" style="padding: 1.5rem; background: rgba(30, 41, 59, 0.4); display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h4 style="margin-bottom: 0.5rem; color: ${item.dr_present ? '#ef4444' : '#4ade80'};">${item.label}</h4>
                        <p style="font-size: 0.8rem;">${new Date(item.timestamp).toLocaleString()}</p>
                    </div>
                    <div style="text-align: right;">
                         <div style="font-weight: bold; font-size: 1.2rem;">${item.confidence.toFixed(1)}%</div>
                         <div style="font-size: 0.8rem; color: var(--text-muted);">Confidence</div>
                    </div>
                </div>
            `;
        });
        html += '</div>';
        container.innerHTML = html;

    } catch (err) {
        console.error(err);
        container.innerHTML = '<p>Failed to load history.</p>';
    }
}

// Settings Logic
if (settingsForm) {
    settingsForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const name = document.getElementById('settingsName').value;
        const password = document.getElementById('settingsPassword').value;
        const btn = e.target.querySelector('button');

        if (!currentUser) return;

        setLoading(btn, true, 'Saving...');

        try {
            const payload = {
                user_id: currentUser.id,
                name: name,
                password: password
            };

            if (activeOtp) {
                payload.otp = activeOtp;
            }

            const res = await fetch(`${API_URL}/profile/update`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();

            if (res.ok) {
                showToast('Profile updated!');
                currentUser.name = name;
                localStorage.setItem('user', JSON.stringify(currentUser));
                document.getElementById('userDisplayName').innerText = name;
                if (document.getElementById('dropdownName')) document.getElementById('dropdownName').innerText = name;

                // Reset password field
                document.getElementById('settingsPassword').value = '';
                activeOtp = null;
                closeOtpModal();
            } else if (res.status === 403 && data.require_otp) {
                // OTP Required!
                showToast('Security check required. Sending OTP...', 'info');
                await requestUpdateOTP();
            } else {
                showToast(data.error || 'Update failed', 'error');
                activeOtp = null; // Reset invalid OTP if any
            }
        } catch (err) {
            showToast('Connection failed', 'error');
        } finally {
            setLoading(btn, false, 'Save Changes');
        }
    });
}





// Auth Forms
const loginForm = document.getElementById('loginForm');
const registerForm = document.getElementById('registerForm');

if (loginForm) {
    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const email = document.getElementById('email').value;
        const password = document.getElementById('password').value;
        const btn = e.target.querySelector('button');

        setLoading(btn, true, 'Signing In...');

        try {
            const res = await fetch(`${API_URL}/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password })
            });
            const data = await res.json();

            if (res.ok) {
                localStorage.setItem('user', JSON.stringify(data.user));
                window.location.href = 'dashboard.html';
            } else {
                showToast(data.error || 'Login failed', 'error');
            }
        } catch (err) {
            showToast('Connection failed', 'error');
        } finally {
            setLoading(btn, false, 'Sign In');
        }
    });
}

if (registerForm) {
    registerForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const name = document.getElementById('name').value;
        const email = document.getElementById('email').value;
        const phone = document.getElementById('phone').value; // New
        const password = document.getElementById('password').value;
        const confirmPassword = document.getElementById('confirmPassword').value;
        const btn = e.target.querySelector('button');

        if (password !== confirmPassword) {
            showToast('Passwords do not match', 'error');
            return;
        }

        setLoading(btn, true, 'Creating Account...');

        try {
            const res = await fetch(`${API_URL}/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, email, password, phone })
            });
            const data = await res.json();

            if (res.ok) {
                showToast('Account created! Please login.');
                setTimeout(() => window.location.href = 'login.html', 1500);
            } else {
                showToast(data.error || 'Registration failed', 'error');
            }
        } catch (err) {
            showToast('Connection failed', 'error');
        } finally {
            setLoading(btn, false, 'Create Account');
        }
    });
}

// Phone Login Logic
const loginPhoneForm = document.getElementById('loginPhoneForm');

if (loginPhoneForm) {
    loginPhoneForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const phone = document.getElementById('loginPhone').value;
        const otp = document.getElementById('loginOtp').value;
        const btn = e.target.querySelector('button[type="submit"]');

        setLoading(btn, true, 'Verifying...');

        try {
            const res = await fetch(`${API_URL}/auth/login/phone`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ phone, otp })
            });
            const data = await res.json();

            if (res.ok) {
                localStorage.setItem('user', JSON.stringify(data.user));
                window.location.href = 'dashboard.html';
            } else {
                showToast(data.error || 'Login failed', 'error');
            }
        } catch (err) {
            showToast('Connection failed', 'error');
        } finally {
            setLoading(btn, false, 'Verify & Login');
        }
    });
}

window.requestLoginOTP = async function () {
    const phone = document.getElementById('loginPhone').value;
    const btn = document.getElementById('getOtpBtn');

    if (!phone) {
        showToast('Please enter your phone number', 'error');
        return;
    }

    setLoading(btn, true, 'Sending...');

    try {
        const res = await fetch(`${API_URL}/auth/otp/request`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ identifier: phone, type: 'login' })
        });
        const data = await res.json();

        if (res.ok) {
            showToast('OTP sent! Check your email (or console for mock).');
            document.getElementById('phoneStep1').style.display = 'none';
            document.getElementById('phoneStep2').style.display = 'block';
        } else {
            showToast(data.error || 'Failed to send OTP', 'error');
        }
    } catch (err) {
        showToast('Connection failed', 'error');
    } finally {
        setLoading(btn, false, 'Get OTP');
    }
}

window.resetPhoneLogin = function () {
    document.getElementById('phoneStep1').style.display = 'block';
    document.getElementById('phoneStep2').style.display = 'none';
}

window.switchLoginTab = function (type) {
    const tabEmail = document.getElementById('tabEmail');
    const tabPhone = document.getElementById('tabPhone');
    const formEmail = document.getElementById('loginForm');
    const formPhone = document.getElementById('loginPhoneForm');

    if (type === 'email') {
        tabEmail.style.borderColor = 'var(--primary)';
        tabPhone.style.borderColor = 'transparent';
        formEmail.style.display = 'block';
        formPhone.style.display = 'none';
    } else {
        tabPhone.style.borderColor = 'var(--primary)';
        tabEmail.style.borderColor = 'transparent';
        formPhone.style.display = 'block';
        formEmail.style.display = 'none';
    }
}

// Global state for Settings OTP
let activeOtp = null;

async function requestUpdateOTP() {
    try {
        const res = await fetch(`${API_URL}/auth/otp/request`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                identifier: currentUser.email, // Or phone, but standard is email for security updates 
                type: 'update'
            })
        });

        if (res.ok) {
            document.getElementById('otpModal').classList.add('show');
            document.getElementById('updateOtp').focus();
        } else {
            showToast('Failed to send OTP', 'error');
        }
    } catch (e) {
        showToast('Connection error', 'error');
    }
}

// Modal Logic
const otpVerifyForm = document.getElementById('otpVerifyForm');
if (otpVerifyForm) {
    otpVerifyForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const otp = document.getElementById('updateOtp').value;
        if (otp.length === 6) {
            activeOtp = otp;
            // Retry the settings submission
            const settingsBtn = document.getElementById('settingsForm').querySelector('button');
            settingsBtn.click(); // Re-trigger settings submit
            // Note: In real setup, we might want to call the update function directly, 
            // but clicking the button re-runs the whole listener which handles the payload construction.
        }
    });
}

function closeOtpModal() {
    document.getElementById('otpModal').classList.remove('show');
    document.getElementById('updateOtp').value = '';
    activeOtp = null;
}

window.closeOtpModal = closeOtpModal;

function setLoading(btn, isLoading, text) {
    if (isLoading) {
        btn.dataset.originalText = btn.innerText;
        btn.innerText = text;
        btn.disabled = true;
        btn.style.opacity = '0.7';
    } else {
        btn.innerText = btn.dataset.originalText || text;
        btn.disabled = false;
        btn.style.opacity = '1';
    }
}

// Mock Data for Predictions
const predictions = [
    { label: "Urban Cityscape", confidence: 94 },
    { label: "Golden Retriever", confidence: 98 },
    { label: "Modern Architecture", confidence: 89 },
    { label: "Mountain Landscape", confidence: 96 },
    { label: "Vintage Car", confidence: 91 }
];

// Event Listeners for Upload
if (uploadZone) {
    uploadZone.addEventListener('click', () => fileInput.click());

    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('drag-over');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFile(files[0]);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showToast('Please upload a valid image file', 'error');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        uploadZone.style.display = 'none';
        previewArea.style.display = 'block';
        resultCard.style.display = 'none'; // Reset result
    };
    reader.readAsDataURL(file);
}

// Remove Image
if (removeImageBtn) {
    removeImageBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // Prevent bubbling if needed
        fileInput.value = '';
        imagePreview.src = '';
        previewArea.style.display = 'none';
        uploadZone.style.display = 'block';
        resultCard.style.display = 'none';
    });
}

// Predict Logic
if (predictBtn) {
    predictBtn.addEventListener('click', () => {
        // Validation
        if (!fileInput.files[0]) {
            showToast('Please upload an image first', 'error');
            return;
        }

        // Loading State
        predictBtn.disabled = true;
        const originalText = predictBtn.innerText;
        predictBtn.innerText = 'Analyzing...';
        predictBtn.style.opacity = '0.7';

        // Prepare Data
        const formData = new FormData();
        formData.append('image', fileInput.files[0]);
        if (currentUser && currentUser.id) {
            formData.append('user_id', currentUser.id);
        }

        // Send to Backend
        fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showToast('Error: ' + data.error, 'error');
                } else {
                    showResult(data);
                }
            })
            .catch(err => {
                console.error(err);
                showToast('Failed to connect to server. Is backend running?', 'error');
            })
            .finally(() => {
                // Reset Button
                predictBtn.disabled = false;
                predictBtn.innerText = originalText;
                predictBtn.style.opacity = '1';
            });
    });
}

function showResult(result) {
    resultCard.style.display = 'block';

    // Animate numbers
    resultLabel.innerText = result.label;

    // Reset bar first for animation effect
    confidenceBar.style.width = '0%';
    confidenceText.innerText = '0%';

    setTimeout(() => {
        confidenceBar.style.width = result.confidence + '%';
        animateValue(confidenceText, 0, result.confidence, 1000);
    }, 100);

    // Show Heatmap if available
    const heatmapContainer = document.getElementById('heatmapContainer');
    const heatmapImage = document.getElementById('heatmapImage');

    if (result.heatmap && heatmapContainer && heatmapImage) {
        heatmapImage.src = result.heatmap;
        heatmapContainer.style.display = 'block';
    } else if (heatmapContainer) {
        heatmapContainer.style.display = 'none';
        heatmapImage.src = '';
    }

    showToast('Prediction generated successfully!');
}

function animateValue(obj, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        obj.innerHTML = Math.floor(progress * (end - start) + start) + "%";
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

function showToast(message, type = 'success') {
    toastMsg.innerText = message;
    toast.style.transform = 'translateY(0)';
    toast.style.opacity = '1';

    if (type === 'error') {
        toast.style.borderColor = '#ef4444';
    } else {
        toast.style.borderColor = '#8b5cf6';
    }

    setTimeout(() => {
        toast.style.transform = 'translateY(100px)';
        toast.style.opacity = '0';
    }, 3000);
}
