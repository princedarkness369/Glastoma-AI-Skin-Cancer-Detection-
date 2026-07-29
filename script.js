// ============================================================
// Glastoma AI · Application logic
// Navigation · Theme · Bilingual EN/AR + RTL · Camera · Mock
// inference · History (localStorage) · Dashboard
// ============================================================

const STORAGE = {
    theme:   'glastoma.theme',
    lang:    'glastoma.lang',
    history: 'glastoma.history',
};

const API_URL = 'http://localhost:8000';
const CLASSES = {
    nv:    { en: 'Melanocytic Nevi',         ar: 'شامات ميلانينية',         tone: 'nv',    risk: 'low'    },
    mel:   { en: 'Melanoma',                 ar: 'الورم الميلانيني',        tone: 'mel',   risk: 'high'   },
    bkl:   { en: 'Benign Keratosis',         ar: 'تقرن حميد',              tone: 'bkl',   risk: 'low'    },
    bcc:   { en: 'Basal Cell Carcinoma',     ar: 'سرطان الخلايا القاعدية',  tone: 'bcc',   risk: 'medium' },
    akiec: { en: 'Actinic Keratoses',        ar: 'تقران أكتيني',           tone: 'akiec', risk: 'medium' },
    vasc:  { en: 'Vascular Lesions',         ar: 'آفات وعائية',            tone: 'vasc',  risk: 'low'    },
    df:    { en: 'Dermatofibroma',           ar: 'ورم ليفي جلدي',          tone: 'df',    risk: 'low'    },
};

const PARAMETERS = [
    { key: 'asymmetry',  en: 'Asymmetry',         ar: 'عدم التناظر',         icon: 'fa-arrows-left-right' },
    { key: 'border',     en: 'Border irregularity', ar: 'عدم انتظام الحدود', icon: 'fa-border-all' },
    { key: 'color',      en: 'Color variation',   ar: 'تباين اللون',          icon: 'fa-palette' },
    { key: 'diameter',   en: 'Diameter (mm)',     ar: 'القطر (ملم)',           icon: 'fa-circle-dot' },
    { key: 'evolution',  en: 'Evolution',         ar: 'التطور',                icon: 'fa-arrows-rotate' },
    { key: 'texture',    en: 'Texture',           ar: 'الملمس',                icon: 'fa-wave-square' },
];

// State
let state = {
    lang: 'en',
    theme: 'dark',
    history: [],
    currentImage: null,
    currentResult: null,
    mediaStream: null,
};

// DOM refs
let dom = {};

// ============================================================
// Init
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    loadState();
    cacheDom();
    bindEvents();
    applyTheme();
    applyLanguage();
    renderHistory();
    renderDashboard();
    renderFeatureImportance();
    setTimeout(initCamera, 800);
    checkServerHealth();
});

function loadState() {
    state.theme   = localStorage.getItem(STORAGE.theme) || 'dark';
    state.lang    = localStorage.getItem(STORAGE.lang)  || 'en';
    state.history = JSON.parse(localStorage.getItem(STORAGE.history) || '[]');
}

function cacheDom() {
    dom = {
        sidenav:        document.getElementById('sidenav'),
        sidenavToggle:  document.getElementById('sidenavToggle'),
        sidenavClose:   document.getElementById('sidenavClose'),
        sidenavOverlay: document.getElementById('sidenavOverlay'),

        themeToggle:    document.getElementById('themeToggle'),
        themeIcon:      document.getElementById('themeIcon'),
        langToggle:     document.getElementById('langToggle'),

        pages:    document.querySelectorAll('.page'),
        navItems: document.querySelectorAll('.navlink'),
        brandLinks: document.querySelectorAll('[data-page]'),

        // Scan
        cameraContainer:   document.getElementById('cameraContainer'),
        cameraVideo:       document.getElementById('cameraVideo'),
        cameraCanvas:      document.getElementById('cameraCanvas'),
        cameraPlaceholder: document.getElementById('cameraPlaceholder'),
        captureBtn:        document.getElementById('captureBtn'),
        uploadBtn:         document.getElementById('uploadBtn'),
        fileInput:         document.getElementById('fileInput'),
        previewWrapper:    document.getElementById('previewWrapper'),
        previewImg:        document.getElementById('previewImg'),
        removePreview:     document.getElementById('removePreview'),
        revealBtn:         document.getElementById('revealBtn'),
        scanAnalyzeBtn:    document.getElementById('scanAnalyzeBtn'),
        scanResults:       document.getElementById('scanResults'),
        predictedClass:    document.getElementById('predictedClass'),
        confidenceFill:    document.getElementById('confidenceFill'),
        confidenceValue:   document.getElementById('confidenceValue'),
        topPredictionsList:document.getElementById('topPredictionsList'),
        parametersGrid:    document.getElementById('parametersGrid'),
        detailedInfo:      document.getElementById('detailedInfo'),
        saveResultBtn:     document.getElementById('saveResultBtn'),

        // History & dashboard
        historyList:       document.getElementById('historyList'),
        clearHistory:      document.getElementById('clearHistory'),
        activityList:      document.getElementById('activityList'),
        featureImportance: document.getElementById('featureImportance'),
        dashboardTotalScans: document.getElementById('dashboardTotalScans'),
        dashboardConcerns:   document.getElementById('dashboardConcerns'),
    };
}

function bindEvents() {
    // Navigation
    dom.sidenavToggle?.addEventListener('click', openSidenav);
    dom.sidenavClose?.addEventListener('click',  closeSidenav);
    dom.sidenavOverlay?.addEventListener('click', closeSidenav);

    dom.navItems.forEach(item => {
        item.addEventListener('click', e => {
            e.preventDefault();
            navigateTo(item.dataset.page);
            closeSidenav();
        });
    });

    // Brand / CTA buttons that carry data-page
    document.querySelectorAll('[data-page]').forEach(el => {
        if (el.classList.contains('navlink') || el.classList.contains('brand')) return;
        el.addEventListener('click', e => {
            e.preventDefault();
            navigateTo(el.dataset.page);
            closeSidenav();
        });
    });

    // Brand click -> home
    document.querySelector('.brand')?.addEventListener('click', e => {
        e.preventDefault(); navigateTo('home'); closeSidenav();
    });

    // Theme & language
    dom.themeToggle?.addEventListener('click', toggleTheme);
    dom.langToggle?.addEventListener('click',  toggleLanguage);

    // Camera / upload
    dom.captureBtn?.addEventListener('click', capturePhoto);
    dom.uploadBtn?.addEventListener('click',  () => dom.fileInput?.click());
    dom.fileInput?.addEventListener('change', handleFileUpload);
    dom.removePreview?.addEventListener('click', clearPreview);
    dom.scanAnalyzeBtn?.addEventListener('click', analyzeImage);

    // Reveal button (hold to unblur)
    if (dom.revealBtn) {
        const add    = () => dom.previewWrapper?.classList.add('revealed');
        const remove = () => dom.previewWrapper?.classList.remove('revealed');

        // Pointer Events cover mouse, touch, and stylus with one API, and
        // pointer capture keeps the "up" event reliably tied to this button
        // even if the finger drifts slightly — this is what was missing.
        dom.revealBtn.addEventListener('pointerdown', (e) => {
            e.preventDefault(); // stops the phone's long-press menu from stealing the gesture
            dom.revealBtn.setPointerCapture?.(e.pointerId);
            add();
        });
        dom.revealBtn.addEventListener('pointerup',     remove);
        dom.revealBtn.addEventListener('pointercancel', remove);
        dom.revealBtn.addEventListener('pointerleave',  remove);

        // Belt-and-suspenders: some mobile browsers still try to open a
        // context menu on a long press even after preventDefault above.
        dom.revealBtn.addEventListener('contextmenu', (e) => e.preventDefault());
    }

    // History actions
    dom.clearHistory?.addEventListener('click', clearAllHistory);
    dom.saveResultBtn?.addEventListener('click', saveCurrentResult);

    // Esc closes sidenav on mobile
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape') closeSidenav();
    });
}

// ============================================================
// Navigation
// ============================================================
function openSidenav()  { dom.sidenav?.classList.add('open'); dom.sidenavOverlay?.classList.add('active'); document.body.style.overflow = 'hidden'; }
function closeSidenav() { dom.sidenav?.classList.remove('open'); dom.sidenavOverlay?.classList.remove('active'); document.body.style.overflow = ''; }

function navigateTo(pageId) {
    dom.pages.forEach(p => p.classList.toggle('active', p.id === `${pageId}Page`));
    dom.navItems.forEach(n => n.classList.toggle('active', n.dataset.page === pageId));
    if (pageId === 'dashboard') renderDashboard();
    if (pageId === 'history')   renderHistory();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

window.navigateToScan = () => navigateTo('scan');

// ============================================================
// Theme
// ============================================================
function applyTheme() {
    document.documentElement.setAttribute('data-theme', state.theme);
    if (dom.themeIcon) {
        dom.themeIcon.className = state.theme === 'dark' ? 'fas fa-moon' : 'fas fa-sun';
    }
    // Update switch label
    const label = dom.themeToggle?.querySelector('.switch-label');
    if (label) {
        label.textContent = state.theme === 'dark'
            ? (state.lang === 'ar' ? 'الوضع الداكن' : 'Dark Mode')
            : (state.lang === 'ar' ? 'الوضع الفاتح' : 'Light Mode');
    }
    dom.themeToggle?.classList.toggle('active', state.theme === 'light');
}

function toggleTheme() {
    state.theme = state.theme === 'dark' ? 'light' : 'dark';
    localStorage.setItem(STORAGE.theme, state.theme);
    applyTheme();
}

// ============================================================
// Language (EN <-> AR) with full RTL
// ============================================================
function applyLanguage() {
    const html = document.documentElement;
    html.lang = state.lang;
    html.dir  = state.lang === 'ar' ? 'rtl' : 'ltr';

    // Translate all data-en / data-ar nodes
    document.querySelectorAll('[data-en]').forEach(el => {
        const text = el.getAttribute(state.lang === 'ar' ? 'data-ar' : 'data-en');
        if (text != null) el.textContent = text;
    });

    // Placeholders
    document.querySelectorAll(`[data-${state.lang}-placeholder]`).forEach(el => {
        el.placeholder = el.getAttribute(`data-${state.lang}-placeholder`);
    });

    // Update lang switch label
    const langLabel = dom.langToggle?.querySelector('.switch-label');
    if (langLabel) langLabel.textContent = state.lang === 'ar' ? 'العربية' : 'English';
    dom.langToggle?.classList.toggle('active', state.lang === 'ar');

    // Theme label
    applyTheme();

    // Re-render dynamic content
    renderHistory();
    renderDashboard();
}

function toggleLanguage() {
    state.lang = state.lang === 'en' ? 'ar' : 'en';
    localStorage.setItem(STORAGE.lang, state.lang);
    applyLanguage();
}

// ============================================================
// Camera
// ============================================================
async function initCamera() {
    try {
        if (!navigator.mediaDevices?.getUserMedia) {
            showCameraError('Camera not supported on this device or browser');
            return;
        }
        let stream;
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: { exact: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } }
            });
        } catch {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
        }
        state.mediaStream = stream;
        if (dom.cameraVideo) {
            dom.cameraVideo.srcObject = stream;
            dom.cameraVideo.style.display = 'block';
            if (dom.cameraPlaceholder) dom.cameraPlaceholder.style.display = 'none';
            await dom.cameraVideo.play();
        }
    } catch (err) {
        console.warn('Camera error:', err);
        showCameraError('Camera access denied. Please use the Upload button instead.');
    }
}

function showCameraError(msg) {
    if (!dom.cameraPlaceholder) return;
    const t = state.lang === 'ar' ? 'استخدم زر الرفع أدناه' : 'Use the Upload button below';
    dom.cameraPlaceholder.innerHTML = `
        <div class="placeholder-icon"><i class="fas fa-triangle-exclamation" style="color:#e8923c"></i></div>
        <p>${msg}</p>
        <small>${t}</small>
    `;
}

function capturePhoto() {
    if (!dom.cameraVideo || dom.cameraVideo.style.display === 'none') {
        alert(state.lang === 'ar' ? 'الكاميرا غير جاهزة. استخدم زر الرفع بدلاً من ذلك.' : 'Camera is not ready. Please use the Upload button instead.');
        return;
    }
    const canvas = dom.cameraCanvas;
    const ctx = canvas.getContext('2d');
    canvas.width  = dom.cameraVideo.videoWidth;
    canvas.height = dom.cameraVideo.videoHeight;
    ctx.drawImage(dom.cameraVideo, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(blob => {
        const file = new File([blob], 'capture.jpg', { type: 'image/jpeg' });
        handleImageFile(file);
    }, 'image/jpeg', 0.9);
}

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file) handleImageFile(file);
    event.target.value = '';
}

function handleImageFile(file) {
    if (!file.type.startsWith('image/')) {
        alert(state.lang === 'ar' ? 'يرجى اختيار ملف صورة صالح' : 'Please choose a valid image file');
        return;
    }
    if (file.size > 10 * 1024 * 1024) {
        alert(state.lang === 'ar' ? 'حجم الصورة يتجاوز 10MB' : 'Image exceeds 10MB limit');
        return;
    }
    const reader = new FileReader();
    reader.onload = e => {
        state.currentImage = e.target.result;
        dom.previewImg.src = e.target.result;
        dom.previewWrapper.style.display = 'block';
        dom.previewWrapper.classList.remove('revealed');
        if (dom.cameraPlaceholder) dom.cameraPlaceholder.style.display = 'none';
        if (dom.cameraVideo) dom.cameraVideo.style.display = 'none';
        dom.scanAnalyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

function clearPreview() {
    state.currentImage = null;
    state.currentResult = null;
    dom.previewWrapper.style.display = 'none';
    dom.previewWrapper.classList.remove('revealed');
    dom.previewImg.src = '';
    dom.scanAnalyzeBtn.disabled = true;
    dom.scanResults.style.display = 'none';
    if (dom.cameraVideo && state.mediaStream) {
        dom.cameraVideo.style.display = 'block';
    } else if (dom.cameraPlaceholder) {
        dom.cameraPlaceholder.style.display = '';
    }
}

// ============================================================
// Mock analysis
// ============================================================
async function analyzeImage() {
    if (!state.currentImage) return;

    dom.scanAnalyzeBtn.disabled = true;
    const original = dom.scanAnalyzeBtn.innerHTML;
    dom.scanAnalyzeBtn.innerHTML = `<i class="fas fa-spinner fa-spin"></i> <span>${state.lang === 'ar' ? 'جاري التحليل...' : 'Analyzing...'}</span>`;

    // Simulate inference
    await new Promise(r => setTimeout(r, 1800));

    const result = generateMockResult();
    state.currentResult = result;
    renderResult(result);

    dom.scanAnalyzeBtn.disabled = false;
    dom.scanAnalyzeBtn.innerHTML = original;
    dom.scanResults.style.display = 'block';
    dom.scanResults.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function generateMockResult() {
    const classKeys = Object.keys(CLASSES);
    const mainKey = classKeys[Math.floor(Math.random() * classKeys.length)];
    const mainConf = 0.65 + Math.random() * 0.3;

    const others = classKeys.filter(k => k !== mainKey).map(k => ({
        key: k,
        prob: Math.random() * (1 - mainConf) * 0.8,
    })).sort((a, b) => b.prob - a.prob).slice(0, 3);

    // Normalize others to fit
    const totalOthers = others.reduce((s, o) => s + o.prob, 0);
    others.forEach(o => o.prob = o.prob / totalOthers * (1 - mainConf));

    const predictions = [
        { key: mainKey, prob: mainConf },
        ...others,
    ];

    return {
        primary: mainKey,
        confidence: mainConf,
        predictions,
        parameters: PARAMETERS.map(p => ({
            ...p,
            value: 0.2 + Math.random() * 0.8,
        })),
        timestamp: Date.now(),
    };
}

function renderResult(result) {
    const cls = CLASSES[result.primary];
    dom.predictedClass.textContent = state.lang === 'ar' ? cls.ar : cls.en;

    const pct = Math.round(result.confidence * 100);
    dom.confidenceValue.textContent = `${pct}%`;
    dom.confidenceFill.style.width = `${pct}%`;

    // Top predictions
    dom.topPredictionsList.innerHTML = result.predictions.map(p => {
        const c = CLASSES[p.key];
        const label = state.lang === 'ar' ? c.ar : c.en;
        const prob = Math.round(p.prob * 100);
        return `
            <div class="prediction-item ${p.key === result.primary ? 'top' : ''}">
                <span class="pred-name">${label}</span>
                <span class="pred-bar"><span class="pred-fill" style="width:${prob}%"></span></span>
                <span class="pred-value">${prob}%</span>
            </div>
        `;
    }).join('');

    // Parameters
    dom.parametersGrid.innerHTML = result.parameters.map(p => {
        const v = Math.round(p.value * 100);
        const label = state.lang === 'ar' ? p.ar : p.en;
        const color = p.value > 0.66 ? '#e5484d' : p.value > 0.4 ? '#d68a2c' : '#2f9e6f';
        return `
            <div class="param-item">
                <div class="param-name"><i class="fas ${p.icon}"></i> ${label}</div>
                <div class="param-bar"><div class="param-fill" style="width:${v}%;background:${color}"></div></div>
                <div class="param-value">${v}%</div>
            </div>
        `;
    }).join('');

    // Detailed info
    const isHigh = cls.risk === 'high';
    const supportTitle = isHigh
        ? (state.lang === 'ar' ? 'يرجى استشارة طبيب جلدية فوراً' : 'Please consult a dermatologist immediately')
        : (state.lang === 'ar' ? 'ابقَ يقظاً وراقب أي تغيرات' : 'Stay vigilant and monitor any changes');
    const supportText = isHigh
        ? (state.lang === 'ar' ? 'هذا التقييم ليس تشخيصاً طبياً. احجز موعداً مع طبيب جلدية في أقرب وقت ممكن.' : 'This is not a medical diagnosis. Please book an appointment with a dermatologist as soon as possible.')
        : (state.lang === 'ar' ? 'تابع أي تغيرات في الحجم أو اللون أو الشكل. الفحص الدوري مهم.' : 'Monitor for any changes in size, color, or shape. Regular screening is important.');

    dom.detailedInfo.innerHTML = `
        <div class="psych-support ${isHigh ? 'urgent' : ''}">
            <i class="fas ${isHigh ? 'fa-triangle-exclamation' : 'fa-hand-holding-heart'}"></i>
            <h4 style="font-size:14px;margin-bottom:6px;">${supportTitle}</h4>
            <p style="font-size:13px;color:var(--text-muted);line-height:1.6;">${supportText}</p>
        </div>
    `;

    dom.saveResultBtn.style.display = 'flex';
}

function saveCurrentResult() {
    if (!state.currentResult || !state.currentImage) return;
    const r = state.currentResult;
    const cls = CLASSES[r.primary];
    const entry = {
        id: r.timestamp,
        classKey: r.primary,
        className: state.lang === 'ar' ? cls.ar : cls.en,
        confidence: r.confidence,
        image: state.currentImage,
        date: new Date(r.timestamp).toISOString(),
    };
    state.history.unshift(entry);
    localStorage.setItem(STORAGE.history, JSON.stringify(state.history));
    renderHistory();
    renderDashboard();

    const msg = state.lang === 'ar' ? 'تم الحفظ في السجل' : 'Saved to history';
    dom.saveResultBtn.innerHTML = `<i class="fas fa-check"></i> <span>${msg}</span>`;
    setTimeout(() => {
        dom.saveResultBtn.innerHTML = `<i class="fas fa-bookmark"></i> <span>${state.lang === 'ar' ? 'حفظ في السجل' : 'Save to History'}</span>`;
    }, 1800);
}

// ============================================================
// History
// ============================================================
function renderHistory() {
    if (!dom.historyList) return;
    if (state.history.length === 0) {
        const t1 = state.lang === 'ar' ? 'لا توجد فحوصات بعد' : 'No scans yet';
        const t2 = state.lang === 'ar' ? 'ابدأ فحصاً جديداً لرؤية النتائج هنا.' : 'Start a new scan to see results here.';
        dom.historyList.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-folder-open"></i>
                <h3>${t1}</h3>
                <p>${t2}</p>
            </div>
        `;
        return;
    }
    dom.historyList.innerHTML = state.history.map(h => {
        const date = new Date(h.date);
        const dateStr = state.lang === 'ar' ? date.toLocaleDateString('ar') : date.toLocaleDateString();
        const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const conf = Math.round(h.confidence * 100);
        return `
            <div class="history-item" data-id="${h.id}">
                <div class="history-thumb"><i class="fas fa-image"></i></div>
                <div class="history-info">
                    <div class="date">${dateStr} · ${timeStr}</div>
                    <div class="diagnosis">${h.className}</div>
                </div>
                <div class="history-meta">
                    <span class="chip ${CLASSES[h.classKey].risk === 'high' ? 'chip-danger' : CLASSES[h.classKey].risk === 'medium' ? 'chip-warning' : 'chip-success'}">
                        ${CLASSES[h.classKey].risk === 'high' ? (state.lang === 'ar' ? 'عالي الخطورة' : 'High risk') : CLASSES[h.classKey].risk === 'medium' ? (state.lang === 'ar' ? 'متوسط' : 'Moderate') : (state.lang === 'ar' ? 'منخفض' : 'Low')}
                    </span>
                    <span class="history-conf">${conf}%</span>
                </div>
            </div>
        `;
    }).join('');
}

function clearAllHistory() {
    const msg = state.lang === 'ar' ? 'هل أنت متأكد من مسح كل السجل؟' : 'Are you sure you want to clear all history?';
    if (!confirm(msg)) return;
    state.history = [];
    localStorage.setItem(STORAGE.history, '[]');
    renderHistory();
    renderDashboard();
}

// ============================================================
// Dashboard
// ============================================================
function renderDashboard() {
    if (!dom.dashboardTotalScans) return;
    const total = state.history.length;
    const concerns = state.history.filter(h => CLASSES[h.classKey].risk === 'high' || CLASSES[h.classKey].risk === 'medium').length;
    dom.dashboardTotalScans.textContent = total;
    dom.dashboardConcerns.textContent = concerns;

    // Activity list
    if (dom.activityList) {
        if (state.history.length === 0) {
            const t1 = state.lang === 'ar' ? 'لا يوجد نشاط بعد' : 'No activity yet';
            const t2 = state.lang === 'ar' ? 'ابدأ أول فحص لرؤية السجل هنا.' : 'Run your first scan to see it logged here.';
            dom.activityList.innerHTML = `
                <div class="empty-state" style="padding:20px 0">
                    <i class="fas fa-clock"></i>
                    <h3>${t1}</h3>
                    <p>${t2}</p>
                </div>
            `;
        } else {
            dom.activityList.innerHTML = state.history.slice(0, 5).map(h => {
                const date = new Date(h.date);
                const conf = Math.round(h.confidence * 100);
                const ago = relativeTime(date, state.lang);
                return `
                    <div class="activity-item">
                        <div class="activity-icon"><i class="fas fa-microscope"></i></div>
                        <div class="activity-text">
                            <div class="activity-title">${h.className}</div>
                            <div class="activity-time">${ago}</div>
                        </div>
                        <div class="activity-conf">${conf}%</div>
                    </div>
                `;
            }).join('');
        }
    }
}

function renderFeatureImportance() {
    if (!dom.featureImportance) return;
    const data = [
        { name: state.lang === 'ar' ? 'عدم التناظر' : 'Asymmetry',         v: 0.92 },
        { name: state.lang === 'ar' ? 'الحدود' : 'Border',                    v: 0.88 },
        { name: state.lang === 'ar' ? 'اللون' : 'Color',                      v: 0.85 },
        { name: state.lang === 'ar' ? 'القطر' : 'Diameter',                   v: 0.74 },
        { name: state.lang === 'ar' ? 'التطور' : 'Evolution',                 v: 0.69 },
        { name: state.lang === 'ar' ? 'الملمس' : 'Texture',                   v: 0.58 },
        { name: state.lang === 'ar' ? 'التصبغ' : 'Pigmentation',              v: 0.51 },
    ];
    dom.featureImportance.innerHTML = data.map(d => `
        <div class="feature-row">
            <span class="feature-name">${d.name}</span>
            <span class="feature-bar"><span class="feature-fill" style="width:${d.v * 100}%"></span></span>
            <span class="feature-pct">${Math.round(d.v * 100)}%</span>
        </div>
    `).join('');
}

function relativeTime(date, lang) {
    const diff = Date.now() - date.getTime();
    const min = Math.floor(diff / 60000);
    const hr  = Math.floor(min / 60);
    const day = Math.floor(hr / 24);
    if (lang === 'ar') {
        if (min < 1)  return 'الآن';
        if (min < 60) return `قبل ${min} دقيقة`;
        if (hr  < 24) return `قبل ${hr} ساعة`;
        if (day < 7)  return `قبل ${day} يوم`;
        return date.toLocaleDateString('ar');
    } else {
        if (min < 1)  return 'just now';
        if (min < 60) return `${min} min ago`;
        if (hr  < 24) return `${hr}h ago`;
        if (day < 7)  return `${day}d ago`;
        return date.toLocaleDateString();
    }
}

// ============================================================
// Health check (no-op in demo)
// ============================================================
async function checkServerHealth() {
    try {
        await fetch(API_URL + '/health', { method: 'GET' });
    } catch (e) {
        // Backend not available — we use local mock inference
    }
}
