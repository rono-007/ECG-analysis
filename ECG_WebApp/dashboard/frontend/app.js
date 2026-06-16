/**
 * CardioScan AI — Dashboard Application Logic
 * Handles API communication, ECG rendering, and all interactive UI.
 */

const API = '';  // Same origin — served by FastAPI

// ── Class color map ──────────────────────────────────────────────
const CLASS_COLORS = {
    Normal: '#16a34a',
    MI:     '#dc2626',
    STTC:   '#ea580c',
    CD:     '#7c3aed',
    HYP:    '#db2777',
};

// ── DOM Refs ─────────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const slider        = $('#patientSlider');
const idxDisplay    = $('#patientIdxDisplay');
const btnAnalyze    = $('#btnAnalyze');
const welcomeState  = $('#welcomeState');
const resultsState  = $('#resultsState');
const loadingOverlay = $('#loadingOverlay');

// ── Patient Slider ───────────────────────────────────────────────
slider.addEventListener('input', () => {
    idxDisplay.textContent = slider.value;
});

// ── Tab Navigation ───────────────────────────────────────────────
let trRevealed = false;

$$('.nav-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        // Update active tab
        $$('.nav-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');

        // Show matching panel
        const panelId = tab.dataset.panel;
        $$('.page-panel').forEach(p => p.classList.remove('active'));
        $(`#panel-${panelId}`).classList.add('active');

        // Update title
        if (panelId === 'diagnostics') {
            $('#pageTitle').textContent = 'Diagnostic Center';
        } else if (panelId === 'technical-review') {
            $('#pageTitle').textContent = 'Technical Review';
            // Show static content and trigger reveals on first visit
            if (!trRevealed) {
                trRevealed = true;
                $('#insightsLoading').style.display = 'none';
                $('#insightsContent').style.display = 'block';
                requestAnimationFrame(() => observeReveals());
            }
        }
    });
});

// ── Accordion Toggles ────────────────────────────────────────────
$$('.tr-toggle').forEach(toggle => {
    toggle.addEventListener('click', () => {
        const targetId = toggle.dataset.target;
        const body = $(`#${targetId}`);
        const isActive = toggle.classList.contains('active');

        // Toggle
        if (isActive) {
            toggle.classList.remove('active');
            body.classList.remove('active');
            toggle.querySelector('.tr-toggle-arrow').textContent = '▸';
        } else {
            toggle.classList.add('active');
            body.classList.add('active');
            toggle.querySelector('.tr-toggle-arrow').textContent = '▾';
        }
    });
});

// ── Gallery ──────────────────────────────────────────────────────
async function loadGallery() {
    try {
        const res = await fetch(`${API}/api/gallery`);
        const data = await res.json();
        const grid = $('#galleryGrid');
        grid.innerHTML = '';

        data.forEach(item => {
            const card = document.createElement('div');
            card.className = 'gallery-card';
            card.setAttribute('data-class', item.class_name);
            card.innerHTML = `
                <h4 style="color:${item.color}">${item.class_name}</h4>
                <p>Case Index: ${item.index}</p>
                <button class="gallery-btn" style="color:${item.color}; border-color:${item.color}">
                    Analyze
                </button>
            `;

            // Inject ::after pseudo-element color via a style tag
            const styleId = `gc-${item.class_name}`;
            if (!document.getElementById(styleId)) {
                const s = document.createElement('style');
                s.id = styleId;
                s.textContent = `.gallery-card[data-class="${item.class_name}"]::after{background:${item.color}}`;
                document.head.appendChild(s);
            }

            card.addEventListener('click', () => {
                slider.value = item.index;
                idxDisplay.textContent = item.index;
                runAnalysis(item.index);
            });

            grid.appendChild(card);
        });
    } catch (e) {
        console.error('Gallery load failed:', e);
    }
}

// ── Analysis ─────────────────────────────────────────────────────
btnAnalyze.addEventListener('click', () => {
    runAnalysis(parseInt(slider.value));
});

async function runAnalysis(patientIdx) {
    // Switch to diagnostics tab first
    $$('.nav-tab').forEach(t => t.classList.remove('active'));
    $('#nav-diagnostics').classList.add('active');
    $$('.page-panel').forEach(p => p.classList.remove('active'));
    $('#panel-diagnostics').classList.add('active');
    $('#pageTitle').textContent = 'Diagnostic Center';

    // UI: loading state
    btnAnalyze.classList.add('loading');
    loadingOverlay.classList.add('active');

    try {
        const res = await fetch(`${API}/api/analyze/${patientIdx}`);
        const data = await res.json();
        renderResults(data);
    } catch (e) {
        console.error('Analysis failed:', e);
        alert('Analysis failed. Is the server running?');
    } finally {
        btnAnalyze.classList.remove('loading');
        loadingOverlay.classList.remove('active');
    }
}

function renderResults(data) {
    // Switch to results view
    welcomeState.style.display = 'none';
    resultsState.style.display = 'block';

    // Hero banner
    $('#caseIdLabel').textContent = `#${data.case_id}`;
    const detEl = $('#metricDetection');
    detEl.textContent = data.prediction;
    detEl.style.color = data.pred_color;
    $('#metricDetectionSub').textContent = `Predicted by ResNet-1D`;
    $('#metricConfidence').textContent = `${data.confidence}%`;
    $('#metricTruth').textContent = data.ground_truth.join(', ');

    // Confidence ring
    drawConfidenceRing(data.confidence, data.pred_color);

    // Donut chart
    renderDonutChart(data.probabilities);

    // ECG Grid
    renderECG(data);
}

// ── Confidence Ring ──────────────────────────────────────────────
function drawConfidenceRing(pct, color) {
    const canvas = $('#confidenceRing');
    const dpr = window.devicePixelRatio || 1;
    const size = 130;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width = size + 'px';
    canvas.style.height = size + 'px';

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    const cx = size / 2, cy = size / 2, r = 52, lw = 10;

    // Background track
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.strokeStyle = '#f0f0f5';
    ctx.lineWidth = lw;
    ctx.lineCap = 'round';
    ctx.stroke();

    // Animated fill arc
    const endAngle = -Math.PI / 2 + (pct / 100) * Math.PI * 2;
    ctx.beginPath();
    ctx.arc(cx, cy, r, -Math.PI / 2, endAngle);
    ctx.strokeStyle = color || '#1e40af';
    ctx.lineWidth = lw;
    ctx.lineCap = 'round';
    ctx.stroke();

    // Glow effect
    ctx.beginPath();
    ctx.arc(cx, cy, r, -Math.PI / 2, endAngle);
    ctx.strokeStyle = color || '#1e40af';
    ctx.lineWidth = lw + 6;
    ctx.globalAlpha = 0.12;
    ctx.lineCap = 'round';
    ctx.stroke();
    ctx.globalAlpha = 1;
}

// ── Donut Pie Chart ──────────────────────────────────────────────
function renderDonutChart(probs) {
    const canvas = $('#probDonut');
    const legend = $('#probLegend');
    const dpr = window.devicePixelRatio || 1;
    const size = 220;

    canvas.width = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width = size + 'px';
    canvas.style.height = size + 'px';

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    const entries = Object.entries(probs);
    const total = entries.reduce((s, [, v]) => s + v, 0) || 1;
    const cx = size / 2, cy = size / 2, outerR = 95, innerR = 58;

    let startAngle = -Math.PI / 2;

    // Draw slices
    entries.forEach(([name, value]) => {
        const color = CLASS_COLORS[name] || '#6b7280';
        const sliceAngle = (value / total) * Math.PI * 2;

        ctx.beginPath();
        ctx.moveTo(cx + innerR * Math.cos(startAngle), cy + innerR * Math.sin(startAngle));
        ctx.arc(cx, cy, outerR, startAngle, startAngle + sliceAngle);
        ctx.arc(cx, cy, innerR, startAngle + sliceAngle, startAngle, true);
        ctx.closePath();
        ctx.fillStyle = color;
        ctx.fill();

        // Thin separator line
        ctx.beginPath();
        ctx.moveTo(cx + innerR * Math.cos(startAngle), cy + innerR * Math.sin(startAngle));
        ctx.lineTo(cx + outerR * Math.cos(startAngle), cy + outerR * Math.sin(startAngle));
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();

        startAngle += sliceAngle;
    });

    // Center circle (to create donut hole)
    ctx.beginPath();
    ctx.arc(cx, cy, innerR - 1, 0, Math.PI * 2);
    ctx.fillStyle = '#fff';
    ctx.fill();

    // Center text
    ctx.fillStyle = '#1f2937';
    ctx.font = '700 1.1rem Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Classes', cx, cy - 8);
    ctx.font = '500 0.7rem Inter, sans-serif';
    ctx.fillStyle = '#9ca3af';
    ctx.fillText(`${entries.length} diagnoses`, cx, cy + 12);

    // Build legend
    legend.innerHTML = '';
    entries.forEach(([name, value]) => {
        const color = CLASS_COLORS[name] || '#6b7280';
        const item = document.createElement('div');
        item.className = 'prob-legend-item';
        item.innerHTML = `
            <div class="prob-legend-dot" style="background:${color}"></div>
            <span class="prob-legend-name">${name}</span>
            <span class="prob-legend-value">${value.toFixed(1)}%</span>
        `;
        legend.appendChild(item);
    });
}


// ── ECG Zoom State ───────────────────────────────────────────────
let currentEcgZoom = 1.0;
let lastEcgData = null;

const btnZoomIn = $('#btnZoomIn');
const btnZoomOut = $('#btnZoomOut');
const btnZoomReset = $('#btnZoomReset');
const zoomLevelDisplay = $('#zoomLevelDisplay');

function applyZoom() {
    // Clamp zoom
    if (currentEcgZoom < 1.0) currentEcgZoom = 1.0;
    if (currentEcgZoom > 5.0) currentEcgZoom = 5.0;

    zoomLevelDisplay.textContent = Math.round(currentEcgZoom * 100) + '%';

    if (lastEcgData) {
        renderECG(lastEcgData);
    }
}

if (btnZoomIn) {
    btnZoomIn.addEventListener('click', () => { currentEcgZoom += 0.25; applyZoom(); });
    btnZoomOut.addEventListener('click', () => { currentEcgZoom -= 0.25; applyZoom(); });
    btnZoomReset.addEventListener('click', () => { currentEcgZoom = 1.0; applyZoom(); });
}

// ── ECG Rendering ────────────────────────────────────────────────
function renderECG(data) {
    // Cache for resizing / zooming
    lastEcgData = data;

    const grid = $('#ecgGrid');
    grid.innerHTML = '';
    
    // Expand grid width based on zoom, causing CSS flex columns to grow
    grid.style.minWidth = (currentEcgZoom * 100) + '%';

    const { signals, reference, saliency, lead_names, time, pred_color } = data;

    lead_names.forEach((leadName, i) => {
        const cell = document.createElement('div');
        cell.className = 'ecg-lead';

        const label = document.createElement('span');
        label.className = 'lead-label';
        label.textContent = leadName;

        const canvas = document.createElement('canvas');
        cell.appendChild(label);
        cell.appendChild(canvas);
        grid.appendChild(cell);

        // Draw after DOM insertion and layout calculation
        requestAnimationFrame(() => drawLead(canvas, signals[i], reference[i], saliency, pred_color));
    });
}

function drawLead(canvas, signal, reference, saliency, predColor) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const w = rect.width - 24; // padding
    const h = 80;

    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    const len = signal.length;
    const xStep = w / len;

    // Find Y range
    let yMin = Infinity, yMax = -Infinity;
    for (let j = 0; j < len; j++) {
        if (signal[j] < yMin) yMin = signal[j];
        if (signal[j] > yMax) yMax = signal[j];
        if (reference[j] < yMin) yMin = reference[j];
        if (reference[j] > yMax) yMax = reference[j];
    }
    const yPad = (yMax - yMin) * 0.15 || 1;
    yMin -= yPad; yMax += yPad;

    const toY = (v) => h - ((v - yMin) / (yMax - yMin)) * h;

    // Grid lines (subtle medical paper look)
    ctx.strokeStyle = '#fde8e8';
    ctx.lineWidth = 0.5;
    for (let gy = 0; gy < h; gy += 16) {
        ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
    }
    for (let gx = 0; gx < w; gx += 20) {
        ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, h); ctx.stroke();
    }

    // Saliency highlight
    if (saliency) {
        for (let j = 0; j < len; j++) {
            if (saliency[j] > 0.68) {
                const alpha = Math.min((saliency[j] - 0.68) * 3, 0.35);
                ctx.fillStyle = predColor + Math.round(alpha * 255).toString(16).padStart(2, '0');
                ctx.fillRect(j * xStep, 0, xStep + 1, h);
            }
        }
    }

    // Reference trace (faint blue fill)
    ctx.beginPath();
    ctx.moveTo(0, toY(reference[0]));
    for (let j = 1; j < len; j++) ctx.lineTo(j * xStep, toY(reference[j]));
    ctx.lineTo(w, h); ctx.lineTo(0, h);
    ctx.closePath();
    ctx.fillStyle = 'rgba(59,130,246,0.06)';
    ctx.fill();

    // Signal trace
    ctx.beginPath();
    ctx.moveTo(0, toY(signal[0]));
    for (let j = 1; j < len; j++) ctx.lineTo(j * xStep, toY(signal[j]));
    ctx.strokeStyle = '#1f2937';
    ctx.lineWidth = 1.2;
    ctx.stroke();
}



// ── Init ─────────────────────────────────────────────────────
loadGallery();

// ── Scroll Reveal (IntersectionObserver) ─────────────────────
const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('visible');
            revealObserver.unobserve(entry.target);
        }
    });
}, {
    threshold: 0.08,
    rootMargin: '0px 0px -40px 0px'
});

function observeReveals() {
    $$('.reveal').forEach((el, idx) => {
        // Add stagger delay based on position
        el.style.transitionDelay = `${idx * 0.06}s`;
        revealObserver.observe(el);
    });
}

// Observe any reveals already in the DOM
observeReveals();
// ── Collapsible Sidebar ──────────────────────────────────────────
const sidebar = $('.sidebar');
const mainContent = $('.main-content');
const sidebarToggle = $('#sidebarToggle');

function toggleSidebar(collapsed) {
    if (collapsed === undefined) {
        collapsed = !sidebar.classList.contains('collapsed');
    }
    
    if (collapsed) {
        sidebar.classList.add('collapsed');
        mainContent.classList.add('sidebar-collapsed');
        sidebarToggle.classList.add('collapsed');
    } else {
        sidebar.classList.remove('collapsed');
        mainContent.classList.remove('sidebar-collapsed');
        sidebarToggle.classList.remove('collapsed');
    }
    
    localStorage.setItem('sidebarCollapsed', collapsed);
}

sidebarToggle.addEventListener('click', () => toggleSidebar());

// Init sidebar state
const isCollapsed = localStorage.getItem('sidebarCollapsed') === 'true';
if (isCollapsed) {
    // Apply immediately without transition for first load
    sidebar.style.transition = 'none';
    mainContent.style.transition = 'none';
    sidebarToggle.style.transition = 'none';
    toggleSidebar(true);
    // Restore transition
    requestAnimationFrame(() => {
        sidebar.style.transition = '';
        mainContent.style.transition = '';
        sidebarToggle.style.transition = '';
    });
}

// ── Mobile Sidebar ───────────────────────────────────────────────
const mobileMenuBtn = $('#mobileMenuBtn');
const sidebarBackdrop = $('#sidebarBackdrop');

function toggleMobileSidebar() {
    sidebar.classList.toggle('mobile-open');
    sidebarBackdrop.classList.toggle('active');
}

if (mobileMenuBtn && sidebarBackdrop) {
    mobileMenuBtn.addEventListener('click', toggleMobileSidebar);
    sidebarBackdrop.addEventListener('click', toggleMobileSidebar);
    
    // Close mobile sidebar when clicking a nav tab
    $$('.nav-tab').forEach(btn => {
        btn.addEventListener('click', () => {
            if (window.innerWidth <= 768 && sidebar.classList.contains('mobile-open')) {
                toggleMobileSidebar();
            }
        });
    });
}
