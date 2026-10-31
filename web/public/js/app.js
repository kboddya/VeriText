// VeriText Hybrid AI Detector Frontend Application Logic

let selectedUnifiedFile = null;
let lastUnifiedResult = null;
let currentParagraphFilter = 'all'; // 'all', 'ai', 'human'
let currentParagraphSort = 'original'; // 'original', 'desc', 'asc'
let currentSourceMode = 'text'; // 'text' or 'file'

let selectedBatchFile = null;
let lastBatchResult = null;

const activeRequests = {
    unified: { controller: null, id: null },
    batch: { controller: null, id: null },
};

function generateRequestId() {
    return 'req_' + Date.now() + '_' + Math.random().toString(36).substring(2, 9);
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    const bgMap = {
        info: 'bg-slate-900 border-blue-500/50 text-blue-300',
        warning: 'bg-slate-900 border-amber-500/50 text-amber-300',
        error: 'bg-slate-900 border-rose-500/50 text-rose-300',
        success: 'bg-slate-900 border-emerald-500/50 text-emerald-300',
    };
    const iconMap = {
        info: `<svg class="w-4 h-4 text-blue-400 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>`,
        warning: `<svg class="w-4 h-4 text-amber-400 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>`,
        error: `<svg class="w-4 h-4 text-rose-400 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>`,
        success: `<svg class="w-4 h-4 text-emerald-400 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>`,
    };

    const colorClass = bgMap[type] || bgMap.info;
    const iconSvg = iconMap[type] || iconMap.info;

    toast.className = `flex items-center gap-2 px-4 py-3 rounded-lg border ${colorClass} shadow-xl text-xs font-medium pointer-events-auto transition-all duration-300 opacity-0 translate-y-2`;
    toast.innerHTML = `${iconSvg}<span>${escapeHtml(message)}</span>`;

    container.appendChild(toast);

    requestAnimationFrame(() => {
        toast.classList.remove('opacity-0', 'translate-y-2');
        toast.classList.add('opacity-100', 'translate-y-0');
    });

    setTimeout(() => {
        toast.classList.add('opacity-0', 'translate-y-2');
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}

const SAMPLES = {
    ru_ai: "В современном мире технологии искусственного интеллекта играют ключевую роль во многих аспектах нашей повседневной жизни. Важно подчеркнуть, что интеграция нейросетевых моделей позволяет существенно оптимизировать рабочие процессы и повысить продуктивность. В заключение стоит отметить, что будущее цифровой трансформации неразрывно связано с развитием передовых алгоритмов.",
    ru_human: "Вчера наконец-то забрал заказ из пункта выдачи. Коробка была слегка помята с одного края, но внутри всё целое. Наушники классные, звук чистый, басы на месте. Правда провод зарядки коротковат, но в остальном покупкой доволен.",
    en_ai: "Artificial intelligence has become an integral part of modern society, driving innovation across various industries. It is worth noting that deep learning frameworks provide sophisticated tools for complex problem-solving. In conclusion, these technologies represent a testament to rapid technological advancement.",
    us_constitution: "We the People of the United States, in Order to form a more perfect Union, establish Justice, insure domestic Tranquility, provide for the common defence, promote the general Welfare, and secure the Blessings of Liberty to ourselves and our Posterity, do ordain and establish this Constitution for the United States of America. All legislative Powers herein granted shall be vested in a Congress of the United States, which shall consist of a Senate and House of Representatives. The House of Representatives shall be composed of Members chosen every second Year by the People of the several States, and the Electors in each State shall have the Qualifications requisite for Electors of the most numerous Branch of the State Legislature.",
};

// ----------------------------------------------------
// INITIALIZATION
// ----------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
    initHealthCheck();
    initDropzones();
    initTextareaCounter();
});

// Switch between Unified view and Batch CSV view
function switchView(view) {
    const viewUnified = document.getElementById('view-unified');
    const viewBatch = document.getElementById('view-batch');
    const btnUnified = document.getElementById('view-btn-unified');
    const btnBatch = document.getElementById('view-btn-batch');

    const activeClass = "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition bg-blue-600 text-white shadow-lg shadow-blue-600/20";
    const inactiveClass = "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition text-slate-400 hover:text-white hover:bg-slate-900/60";

    if (view === 'unified') {
        viewUnified.classList.remove('hidden');
        viewBatch.classList.add('hidden');
        btnUnified.className = activeClass;
        btnBatch.className = inactiveClass;
    } else {
        viewUnified.classList.add('hidden');
        viewBatch.classList.remove('hidden');
        btnUnified.className = inactiveClass;
        btnBatch.className = activeClass;
    }
}

// Health check and Hardware Acceleration indicator
async function initHealthCheck() {
    const statusDot = document.getElementById('status-dot');
    const statusPing = document.getElementById('status-ping');
    const deviceLabel = document.getElementById('device-label');

    async function check() {
        try {
            const res = await fetch('/api/health');
            if (!res.ok) throw new Error('Backend HTTP ' + res.status);
            const data = await res.json();
            
            statusDot.className = "relative inline-flex rounded-full h-2 w-2 bg-emerald-500";
            statusPing.className = "animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75";
            
            if (data.cuda_available) {
                deviceLabel.innerHTML = `<span class="text-emerald-400 font-semibold">NVIDIA GPU:</span> ${data.device_name}`;
            } else if (data.mps_available) {
                deviceLabel.innerHTML = `<span class="text-emerald-400 font-semibold">Apple GPU:</span> Metal (MPS)`;
            } else {
                deviceLabel.innerHTML = `<span class="text-blue-400 font-semibold">CPU:</span> ${data.device_name || 'Active'}`;
            }
        } catch (err) {
            statusDot.className = "relative inline-flex rounded-full h-2 w-2 bg-rose-500";
            statusPing.className = "hidden";
            deviceLabel.innerHTML = `<span class="text-rose-400 font-semibold">Offline</span> (waiting for backend)`;
        }
    }

    check();
    setInterval(check, 8000);
}

// Text counter
function initTextareaCounter() {
    const textarea = document.getElementById('text-input');
    const counter = document.getElementById('char-word-count');

    function update() {
        const val = textarea.value.trim();
        const chars = val.length;
        const words = val ? val.split(/\s+/).length : 0;
        counter.textContent = `Characters: ${chars} | Words: ${words}`;
    }

    textarea.addEventListener('input', update);
}

function loadSample(type) {
    setAnalysisSourceMode('text');
    clearSelectedFile();
    const textarea = document.getElementById('text-input');
    textarea.value = SAMPLES[type] || '';
    textarea.dispatchEvent(new Event('input'));
    showToast(`Loaded sample: ${type.toUpperCase()}`, 'info');
}

function setAnalysisSourceMode(mode) {
    currentSourceMode = mode;
    const btnText = document.getElementById('mode-btn-text');
    const btnFile = document.getElementById('mode-btn-file');
    const panelText = document.getElementById('input-panel-text');
    const panelFile = document.getElementById('input-panel-file');
    const samplesContainer = document.getElementById('quick-samples-container');
    const btnRunLabel = document.getElementById('btn-run-label');

    const activeClass = "flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-600 text-white font-semibold transition shadow-sm";
    const inactiveClass = "flex items-center gap-2 px-4 py-2 rounded-lg text-slate-400 hover:text-white transition";

    if (mode === 'text') {
        if (btnText) btnText.className = activeClass;
        if (btnFile) btnFile.className = inactiveClass;
        if (panelText) panelText.classList.remove('hidden');
        if (panelFile) panelFile.classList.add('hidden');
        if (samplesContainer) samplesContainer.classList.remove('hidden');
        if (btnRunLabel) btnRunLabel.textContent = 'Run Hybrid Analysis (Text)';
    } else {
        if (btnText) btnText.className = inactiveClass;
        if (btnFile) btnFile.className = activeClass;
        if (panelText) panelText.classList.add('hidden');
        if (panelFile) panelFile.classList.remove('hidden');
        if (samplesContainer) samplesContainer.classList.add('hidden');
        if (btnRunLabel) {
            btnRunLabel.textContent = selectedUnifiedFile 
                ? `Run Hybrid Analysis (${selectedUnifiedFile.name})`
                : 'Select a Document to Analyze';
        }
    }
}

function clearUnifiedInput() {
    clearSelectedFile();
    const textarea = document.getElementById('text-input');
    textarea.value = '';
    textarea.dispatchEvent(new Event('input'));
}

// ----------------------------------------------------
// DROPZONE LOGIC
// ----------------------------------------------------
function initDropzones() {
    setupDropzone('unified-dropzone', handleSelectedUnifiedFile);
    setupDropzone('file-dropzone', handleSelectedBatchFile);
}

function setupDropzone(dropzoneId, onFileDrop) {
    const dropzone = document.getElementById(dropzoneId);
    if (!dropzone) return;

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dropzone-active');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dropzone-active');
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            onFileDrop(files[0]);
        }
    });
}

function handleUnifiedFileSelected(event) {
    const file = event.target.files[0];
    if (file) {
        handleSelectedUnifiedFile(file);
    }
}

function handleSelectedUnifiedFile(file) {
    selectedUnifiedFile = file;
    const badge = document.getElementById('selected-file-badge');
    const fileNameEl = document.getElementById('selected-file-name');
    const fileSizeEl = document.getElementById('selected-file-size');
    const fileExtEl = document.getElementById('selected-file-ext');
    const dropzoneLabel = document.getElementById('dropzone-label');
    const btnRunLabel = document.getElementById('btn-run-label');

    const sizeKb = (file.size / 1024).toFixed(1);
    const ext = file.name.split('.').pop().toUpperCase();

    if (fileExtEl) fileExtEl.textContent = ext;
    if (fileNameEl) fileNameEl.textContent = file.name;
    if (fileSizeEl) fileSizeEl.textContent = `${sizeKb} KB`;
    if (badge) badge.classList.remove('hidden');

    if (dropzoneLabel) dropzoneLabel.textContent = `Attached: ${file.name} (${sizeKb} KB)`;
    if (btnRunLabel) btnRunLabel.textContent = `Run Hybrid Analysis (${file.name})`;
    showToast(`Attached document: ${file.name}`, 'info');
}

function clearSelectedFile() {
    selectedUnifiedFile = null;
    const badge = document.getElementById('selected-file-badge');
    const fileInput = document.getElementById('unified-file-input');
    const dropzoneLabel = document.getElementById('dropzone-label');
    const btnRunLabel = document.getElementById('btn-run-label');

    if (badge) badge.classList.add('hidden');
    if (fileInput) fileInput.value = '';
    if (dropzoneLabel) dropzoneLabel.textContent = 'Click to select or drag and drop a document file here';
    if (btnRunLabel) {
        btnRunLabel.textContent = (currentSourceMode === 'file') ? 'Select a Document to Analyze' : 'Run Hybrid Analysis (Text)';
    }
}

// ----------------------------------------------------
// HYBRID ANALYSIS EXECUTION
// ----------------------------------------------------
async function submitHybridAnalysis() {
    if (currentSourceMode === 'file') {
        if (!selectedUnifiedFile) {
            alert('Please select or drag & drop a document file (.pdf, .docx, .md, .txt) first.');
            return;
        }
    } else {
        const textVal = document.getElementById('text-input').value.trim();
        if (!textVal) {
            alert('Please enter or paste text to analyze.');
            return;
        }
    }

    const mode = document.getElementById('model-select').value;
    const btn = document.getElementById('btn-run-analysis');
    const btnLabel = document.getElementById('btn-run-label');
    const btnAbort = document.getElementById('btn-abort-analysis');
    const emptyState = document.getElementById('unified-empty-state');
    const resultsCard = document.getElementById('unified-results-card');

    const controller = new AbortController();
    const reqId = generateRequestId();
    activeRequests.unified = { controller, id: reqId };

    btn.disabled = true;
    btnLabel.textContent = 'Running Hybrid Inference...';
    btnAbort.classList.remove('hidden');

    const progressCard = document.getElementById('live-progress-card');
    const stepBadge = document.getElementById('live-step-badge');
    const progressMsg = document.getElementById('live-progress-message');
    const progressPct = document.getElementById('live-progress-pct');
    const progressFill = document.getElementById('live-progress-fill');

    if (progressCard) {
        progressCard.classList.remove('hidden');
        stepBadge.textContent = 'Step 1/4';
        progressMsg.textContent = (currentSourceMode === 'file') ? `Extracting ${selectedUnifiedFile.name}...` : 'Preparing text input...';
        progressPct.textContent = '10%';
        progressFill.style.width = '10%';
    }

    const progressTimer = setInterval(async () => {
        try {
            const pollRes = await fetch(`/api/status/${reqId}`);
            if (pollRes.ok) {
                const pollData = await pollRes.json();
                if (pollData && pollData.status === 'running') {
                    if (stepBadge) stepBadge.textContent = `Step ${pollData.step || 1}/${pollData.total_steps || 4}`;
                    if (progressMsg) progressMsg.textContent = pollData.message || 'Processing...';
                    const pct = pollData.percent || 15;
                    if (progressPct) progressPct.textContent = `${pct}%`;
                    if (progressFill) progressFill.style.width = `${pct}%`;
                }
            }
        } catch (_) {}
    }, 250);

    try {
        let res;
        if (currentSourceMode === 'file') {
            // Upload file to document analysis endpoint
            const formData = new FormData();
            formData.append('file', selectedUnifiedFile);
            formData.append('mode', mode);
            formData.append('request_id', reqId);

            res = await fetch('/api/score-document', {
                method: 'POST',
                headers: { 'X-Request-ID': reqId },
                body: formData,
                signal: controller.signal,
            });
        } else {
            // Send direct text
            const textVal = document.getElementById('text-input').value.trim();
            res = await fetch('/api/score', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Request-ID': reqId,
                },
                body: JSON.stringify({
                    text: textVal,
                    mode: mode,
                    request_id: reqId,
                }),
                signal: controller.signal,
            });
        }

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Inference failed');
        }

        const data = await res.json();
        lastUnifiedResult = data;
        renderUnifiedResults(data);

        emptyState.classList.add('hidden');
        resultsCard.classList.remove('hidden');
        showToast('Hybrid analysis completed successfully', 'success');

        // Smooth scroll to results
        resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });

    } catch (e) {
        if (e.name === 'AbortError') {
            showToast('Analysis cancelled by user', 'warning');
        } else {
            alert('Analysis error: ' + e.message);
            showToast('Error: ' + e.message, 'error');
        }
    } finally {
        clearInterval(progressTimer);
        if (progressFill) progressFill.style.width = '100%';
        if (progressPct) progressPct.textContent = '100%';
        if (progressMsg) progressMsg.textContent = 'Inference finished';
        setTimeout(() => {
            if (progressCard) progressCard.classList.add('hidden');
        }, 500);

        btn.disabled = false;
        btnLabel.textContent = 'Run Hybrid Analysis';
        btnAbort.classList.add('hidden');
        activeRequests.unified = { controller: null, id: null };
    }
}

function abortUnifiedAnalysis() {
    const item = activeRequests.unified;
    if (!item) return;

    if (item.id) {
        fetch(`/api/abort/${item.id}`, { method: 'POST' }).catch(() => {});
    }
    if (item.controller) {
        item.controller.abort();
    }
    activeRequests.unified = { controller: null, id: null };
}

// ----------------------------------------------------
// RENDERING RESULTS DASHBOARD
// ----------------------------------------------------
function renderUnifiedResults(data) {
    const summary = data.summary || {};
    const isAI = (data.label === 'ai') || (summary.overall_label === 'ai');

    // 1. Top Verdict Banner
    const verdictContainer = document.getElementById('main-verdict-container');
    const verdictIcon = document.getElementById('main-verdict-icon');
    const verdictText = document.getElementById('main-verdict-text');
    const modelBadge = document.getElementById('active-model-badge');
    const deviceBadge = document.getElementById('active-device-badge');

    if (isAI) {
        verdictContainer.className = "mt-2 inline-flex items-center gap-2.5 px-6 py-2 rounded-full border border-rose-500/40 bg-rose-500/10 text-rose-400 text-base font-bold shadow-lg shadow-rose-500/10";
        verdictIcon.innerHTML = `<svg class="w-5 h-5 text-rose-400" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/></svg>`;
        verdictText.textContent = "AI-Generated Content (AI)";
    } else {
        verdictContainer.className = "mt-2 inline-flex items-center gap-2.5 px-6 py-2 rounded-full border border-emerald-500/40 bg-emerald-500/10 text-emerald-400 text-base font-bold shadow-lg shadow-emerald-500/10";
        verdictIcon.innerHTML = `<svg class="w-5 h-5 text-emerald-400" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/></svg>`;
        verdictText.textContent = "Human-Written Content (Human)";
    }

    modelBadge.textContent = `Model: ${data.model_used || 'RuBERT-AI'}`;
    deviceBadge.textContent = `Device: ${(data.device || 'Auto').toUpperCase()}`;

    const sourceBadge = document.getElementById('active-source-badge');
    if (sourceBadge) {
        if (data.filename || currentSourceMode === 'file') {
            sourceBadge.textContent = `Source: File (${data.filename || (selectedUnifiedFile ? selectedUnifiedFile.name : 'Document')})`;
            sourceBadge.className = "px-3 py-1.5 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 font-mono";
        } else {
            sourceBadge.textContent = `Source: Direct Text`;
            sourceBadge.className = "px-3 py-1.5 rounded-lg bg-blue-500/10 border border-blue-500/20 text-blue-300 font-mono";
        }
    }

    // Confidence & Risk Level Status
    const confBadge = document.getElementById('confidence-status-badge');
    const confLabel = document.getElementById('confidence-status-label');
    const confDesc = document.getElementById('confidence-status-desc');

    const conf = data.confidence_status || (summary && summary.confidence_status) || {
        grade: isAI ? 'Grade F (Confirmed AI)' : 'Grade A (Verified Human)',
        badge_class: isAI ? 'bg-rose-500/10 text-rose-400 border-rose-500/30' : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
        description: isAI ? 'High probability synthetic phrasing detected.' : 'High confidence human authorship.'
    };

    if (confBadge && confLabel && confDesc) {
        confBadge.className = `inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full border text-xs font-bold shadow-sm ${conf.badge_class || ''}`;
        confLabel.textContent = conf.grade || conf.label;
        confDesc.textContent = conf.description || '';
    }

    // 2. Dual Meters: Monolithic vs Paragraph-Weighted
    const monolithicVal = (data.monolithic_scale !== undefined) ? data.monolithic_scale : (data.human_ai_scale || 50.0);
    const paragraphVal = (data.paragraph_scale !== undefined) ? data.paragraph_scale : (summary.overall_scale || monolithicVal);

    const monoClamped = Math.max(0, Math.min(100, monolithicVal));
    const paraClamped = Math.max(0, Math.min(100, paragraphVal));

    document.getElementById('meter-monolithic-val').textContent = `${monoClamped.toFixed(2)}%`;
    document.getElementById('meter-monolithic-indicator').style.left = `${monoClamped}%`;

    document.getElementById('meter-paragraph-val').textContent = `${paraClamped.toFixed(2)}%`;
    document.getElementById('meter-paragraph-indicator').style.left = `${paraClamped}%`;

    // 3. Stats Row
    const words = summary.word_count || (data.full_text ? data.full_text.split(/\s+/).length : 0);
    const chars = summary.char_count || (data.full_text ? data.full_text.length : 0);
    const totalParas = summary.total_paragraphs || (data.paragraphs ? data.paragraphs.length : 1);
    const aiParas = summary.ai_paragraphs || (isAI ? 1 : 0);
    const humanParas = summary.human_paragraphs || (isAI ? 0 : 1);
    const aiPct = summary.ai_paragraph_pct !== undefined ? summary.ai_paragraph_pct : (isAI ? 100 : 0);

    document.getElementById('stat-words-chars').textContent = `${words.toLocaleString()} / ${chars.toLocaleString()}`;
    document.getElementById('stat-total-paras').textContent = totalParas;
    document.getElementById('stat-ai-paras').textContent = `${aiParas} (${aiPct}%)`;
    document.getElementById('stat-human-paras').textContent = `${humanParas} (${(100 - aiPct).toFixed(1)}%)`;

    // 4. Stylistic Markers
    const markersBox = document.getElementById('markers-box');
    const markersList = document.getElementById('markers-list');
    if (data.markers && data.markers.length > 0) {
        markersBox.classList.remove('hidden');
        markersList.innerHTML = '';
        data.markers.forEach(m => {
            const item = document.createElement('div');
            item.className = "p-2 rounded bg-slate-900 border border-slate-800 text-xs flex justify-between items-center";
            item.innerHTML = `
                <div>
                    <span class="text-rose-400 font-mono font-semibold">«${escapeHtml(m.phrase)}»</span>
                    <span class="text-slate-400 block text-[11px]">${escapeHtml(m.description)}</span>
                </div>
                <span class="text-slate-500 text-[11px] font-mono font-bold ml-2">x${m.count}</span>
            `;
            markersList.appendChild(item);
        });
    } else {
        markersBox.classList.add('hidden');
    }

    // 5. Update Filters Count
    const paragraphs = data.paragraphs || [];
    document.getElementById('paras-count-badge').textContent = `${paragraphs.length} sections`;
    document.getElementById('filter-count-all').textContent = paragraphs.length;
    document.getElementById('filter-count-ai').textContent = aiParas;
    document.getElementById('filter-count-human').textContent = humanParas;

    // 6. Render Paragraphs List with Current Filter and Sorting
    renderParagraphsList();
}

// ----------------------------------------------------
// PARAGRAPH FILTERING & SORTING LOGIC
// ----------------------------------------------------
function setParagraphFilter(filter) {
    currentParagraphFilter = filter;
    
    const btnAll = document.getElementById('filter-btn-all');
    const btnAi = document.getElementById('filter-btn-ai');
    const btnHuman = document.getElementById('filter-btn-human');

    const activeClass = "px-3 py-1 rounded bg-blue-600 text-white font-medium transition";
    const inactiveClass = "px-3 py-1 rounded text-slate-400 hover:text-white transition";

    btnAll.className = (filter === 'all') ? activeClass : inactiveClass;
    btnAi.className = (filter === 'ai') ? activeClass : inactiveClass;
    btnHuman.className = (filter === 'human') ? activeClass : inactiveClass;

    renderParagraphsList();
}

function changeParagraphSort(sortType) {
    currentParagraphSort = sortType;
    renderParagraphsList();
}

function renderParagraphsList() {
    if (!lastUnifiedResult || !lastUnifiedResult.paragraphs) return;

    const container = document.getElementById('paragraphs-container');
    container.innerHTML = '';

    let list = [...lastUnifiedResult.paragraphs];

    // 1. Filter
    if (currentParagraphFilter === 'ai') {
        list = list.filter(p => p.label === 'ai');
    } else if (currentParagraphFilter === 'human') {
        list = list.filter(p => p.label !== 'ai');
    }

    // 2. Sort
    if (currentParagraphSort === 'desc') {
        // Highest AI % First (спадание)
        list.sort((a, b) => (b.human_ai_scale || 0) - (a.human_ai_scale || 0));
    } else if (currentParagraphSort === 'asc') {
        // Lowest AI % First (возрастание)
        list.sort((a, b) => (a.human_ai_scale || 0) - (b.human_ai_scale || 0));
    } else {
        // Original text order (#1, #2, #3...)
        list.sort((a, b) => (a.index || 0) - (b.index || 0));
    }

    if (list.length === 0) {
        container.innerHTML = `<div class="text-center py-8 text-xs text-slate-500">No paragraphs match the selected filter.</div>`;
        return;
    }

    list.forEach(p => {
        const isAI = p.label === 'ai';
        const card = document.createElement('div');
        const borderClass = isAI ? 'border-rose-500/30 bg-rose-950/10' : 'border-emerald-500/30 bg-emerald-950/10';
        const badgeClass = isAI ? 'bg-rose-500/20 text-rose-400 border-rose-500/30' : 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
        const scaleVal = (p.human_ai_scale !== undefined) ? p.human_ai_scale.toFixed(1) : '50.0';
        const wordCount = p.word_count || (p.text ? p.text.split(/\s+/).length : 0);

        card.className = `p-4 rounded-xl border ${borderClass} space-y-2.5 transition`;
        card.innerHTML = `
            <div class="flex items-center justify-between text-xs">
                <div class="flex items-center gap-2">
                    <span class="font-mono text-slate-400 font-semibold">#${p.index}</span>
                    <span class="px-2 py-0.5 rounded text-[11px] font-bold border ${badgeClass}">
                        ${isAI ? 'AI-Generated' : 'Human-Written'}
                    </span>
                    <span class="text-[11px] text-slate-500 font-mono">${wordCount} words</span>
                </div>
                <div class="flex items-center gap-2">
                    <span class="text-slate-400 font-mono text-[11px]">AI Probability:</span>
                    <span class="font-mono font-bold ${isAI ? 'text-rose-400' : 'text-emerald-400'} text-xs">${scaleVal}%</span>
                </div>
            </div>

            <!-- Mini Progress bar for this paragraph -->
            <div class="w-full bg-slate-900 rounded-full h-1.5 overflow-hidden">
                <div class="${isAI ? 'bg-rose-500' : 'bg-emerald-500'} h-1.5 rounded-full" style="width: ${scaleVal}%"></div>
            </div>

            <p class="text-xs text-slate-200 leading-relaxed font-normal">${escapeHtml(p.text)}</p>
        `;
        container.appendChild(card);
    });
}

// ----------------------------------------------------
// BATCH CSV / TSV ANALYSIS
// ----------------------------------------------------
function handleFileSelected(event) {
    const file = event.target.files[0];
    if (file) {
        handleSelectedBatchFile(file);
    }
}

function handleSelectedBatchFile(file) {
    selectedBatchFile = file;
    const dropzoneText = document.getElementById('batch-dropzone-text');
    const btn = document.getElementById('btn-run-batch');
    const btnLabel = document.getElementById('btn-run-batch-label');

    const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
    dropzoneText.textContent = `Selected: ${file.name} (${sizeMb} MB)`;

    btn.disabled = false;
    btnLabel.textContent = `Process Dataset (${file.name})`;
}

async function submitBatchAnalysis() {
    if (!selectedBatchFile) {
        alert('Please select a .csv or .tsv file.');
        return;
    }

    const textCol = document.getElementById('batch-text-column').value.trim();
    const mode = document.getElementById('batch-mode-select').value;
    const batchSize = parseInt(document.getElementById('batch-size-input').value) || 32;

    const btn = document.getElementById('btn-run-batch');
    const btnLabel = document.getElementById('btn-run-batch-label');
    const btnAbort = document.getElementById('btn-abort-batch');
    const resultsContainer = document.getElementById('batch-results-container');

    const controller = new AbortController();
    const reqId = generateRequestId();
    activeRequests.batch = { controller, id: reqId };

    btn.disabled = true;
    btnLabel.textContent = 'Processing Batch Dataset...';
    btnAbort.classList.remove('hidden');

    const formData = new FormData();
    formData.append('file', selectedBatchFile);
    formData.append('mode', mode);
    formData.append('batch_size', batchSize);
    formData.append('request_id', reqId);
    if (textCol) {
        formData.append('text_column', textCol);
    }

    try {
        const res = await fetch('/api/score-file', {
            method: 'POST',
            headers: { 'X-Request-ID': reqId },
            body: formData,
            signal: controller.signal,
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Batch processing failed');
        }

        const data = await res.json();
        lastBatchResult = data;
        renderBatchResult(data);
        resultsContainer.classList.remove('hidden');
        showToast('Batch processing finished', 'success');

    } catch (e) {
        if (e.name === 'AbortError') {
            showToast('Batch analysis aborted', 'warning');
        } else {
            alert('Error: ' + e.message);
            showToast('Error: ' + e.message, 'error');
        }
    } finally {
        btn.disabled = false;
        btnLabel.textContent = `Process Dataset (${selectedBatchFile.name})`;
        btnAbort.classList.add('hidden');
        activeRequests.batch = { controller: null, id: null };
    }
}

function abortAnalysis(type) {
    const item = activeRequests[type];
    if (!item) return;

    if (item.id) {
        fetch(`/api/abort/${item.id}`, { method: 'POST' }).catch(() => {});
    }
    if (item.controller) {
        item.controller.abort();
    }
    activeRequests[type] = { controller: null, id: null };
}

function renderBatchResult(data) {
    document.getElementById('batch-total-rows').textContent = data.total_rows.toLocaleString();
    document.getElementById('batch-ai-count').textContent = data.ai_rows.toLocaleString();
    document.getElementById('batch-ai-pct').textContent = `(${data.ai_percentage}%)`;
    document.getElementById('batch-human-count').textContent = data.human_rows.toLocaleString();
    document.getElementById('batch-human-pct').textContent = `(${(100 - data.ai_percentage).toFixed(1)}%)`;

    const thead = document.getElementById('batch-table-head');
    const tbody = document.getElementById('batch-table-body');
    thead.innerHTML = '';
    tbody.innerHTML = '';

    if (!data.preview || data.preview.length === 0) return;

    const cols = Object.keys(data.preview[0]);
    const headerRow = document.createElement('tr');
    cols.forEach(col => {
        const th = document.createElement('th');
        th.className = "px-4 py-2 font-semibold";
        th.textContent = col;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);

    data.preview.forEach(row => {
        const tr = document.createElement('tr');
        tr.className = "hover:bg-slate-800/40";
        cols.forEach(col => {
            const td = document.createElement('td');
            td.className = "px-4 py-2 truncate max-w-xs";
            const val = row[col];
            if (col === 'ai_prediction') {
                const isAI = String(val).toLowerCase() === 'ai';
                td.innerHTML = `<span class="px-2 py-0.5 rounded text-[10px] font-bold ${isAI ? 'bg-rose-500/20 text-rose-400' : 'bg-emerald-500/20 text-emerald-400'}">${val}</span>`;
            } else if (col === 'human_ai_scale') {
                td.textContent = (typeof val === 'number') ? `${val.toFixed(1)}%` : val;
            } else {
                td.textContent = val !== null ? String(val) : '';
            }
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
}

function downloadScoredCSV() {
    if (!lastBatchResult || !lastBatchResult.download_token) {
        alert('No scored data available for download.');
        return;
    }
    window.location.href = `/api/download/${lastBatchResult.download_token}`;
}

// ----------------------------------------------------
// UTILITIES
// ----------------------------------------------------
function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}
