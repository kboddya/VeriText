// VeriText Hybrid AI Detector & Text Humanizer Frontend Application Logic

let selectedUnifiedFile = null;
let lastUnifiedResult = null;
let currentParagraphFilter = 'all'; // 'all', 'ai', 'human'
let currentParagraphSort = 'original'; // 'original', 'desc', 'asc'
let currentSourceMode = 'text'; // 'text' or 'file'

let lastHumanizeResult = null;
let selectedHumanizeFile = null;
let currentHumanizeInputMode = 'text'; // 'text' or 'file'

const activeRequests = {
    unified: { controller: null, id: null },
    humanize: { controller: null, id: null },
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

const HUMANIZE_SAMPLES = {
    ru_ai: "Важно отметить, что в современном мире искусственный интеллект играет ключевую роль в обработке данных. Таким образом, можно сделать вывод, что интеграция алгоритмов является неотъемлемой частью технологического прогресса. Подводя итоги, следует подчеркнуть необходимость дальнейших исследований в этой динамично развивающейся области.",
    en_ai: "In today's fast-paced digital world, it is important to remember that artificial intelligence has become a cornerstone of technology. Furthermore, it is worth noting that machine learning models provide sophisticated automation tools. In conclusion, navigating this complex landscape requires continuous engineering vigilance.",
    cliche: "Важно отметить, что в конечном итоге необходимо подчеркнуть значимость данного подхода. Таким образом, можно сделать вывод, что это играет ключевую роль. Подводя итоги, следует отметить дальнейшие перспективы и комплексное развитие системы.",
};

// ----------------------------------------------------
// INITIALIZATION
// ----------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
    initHealthCheck();
    initDropzones();
    initTextareaCounter();
});

// Switch view logic
function switchView(view) {
    const viewUnified = document.getElementById('view-unified');
    const viewHumanize = document.getElementById('view-humanize');
    const btnUnified = document.getElementById('view-btn-unified');
    const btnHumanize = document.getElementById('view-btn-humanize');

    const activeClass = "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition bg-blue-600 text-white shadow-lg shadow-blue-600/20";
    const activeHumanizeClass = "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition bg-indigo-600 text-white shadow-lg shadow-indigo-600/20";
    const inactiveClass = "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition text-slate-400 hover:text-white hover:bg-slate-900/60";

    if (view === 'humanize') {
        if (viewUnified) viewUnified.classList.add('hidden');
        if (viewHumanize) viewHumanize.classList.remove('hidden');
        if (btnUnified) btnUnified.className = inactiveClass;
        if (btnHumanize) btnHumanize.className = activeHumanizeClass;
    } else {
        if (viewUnified) viewUnified.classList.remove('hidden');
        if (viewHumanize) viewHumanize.classList.add('hidden');
        if (btnUnified) btnUnified.className = activeClass;
        if (btnHumanize) btnHumanize.className = inactiveClass;
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

            // Update Humanizer engine badge if present
            const hEngineLabel = document.getElementById('humanize-engine-label');
            const hEngineDot = document.getElementById('humanize-engine-dot');
            if (hEngineLabel && data.humanizer_info) {
                const info = data.humanizer_info;
                if (info.active) {
                    hEngineLabel.innerHTML = `Generator: <span class="text-emerald-400 font-semibold">${escapeHtml(info.provider)}</span> (Local T5 bypassed)`;
                    if (hEngineDot) hEngineDot.className = "w-2 h-2 rounded-full bg-emerald-400";
                } else {
                    hEngineLabel.innerHTML = `Generator: <span class="text-indigo-400 font-semibold">${escapeHtml(info.provider || 'Local T5')}</span>`;
                    if (hEngineDot) hEngineDot.className = "w-2 h-2 rounded-full bg-indigo-400";
                }
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

    if (textarea && counter) {
        function update() {
            const val = textarea.value.trim();
            const chars = val.length;
            const words = val ? val.split(/\s+/).length : 0;
            counter.textContent = `Characters: ${chars} | Words: ${words}`;
        }
        textarea.addEventListener('input', update);
    }

    const humanizeTextarea = document.getElementById('humanize-text-input');
    const humanizeCounter = document.getElementById('humanize-char-word-count');

    if (humanizeTextarea && humanizeCounter) {
        function updateHumanize() {
            const val = humanizeTextarea.value.trim();
            const chars = val.length;
            const words = val ? val.split(/\s+/).length : 0;
            humanizeCounter.textContent = `Characters: ${chars} | Words: ${words}`;
        }
        humanizeTextarea.addEventListener('input', updateHumanize);
    }
}

function loadSample(type) {
    setAnalysisSourceMode('text');
    clearSelectedFile();
    const textarea = document.getElementById('text-input');
    textarea.value = SAMPLES[type] || '';
    textarea.dispatchEvent(new Event('input'));
    showToast(`Loaded sample: ${type.toUpperCase()}`, 'info');
}

function loadHumanizeSample(type) {
    setHumanizeInputMode('text');
    clearSelectedHumanizeFile();
    const textarea = document.getElementById('humanize-text-input');
    if (textarea) {
        textarea.value = HUMANIZE_SAMPLES[type] || '';
        textarea.dispatchEvent(new Event('input'));
        showToast(`Loaded sample: ${type.toUpperCase()}`, 'info');
    }
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

function setHumanizeInputMode(mode) {
    currentHumanizeInputMode = mode;
    const btnText = document.getElementById('humanize-mode-btn-text');
    const btnFile = document.getElementById('humanize-mode-btn-file');
    const panelText = document.getElementById('humanize-input-panel-text');
    const panelFile = document.getElementById('humanize-input-panel-file');
    const samplesContainer = document.getElementById('humanize-quick-samples-container');
    const btnRunLabel = document.getElementById('btn-run-humanize-label');

    const activeClass = "flex items-center gap-2 px-4 py-2 rounded-lg bg-indigo-600 text-white font-semibold transition shadow-sm";
    const inactiveClass = "flex items-center gap-2 px-4 py-2 rounded-lg text-slate-400 hover:text-white transition";

    if (mode === 'text') {
        if (btnText) btnText.className = activeClass;
        if (btnFile) btnFile.className = inactiveClass;
        if (panelText) panelText.classList.remove('hidden');
        if (panelFile) panelFile.classList.add('hidden');
        if (samplesContainer) samplesContainer.classList.remove('hidden');
        if (btnRunLabel) btnRunLabel.textContent = '✨ Run AI Humanization (Text)';
    } else {
        if (btnText) btnText.className = inactiveClass;
        if (btnFile) btnFile.className = activeClass;
        if (panelText) panelText.classList.add('hidden');
        if (panelFile) panelFile.classList.remove('hidden');
        if (samplesContainer) samplesContainer.classList.add('hidden');
        if (btnRunLabel) {
            btnRunLabel.textContent = selectedHumanizeFile 
                ? `✨ Humanize Document (${selectedHumanizeFile.name})`
                : 'Select a Document to Humanize';
        }
    }
}

function clearUnifiedInput() {
    clearSelectedFile();
    const textarea = document.getElementById('text-input');
    textarea.value = '';
    textarea.dispatchEvent(new Event('input'));
}

function clearHumanizeInput() {
    clearSelectedHumanizeFile();
    const textarea = document.getElementById('humanize-text-input');
    if (textarea) {
        textarea.value = '';
        textarea.dispatchEvent(new Event('input'));
    }
}

// ----------------------------------------------------
// DROPZONE LOGIC
// ----------------------------------------------------
function initDropzones() {
    setupDropzone('unified-dropzone', handleSelectedUnifiedFile);
    setupDropzone('humanize-dropzone', handleSelectedHumanizeFile);
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
    const file = event.target.files && event.target.files[0];
    if (file) {
        handleSelectedUnifiedFile(file);
    }
    event.target.value = '';
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

function handleHumanizeFileSelected(event) {
    const file = event.target.files && event.target.files[0];
    if (file) {
        handleSelectedHumanizeFile(file);
    }
    event.target.value = '';
}

function handleSelectedHumanizeFile(file) {
    selectedHumanizeFile = file;
    const badge = document.getElementById('humanize-selected-file-badge');
    const fileNameEl = document.getElementById('humanize-selected-file-name');
    const fileSizeEl = document.getElementById('humanize-selected-file-size');
    const fileExtEl = document.getElementById('humanize-selected-file-ext');
    const dropzoneLabel = document.getElementById('humanize-dropzone-label');
    const btnRunLabel = document.getElementById('btn-run-humanize-label');

    const sizeKb = (file.size / 1024).toFixed(1);
    const ext = file.name.split('.').pop().toUpperCase();

    if (fileExtEl) fileExtEl.textContent = ext;
    if (fileNameEl) fileNameEl.textContent = file.name;
    if (fileSizeEl) fileSizeEl.textContent = `${sizeKb} KB`;
    if (badge) badge.classList.remove('hidden');

    if (dropzoneLabel) dropzoneLabel.textContent = `Attached: ${file.name} (${sizeKb} KB)`;
    if (btnRunLabel) btnRunLabel.textContent = `✨ Humanize Document (${file.name})`;
    showToast(`Attached document: ${file.name}`, 'info');
}

function clearSelectedHumanizeFile() {
    selectedHumanizeFile = null;
    const badge = document.getElementById('humanize-selected-file-badge');
    const fileInput = document.getElementById('humanize-file-input');
    const dropzoneLabel = document.getElementById('humanize-dropzone-label');
    const btnRunLabel = document.getElementById('btn-run-humanize-label');

    if (badge) badge.classList.add('hidden');
    if (fileInput) fileInput.value = '';
    if (dropzoneLabel) dropzoneLabel.textContent = 'Click to select or drag and drop a document file here';
    if (btnRunLabel) {
        btnRunLabel.textContent = (currentHumanizeInputMode === 'file') ? 'Select a Document to Humanize' : '✨ Run AI Humanization';
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
// AI TEXT HUMANIZER EXECUTION
// ----------------------------------------------------
async function submitHumanizeAnalysis() {
    if (currentHumanizeInputMode === 'file') {
        if (!selectedHumanizeFile) {
            alert('Please select or drag & drop a document file (.pdf, .docx, .md, .txt) first.');
            return;
        }
    } else {
        const textVal = (document.getElementById('humanize-text-input')?.value || '').trim();
        if (!textVal) {
            alert('Please enter or paste text to humanize.');
            return;
        }
    }

    const mode = document.getElementById('humanize-mode-select')?.value || 'auto';
    const strength = document.getElementById('humanize-strength-select')?.value || 'balanced';
    const btn = document.getElementById('btn-run-humanize');
    const btnLabel = document.getElementById('btn-run-humanize-label');
    const btnAbort = document.getElementById('btn-abort-humanize');
    const emptyState = document.getElementById('humanize-empty-state');
    const resultsCard = document.getElementById('humanize-results-card');

    const controller = new AbortController();
    const reqId = generateRequestId();
    activeRequests.humanize = { controller, id: reqId };

    if (btn) btn.disabled = true;
    if (btnLabel) btnLabel.textContent = 'Transforming & Paraphrasing...';
    if (btnAbort) btnAbort.classList.remove('hidden');

    const progressCard = document.getElementById('humanize-live-progress-card');
    const stepBadge = document.getElementById('humanize-step-badge');
    const progressMsg = document.getElementById('humanize-progress-message');
    const progressPct = document.getElementById('humanize-progress-pct');
    const progressFill = document.getElementById('humanize-progress-fill');

    const paraBadge = document.getElementById('humanize-para-badge');
    if (progressCard) {
        progressCard.classList.remove('hidden');
        if (stepBadge) stepBadge.textContent = 'Step 1/4';
        if (paraBadge) paraBadge.classList.add('hidden');
        if (progressMsg) progressMsg.textContent = (currentHumanizeInputMode === 'file') ? `Extracting ${selectedHumanizeFile.name}...` : 'Analyzing text & language...';
        if (progressPct) progressPct.textContent = '10%';
        if (progressFill) progressFill.style.width = '10%';
    }

    const progressTimer = setInterval(async () => {
        try {
            const pollRes = await fetch(`/api/status/${reqId}`);
            if (pollRes.ok) {
                const pollData = await pollRes.json();
                if (pollData && pollData.status === 'running') {
                    if (stepBadge) stepBadge.textContent = `Step ${pollData.step || 1}/${pollData.total_steps || 4}`;
                    if (paraBadge) {
                        if (pollData.current_paragraph && pollData.total_paragraphs) {
                            paraBadge.textContent = `Para ${pollData.current_paragraph}/${pollData.total_paragraphs}`;
                            paraBadge.classList.remove('hidden');
                        } else {
                            paraBadge.classList.add('hidden');
                        }
                    }
                    if (progressMsg) progressMsg.textContent = pollData.message || 'Transforming...';
                    const pct = pollData.percent || 15;
                    if (progressPct) progressPct.textContent = `${pct}%`;
                    if (progressFill) progressFill.style.width = `${pct}%`;
                }
            }
        } catch (_) {}
    }, 250);

    try {
        let res;
        if (currentHumanizeInputMode === 'file') {
            const formData = new FormData();
            formData.append('file', selectedHumanizeFile);
            formData.append('mode', mode);
            formData.append('strength', strength);
            formData.append('request_id', reqId);

            res = await fetch('/api/humanize-document', {
                method: 'POST',
                headers: { 'X-Request-ID': reqId },
                body: formData,
                signal: controller.signal,
            });
        } else {
            const textVal = document.getElementById('humanize-text-input').value.trim();
            res = await fetch('/api/humanize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Request-ID': reqId,
                },
                body: JSON.stringify({
                    text: textVal,
                    mode: mode,
                    strength: strength,
                    request_id: reqId,
                }),
                signal: controller.signal,
            });
        }

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `Server returned HTTP ${res.status}`);
        }

        const data = await res.json();
        lastHumanizeResult = data;
        renderHumanizeResult(data);

        if (emptyState) emptyState.classList.add('hidden');
        if (resultsCard) resultsCard.classList.remove('hidden');
        showToast('Text humanization completed successfully!', 'success');

        resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });

    } catch (e) {
        if (e.name === 'AbortError') {
            showToast('Humanization cancelled by user', 'warning');
        } else {
            alert('Humanization error: ' + e.message);
            showToast('Error: ' + e.message, 'error');
        }
    } finally {
        clearInterval(progressTimer);
        if (progressFill) progressFill.style.width = '100%';
        if (progressPct) progressPct.textContent = '100%';
        if (progressMsg) progressMsg.textContent = 'Transformation finished';
        setTimeout(() => {
            if (progressCard) progressCard.classList.add('hidden');
        }, 500);

        if (btn) btn.disabled = false;
        if (btnLabel) btnLabel.textContent = '✨ Run AI Humanization';
        if (btnAbort) btnAbort.classList.add('hidden');
        activeRequests.humanize = { controller: null, id: null };
    }
}

function abortHumanizeAnalysis() {
    const item = activeRequests.humanize;
    if (!item) return;

    if (item.id) {
        fetch(`/api/cancel/${item.id}`, { method: 'POST' }).catch(() => {
            fetch(`/api/abort/${item.id}`, { method: 'POST' }).catch(() => {});
        });
    }
    if (item.controller) {
        item.controller.abort();
    }
    activeRequests.humanize = { controller: null, id: null };
}

function getFallbackConfidence(pctVal) {
    const pct = parseFloat(pctVal) || 50.0;
    if (pct < 20.0) {
        return {
            code: 'verified_human',
            grade: 'Grade A (Verified Human)',
            label: 'Verified Human',
            badge_class: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
            description: 'High confidence human authorship with natural lexical variety and burstiness.',
        };
    } else if (pct < 45.0) {
        return {
            code: 'likely_human',
            grade: 'Grade B (Likely Human)',
            label: 'Likely Human / Formal Style',
            badge_class: 'bg-teal-500/10 text-teal-300 border-teal-500/30',
            description: 'Predominantly human characteristics with structured or formal terminology.',
        };
    } else if (pct <= 60.0) {
        return {
            code: 'uncertain_borderline',
            grade: 'Grade C (Borderline Content)',
            label: 'Uncertain / Borderline Content',
            badge_class: 'bg-amber-500/10 text-amber-300 border-amber-500/30',
            description: 'Score near 50% threshold. Typical for legal/historical charters, formulaic encyclopedic prose, or mixed hybrid edits.',
        };
    } else if (pct <= 85.0) {
        return {
            code: 'likely_ai',
            grade: 'Grade D (Likely AI)',
            label: 'Likely AI-Generated',
            badge_class: 'bg-orange-500/10 text-orange-400 border-orange-500/30',
            description: 'Substantial synthetic phrasing, uniform rhythm, and low perplexity typical of LLMs.',
        };
    } else {
        return {
            code: 'high_confidence_ai',
            grade: 'Grade F (Confirmed AI)',
            label: 'High-Confidence AI Generation',
            badge_class: 'bg-rose-500/10 text-rose-400 border-rose-500/30',
            description: 'Unmistakable synthetic signatures (ChatGPT, Claude, DeepSeek, or Llama).',
        };
    }
}

// ----------------------------------------------------
// RENDERING RESULTS DASHBOARD
// ----------------------------------------------------
function renderUnifiedResults(data) {
    const summary = data.summary || {};
    const monolithicVal = (data.monolithic_scale !== undefined) ? data.monolithic_scale : (data.human_ai_scale || 50.0);
    const paragraphVal = (data.paragraph_scale !== undefined) ? data.paragraph_scale : (summary.overall_scale || monolithicVal);

    // 1. Verdict Confidence Status (Sole Primary Verdict - No Binary AI/Human)
    const conf = data.confidence_status || (summary && summary.confidence_status) || getFallbackConfidence(monolithicVal);

    let iconSvg = '';
    if (conf.code === 'verified_human') {
        iconSvg = `<svg class="w-5 h-5 text-emerald-400" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/></svg>`;
    } else if (conf.code === 'likely_human') {
        iconSvg = `<svg class="w-5 h-5 text-teal-300" fill="currentColor" viewBox="0 0 20 20"><path d="M9 4.804A7.968 7.968 0 005.5 4c-1.255 0-2.443.29-3.5.804v10A7.969 7.969 0 015.5 14c1.669 0 3.218.51 4.5 1.385A7.962 7.962 0 0114.5 14c1.255 0 2.443.29 3.5.804v-10A7.968 7.968 0 0014.5 4c-1.255 0-2.443.29-3.5.804V12a1 1 0 11-2 0V4.804z"/></svg>`;
    } else if (conf.code === 'uncertain_borderline') {
        iconSvg = `<svg class="w-5 h-5 text-amber-300" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/></svg>`;
    } else if (conf.code === 'likely_ai') {
        iconSvg = `<svg class="w-5 h-5 text-orange-400" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/></svg>`;
    } else {
        iconSvg = `<svg class="w-5 h-5 text-rose-400" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/></svg>`;
    }

    const statusContainer = document.getElementById('confidence-status-container');
    const statusIcon = document.getElementById('confidence-status-icon');
    const statusText = document.getElementById('confidence-status-text');
    const statusDesc = document.getElementById('confidence-status-desc');

    if (statusContainer) {
        statusContainer.className = `inline-flex items-center gap-2.5 px-6 py-2.5 rounded-xl border text-base font-bold shadow-lg transition-all ${conf.badge_class || 'border-blue-500/40 bg-blue-500/10 text-blue-300'}`;
    }
    if (statusIcon) statusIcon.innerHTML = iconSvg;
    if (statusText) statusText.textContent = conf.grade || conf.label;
    if (statusDesc) statusDesc.textContent = conf.description || '';

    const modelBadge = document.getElementById('active-model-badge');
    const deviceBadge = document.getElementById('active-device-badge');
    if (modelBadge) modelBadge.textContent = `Model: ${data.model_used || 'RuBERT-AI'}`;
    if (deviceBadge) deviceBadge.textContent = `Device: ${(data.device || 'Auto').toUpperCase()}`;

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

    // 2. Dual Meters: Monolithic vs Paragraph-Weighted
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
    const isSuspicious = (monolithicVal >= 50.0);
    const aiParas = (summary.ai_paragraphs !== undefined) ? summary.ai_paragraphs : (isSuspicious ? 1 : 0);
    const humanParas = (summary.human_paragraphs !== undefined) ? summary.human_paragraphs : (isSuspicious ? 0 : 1);
    const aiPct = (summary.ai_paragraph_pct !== undefined) ? summary.ai_paragraph_pct : (isSuspicious ? 100 : 0);

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
        const scaleVal = (p.human_ai_scale !== undefined) ? p.human_ai_scale.toFixed(1) : '50.0';
        const paraConf = getFallbackConfidence(p.human_ai_scale);
        const card = document.createElement('div');
        card.id = `para-card-${p.index}`;
        const isSuspicious = (p.human_ai_scale || 0) >= 50.0;
        const borderClass = isSuspicious ? 'border-rose-500/30 bg-rose-950/10' : 'border-emerald-500/30 bg-emerald-950/10';
        const wordCount = p.word_count || (p.text ? p.text.split(/\s+/).length : 0);

        let humanizeBtnHtml = '';
        if (isSuspicious || p.label === 'ai' || (p.human_ai_scale || 0) >= 45.0) {
            humanizeBtnHtml = `
                <button onclick="humanizeIndividualParagraph(${p.index})" id="btn-humanize-para-${p.index}" class="flex items-center gap-1 px-2.5 py-1 rounded bg-indigo-500/10 hover:bg-indigo-500/20 active:bg-indigo-500/30 text-indigo-300 border border-indigo-500/30 text-[11px] font-semibold transition" title="Humanize this paragraph in-place">
                    <svg class="w-3.5 h-3.5 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/>
                    </svg>
                    <span>✨ Humanize Paragraph</span>
                </button>
            `;
        }

        card.className = `p-4 rounded-xl border ${borderClass} space-y-2.5 transition`;
        card.innerHTML = `
            <div class="flex items-center justify-between text-xs">
                <div class="flex items-center gap-2">
                    <span class="font-mono text-slate-400 font-semibold">#${p.index}</span>
                    <span class="px-2.5 py-0.5 rounded text-[11px] font-bold border ${paraConf.badge_class}">
                        ${paraConf.grade}
                    </span>
                    <span class="text-[11px] text-slate-500 font-mono">${wordCount} words</span>
                </div>
                <div class="flex items-center gap-2.5">
                    ${humanizeBtnHtml}
                    <span class="text-slate-400 font-mono text-[11px]">AI Probability:</span>
                    <span class="font-mono font-bold ${isSuspicious ? 'text-rose-400' : 'text-emerald-400'} text-xs">${scaleVal}%</span>
                </div>
            </div>

            <!-- Mini Progress bar for this paragraph -->
            <div class="w-full bg-slate-900 rounded-full h-1.5 overflow-hidden">
                <div class="${isSuspicious ? 'bg-rose-500' : 'bg-emerald-500'} h-1.5 rounded-full" style="width: ${scaleVal}%"></div>
            </div>

            <p class="text-xs text-slate-200 leading-relaxed font-normal">${escapeHtml(p.text)}</p>
        `;
        container.appendChild(card);
    });
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

// ----------------------------------------------------
// MARKDOWN REPORT EXPORT (WEB MODE)
// ----------------------------------------------------
async function getUnifiedMarkdownReportText() {
    if (!lastUnifiedResult) {
        throw new Error('No inspection results available to generate report.');
    }

    const payload = {
        data: lastUnifiedResult,
        report_type: 'single',
        filename: lastUnifiedResult.filename || (selectedUnifiedFile ? selectedUnifiedFile.name : 'direct_text_analysis')
    };

    const res = await fetch('/api/report/markdown', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });

    if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Server returned HTTP ${res.status}`);
    }

    return await res.text();
}

async function downloadUnifiedMarkdownReport() {
    if (!lastUnifiedResult) {
        showToast('Please run an analysis first.', 'warning');
        return;
    }

    try {
        showToast('Generating Markdown report...', 'info');
        const mdText = await getUnifiedMarkdownReportText();
        const baseName = lastUnifiedResult.filename || (selectedUnifiedFile ? selectedUnifiedFile.name : 'text_inspection');
        const cleanBase = baseName.replace(/\.[^/.]+$/, "");
        const filename = `report_${cleanBase}.md`;

        const blob = new Blob([mdText], { type: 'text/markdown;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        showToast('Markdown report downloaded successfully', 'success');
    } catch (err) {
        showToast('Failed to download report: ' + err.message, 'error');
    }
}

async function copyUnifiedMarkdownReport() {
    if (!lastUnifiedResult) {
        showToast('Please run an analysis first.', 'warning');
        return;
    }

    try {
        showToast('Generating Markdown report...', 'info');
        const mdText = await getUnifiedMarkdownReportText();
        await navigator.clipboard.writeText(mdText);
        showToast('Markdown report copied to clipboard!', 'success');
    } catch (err) {
        showToast('Failed to copy report: ' + err.message, 'error');
    }
}

// ----------------------------------------------------
// UTILITIES & EXPORTS
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

function escapeJsString(str) {
    if (!str) return '';
    return str.replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/\n/g, '\\n').replace(/\r/g, '');
}

function copyTextToClipboard(text, successMsg = 'Copied to clipboard!') {
    if (!text) return;
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(() => {
            showToast(successMsg, 'success');
        }).catch(err => {
            showToast('Copy failed: ' + err.message, 'error');
        });
    } else {
        const ta = document.createElement('textarea');
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        showToast(successMsg, 'success');
    }
}

// ----------------------------------------------------
// HUMANIZER RESULTS RENDERING & ACTIONS
// ----------------------------------------------------
function renderHumanizeResult(data) {
    if (!data) return;

    // 1. Dual Comparison Scores & Delta Badge
    const beforeScore = data.before_scale !== undefined ? data.before_scale : 0;
    const afterScore = data.after_scale !== undefined ? data.after_scale : 0;
    const delta = data.delta_scale !== undefined ? data.delta_scale : (beforeScore - afterScore);

    const beforeEl = document.getElementById('humanize-before-score');
    const afterEl = document.getElementById('humanize-after-score');
    const deltaEl = document.getElementById('humanize-delta-badge');
    const beforeGradeEl = document.getElementById('humanize-before-grade');
    const afterGradeEl = document.getElementById('humanize-after-grade');

    if (beforeEl) beforeEl.textContent = `${beforeScore.toFixed(1)}%`;
    if (afterEl) afterEl.textContent = `${afterScore.toFixed(1)}%`;

    const beforeConf = data.before_confidence || getFallbackConfidence(beforeScore);
    const afterConf = data.after_confidence || getFallbackConfidence(afterScore);

    if (beforeGradeEl) {
        beforeGradeEl.textContent = beforeConf.grade || (beforeScore >= 50 ? 'Grade D' : 'Grade A');
        beforeGradeEl.className = `px-2 py-0.5 rounded text-[11px] font-bold border ${beforeConf.badge_class || 'border-rose-500/30 text-rose-300 bg-rose-500/10'}`;
    }
    if (afterGradeEl) {
        afterGradeEl.textContent = afterConf.grade || (afterScore >= 50 ? 'Grade D' : 'Grade A');
        afterGradeEl.className = `px-2 py-0.5 rounded text-[11px] font-bold border ${afterConf.badge_class || 'border-emerald-500/30 text-emerald-300 bg-emerald-500/10'}`;
    }

    if (deltaEl) {
        const sign = delta >= 0 ? '-' : '+';
        deltaEl.innerHTML = `
            <svg class="w-3.5 h-3.5 ${delta >= 0 ? 'text-emerald-400' : 'text-rose-400'}" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="${delta >= 0 ? 'M19 14l-7 7m0 0l-7-7m7 7V3' : 'M5 10l7-7m0 0l7 7m-7-7v18'}"/>
            </svg>
            <span>${sign}${Math.abs(delta).toFixed(1)}% AI Reduction</span>
        `;
        deltaEl.className = delta >= 0 
            ? "px-3.5 py-1.5 rounded-full text-xs font-bold font-mono bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 flex items-center gap-1.5 shadow-sm"
            : "px-3.5 py-1.5 rounded-full text-xs font-bold font-mono bg-rose-500/20 text-rose-300 border border-rose-500/40 flex items-center gap-1.5 shadow-sm";
    }

    // 2. Summary KPI stats
    const summary = data.summary || {};
    const genResultBadge = document.getElementById('humanize-generator-result-badge');
    if (genResultBadge) {
        const provider = summary.generator_provider || data.generator_model || 'T5 Paraphraser';
        const isExt = summary.external_llm_active !== undefined ? summary.external_llm_active : data.external_llm_active;
        if (isExt) {
            genResultBadge.innerHTML = `<svg class="w-3 h-3 text-emerald-400 inline mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>${escapeHtml(provider)} (Local T5 bypassed)`;
            genResultBadge.className = "inline-flex items-center px-2.5 py-0.5 rounded-full text-[11px] font-mono font-medium bg-emerald-500/10 text-emerald-300 border border-emerald-500/30";
        } else {
            genResultBadge.innerHTML = `<svg class="w-3 h-3 text-indigo-400 inline mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/></svg>${escapeHtml(provider)} (Local)`;
            genResultBadge.className = "inline-flex items-center px-2.5 py-0.5 rounded-full text-[11px] font-mono font-medium bg-indigo-500/10 text-indigo-300 border border-indigo-500/30";
        }
    }
    const modifiedParas = summary.modified_paragraphs_count !== undefined ? summary.modified_paragraphs_count : (data.paragraphs ? data.paragraphs.filter(p => p.changed).length : 0);
    const totalParas = summary.total_paragraphs !== undefined ? summary.total_paragraphs : (data.paragraphs ? data.paragraphs.length : 1);
    const removedCliches = summary.removed_cliches_count !== undefined ? summary.removed_cliches_count : (data.removed_markers ? data.removed_markers.length : 0);
    const wordsBefore = summary.word_count_before || (data.original_text ? data.original_text.split(/\s+/).length : 0);
    const wordsAfter = summary.word_count_after || (data.humanized_text ? data.humanized_text.split(/\s+/).length : 0);
    const charsBefore = summary.char_count_before || (data.original_text ? data.original_text.length : 0);
    const charsAfter = summary.char_count_after || (data.humanized_text ? data.humanized_text.length : 0);

    const statParasEl = document.getElementById('humanize-stat-modified-paras');
    const statClichesEl = document.getElementById('humanize-stat-removed-cliches');
    const statWordsEl = document.getElementById('humanize-stat-words');
    const statCharsEl = document.getElementById('humanize-stat-chars');

    if (statParasEl) statParasEl.textContent = `${modifiedParas} / ${totalParas}`;
    if (statClichesEl) statClichesEl.textContent = `${removedCliches} scrubbed`;
    if (statWordsEl) statWordsEl.textContent = `${wordsBefore.toLocaleString()} → ${wordsAfter.toLocaleString()}`;
    if (statCharsEl) statCharsEl.textContent = `${charsBefore.toLocaleString()} → ${charsAfter.toLocaleString()}`;

    // 3. Side-by-Side Comparison Panes
    const origViewer = document.getElementById('humanize-original-viewer');
    const resViewer = document.getElementById('humanize-result-viewer');
    if (origViewer) origViewer.textContent = data.original_text || '';
    if (resViewer) resViewer.textContent = data.humanized_text || '';

    // 4. Paragraph Breakdown List
    const parasContainer = document.getElementById('humanize-paragraphs-container');
    const parasBadge = document.getElementById('humanize-paras-count-badge');
    const paragraphs = data.paragraphs || [];

    if (parasBadge) parasBadge.textContent = `${paragraphs.length} paragraphs`;
    if (parasContainer) {
        parasContainer.innerHTML = '';
        if (paragraphs.length === 0) {
            parasContainer.innerHTML = `<div class="text-center py-8 text-xs text-slate-500">No paragraph diffs available.</div>`;
        } else {
            paragraphs.forEach(p => {
                const card = document.createElement('div');
                const pBefore = (p.before_scale !== undefined) ? p.before_scale.toFixed(1) : '0.0';
                const pAfter = (p.after_scale !== undefined) ? p.after_scale.toFixed(1) : '0.0';
                const pDelta = (p.delta_scale !== undefined) ? p.delta_scale.toFixed(1) : (p.before_scale - p.after_scale).toFixed(1);
                const isChanged = p.changed !== false;
                const borderClass = isChanged ? 'border-indigo-500/30 bg-indigo-950/10' : 'border-slate-800 bg-slate-950/40';

                card.className = `p-4 rounded-xl border ${borderClass} space-y-3 transition`;
                card.innerHTML = `
                    <div class="flex flex-wrap items-center justify-between gap-2 text-xs border-b border-slate-800 pb-2.5">
                        <div class="flex items-center gap-2">
                            <span class="font-mono text-slate-400 font-semibold">#${p.index}</span>
                            <span class="px-2 py-0.5 rounded text-[11px] font-bold border ${p.before_scale >= 50 ? 'border-rose-500/30 text-rose-300 bg-rose-500/10' : 'border-emerald-500/30 text-emerald-300 bg-emerald-500/10'}">
                                Before: ${pBefore}%
                            </span>
                            <span class="text-slate-500 text-xs">→</span>
                            <span class="px-2 py-0.5 rounded text-[11px] font-bold border ${p.after_scale >= 50 ? 'border-rose-500/30 text-rose-300 bg-rose-500/10' : 'border-emerald-500/30 text-emerald-300 bg-emerald-500/10'}">
                                After: ${pAfter}%
                            </span>
                            <span class="px-2 py-0.5 rounded text-[11px] font-mono font-bold ${parseFloat(pDelta) >= 0 ? 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/30' : 'bg-slate-800 text-slate-400'}">
                                ${parseFloat(pDelta) >= 0 ? '-' : '+'}${Math.abs(parseFloat(pDelta))}% AI
                            </span>
                        </div>
                        <div class="flex items-center gap-2">
                            ${p.is_code || p.stage_applied === 'code' ? '<span class="px-2 py-0.5 rounded text-[10px] font-mono text-cyan-300 bg-cyan-500/10 border border-cyan-500/20">Code (Preserved)</span>' : ''}
                            <span class="px-2 py-0.5 rounded text-[10px] font-mono text-slate-400 bg-slate-900 border border-slate-800">
                                Stage: ${escapeHtml(p.stage_applied || 'full')}
                            </span>
                            <span class="px-2 py-0.5 rounded text-[10px] font-semibold ${isChanged ? 'text-indigo-300 bg-indigo-500/20' : 'text-slate-400 bg-slate-800'}">
                                ${isChanged ? 'Modified' : 'Unchanged'}
                            </span>
                            <button onclick="copyTextToClipboard('${escapeJsString(p.humanized_text)}', 'Paragraph #${p.index} copied')" class="p-1 rounded text-slate-400 hover:text-white transition" title="Copy humanized paragraph">
                                <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3"/>
                                </svg>
                            </button>
                        </div>
                    </div>

                    <div class="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                        <div class="space-y-1">
                            <span class="text-[11px] font-semibold text-rose-400 block">${p.is_code ? 'Original Code Block:' : 'Original Paragraph:'}</span>
                            ${p.is_code ? `<pre class="font-mono text-[11px] overflow-x-auto text-slate-300 bg-slate-950 p-2.5 rounded-lg border border-slate-900 whitespace-pre-wrap"><code>${escapeHtml(p.original_text)}</code></pre>` : `<p class="text-slate-300 leading-relaxed font-normal bg-slate-950 p-2.5 rounded-lg border border-slate-900">${escapeHtml(p.original_text)}</p>`}
                        </div>
                        <div class="space-y-1">
                            <span class="text-[11px] font-semibold text-emerald-400 block">${p.is_code ? 'Preserved Code Block:' : 'Humanized Paragraph:'}</span>
                            ${p.is_code ? `<pre class="font-mono text-[11px] overflow-x-auto text-emerald-300 bg-slate-950 p-2.5 rounded-lg border border-slate-900 whitespace-pre-wrap"><code>${escapeHtml(p.humanized_text)}</code></pre>` : `<p class="text-slate-100 leading-relaxed font-normal bg-slate-950 p-2.5 rounded-lg border border-slate-900">${escapeHtml(p.humanized_text)}</p>`}
                        </div>
                    </div>
                `;
                parasContainer.appendChild(card);
            });
        }
    }
}

async function copyHumanizedText() {
    if (!lastHumanizeResult || !lastHumanizeResult.humanized_text) {
        showToast('No humanized text available to copy.', 'warning');
        return;
    }
    copyTextToClipboard(lastHumanizeResult.humanized_text, 'Humanized text copied to clipboard!');
}

function downloadHumanizedTextFile(ext = 'txt') {
    if (!lastHumanizeResult || !lastHumanizeResult.humanized_text) {
        showToast('No humanized text available to download.', 'warning');
        return;
    }

    const text = lastHumanizeResult.humanized_text;
    const base = lastHumanizeResult.filename || (selectedHumanizeFile ? selectedHumanizeFile.name : 'humanized_text');
    const cleanBase = base.replace(/\.[^/.]+$/, "");
    const filename = `humanized_${cleanBase}.${ext}`;
    const mimeType = ext === 'md' ? 'text/markdown;charset=utf-8;' : 'text/plain;charset=utf-8;';

    const blob = new Blob([text], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showToast(`Downloaded ${filename} successfully`, 'success');
}

async function downloadHumanizeMarkdownReport() {
    if (!lastHumanizeResult) {
        showToast('Please run humanization first.', 'warning');
        return;
    }

    try {
        showToast('Generating Humanization Markdown report...', 'info');
        const baseName = lastHumanizeResult.filename || (selectedHumanizeFile ? selectedHumanizeFile.name : 'humanized_session');
        const cleanBase = baseName.replace(/\.[^/.]+$/, "");
        const filename = `report_humanized_${cleanBase}.md`;

        const payload = {
            data: lastHumanizeResult,
            report_type: 'humanize',
            filename: filename,
        };

        const res = await fetch('/api/report/markdown', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!res.ok) {
            const errJson = await res.json().catch(() => ({}));
            throw new Error(errJson.detail || `Server returned HTTP ${res.status}`);
        }

        const mdText = await res.text();
        const blob = new Blob([mdText], { type: 'text/markdown;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        showToast('Humanization audit report downloaded successfully', 'success');
    } catch (err) {
        showToast('Failed to download report: ' + err.message, 'error');
    }
}

function reverifyInInspector() {
    if (!lastHumanizeResult || !lastHumanizeResult.humanized_text) {
        showToast('No humanized text available to verify.', 'warning');
        return;
    }

    const text = lastHumanizeResult.humanized_text;
    const input = document.getElementById('text-input');
    if (input) {
        input.value = text;
        input.dispatchEvent(new Event('input'));
    }

    setAnalysisSourceMode('text');
    switchView('unified');

    showToast('Re-verifying humanized text in Inspector...', 'info');
    submitHybridAnalysis();
}

// ----------------------------------------------------
// INSPECTOR INTEGRATION FUNCTIONS
// ----------------------------------------------------
function transferTextToHumanizer() {
    if (!lastUnifiedResult) {
        showToast('Please run an analysis first.', 'warning');
        return;
    }

    const fullText = lastUnifiedResult.full_text || (lastUnifiedResult.paragraphs ? lastUnifiedResult.paragraphs.map(p => p.text).join('\n\n') : '');
    if (!fullText) {
        showToast('No text available to transfer.', 'warning');
        return;
    }

    const humanizeInput = document.getElementById('humanize-text-input');
    if (humanizeInput) {
        humanizeInput.value = fullText;
        humanizeInput.dispatchEvent(new Event('input'));
    }

    if (lastUnifiedResult.mode) {
        const modeSelect = document.getElementById('humanize-mode-select');
        if (modeSelect) modeSelect.value = lastUnifiedResult.mode;
    }

    setHumanizeInputMode('text');
    clearSelectedHumanizeFile();
    switchView('humanize');

    showToast('Text transferred to AI Humanizer Workspace', 'success');
}

async function humanizeIndividualParagraph(index) {
    if (!lastUnifiedResult || !lastUnifiedResult.paragraphs) return;

    const p = lastUnifiedResult.paragraphs.find(item => item.index === index);
    if (!p || !p.text) return;

    const btn = document.getElementById(`btn-humanize-para-${index}`);
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = `
            <svg class="animate-spin w-3.5 h-3.5 text-indigo-400" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <span>Humanizing...</span>
        `;
    }

    try {
        const res = await fetch('/api/humanize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: p.text,
                mode: lastUnifiedResult.mode || 'auto',
                strength: 'balanced',
            }),
        });

        if (!res.ok) {
            const errJson = await res.json().catch(() => ({}));
            throw new Error(errJson.detail || `Server error HTTP ${res.status}`);
        }

        const data = await res.json();

        // Update paragraph in lastUnifiedResult
        const oldScale = p.human_ai_scale || 0;
        p.text = data.humanized_text;
        p.human_ai_scale = data.after_scale;
        p.label = data.after_scale >= 50.0 ? 'ai' : 'human';
        p.ai_probability = data.after_scale / 100.0;
        p.word_count = data.humanized_text.split(/\s+/).length;

        // Recalculate document-level weighted score
        let totalWeighted = 0;
        let totalWords = 0;
        lastUnifiedResult.paragraphs.forEach(item => {
            const words = item.word_count || (item.text ? item.text.split(/\s+/).length : 1);
            totalWeighted += (item.human_ai_scale || 0) * words;
            totalWords += words;
        });
        const newWeightedScale = totalWords > 0 ? (totalWeighted / totalWords) : data.after_scale;
        lastUnifiedResult.paragraph_weighted_scale = newWeightedScale;
        if (lastUnifiedResult.summary) {
            lastUnifiedResult.summary.paragraph_weighted_scale = newWeightedScale;
            lastUnifiedResult.summary.ai_paragraphs = lastUnifiedResult.paragraphs.filter(item => item.label === 'ai').length;
            lastUnifiedResult.summary.human_paragraphs = lastUnifiedResult.paragraphs.filter(item => item.label !== 'ai').length;
            lastUnifiedResult.summary.ai_paragraph_pct = Math.round((lastUnifiedResult.summary.ai_paragraphs / lastUnifiedResult.paragraphs.length) * 100);
        }

        // Update full_text in lastUnifiedResult
        lastUnifiedResult.full_text = lastUnifiedResult.paragraphs.map(item => item.text).join('\n\n');

        renderUnifiedResults(lastUnifiedResult);

        // Flash highlight on updated card
        const cardId = `para-card-${index}`;
        const updatedCard = document.getElementById(cardId);
        if (updatedCard) {
            updatedCard.classList.add('ring-2', 'ring-emerald-500', 'transition-all');
            setTimeout(() => updatedCard.classList.remove('ring-2', 'ring-emerald-500'), 2000);
        }

        showToast(`Paragraph #${index} successfully humanized: ${oldScale.toFixed(1)}% → ${data.after_scale.toFixed(1)}%`, 'success');

    } catch (err) {
        showToast(`Failed to humanize paragraph #${index}: ${err.message}`, 'error');
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = `
                <svg class="w-3.5 h-3.5 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/>
                </svg>
                <span>✨ Humanize Paragraph</span>
            `;
        }
    }
}
