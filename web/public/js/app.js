// VeriText Modern AI Detector Frontend Application Logic

let selectedBatchFile = null;
let lastBatchResult = null;
let selectedDocFile = null;
let lastDocResult = null;
let currentDocFilter = 'all';

const SAMPLES = {
    ru_ai: "В современном мире технологии искусственного интеллекта играют ключевую роль во многих аспектах нашей повседневной жизни. Важно подчеркнуть, что интеграция нейросетевых моделей позволяет существенно оптимизировать рабочие процессы и повысить продуктивность. В заключение стоит отметить, что будущее цифровой трансформации неразрывно связано с развитием передовых алгоритмов.",
    ru_human: "Вчера наконец-то забрал заказ из пункта выдачи. Коробка была слегка помята с одного края, но внутри всё целое. Наушники классные, звук чистый, басы на месте. Правда провод зарядки коротковат, но в остальном покупкой доволен.",
    en_ai: "Artificial intelligence has become an integral part of modern society, driving innovation across various industries. It is worth noting that deep learning frameworks provide sophisticated tools for complex problem-solving. In conclusion, these technologies represent a testament to rapid technological advancement.",
};

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    initHealthCheck();
    initDropzones();
    initTextareaCounter();
});

// Tab navigation (text, doc, batch)
function switchTab(tab) {
    const textTab = document.getElementById('tab-text');
    const docTab = document.getElementById('tab-doc');
    const batchTab = document.getElementById('tab-batch');

    const btnText = document.getElementById('tab-btn-text');
    const btnDoc = document.getElementById('tab-btn-doc');
    const btnBatch = document.getElementById('tab-btn-batch');

    const activeClass = "flex items-center gap-2 px-5 py-3 border-b-2 font-medium text-sm transition-all border-blue-500 text-blue-400 bg-slate-900/40 rounded-t-lg whitespace-nowrap";
    const inactiveClass = "flex items-center gap-2 px-5 py-3 border-b-2 font-medium text-sm transition-all border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-900/20 rounded-t-lg whitespace-nowrap";

    textTab.classList.add('hidden');
    docTab.classList.add('hidden');
    batchTab.classList.add('hidden');

    btnText.className = inactiveClass;
    btnDoc.className = inactiveClass;
    btnBatch.className = inactiveClass;

    if (tab === 'text') {
        textTab.classList.remove('hidden');
        btnText.className = activeClass;
    } else if (tab === 'doc') {
        docTab.classList.remove('hidden');
        btnDoc.className = activeClass;
    } else if (tab === 'batch') {
        batchTab.classList.remove('hidden');
        btnBatch.className = activeClass;
    }
}

// Health check and GPU indicator
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
                deviceLabel.innerHTML = `<span class="text-blue-400 font-semibold">CPU:</span> ${data.device_name}`;
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
    const textarea = document.getElementById('text-input');
    textarea.value = SAMPLES[type] || '';
    textarea.dispatchEvent(new Event('input'));
}

function clearText() {
    const textarea = document.getElementById('text-input');
    textarea.value = '';
    textarea.dispatchEvent(new Event('input'));
}

// Submit single text analysis
async function submitTextAnalysis() {
    const text = document.getElementById('text-input').value.trim();
    if (!text) {
        alert('Please enter text to analyze.');
        return;
    }

    const mode = document.getElementById('mode-select').value;
    const btn = document.getElementById('btn-submit-text');
    const btnLabel = document.getElementById('btn-submit-text-label');
    const emptyState = document.getElementById('results-empty-state');
    const resultsCard = document.getElementById('results-card');

    btn.disabled = true;
    btnLabel.textContent = 'Running inference...';

    try {
        const res = await fetch('/api/score', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: text,
                mode: mode,
            })
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Classification error');
        }

        const data = await res.json();
        renderTextResult(data);

        emptyState.classList.add('hidden');
        resultsCard.classList.remove('hidden');
    } catch (e) {
        alert('Analysis error: ' + e.message);
    } finally {
        btn.disabled = false;
        btnLabel.textContent = 'Analyze Text';
    }
}

function renderTextResult(data) {
    const isAI = data.label === 'ai';
    const verdictContainer = document.getElementById('verdict-container');
    const verdictIcon = document.getElementById('verdict-icon');
    const verdictText = document.getElementById('verdict-text');
    const activeModelBadge = document.getElementById('active-model-badge');
    const scaleIndicator = document.getElementById('scale-indicator');
    const scaleValue = document.getElementById('scale-value');
    const metricScore = document.getElementById('metric-score');
    const metricDevice = document.getElementById('metric-device');
    const markersContainer = document.getElementById('markers-container');
    const markersList = document.getElementById('markers-list');

    if (isAI) {
        verdictContainer.className = "mt-2 inline-flex items-center gap-2 px-6 py-2.5 rounded-full border border-rose-500/30 bg-rose-500/10 text-rose-400 text-base font-bold shadow-lg shadow-rose-500/10";
        verdictIcon.innerHTML = `<svg class="w-5 h-5 text-rose-400" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/></svg>`;
        verdictText.textContent = "AI-Generated (AI)";
    } else {
        verdictContainer.className = "mt-2 inline-flex items-center gap-2 px-6 py-2.5 rounded-full border border-emerald-500/30 bg-emerald-500/10 text-emerald-400 text-base font-bold shadow-lg shadow-emerald-500/10";
        verdictIcon.innerHTML = `<svg class="w-5 h-5 text-emerald-400" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/></svg>`;
        verdictText.textContent = "Human-Written (Human)";
    }

    if (activeModelBadge) {
        activeModelBadge.textContent = `Model: ${data.model_used || 'RuBERT-AI'}`;
    }

    const scale = Math.max(0, Math.min(100, data.human_ai_scale || 50));
    scaleIndicator.style.left = `${scale}%`;
    scaleValue.textContent = `${scale.toFixed(2)}%`;

    metricScore.textContent = data.score.toFixed(3);
    metricDevice.textContent = data.device;

    // Render detected stylistic markers
    if (data.markers && data.markers.length > 0) {
        markersContainer.classList.remove('hidden');
        markersList.innerHTML = '';
        data.markers.forEach(m => {
            const item = document.createElement('div');
            item.className = "p-2 rounded bg-slate-950 border border-slate-800 text-xs flex justify-between items-center";
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
        markersContainer.classList.add('hidden');
    }
}

// Dropzone logic
function initDropzones() {
    setupDropzone('file-dropzone', handleSelectedFile);
    setupDropzone('doc-dropzone', handleSelectedDocFile);
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

// ----------------------------------------------------
// DOCUMENT ANALYSIS LOGIC (.md, .docx, .pdf, .txt)
// ----------------------------------------------------

function handleDocFileSelected(event) {
    const file = event.target.files[0];
    if (file) {
        handleSelectedDocFile(file);
    }
}

function handleSelectedDocFile(file) {
    selectedDocFile = file;
    const dropzoneText = document.getElementById('doc-dropzone-text');
    const dropzoneSubtext = document.getElementById('doc-dropzone-subtext');
    const btn = document.getElementById('btn-run-doc');
    const btnLabel = document.getElementById('btn-run-doc-label');

    const sizeKb = (file.size / 1024).toFixed(1);
    const ext = file.name.split('.').pop().toUpperCase();
    dropzoneText.textContent = `Selected document: [${ext}] ${file.name} (${sizeKb} KB)`;
    dropzoneSubtext.textContent = 'Click "Inspect Document" below to extract text and analyze';

    btn.disabled = false;
    btnLabel.textContent = `Inspect Document (${file.name})`;
}

async function submitDocAnalysis() {
    if (!selectedDocFile) {
        alert('Please select a valid document file (.md, .docx, .pdf, .txt)');
        return;
    }

    const mode = document.getElementById('doc-mode-select').value;
    const btn = document.getElementById('btn-run-doc');
    const btnLabel = document.getElementById('btn-run-doc-label');
    const resultsContainer = document.getElementById('doc-results-container');

    btn.disabled = true;
    btnLabel.textContent = 'Extracting text and analyzing...';

    const formData = new FormData();
    formData.append('file', selectedDocFile);
    formData.append('mode', mode);

    try {
        const res = await fetch('/api/score-document', {
            method: 'POST',
            body: formData
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Document analysis error');
        }

        const data = await res.json();
        lastDocResult = data;
        renderDocResult(data);
        resultsContainer.classList.remove('hidden');
    } catch (e) {
        alert('Error: ' + e.message);
    } finally {
        btn.disabled = false;
        btnLabel.textContent = `Inspect Document (${selectedDocFile.name})`;
    }
}

function renderDocResult(data) {
    const summary = data.summary;
    const isAI = summary.overall_label === 'ai';

    document.getElementById('doc-result-filename').textContent = data.filename;
    document.getElementById('doc-result-stats').textContent = 
        `${summary.word_count.toLocaleString()} words | ${summary.char_count.toLocaleString()} chars | ${summary.total_paragraphs} paragraphs`;

    const verdictContainer = document.getElementById('doc-verdict-container');
    const verdictIcon = document.getElementById('doc-verdict-icon');
    const verdictText = document.getElementById('doc-verdict-text');
    const modelMetric = document.getElementById('doc-model-metric');

    if (isAI) {
        verdictContainer.className = "inline-flex items-center gap-2 px-5 py-2 rounded-full border border-rose-500/30 bg-rose-500/10 text-rose-400 text-sm font-bold shadow-lg shadow-rose-500/10";
        verdictIcon.innerHTML = `<svg class="w-4 h-4 text-rose-400" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/></svg>`;
        verdictText.textContent = "AI-Generated (AI)";
    } else {
        verdictContainer.className = "inline-flex items-center gap-2 px-5 py-2 rounded-full border border-emerald-500/30 bg-emerald-500/10 text-emerald-400 text-sm font-bold shadow-lg shadow-emerald-500/10";
        verdictIcon.innerHTML = `<svg class="w-4 h-4 text-emerald-400" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/></svg>`;
        verdictText.textContent = "Human-Written (Human)";
    }

    if (modelMetric) {
        modelMetric.textContent = data.model_used || 'RuBERT-AI';
    }

    const scale = Math.max(0, Math.min(100, summary.overall_scale || 50));
    document.getElementById('doc-scale-indicator').style.left = `${scale}%`;
    document.getElementById('doc-scale-value').textContent = `${scale.toFixed(2)}%`;

    document.getElementById('doc-ai-paras').textContent = summary.ai_paragraphs;
    document.getElementById('doc-ai-pct').textContent = `(${summary.ai_paragraph_pct}%)`;
    document.getElementById('doc-human-paras').textContent = summary.human_paragraphs;
    document.getElementById('doc-human-pct').textContent = `(${(100 - summary.ai_paragraph_pct).toFixed(2)}%)`;

    document.getElementById('count-filter-all').textContent = data.paragraphs.length;
    document.getElementById('count-filter-ai').textContent = summary.ai_paragraphs;
    document.getElementById('count-filter-human').textContent = summary.human_paragraphs;

    renderDocParagraphsList(data.paragraphs);
}

function renderDocParagraphsList(paragraphs) {
    const listContainer = document.getElementById('doc-paragraphs-list');
    listContainer.innerHTML = '';

    paragraphs.forEach(p => {
        const isAI = p.label === 'ai';
        if (currentDocFilter === 'ai' && !isAI) return;
        if (currentDocFilter === 'human' && isAI) return;

        const card = document.createElement('div');
        const borderClass = isAI ? 'border-rose-500/30 bg-rose-950/10' : 'border-emerald-500/30 bg-emerald-950/10';
        const badgeClass = isAI ? 'bg-rose-500/20 text-rose-400 border-rose-500/30' : 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
        const scaleVal = p.human_ai_scale ? p.human_ai_scale.toFixed(1) : '50.0';

        card.className = `p-4 rounded-xl border ${borderClass} space-y-2 transition`;
        card.innerHTML = `
            <div class="flex items-center justify-between text-xs">
                <div class="flex items-center gap-2">
                    <span class="font-mono text-slate-400">#${p.index}</span>
                    <span class="px-2 py-0.5 rounded text-[11px] font-bold border ${badgeClass}">
                        ${isAI ? 'AI-Generated (AI)' : 'Human-Written (Human)'}
                    </span>
                </div>
                <div class="text-slate-400 font-mono text-[11px]">
                    AI Probability: <span class="font-semibold text-slate-200">${scaleVal}%</span>
                </div>
            </div>
            <p class="text-xs text-slate-200 leading-relaxed font-normal">${escapeHtml(p.text)}</p>
        `;
        listContainer.appendChild(card);
    });

    if (listContainer.children.length === 0) {
        listContainer.innerHTML = `<div class="text-center py-6 text-xs text-slate-500">No sections match the selected filter.</div>`;
    }
}

function filterDocParagraphs(filter) {
    currentDocFilter = filter;
    const btnAll = document.getElementById('doc-filter-all');
    const btnAi = document.getElementById('doc-filter-ai');
    const btnHuman = document.getElementById('doc-filter-human');

    const activeClass = "text-xs px-3 py-1.5 rounded-lg bg-blue-600 text-white font-medium transition";
    const inactiveClass = "text-xs px-3 py-1.5 rounded-lg bg-slate-800 text-slate-300 hover:text-white transition";

    btnAll.className = filter === 'all' ? activeClass : inactiveClass;
    btnAi.className = filter === 'ai' ? activeClass : inactiveClass;
    btnHuman.className = filter === 'human' ? activeClass : inactiveClass;

    if (lastDocResult && lastDocResult.paragraphs) {
        renderDocParagraphsList(lastDocResult.paragraphs);
    }
}

function copyDocTextToEditor() {
    if (!lastDocResult || !lastDocResult.full_text) return;
    const textarea = document.getElementById('text-input');
    textarea.value = lastDocResult.full_text;
    textarea.dispatchEvent(new Event('input'));
    switchTab('text');
}

// ----------------------------------------------------
// BATCH CSV / TSV ANALYSIS LOGIC
// ----------------------------------------------------

function handleFileSelected(event) {
    const file = event.target ? event.target.files[0] : event;
    if (file) {
        handleSelectedFile(file);
    }
}

function handleSelectedFile(file) {
    selectedBatchFile = file;
    const dropzoneText = document.getElementById('dropzone-text');
    const dropzoneSubtext = document.getElementById('dropzone-subtext');
    const btn = document.getElementById('btn-run-batch');
    const btnLabel = document.getElementById('btn-run-batch-label');

    const sizeKb = (file.size / 1024).toFixed(1);
    dropzoneText.textContent = `Selected file: ${file.name} (${sizeKb} KB)`;
    dropzoneSubtext.textContent = 'Click "Run Batch Scoring" below to begin inference';

    btn.disabled = false;
    btnLabel.textContent = `Run Batch Scoring (${file.name})`;
}

async function submitBatchAnalysis() {
    if (!selectedBatchFile) {
        alert('Please select a CSV or TSV file');
        return;
    }

    const textColumn = document.getElementById('batch-text-column').value.trim();
    const mode = document.getElementById('batch-mode-select').value;
    const batchSize = document.getElementById('batch-size-input').value || 32;

    const btn = document.getElementById('btn-run-batch');
    const btnLabel = document.getElementById('btn-run-batch-label');
    const resultsContainer = document.getElementById('batch-results-container');

    btn.disabled = true;
    btnLabel.textContent = 'Processing dataset batches on model... Please wait';

    const formData = new FormData();
    formData.append('file', selectedBatchFile);
    if (textColumn) formData.append('text_column', textColumn);
    formData.append('mode', mode);
    formData.append('batch_size', batchSize);

    try {
        const res = await fetch('/api/score-file', {
            method: 'POST',
            body: formData
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Dataset processing error');
        }

        const data = await res.json();
        lastBatchResult = data;
        renderBatchResult(data);
        resultsContainer.classList.remove('hidden');
    } catch (e) {
        alert('File processing error: ' + e.message);
    } finally {
        btn.disabled = false;
        btnLabel.textContent = `Run Batch Scoring (${selectedBatchFile.name})`;
    }
}

function renderBatchResult(data) {
    const summary = data.summary;
    document.getElementById('batch-total-rows').textContent = summary.total_rows.toLocaleString();
    document.getElementById('batch-ai-count').textContent = summary.ai_count.toLocaleString();
    document.getElementById('batch-ai-pct').textContent = `(${summary.ai_percentage}%)`;
    document.getElementById('batch-human-count').textContent = summary.human_count.toLocaleString();
    document.getElementById('batch-human-pct').textContent = `(${(100 - summary.ai_percentage).toFixed(2)}%)`;

    const thead = document.getElementById('batch-table-head');
    const tbody = document.getElementById('batch-table-body');
    thead.innerHTML = '';
    tbody.innerHTML = '';

    const trHead = document.createElement('tr');
    data.preview_columns.forEach(col => {
        const th = document.createElement('th');
        th.className = "p-2.5 whitespace-nowrap";
        if (col.startsWith('ai_')) {
            th.className += " text-blue-400 bg-blue-950/30";
        }
        th.textContent = col;
        trHead.appendChild(th);
    });
    thead.appendChild(trHead);

    data.preview_rows.forEach(row => {
        const tr = document.createElement('tr');
        data.preview_columns.forEach(col => {
            const td = document.createElement('td');
            td.className = "p-2.5 max-w-xs truncate border-slate-800";
            const val = row[col];
            
            if (col === 'ai_detector_label') {
                if (val === 'ai') {
                    td.innerHTML = `<span class="px-2 py-0.5 rounded bg-rose-500/20 text-rose-400 font-bold">AI</span>`;
                } else {
                    td.innerHTML = `<span class="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 font-bold">Human</span>`;
                }
            } else if (col === 'ai_probability_pct' || col === 'ai_score') {
                td.textContent = typeof val === 'number' ? (col === 'ai_probability_pct' ? `${val.toFixed(1)}%` : val.toFixed(3)) : val;
                td.className += " font-semibold text-slate-200";
            } else {
                td.textContent = val !== null && val !== undefined ? String(val) : '';
            }
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
}

function downloadScoredCSV() {
    if (!lastBatchResult || !lastBatchResult.csv_content) {
        alert('No scored dataset available to download');
        return;
    }
    const blob = new Blob([lastBatchResult.csv_content], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = lastBatchResult.filename || 'scored_dataset.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function escapeHtml(str) {
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
