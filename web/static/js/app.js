/* DeepVision Forensics - Frontend Logic */

document.addEventListener('DOMContentLoaded', () => {
    // State
    let selectedFile = null;
    let isLoading = false;
    let currentVisuals = {};

    // Elements
    const dropzone = document.getElementById('dropzone');
    const dropzoneLabel = document.getElementById('dropzoneLabel');
    const fileInput = document.getElementById('fileInput');
    const btnAnalyze = document.getElementById('btnAnalyze');
    const btnSample = document.getElementById('btnSample');
    const loader = document.getElementById('loader');
    const resultsSection = document.getElementById('results');

    // Image elements
    const origImg = document.getElementById('origImg');
    const elaImg = document.getElementById('elaImg');
    const heatmapImg = document.getElementById('heatmapImg');
    const maskImg = document.getElementById('maskImg');

    // UI helpers
    const verdictBadge = document.getElementById('verdictBadge');
    const attributedMethod = document.getElementById('attributedMethod');
    const tamperProb = document.getElementById('tamperProb');
    const methodConf = document.getElementById('methodConf');

    // ----- Helper Functions -----
    function showLoading(flag) {
        isLoading = flag;
        loader.style.display = flag ? 'block' : 'none';
        resultsSection.style.display = flag ? 'none' : 'block';
        btnAnalyze.disabled = flag || !selectedFile;
    }

    // ----- File handling -----
    dropzone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', e => {
        if (e.target.files && e.target.files[0]) {
            handleFileSelect(e.target.files[0]);
        }
    });

    ['dragenter', 'dragover'].forEach(ev => {
        dropzone.addEventListener(ev, e => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(ev => {
        dropzone.addEventListener(ev, e => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
        });
    });

    dropzone.addEventListener('drop', e => {
        const dt = e.dataTransfer;
        if (dt.files && dt.files[0]) {
            handleFileSelect(dt.files[0]);
        }
    });

    function handleFileSelect(file) {
        selectedFile = file;
        btnAnalyze.disabled = false;
        dropzoneLabel.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        // Show file info preview
        const fileInfo = document.getElementById('fileInfo');
        fileInfo.style.display = 'block';
        document.getElementById('fileName').textContent = `File: ${file.name}`;
        document.getElementById('fileSize').textContent = `Size: ${(file.size / 1024).toFixed(1)} KB`;
        document.getElementById('previewImg').src = URL.createObjectURL(file);
        // Clear previous visuals
        origImg.src = '';
        elaImg.src = '';
        heatmapImg.src = '';
        maskImg.src = '';
        // Hide results until analysis runs
        resultsSection.style.display = 'none';
    }

    // ----- Tab handling -----
    const tabButtons = document.querySelectorAll('.tab-btn');
    const visualsTab = document.getElementById('visualsTab');
    const evidenceTab = document.getElementById('evidenceTab');
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            tabButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            if (btn.dataset.tab === 'visuals') {
                visualsTab.style.display = 'block';
                evidenceTab.style.display = 'none';
            } else {
                visualsTab.style.display = 'none';
                evidenceTab.style.display = 'block';
            }
        });
    });

    // ----- API calls -----
    btnAnalyze.addEventListener('click', async () => {
        if (!selectedFile) return;
        const formData = new FormData();
        formData.append('image', selectedFile);
        showLoading(true);
        try {
            const resp = await fetch('/api/analyze', { method: 'POST', body: formData });
            const data = await resp.json();
            if (data.success) {
                renderResults(data);
            } else {
                alert('Analysis failed: ' + (data.error || 'Unknown error'));
            }
        } catch (err) {
            console.error(err);
            alert('Connection error during analysis.');
        } finally {
            showLoading(false);
        }
    });

    btnSample.addEventListener('click', async () => {
        showLoading(true);
        try {
            const resp = await fetch('/api/sample');
            const data = await resp.json();
            if (data.success) {
                renderResults(data);
                dropzoneLabel.textContent = `Loaded Sample: ${data.sample_name}`;
            } else {
                alert('Sample load failed: ' + (data.error || ''));
            }
        } catch (err) {
            console.error(err);
            alert('Error loading sample image.');
        } finally {
            showLoading(false);
        }
    });

    // ----- Render results -----
    function renderResults(data) {
        // Hide file preview
        const fileInfo = document.getElementById('fileInfo');
        if (fileInfo) fileInfo.style.display = 'none';
        currentVisuals = data.visuals;
        const isTampered = data.is_tampered;
        verdictBadge.textContent = isTampered ? 'TAMPERED' : 'AUTHENTIC';
        verdictBadge.className = 'verdict-tag ' + (isTampered ? 'tampered' : 'authentic');
        attributedMethod.textContent = data.predicted_type + (isTampered ? ' Forgery' : ' Image');
        tamperProb.textContent = (data.tamper_probability * 100).toFixed(1) + '%';
        methodConf.textContent = (data.type_confidence * 100).toFixed(1) + '%';
        // Populate visual tab
        origImg.src = data.visuals.original;
        elaImg.src = data.visuals.ela;
        heatmapImg.src = data.visuals.overlay;
        maskImg.src = data.visuals.mask;
        // Populate forensic evidence mini‑cards and quantitative metrics
        const evidenceBox = document.getElementById('evidenceBox');
        evidenceBox.innerHTML = '';
        // Textual evidence summary
        if (Array.isArray(data.forensic_dossier?.forensic_evidence_summary)) {
            data.forensic_dossier.forensic_evidence_summary.forEach(point => {
                const card = document.createElement('div');
                card.className = 'evidence-card';
                card.textContent = point;
                evidenceBox.appendChild(card);
            });
        }
        // Quantitative forensic metrics (ELA, JPEG compression, PRNU, etc.)
        const metrics = [];
        const d = data.forensic_dossier || {};
        if (d.ela_metrics) {
            metrics.push(`ELA DCT Grid Error: ${d.ela_metrics.dct_grid_inconsistency?.toFixed(2)}`);
        }
        if (d.sensor_noise_metrics) {
            metrics.push(`PRNU Noise Std Dev: ${d.sensor_noise_metrics.noise_residual_std?.toFixed(2)}`);
        }
        if (d.copy_move_metrics) {
            metrics.push(`Copy‑Move Matches: ${d.copy_move_metrics.copy_move_duplicate_pairs}`);
        }
        if (d.boundary_metrics) {
            metrics.push(`Boundary Gradient: ${d.boundary_metrics.boundary_gradient_discontinuity?.toFixed(2)}x`);
        }
        if (d.spectral_metrics) {
            metrics.push(`FFT Spectral Peak: ${d.spectral_metrics.fft_peak_to_average_ratio?.toFixed(2)}`);
        }
        metrics.forEach(txt => {
            const card = document.createElement('div');
            card.className = 'evidence-card metric-card';
            card.textContent = txt;
            evidenceBox.appendChild(card);
        });
        // Show results and activate visual tab
        resultsSection.style.display = 'block';
        const visualBtn = document.querySelector('.tab-btn[data-tab="visuals"]');
        if (visualBtn) visualBtn.classList.add('active');
        visualsTab.style.display = 'block';
        evidenceTab.style.display = 'none';
    }
});
