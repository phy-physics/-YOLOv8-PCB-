/**
 * PCB缺陷检测智能系统 - 前端脚本
 * 2026广东省大学生计算机设计大赛
 */

const API = '';
const CLASS_NAMES = ['mouse_bite', 'spur', 'missing_hole', 'short', 'open_circuit', 'spurious_copper'];
const CLASS_LABELS = ['鼠咬痕', '毛刺', '缺孔', '短路', '开路', '残铜'];
const CLASS_COLORS = ['#0d904f', '#1a73e8', '#e37400', '#c5221f', '#9c27b0', '#00bcd4'];

let currentPage = 'dashboard';
let trainingInterval = null;
let systemInterval = null;
let metricsChart = null;
let classChart = null;
let lossChart = null;

// ============ 初始化 ============
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    switchPage('dashboard');
    startSystemMonitor();
});

// ============ 导航 ============
function initNavigation() {
    document.querySelectorAll('.sidebar-nav a').forEach(link => {
        link.addEventListener('click', e => {
            e.preventDefault();
            const page = link.dataset.page;
            if (page) switchPage(page);
        });
    });
}

function switchPage(page) {
    currentPage = page;
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    const target = document.getElementById('page-' + page);
    if (target) target.classList.add('active');

    document.querySelectorAll('.sidebar-nav a').forEach(a => {
        a.classList.toggle('active', a.dataset.page === page);
    });

    // 加载页面数据
    switch (page) {
        case 'dashboard': loadDashboard(); break;
        case 'dataset': loadDataset(); break;
        case 'training': loadTrainingStatus(); refreshPretrainedModels(); break;
        case 'detection': loadDetectionModels(); break;
        case 'diagnosis': loadAlerts(); break;
        case 'models': loadModels(); loadPretrainedModelsList(); loadInnovations(); loadActiveModel(); break;
        case 'monitor': loadSystemStatus(); break;
    }
}

// ============ 仪表板 ============
async function loadDashboard() {
    try {
        // 使用 allSettled 而不是 all，这样一个失败不会导致全部失败
        const results = await Promise.allSettled([
            fetch(API + '/api/dataset/stats').then(r => r.json()),
            fetch(API + '/api/system/status').then(r => r.json()),
            fetch(API + '/api/alerts?limit=5').then(r => r.json()),
        ]);

        // 处理数据集统计
        if (results[0].status === 'fulfilled' && results[0].value.success) {
            const stats = results[0].value;
            const d = stats.data;
            const totalImages = Object.values(d.total_images).reduce((a, b) => a + b, 0);
            document.getElementById('dash-total-images').textContent = totalImages.toLocaleString();
            document.getElementById('dash-total-annotations').textContent = d.total_annotations.toLocaleString();
            document.getElementById('dash-total-classes').textContent = '6';
            renderClassDistribution(d.classes);
        }

        // 处理系统状态
        if (results[1].status === 'fulfilled' && results[1].value.success) {
            const sys = results[1].value;
            document.getElementById('dash-gpu').textContent = sys.data.gpu ? sys.data.gpu.name : 'CPU模式';
            updateSystemBrief(sys.data);
        }

        // 处理告警
        if (results[2].status === 'fulfilled' && results[2].value.success) {
            const alerts = results[2].value;
            renderDashAlerts(alerts.data);
        }

        loadTrainingStatus();
    } catch (e) {
        console.error('Dashboard load error:', e);
    }
}

function renderClassDistribution(classes) {
    const container = document.getElementById('class-distribution');
    if (!container) return;

    const total = Object.values(classes).reduce((a, b) => a + b, 0);
    let html = '';
    CLASS_NAMES.forEach((name, i) => {
        const count = classes[name] || 0;
        const pct = total ? (count / total * 100).toFixed(1) : 0;
        html += `
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
            <div style="width:12px;height:12px;border-radius:3px;background:${CLASS_COLORS[i]}"></div>
            <span style="width:60px;font-size:13px">${CLASS_LABELS[i]}</span>
            <div style="flex:1;background:#e8eaed;height:20px;border-radius:4px;overflow:hidden">
                <div style="width:${pct}%;height:100%;background:${CLASS_COLORS[i]};border-radius:4px;transition:width 0.5s"></div>
            </div>
            <span style="font-size:13px;color:#5f6368;min-width:80px;text-align:right">${count} (${pct}%)</span>
        </div>`;
    });
    container.innerHTML = html;
}

function renderDashAlerts(alerts) {
    const container = document.getElementById('dash-alerts');
    if (!container) return;
    if (!alerts.length) {
        container.innerHTML = '<div class="empty-state"><p>暂无告警</p></div>';
        return;
    }
    container.innerHTML = alerts.map(a => {
        const levelClass = a.level === 'critical' ? 'danger' : a.level;
        const icon = a.level === 'critical' ? '🔴' : a.level === 'warning' ? '🟡' : 'ℹ️';
        const time = new Date(a.time).toLocaleString('zh-CN');
        return `<div class="alert alert-${levelClass}">${icon} <div><div style="font-weight:500">${a.message}</div><div style="font-size:12px;color:#5f6368;margin-top:2px">${time}</div></div></div>`;
    }).join('');
}

function updateSystemBrief(data) {
    const el = document.getElementById('dash-system-brief');
    if (!el) return;
    el.innerHTML = `
        <div style="display:flex;gap:20px;font-size:13px">
            <span>CPU: <strong>${data.cpu.percent}%</strong></span>
            <span>内存: <strong>${data.memory.percent}%</strong></span>
            <span>磁盘: <strong>${data.disk.percent}%</strong></span>
            ${data.gpu ? `<span>GPU: <strong>${data.gpu.name}</strong></span>` : ''}
        </div>`;
}

// ============ 数据集 ============
async function loadDataset() {
    try {
        const res = await fetch(API + '/api/dataset/stats');
        const data = await res.json();
        if (!data.success) return;

        const stats = data.data;
        document.getElementById('ds-train-count').textContent = stats.total_images.train || 0;
        document.getElementById('ds-val-count').textContent = stats.total_images.val || 0;
        document.getElementById('ds-test-count').textContent = stats.total_images.test || 0;

        // 类别统计表格
        const tbody = document.getElementById('ds-class-table');
        if (tbody) {
            const total = Object.values(stats.classes).reduce((a, b) => a + b, 0);
            tbody.innerHTML = CLASS_NAMES.map((name, i) => {
                const count = stats.classes[name] || 0;
                const pct = total ? (count / total * 100).toFixed(1) : 0;
                return `<tr>
                    <td><span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:${CLASS_COLORS[i]};vertical-align:middle;margin-right:6px"></span>${CLASS_LABELS[i]}</td>
                    <td>${name}</td>
                    <td>${count}</td>
                    <td>${pct}%</td>
                </tr>`;
            }).join('');
        }
    } catch (e) {
        console.error(e);
    }
}

async function loadDatasetSamples(defectType = 'all') {
    const container = document.getElementById('ds-samples');
    if (!container) return;
    container.innerHTML = '<div style="text-align:center;padding:20px"><div class="spinner" style="margin:0 auto"></div></div>';

    try {
        const res = await fetch(API + `/api/dataset/samples?split=train&count=12&defect_type=${defectType}`);
        const data = await res.json();
        if (!data.success || !data.data.length) {
            container.innerHTML = '<div class="empty-state"><p>暂无样本数据</p></div>';
            return;
        }
        container.innerHTML = '<div class="image-grid">' +
            data.data.map(s => `<img src="data:image/jpeg;base64,${s.image}" alt="${s.filename}" title="${s.filename}" onclick="showImageModal(this.src, '${s.filename}')">`).join('') +
            '</div>';
    } catch (e) {
        container.innerHTML = '<div class="empty-state"><p>加载失败</p></div>';
    }
}

// ============ 模型训练 ============
async function startTraining() {
    const trainMode = document.getElementById('train-mode-select')?.value || 'retrain';
    let usePretrainedPath = null;

    if (trainMode === 'pretrained') {
        // 仅加载预训练模型，不训练
        const select = document.getElementById('pretrained-model-select');
        const modelPath = select?.value;
        if (!modelPath) {
            showToast('请先选择一个预训练模型或切换到训练模式', 'warning');
            return;
        }
        try {
            const res = await fetch(API + '/api/pretrained/load', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_path: modelPath }),
            });
            const data = await res.json();
            showToast(data.success ? '预训练模型已加载，可直接用于检测' : data.message, data.success ? 'success' : 'error');
        } catch (e) {
            showToast('加载失败: ' + e.message, 'error');
        }
        return;
    }

    if (trainMode === 'finetune') {
        const select = document.getElementById('pretrained-model-select');
        usePretrainedPath = select?.value || null;
        if (!usePretrainedPath) {
            showToast('微调模式需要选择一个预训练模型', 'warning');
            return;
        }
    }

    const params = {
        model_size: document.getElementById('train-model-size').value,
        epochs: parseInt(document.getElementById('train-epochs').value),
        batch_size: parseInt(document.getElementById('train-batch').value),
        img_size: parseInt(document.getElementById('train-imgsize').value),
        lr0: parseFloat(document.getElementById('train-lr').value),
        optimizer: document.getElementById('train-optimizer').value,
        patience: parseInt(document.getElementById('train-patience').value),
        augment: document.getElementById('train-augment').checked,
        cos_lr: document.getElementById('train-cos-lr')?.checked ?? true,
        multi_scale: document.getElementById('train-multi-scale')?.checked ?? false,
        label_smoothing: parseFloat(document.getElementById('train-label-smoothing')?.value || '0.05'),
        use_pretrained: usePretrainedPath,
    };

    try {
        const res = await fetch(API + '/api/train/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params),
        });
        const data = await res.json();
        if (data.success) {
            showToast('训练已启动', 'success');
            startTrainingMonitor();
        } else {
            showToast(data.message, 'error');
        }
    } catch (e) {
        showToast('启动训练失败: ' + e.message, 'error');
    }
}

async function stopTraining() {
    try {
        const res = await fetch(API + '/api/train/stop', { method: 'POST' });
        const data = await res.json();
        showToast(data.message, data.success ? 'warning' : 'error');
        if (data.success) stopTrainingMonitor();
    } catch (e) {
        showToast('停止失败', 'error');
    }
}

function startTrainingMonitor() {
    if (trainingInterval) clearInterval(trainingInterval);
    trainingInterval = setInterval(loadTrainingStatus, 3000);
}

function stopTrainingMonitor() {
    if (trainingInterval) {
        clearInterval(trainingInterval);
        trainingInterval = null;
    }
}

async function loadTrainingStatus() {
    try {
        const res = await fetch(API + '/api/train/status');
        const data = await res.json();
        if (!data.success) return;

        const state = data.data;
        const statusEl = document.getElementById('train-status-text');
        const progressEl = document.getElementById('train-progress-fill');
        const epochEl = document.getElementById('train-epoch-text');
        const btnStart = document.getElementById('btn-start-train');
        const btnStop = document.getElementById('btn-stop-train');

        if (statusEl) {
            const statusMap = { idle: '空闲', training: '训练中', completed: '已完成', error: '出错', stopped: '已停止' };
            statusEl.textContent = statusMap[state.status] || state.status;
            statusEl.className = 'badge badge-' + (state.status === 'training' ? 'warning' : state.status === 'completed' ? 'success' : state.status === 'error' ? 'danger' : 'info');
        }
        if (progressEl) progressEl.style.width = state.progress + '%';
        if (epochEl) epochEl.textContent = `${state.current_epoch} / ${state.total_epochs}`;
        if (btnStart) btnStart.disabled = state.is_training;
        if (btnStop) btnStop.disabled = !state.is_training;

        // 更新导航栏状态
        const navStatus = document.getElementById('nav-train-status');
        if (navStatus) {
            navStatus.className = 'status-dot ' + (state.is_training ? 'training' : 'online');
        }

        // 如果训练中，加载指标图表
        if (state.status === 'training' || state.status === 'completed') {
            loadTrainingMetrics();
            if (!trainingInterval && state.is_training) startTrainingMonitor();
            if (!state.is_training) stopTrainingMonitor();
        }

        // 显示最新指标
        if (state.latest_metrics) {
            renderLatestMetrics(state.latest_metrics);
        }
    } catch (e) {
        console.error('Training status error:', e);
    }
}

function renderLatestMetrics(m) {
    const el = document.getElementById('train-latest-metrics');
    if (!el) return;
    const keys = Object.keys(m).filter(k => k.includes('map') || k.includes('loss') || k.includes('precision') || k.includes('recall'));
    el.innerHTML = '<div class="grid grid-4">' + keys.slice(0, 8).map(k => {
        const v = typeof m[k] === 'number' ? m[k].toFixed(4) : m[k];
        return `<div class="stat-card"><div class="stat-info"><h4>${v}</h4><p>${k}</p></div></div>`;
    }).join('') + '</div>';
}

async function loadTrainingMetrics() {
    try {
        const res = await fetch(API + '/api/train/metrics');
        const data = await res.json();
        if (!data.success) return;

        renderTrainingCharts(data.data);
    } catch (e) {
        console.error(e);
    }
}

function renderTrainingCharts(metrics) {
    if (!metrics || !metrics.length) return;

    const epochs = metrics.map((_, i) => i + 1);

    // 损失图表
    const lossCanvas = document.getElementById('chart-loss');
    if (lossCanvas) {
        const ctx = lossCanvas.getContext('2d');
        const lossKeys = Object.keys(metrics[0]).filter(k => k.toLowerCase().includes('loss'));

        if (lossChart) lossChart.destroy();

        const datasets = lossKeys.map((key, i) => ({
            label: key.replace(/\s+/g, ''),
            data: metrics.map(m => m[key]),
            borderColor: CLASS_COLORS[i % CLASS_COLORS.length],
            backgroundColor: 'transparent',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.3,
        }));

        lossChart = new Chart(ctx, {
            type: 'line',
            data: { labels: epochs, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: 'top', labels: { font: { size: 11 } } } },
                scales: {
                    x: { title: { display: true, text: 'Epoch' } },
                    y: { title: { display: true, text: 'Loss' } },
                },
            },
        });
    }

    // mAP图表
    const mapCanvas = document.getElementById('chart-map');
    if (mapCanvas) {
        const ctx = mapCanvas.getContext('2d');
        const mapKeys = Object.keys(metrics[0]).filter(k =>
            k.toLowerCase().includes('map') || k.toLowerCase().includes('precision') || k.toLowerCase().includes('recall')
        );

        if (metricsChart) metricsChart.destroy();

        const datasets = mapKeys.map((key, i) => ({
            label: key.replace(/\s+/g, ''),
            data: metrics.map(m => m[key]),
            borderColor: CLASS_COLORS[i % CLASS_COLORS.length],
            backgroundColor: 'transparent',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.3,
        }));

        metricsChart = new Chart(ctx, {
            type: 'line',
            data: { labels: epochs, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: 'top', labels: { font: { size: 11 } } } },
                scales: {
                    x: { title: { display: true, text: 'Epoch' } },
                    y: { title: { display: true, text: 'Value' }, min: 0, max: 1 },
                },
            },
        });
    }
}

// ============ 缺陷检测 ============
function initDetectionUpload() {
    const zone = document.getElementById('upload-zone');
    const input = document.getElementById('upload-input');
    if (!zone || !input) return;

    zone.addEventListener('click', () => input.click());
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', e => {
        e.preventDefault();
        zone.classList.remove('dragover');
        if (e.dataTransfer.files.length) handleDetectionUpload(e.dataTransfer.files[0]);
    });
    input.addEventListener('change', () => { if (input.files.length) handleDetectionUpload(input.files[0]); });
}

async function handleDetectionUpload(file) {
    const resultContainer = document.getElementById('detection-result');
    resultContainer.innerHTML = '<div style="text-align:center;padding:40px"><div class="spinner" style="margin:0 auto"></div><p style="margin-top:12px;color:#5f6368">正在检测中...</p></div>';

    const formData = new FormData();
    formData.append('image', file);
    formData.append('conf', document.getElementById('detect-conf')?.value || '0.25');
    formData.append('iou', document.getElementById('detect-iou')?.value || '0.45');
    const detectModel = document.getElementById('detect-model')?.value;
    if (detectModel) formData.append('model_path', detectModel);

    try {
        const res = await fetch(API + '/api/detect', { method: 'POST', body: formData });
        const data = await res.json();

        if (!data.success) {
            resultContainer.innerHTML = `<div class="alert alert-danger">⚠️ ${data.message}</div>`;
            return;
        }

        const r = data.data;
        let defectsHtml = '';
        if (r.detections.length) {
            defectsHtml = r.detections.map(d => {
                const idx = CLASS_NAMES.indexOf(d.class_name);
                const color = idx >= 0 ? CLASS_COLORS[idx] : '#999';
                const label = idx >= 0 ? CLASS_LABELS[idx] : d.class_name;
                return `<div class="defect-item">
                    <span><span class="defect-color" style="display:inline-block;background:${color}"></span> ${label}</span>
                    <span class="badge badge-${d.confidence > 0.7 ? 'danger' : 'warning'}">${(d.confidence * 100).toFixed(1)}%</span>
                </div>`;
            }).join('');
        } else {
            defectsHtml = '<div class="empty-state" style="padding:20px"><p>未检测到缺陷 ✓</p></div>';
        }

        resultContainer.innerHTML = `
            <div class="detection-result fade-in">
                <div>
                    <img src="data:image/jpeg;base64,${r.image}" class="detection-image" alt="检测结果">
                    <p style="text-align:center;font-size:13px;color:#5f6368;margin-top:8px">图像尺寸: ${r.image_size[0]}×${r.image_size[1]}</p>
                </div>
                <div>
                    <div class="card">
                        <div class="card-header"><h3>🔍 检测结果</h3><span class="badge badge-${r.count ? 'danger' : 'success'}">${r.count} 个缺陷</span></div>
                        <div class="card-body">${defectsHtml}</div>
                    </div>
                    ${r.count ? `<button class="btn btn-primary" onclick='runDiagnosis(${JSON.stringify(r.detections)})'>🔧 故障诊断分析</button>` : ''}
                </div>
            </div>`;

        if (r.count) showToast(`检测到 ${r.count} 个缺陷`, 'warning');
    } catch (e) {
        resultContainer.innerHTML = `<div class="alert alert-danger">⚠️ 检测失败: ${e.message}</div>`;
    }
}

// ============ 故障诊断 ============
async function runDiagnosis(detections) {
    const container = document.getElementById('diagnosis-result');
    if (!container) {
        switchPage('diagnosis');
        await new Promise(r => setTimeout(r, 200));
    }

    const resultEl = document.getElementById('diagnosis-result');
    if (!resultEl) return;
    resultEl.innerHTML = '<div style="text-align:center;padding:20px"><div class="spinner" style="margin:0 auto"></div></div>';

    try {
        const res = await fetch(API + '/api/diagnosis/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ detections }),
        });
        const data = await res.json();
        if (!data.success) return;

        const d = data.data;
        const riskColors = { critical: 'danger', warning: 'warning', normal: 'success' };
        const riskLabels = { critical: '严重', warning: '警告', normal: '正常' };

        let html = `
            <div class="card fade-in">
                <div class="card-header">
                    <h3>📊 诊断结果</h3>
                    <span class="badge badge-${riskColors[d.risk_level]}">${riskLabels[d.risk_level]}</span>
                </div>
                <div class="card-body">
                    <div class="grid grid-3" style="margin-bottom:20px">
                        <div class="stat-card"><div class="stat-icon red">⚠️</div><div class="stat-info"><h4>${d.total_defects}</h4><p>缺陷总数</p></div></div>
                        <div class="stat-card"><div class="stat-icon orange">📋</div><div class="stat-info"><h4>${d.issues.length}</h4><p>问题类型</p></div></div>
                        <div class="stat-card"><div class="stat-icon blue">🔧</div><div class="stat-info"><h4>${d.suggestions.length}</h4><p>修复建议</p></div></div>
                    </div>`;

        // 缺陷详情
        if (d.issues.length) {
            html += '<h4 style="margin-bottom:12px">📌 缺陷详情</h4>';
            d.issues.forEach(issue => {
                html += `<div class="diagnosis-card ${issue.severity}">
                    <h4>${issue.description} (${issue.defect}) × ${issue.count}</h4>
                    <p>严重等级: ${issue.severity === 'critical' ? '🔴 严重' : issue.severity === 'warning' ? '🟡 警告' : 'ℹ️ 提示'}</p>
                </div>`;
            });
        }

        // 修复建议
        if (d.suggestions.length) {
            html += '<h4 style="margin:20px 0 12px">🔧 修复建议</h4>';
            d.suggestions.forEach((s, i) => {
                html += `<div class="alert alert-info">💡 ${i + 1}. ${s}</div>`;
            });
        }

        // 根因分析
        if (d.root_causes && d.root_causes.length) {
            html += '<h4 style="margin:20px 0 12px">🔬 根因分析</h4>';
            d.root_causes.forEach(rc => {
                html += `<div class="alert alert-warning">⚙️ ${rc}</div>`;
            });
        }

        html += '</div></div>';
        resultEl.innerHTML = html;
    } catch (e) {
        resultEl.innerHTML = `<div class="alert alert-danger">诊断失败: ${e.message}</div>`;
    }
}

async function loadAlerts() {
    try {
        const res = await fetch(API + '/api/alerts?limit=20');
        const data = await res.json();
        const container = document.getElementById('alerts-list');
        if (!container || !data.success) return;

        if (!data.data.length) {
            container.innerHTML = '<div class="empty-state"><div class="icon">🔔</div><h4>暂无告警</h4><p>系统运行正常</p></div>';
            return;
        }

        container.innerHTML = data.data.map(a => {
            const levelClass = a.level === 'critical' ? 'danger' : a.level;
            const icon = a.level === 'critical' ? '🔴' : a.level === 'warning' ? '🟡' : a.level === 'error' ? '❌' : 'ℹ️';
            const time = new Date(a.time).toLocaleString('zh-CN');
            return `<div class="alert alert-${levelClass}">${icon} <div><strong>${a.message}</strong><div style="font-size:12px;color:#5f6368;margin-top:2px">${time} · ${a.type}</div></div></div>`;
        }).join('');
    } catch (e) {
        console.error(e);
    }
}

async function clearAlerts() {
    await fetch(API + '/api/alerts/clear', { method: 'POST' });
    loadAlerts();
    showToast('告警已清除', 'success');
}

// ============ 模型管理 ============
async function loadModels() {
    try {
        const res = await fetch(API + '/api/models');
        const data = await res.json();
        const container = document.getElementById('models-list');
        if (!container || !data.success) return;

        if (!data.data.length) {
            container.innerHTML = '<div class="empty-state"><div class="icon">📦</div><h4>暂无模型</h4><p>请先训练模型</p></div>';
            return;
        }

        container.innerHTML = `<table><thead><tr><th>模型名称</th><th>类型</th><th>大小</th><th>更新时间</th><th>操作</th></tr></thead><tbody>` +
            data.data.map(m => `<tr>
                <td>${m.name}</td>
                <td><span class="badge badge-${m.type === 'best' ? 'success' : m.type === 'exported' ? 'info' : 'warning'}">${m.type}</span></td>
                <td>${m.size_mb} MB</td>
                <td>${new Date(m.modified).toLocaleString('zh-CN')}</td>
                <td><button class="btn btn-sm btn-outline" onclick="exportModel('${m.path.replace(/\\/g, '\\\\')}')">导出</button></td>
            </tr>`).join('') + '</tbody></table>';
    } catch (e) {
        console.error(e);
    }
}

async function exportModel(modelPath) {
    showToast('正在导出ONNX模型...', 'info');
    try {
        const res = await fetch(API + '/api/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_path: modelPath, formats: ['onnx'] }),
        });
        const data = await res.json();
        if (data.success) {
            showToast('模型导出成功', 'success');
            loadModels();
        } else {
            showToast(data.message, 'error');
        }
    } catch (e) {
        showToast('导出失败', 'error');
    }
}

// ============ 系统监控 ============
function startSystemMonitor() {
    loadSystemStatus();
    if (systemInterval) clearInterval(systemInterval);
    systemInterval = setInterval(loadSystemStatus, 5000);
}

async function loadSystemStatus() {
    try {
        const res = await fetch(API + '/api/system/status');
        const data = await res.json();
        if (!data.success) return;

        const s = data.data;

        // 更新仪表盘上的系统简况
        updateSystemBrief(s);

        // 更新监控页面
        updateGauge('cpu-gauge', s.cpu.percent, `${s.cpu.percent}%`, `CPU (${s.cpu.cores} 核)`);
        updateGauge('mem-gauge', s.memory.percent, `${s.memory.percent}%`, `内存 ${s.memory.used}/${s.memory.total}GB`);
        updateGauge('disk-gauge', s.disk.percent, `${s.disk.percent}%`, `磁盘 ${s.disk.used}/${s.disk.total}GB`);

        const gpuEl = document.getElementById('gpu-info');
        if (gpuEl) {
            if (s.gpu) {
                gpuEl.innerHTML = `<div class="stat-card"><div class="stat-icon green">🎮</div><div class="stat-info"><h4>${s.gpu.name}</h4><p>显存: ${s.gpu.memory_used}/${s.gpu.memory_total} GB</p></div></div>`;
            } else {
                gpuEl.innerHTML = '<div class="stat-card"><div class="stat-icon blue">💻</div><div class="stat-info"><h4>CPU模式</h4><p>未检测到GPU</p></div></div>';
            }
        }
    } catch (e) {
        // 静默处理
    }
}

function updateGauge(id, percent, text, label) {
    const el = document.getElementById(id);
    if (!el) return;
    const color = percent > 80 ? '#c5221f' : percent > 60 ? '#e37400' : '#0d904f';
    el.innerHTML = `
        <div class="gauge-circle" style="background:conic-gradient(${color} ${percent * 3.6}deg, #e8eaed ${percent * 3.6}deg)">
            <span class="gauge-value" style="color:${color}">${text}</span>
        </div>
        <div class="gauge-label">${label}</div>`;
}

// ============ 训练历史 ============
async function loadTrainingHistory() {
    try {
        const res = await fetch(API + '/api/train/history');
        const data = await res.json();
        const container = document.getElementById('training-history');
        if (!container || !data.success) return;

        if (!data.data.length) {
            container.innerHTML = '<div class="empty-state"><p>暂无训练历史</p></div>';
            return;
        }

        container.innerHTML = `<table><thead><tr><th>训练名称</th><th>轮次</th><th>状态</th><th>时间</th></tr></thead><tbody>` +
            data.data.map(h => `<tr>
                <td>${h.name}</td>
                <td>${h.epochs || '-'}</td>
                <td><span class="badge badge-${h.has_best ? 'success' : 'info'}">${h.has_best ? '完成' : '进行中'}</span></td>
                <td>${new Date(h.modified).toLocaleString('zh-CN')}</td>
            </tr>`).join('') + '</tbody></table>';
    } catch (e) {
        console.error(e);
    }
}

// ============ 工具函数 ============
function showToast(msg, type = 'info') {
    const toast = document.createElement('div');
    const colors = { success: '#0d904f', error: '#c5221f', warning: '#e37400', info: '#1a73e8' };
    toast.style.cssText = `position:fixed;top:72px;right:24px;padding:12px 20px;background:${colors[type]||'#333'};color:white;border-radius:8px;font-size:14px;z-index:10000;box-shadow:0 4px 12px rgba(0,0,0,0.15);animation:fadeIn 0.3s ease;max-width:360px`;
    toast.textContent = msg;
    document.body.appendChild(toast);
    setTimeout(() => { toast.style.opacity = '0'; setTimeout(() => toast.remove(), 300); }, 3000);
}

function showImageModal(src, title) {
    const overlay = document.createElement('div');
    overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.8);display:flex;align-items:center;justify-content:center;z-index:10000;cursor:pointer';
    overlay.innerHTML = `<div style="max-width:90vw;max-height:90vh"><img src="${src}" style="max-width:100%;max-height:85vh;border-radius:8px"><p style="text-align:center;color:white;margin-top:8px;font-size:13px">${title || ''}</p></div>`;
    overlay.addEventListener('click', () => overlay.remove());
    document.body.appendChild(overlay);
}

// 生成诊断报告
async function generateReport() {
    try {
        const res = await fetch(API + '/api/diagnosis/report');
        const data = await res.json();
        if (!data.success) return;

        const report = data.data;
        const reportEl = document.getElementById('diagnosis-report');
        if (!reportEl) return;

        reportEl.innerHTML = `
            <div class="card fade-in">
                <div class="card-header"><h3>📑 系统诊断报告</h3><span style="font-size:12px;color:#5f6368">${new Date(report.generated_at).toLocaleString('zh-CN')}</span></div>
                <div class="card-body">
                    <h4>数据集状态</h4>
                    <p style="margin:8px 0">总标注数: ${report.dataset.total_annotations} | 训练集: ${report.dataset.total_images.train || 0} | 验证集: ${report.dataset.total_images.val || 0} | 测试集: ${report.dataset.total_images.test || 0}</p>
                    <h4 style="margin-top:16px">训练状态</h4>
                    <p style="margin:8px 0">状态: ${report.training.status} | 模型: YOLOv8${report.training.model_size || '-'}</p>
                    <h4 style="margin-top:16px">系统健康</h4>
                    <p style="margin:8px 0">CPU: ${report.system_health.cpu}% | 内存: ${report.system_health.memory}%</p>
                    <h4 style="margin-top:16px">最近告警 (${report.recent_alerts.length})</h4>
                    ${report.recent_alerts.length ? report.recent_alerts.map(a => `<div class="alert alert-${a.level === 'critical' ? 'danger' : a.level}" style="margin-top:8px">${a.message}</div>`).join('') : '<p style="color:#5f6368">无告警记录</p>'}
                </div>
            </div>`;
    } catch (e) {
        showToast('生成报告失败', 'error');
    }
}

// 初始化检测上传区域
setTimeout(initDetectionUpload, 500);

// ============ 预训练模型管理 ============
async function refreshPretrainedModels() {
    try {
        const [preRes, modelsRes] = await Promise.all([
            fetch(API + '/api/pretrained'),
            fetch(API + '/api/models'),
        ]);
        const preData = await preRes.json();
        const modelsData = await modelsRes.json();

        const select = document.getElementById('pretrained-model-select');
        if (!select) return;

        select.innerHTML = '';

        // 添加预训练模型
        const preModels = preData.success ? preData.data : [];
        if (preModels.length) {
            const group1 = document.createElement('optgroup');
            group1.label = '⭐ 预训练模型';
            preModels.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m.path;
                opt.textContent = `${m.name} (${m.size_mb}MB)`;
                group1.appendChild(opt);
            });
            select.appendChild(group1);
        }

        // 添加训练产出模型
        const trainModels = modelsData.success ? modelsData.data.filter(m => m.type === 'best') : [];
        if (trainModels.length) {
            const group2 = document.createElement('optgroup');
            group2.label = '📁 训练产出';
            trainModels.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m.path;
                opt.textContent = `${m.name} (${m.size_mb}MB)`;
                group2.appendChild(opt);
            });
            select.appendChild(group2);
        }

        if (!preModels.length && !trainModels.length) {
            select.innerHTML = '<option value="">暂无可用模型</option>';
        }

        // 更新当前模型信息
        const curRes = await fetch(API + '/api/pretrained/current');
        const curData = await curRes.json();
        const infoEl = document.getElementById('current-model-info');
        if (infoEl && curData.success && curData.data) {
            const d = curData.data;
            infoEl.innerHTML = `<div class="alert alert-success">🟢 当前激活模型: <strong>${d.name}</strong> (${d.size_mb}MB, 来源: ${d.source === 'pretrained' ? '预训练' : '训练产出'})</div>`;
        } else if (infoEl) {
            infoEl.innerHTML = '<div class="alert alert-warning">⚠️ 暂无激活模型，请训练或加载预训练模型</div>';
        }
    } catch (e) {
        console.error('Load pretrained models error:', e);
    }
}

function onTrainModeChange() {
    const mode = document.getElementById('train-mode-select')?.value;
    const preGroup = document.getElementById('pretrained-select-group');
    const configCard = document.getElementById('training-config-card');
    const infoEl = document.getElementById('pretrained-info');
    const btnStart = document.getElementById('btn-start-train');

    if (preGroup) preGroup.style.display = (mode === 'retrain') ? 'none' : '';
    if (configCard) configCard.style.display = (mode === 'pretrained') ? 'none' : '';

    if (infoEl) {
        const msgs = {
            pretrained: '💡 预训练模型已在PCB缺陷数据集上训练优化，可直接用于检测。选择模型后点击"加载模型"即可使用。',
            retrain: '🔧 将使用YOLOv8官方预训练权重从头训练PCB缺陷检测模型。训练完成后自动保存到预训练目录。',
            finetune: '🎯 基于已有预训练模型继续微调，收敛更快。适用于数据集更新或超参调优。',
        };
        infoEl.innerHTML = msgs[mode] || '';
    }

    if (btnStart) {
        btnStart.textContent = mode === 'pretrained' ? '📥 加载模型' : '▶ 开始训练';
    }
}

async function loadDetectionModels() {
    try {
        const [preRes, modelsRes] = await Promise.all([
            fetch(API + '/api/pretrained'),
            fetch(API + '/api/models'),
        ]);
        const preData = await preRes.json();
        const modelsData = await modelsRes.json();

        const select = document.getElementById('detect-model');
        if (!select) return;

        // 保留第一个"自动选择"项
        while (select.children.length > 1) select.removeChild(select.lastChild);

        const allModels = [
            ...(preData.success ? preData.data.map(m => ({ ...m, tag: '预训练' })) : []),
            ...(modelsData.success ? modelsData.data.filter(m => m.type === 'best').map(m => ({ ...m, tag: '训练' })) : []),
        ];

        allModels.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m.path;
            opt.textContent = `[${m.tag}] ${m.name} (${m.size_mb}MB)`;
            select.appendChild(opt);
        });
    } catch (e) {
        console.error(e);
    }
}

async function loadPretrainedModelsList() {
    try {
        const res = await fetch(API + '/api/pretrained');
        const data = await res.json();
        const container = document.getElementById('pretrained-models-list');
        if (!container || !data.success) return;

        if (!data.data.length) {
            container.innerHTML = '<div class="empty-state"><div class="icon">⭐</div><h4>暂无预训练模型</h4><p>在训练页面执行训练后自动保存</p></div>';
            return;
        }

        container.innerHTML = `<table><thead><tr><th>模型名称</th><th>大小</th><th>更新时间</th><th>操作</th></tr></thead><tbody>` +
            data.data.map(m => `<tr>
                <td>⭐ ${m.name}</td>
                <td>${m.size_mb} MB</td>
                <td>${new Date(m.modified).toLocaleString('zh-CN')}</td>
                <td><button class="btn btn-sm btn-success" onclick="loadPretrainedAsActive('${m.path.replace(/\\/g, '\\\\')}')">📥 激活</button>
                    <button class="btn btn-sm btn-outline" onclick="exportModel('${m.path.replace(/\\/g, '\\\\')}')">导出</button></td>
            </tr>`).join('') + '</tbody></table>';
    } catch (e) {
        console.error(e);
    }
}

async function loadPretrainedAsActive(modelPath) {
    try {
        const res = await fetch(API + '/api/pretrained/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_path: modelPath }),
        });
        const data = await res.json();
        showToast(data.success ? '模型已激活' : data.message, data.success ? 'success' : 'error');
        if (data.success) loadActiveModel();
    } catch (e) {
        showToast('加载失败', 'error');
    }
}

async function loadActiveModel() {
    try {
        const res = await fetch(API + '/api/pretrained/current');
        const data = await res.json();
        const container = document.getElementById('active-model-card');
        if (!container || !data.success) return;

        if (data.data) {
            const d = data.data;
            container.innerHTML = `
                <div class="stat-card" style="background:#e8f5e9">
                    <div class="stat-icon green">🟢</div>
                    <div class="stat-info">
                        <h4>${d.name}</h4>
                        <p>${d.size_mb} MB · 来源: ${d.source === 'pretrained' ? '预训练模型' : '训练产出'}</p>
                    </div>
                </div>`;
        } else {
            container.innerHTML = '<div class="empty-state"><div class="icon">📦</div><h4>暂无激活模型</h4><p>请先训练或加载预训练模型</p></div>';
        }
    } catch (e) {
        console.error(e);
    }
}

async function loadInnovations() {
    try {
        const res = await fetch(API + '/api/innovation');
        const data = await res.json();
        const container = document.getElementById('innovation-list');
        if (!container || !data.success) return;

        container.innerHTML = '<div class="grid grid-2">' +
            data.data.map(inn => `
                <div class="diagnosis-card info" style="border-left-color:${['#1a73e8','#0d904f','#e37400','#9c27b0','#00bcd4','#c5221f'][inn.id-1]}">
                    <h4>${inn.title}</h4>
                    <span class="badge badge-info" style="font-size:11px;margin-bottom:6px">${inn.category}</span>
                    <p style="font-size:13px;color:#5f6368">${inn.description}</p>
                </div>`).join('') +
            '</div>';
    } catch (e) {
        console.error(e);
    }
}
