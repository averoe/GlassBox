/**
 * GlassBox RAG - Dashboard JavaScript
 * Handles real-time trace visualization and metrics
 */

class TraceViewer {
    constructor() {
        this.apiBase = '/api';
        this.traces = [];
        this.currentTrace = null;
        this.initEventListeners();
        this.loadTraces();
    }

    initEventListeners() {
        // Auto-refresh traces every 5 seconds
        setInterval(() => this.loadTraces(), 5000);
    }

    async loadTraces() {
        try {
            const response = await fetch(`${this.apiBase}/traces?limit=10`);
            const data = await response.json();
            this.traces = data.traces || [];
            this.renderTraceList();
        } catch (error) {
            console.error('Failed to load traces:', error);
        }
    }

    renderTraceList() {
        const traceList = document.getElementById('traceList');
        if (!traceList) return;

        traceList.innerHTML = this.traces.map(trace => `
            <div class="list-item" onclick="viewer.selectTrace('${trace.trace_id}')">
                <div style="font-family: 'Courier New', monospace; font-size: 12px; color: var(--primary-color); font-weight: 500;">
                    ${trace.trace_id.substring(0, 16)}...
                </div>
                <div style="font-size: 12px; color: var(--text-light); margin-top: 5px;">
                    Duration: ${trace.duration_ms.toFixed(0)}ms
                </div>
            </div>
        `).join('');
    }

    async selectTrace(traceId) {
        try {
            const response = await fetch(`${this.apiBase}/traces/${traceId}`);
            this.currentTrace = await response.json();
            this.renderTraceDetail();
        } catch (error) {
            console.error('Failed to load trace:', error);
        }
    }

    async renderTraceDetail() {
        const detail = document.getElementById('traceDetail');
        if (!detail || !this.currentTrace) return;

        const traceTitle = document.getElementById('traceTitle');
        traceTitle.textContent = `Trace: ${this.currentTrace.trace_id}`;

        const timeline = this.renderTimeline(this.currentTrace.root_step);
        document.getElementById('traceTimeline').innerHTML = timeline;

        // Hide main view, show detail
        document.getElementById('mainView').style.display = 'none';
        detail.classList.add('active');
    }

    renderTimeline(step, level = 0) {
        if (!step) return '';

        const indent = level * 20;
        let html = `
            <div class="timeline-item" style="margin-left: ${indent}px;">
                <div class="timeline-step">
                    <div class="step-name">${step.name}</div>
                    <div class="step-duration">
                        ⏱️ ${step.duration_ms.toFixed(2)}ms
                        ${step.error ? `<span style="color: var(--error-color);"> ❌ ${step.error}</span>` : ''}
                    </div>
                </div>
            </div>
        `;

        if (step.children && step.children.length > 0) {
            for (const child of step.children) {
                html += this.renderTimeline(child, level + 1);
            }
        }

        return html;
    }

    backToMain() {
        document.getElementById('mainView').style.display = 'grid';
        document.getElementById('traceDetail').classList.remove('active');
        this.currentTrace = null;
    }
}

class MetricsView {
    constructor() {
        this.apiBase = '/api';
        this.loadMetrics();
    }

    async loadMetrics() {
        try {
            // Load metrics from server
            // This would be implemented based on your metrics API
            this.updateMetrics();
        } catch (error) {
            console.error('Failed to load metrics:', error);
        }
    }

    updateMetrics() {
        // Update metric cards
        const metrics = {
            'Total Requests': '1,234',
            'Avg Latency': '245ms',
            'Total Cost': '$12.45',
            'Success Rate': '99.8%'
        };

        const metricsContainer = document.querySelector('.metrics-grid');
        if (!metricsContainer) return;

        metricsContainer.innerHTML = Object.entries(metrics).map(([label, value]) => `
            <div class="metric">
                <div class="metric-label">${label}</div>
                <div class="metric-value">${value}</div>
            </div>
        `).join('');
    }
}

// Initialize views when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.viewer = new TraceViewer();
    window.metrics = new MetricsView();
});

// Export functions for inline calls
function selectTrace(traceId) {
    if (window.viewer) {
        window.viewer.selectTrace(traceId);
    }
}

function backToMain() {
    if (window.viewer) {
        window.viewer.backToMain();
    }
}
