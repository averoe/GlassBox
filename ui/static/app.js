/**
 * GlassBox RAG - Enhanced Dashboard JavaScript
 * Handles navigation, telemetry, pipeline visualization, and debugging
 */

class GlassBoxDashboard {
    constructor() {
        this.apiBase = window.location.origin;
        this.currentPage = 'overview';
        this.charts = {};
        this.pipelineData = null;
        this.traces = [];
        this.currentTrace = null;

        this.init();
    }

    init() {
        this.setupNavigation();
        this.setupEventListeners();
        this.loadInitialData();
        this.startAutoRefresh();
    }

    setupNavigation() {
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const page = e.target.dataset.page;
                this.switchPage(page);
            });
        });
    }

    setupEventListeners() {
        // Refresh button
        document.getElementById('refreshBtn')?.addEventListener('click', () => {
            this.refreshAll();
        });

        // Pipeline controls
        document.getElementById('runPipelineBtn')?.addEventListener('click', () => {
            this.runPipelineTest();
        });

        document.getElementById('resetPipelineBtn')?.addEventListener('click', () => {
            this.resetPipeline();
        });

        // Trace debugging
        document.getElementById('traceFilter')?.addEventListener('change', (e) => {
            this.filterTraces(e.target.value);
        });

        document.getElementById('exportTraceBtn')?.addEventListener('click', () => {
            this.exportTrace();
        });
    }

    switchPage(page) {
        // Update navigation
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-page="${page}"]`).classList.add('active');

        // Update page content
        document.querySelectorAll('.page').forEach(p => {
            p.classList.remove('active');
        });
        document.getElementById(page).classList.add('active');

        this.currentPage = page;

        // Load page-specific data
        switch(page) {
            case 'overview':
                this.loadOverviewData();
                break;
            case 'pipeline':
                this.loadPipelineData();
                break;
            case 'debugger':
                this.loadTraceData();
                break;
            case 'telemetry':
                this.loadTelemetryData();
                break;
        }
    }

    async loadInitialData() {
        await Promise.all([
            this.loadOverviewData(),
            this.loadHealthStatus()
        ]);
    }

    async loadOverviewData() {
        try {
            const [healthRes, metricsRes] = await Promise.all([
                fetch(`${this.apiBase}/health`),
                fetch(`${this.apiBase}/metrics/prometheus`)
            ]);

            const health = await healthRes.json();
            const metrics = await metricsRes.text();

            this.updateHealthStatus(health);
            this.updateMetrics(metrics);
            this.updateComponents(health.components);
            this.updateActivityFeed();
        } catch (error) {
            console.error('Failed to load overview data:', error);
        }
    }

    async loadHealthStatus() {
        try {
            const response = await fetch(`${this.apiBase}/health`);
            const health = await response.json();
            this.updateHealthStatus(health);
        } catch (error) {
            console.error('Failed to load health status:', error);
        }
    }

    updateHealthStatus(health) {
        const indicator = document.getElementById('healthStatus');
        const dot = indicator.querySelector('.status-dot');
        const text = indicator.querySelector('span');

        const isHealthy = health.status === 'healthy';
        dot.className = `status-dot ${isHealthy ? 'healthy' : 'unhealthy'}`;
        text.textContent = isHealthy ? 'Healthy' : 'Degraded';
    }

    updateMetrics(prometheusData) {
        // Parse Prometheus metrics and update dashboard
        const metrics = this.parsePrometheusMetrics(prometheusData);

        document.getElementById('totalRequests').textContent = metrics.totalRequests || '0';
        document.getElementById('avgLatency').textContent = `${(metrics.avgLatency || 0).toFixed(0)}ms`;
        document.getElementById('totalTokens').textContent = (metrics.totalTokens || 0).toLocaleString();
        document.getElementById('totalCost').textContent = `$${(metrics.totalCost || 0).toFixed(4)}`;

        // Update trends (mock data for now)
        this.updateTrends();
    }

    parsePrometheusMetrics(data) {
        // Simple Prometheus metrics parser
        const lines = data.split('\n');
        const metrics = {};

        lines.forEach(line => {
            if (line.startsWith('#') || !line.trim()) return;

            const [name, value] = line.split(' ');
            if (name && value) {
                metrics[name] = parseFloat(value) || 0;
            }
        });

        return metrics;
    }

    updateTrends() {
        // Mock trend updates
        const trends = ['requestsTrend', 'latencyTrend', 'tokensTrend', 'costTrend'];
        trends.forEach(trendId => {
            const element = document.getElementById(trendId);
            const isPositive = Math.random() > 0.5;
            const change = (Math.random() * 20).toFixed(1);
            element.textContent = `${isPositive ? '+' : '-'}${change}%`;
            element.className = `metric-trend ${isPositive ? 'positive' : 'negative'}`;
        });
    }

    updateComponents(components) {
        const container = document.getElementById('componentStatus');
        container.innerHTML = '';

        Object.entries(components).forEach(([name, status]) => {
            const item = document.createElement('div');
            item.className = 'component-item';

            const displayName = name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            const statusClass = status === 'operational' ? 'status-operational' : 'status-error';

            item.innerHTML = `
                <span class="component-name">${displayName}</span>
                <span class="component-status ${statusClass}">${status}</span>
            `;

            container.appendChild(item);
        });
    }

    updateActivityFeed() {
        const container = document.getElementById('activityFeed');
        const activities = [
            { icon: 'search', title: 'Query processed', time: '2 minutes ago' },
            { icon: 'upload', title: 'Documents ingested', time: '5 minutes ago' },
            { icon: 'zap', title: 'Model inference completed', time: '8 minutes ago' },
            { icon: 'database', title: 'Vector store updated', time: '12 minutes ago' }
        ];

        container.innerHTML = activities.map(activity => `
            <div class="activity-item">
                <div class="activity-icon">
                    <i data-lucide="${activity.icon}"></i>
                </div>
                <div class="activity-content">
                    <div class="activity-title">${activity.title}</div>
                    <div class="activity-time">${activity.time}</div>
                </div>
            </div>
        `).join('');

        // Re-initialize Lucide icons for new elements
        lucide.createIcons();
    }

    async loadPipelineData() {
        // Load pipeline configuration and render visualization
        this.renderPipeline();
    }

    renderPipeline() {
        const svg = document.getElementById('pipelineSvg');
        if (!svg) return;

        // Clear existing content
        svg.innerHTML = '';

        // Define pipeline steps
        const steps = [
            { id: 'input', label: 'Query Input', x: 100, y: 150, icon: 'message-square' },
            { id: 'encode', label: 'Encode Query', x: 300, y: 150, icon: 'cpu' },
            { id: 'retrieve', label: 'Retrieve Documents', x: 500, y: 150, icon: 'search' },
            { id: 'rerank', label: 'Rerank Results', x: 700, y: 150, icon: 'filter' },
            { id: 'generate', label: 'Generate Response', x: 900, y: 150, icon: 'zap' },
            { id: 'output', label: 'Final Output', x: 1100, y: 150, icon: 'send' }
        ];

        // Draw connections
        for (let i = 0; i < steps.length - 1; i++) {
            const step1 = steps[i];
            const step2 = steps[i + 1];

            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', step1.x + 40);
            line.setAttribute('y1', step1.y + 20);
            line.setAttribute('x2', step2.x - 40);
            line.setAttribute('y2', step2.y + 20);
            line.setAttribute('stroke', 'var(--border)');
            line.setAttribute('stroke-width', '2');
            line.setAttribute('marker-end', 'url(#arrowhead)');
            svg.appendChild(line);
        }

        // Draw steps
        steps.forEach(step => {
            const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            group.classList.add('pipeline-step');
            group.setAttribute('data-step', step.id);

            // Circle background
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', step.x);
            circle.setAttribute('cy', step.y);
            circle.setAttribute('r', '40');
            circle.setAttribute('fill', 'var(--bg-elevated)');
            circle.setAttribute('stroke', 'var(--border)');
            circle.setAttribute('stroke-width', '2');
            group.appendChild(circle);

            // Icon
            const icon = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            icon.setAttribute('x', step.x);
            icon.setAttribute('y', step.y - 5);
            icon.setAttribute('text-anchor', 'middle');
            icon.setAttribute('font-size', '20');
            icon.setAttribute('fill', 'var(--accent-indigo)');
            icon.textContent = step.icon; // Simplified - would need proper icon rendering
            group.appendChild(icon);

            // Label
            const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            label.setAttribute('x', step.x);
            label.setAttribute('y', step.y + 60);
            label.setAttribute('text-anchor', 'middle');
            label.setAttribute('font-size', '12');
            label.setAttribute('fill', 'var(--text-secondary)');
            label.textContent = step.label;
            group.appendChild(label);

            group.addEventListener('click', () => this.showStepDetails(step));
            svg.appendChild(group);
        });

        // Add arrow marker
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        const marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
        marker.setAttribute('id', 'arrowhead');
        marker.setAttribute('markerWidth', '10');
        marker.setAttribute('markerHeight', '7');
        marker.setAttribute('refX', '9');
        marker.setAttribute('refY', '3.5');
        marker.setAttribute('orient', 'auto');

        const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        polygon.setAttribute('points', '0 0, 10 3.5, 0 7');
        polygon.setAttribute('fill', 'var(--border)');
        marker.appendChild(polygon);
        defs.appendChild(marker);
        svg.appendChild(defs);
    }

    showStepDetails(step) {
        const details = document.getElementById('stepDetails');
        details.innerHTML = `
            <h4>${step.label}</h4>
            <p><strong>Status:</strong> <span class="status-operational">Operational</span></p>
            <p><strong>Avg Latency:</strong> 45ms</p>
            <p><strong>Success Rate:</strong> 99.8%</p>
            <p><strong>Description:</strong> ${this.getStepDescription(step.id)}</p>
        `;
    }

    getStepDescription(stepId) {
        const descriptions = {
            'input': 'Accepts and validates user queries',
            'encode': 'Converts text queries to vector embeddings',
            'retrieve': 'Searches vector store for relevant documents',
            'rerank': 'Reorders results by relevance using cross-encoder',
            'generate': 'Produces final response using retrieved context',
            'output': 'Formats and returns response to user'
        };
        return descriptions[stepId] || 'Pipeline processing step';
    }

    async runPipelineTest() {
        try {
            const response = await fetch(`${this.apiBase}/retrieve`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: 'What is GlassBox RAG?',
                    top_k: 3
                })
            });

            const result = await response.json();
            this.updatePipelineExecution(result);
        } catch (error) {
            console.error('Pipeline test failed:', error);
        }
    }

    updatePipelineExecution(result) {
        const timeline = document.getElementById('executionTimeline');
        timeline.innerHTML = `
            <div class="execution-step">
                <div class="step-name">Query Processing</div>
                <div class="step-duration">12ms</div>
            </div>
            <div class="execution-step">
                <div class="step-name">Encoding</div>
                <div class="step-duration">45ms</div>
            </div>
            <div class="execution-step">
                <div class="step-name">Retrieval</div>
                <div class="step-duration">89ms</div>
            </div>
            <div class="execution-step">
                <div class="step-name">Reranking</div>
                <div class="step-duration">23ms</div>
            </div>
            <div class="execution-step">
                <div class="step-name">Generation</div>
                <div class="step-duration">156ms</div>
            </div>
            <div class="execution-total">
                <strong>Total: ${result.execution_time_ms}ms</strong>
            </div>
        `;
    }

    resetPipeline() {
        document.getElementById('executionTimeline').innerHTML = '<p class="placeholder">Run a query to see execution timeline</p>';
        document.getElementById('stepDetails').innerHTML = '<p class="placeholder">Click on a pipeline step to view details</p>';
    }

    async loadTraceData() {
        try {
            // Mock trace data - in real implementation, fetch from API
            this.traces = [
                {
                    trace_id: 'abc123',
                    query: 'What is RAG?',
                    duration_ms: 234,
                    status: 'success',
                    timestamp: new Date().toISOString()
                },
                {
                    trace_id: 'def456',
                    query: 'How does vector search work?',
                    duration_ms: 189,
                    status: 'success',
                    timestamp: new Date(Date.now() - 300000).toISOString()
                }
            ];

            this.renderTraceList();
        } catch (error) {
            console.error('Failed to load trace data:', error);
        }
    }

    renderTraceList() {
        const container = document.getElementById('traceList');
        container.innerHTML = this.traces.map(trace => `
            <div class="trace-item" onclick="dashboard.selectTrace('${trace.trace_id}')">
                <div class="trace-meta">
                    <span class="trace-id">${trace.trace_id.substring(0, 8)}...</span>
                    <span class="trace-duration">${trace.duration_ms}ms</span>
                </div>
                <div class="trace-query">${trace.query}</div>
                <span class="trace-status status-${trace.status}">${trace.status}</span>
            </div>
        `).join('');
    }

    selectTrace(traceId) {
        const trace = this.traces.find(t => t.trace_id === traceId);
        if (!trace) return;

        this.currentTrace = trace;

        // Update UI
        document.querySelectorAll('.trace-item').forEach(item => {
            item.classList.remove('active');
        });
        event.currentTarget.classList.add('active');

        this.renderTraceVisualization(trace);
    }

    renderTraceVisualization(trace) {
        const container = document.getElementById('traceVisualization');
        container.innerHTML = `
            <div class="trace-header">
                <h4>Trace: ${trace.trace_id}</h4>
                <div class="trace-summary">
                    <span>Duration: ${trace.duration_ms}ms</span>
                    <span>Status: <span class="status-${trace.status}">${trace.status}</span></span>
                </div>
            </div>
            <div class="trace-steps">
                <div class="trace-step">
                    <div class="step-icon">📥</div>
                    <div class="step-content">
                        <div class="step-name">Input Processing</div>
                        <div class="step-details">Query: "${trace.query}"</div>
                        <div class="step-time">+12ms</div>
                    </div>
                </div>
                <div class="trace-step">
                    <div class="step-icon">🧠</div>
                    <div class="step-content">
                        <div class="step-name">Encoding</div>
                        <div class="step-details">Generated embeddings for query</div>
                        <div class="step-time">+45ms</div>
                    </div>
                </div>
                <div class="trace-step">
                    <div class="step-icon">🔍</div>
                    <div class="step-content">
                        <div class="step-name">Retrieval</div>
                        <div class="step-details">Found 5 relevant documents</div>
                        <div class="step-time">+89ms</div>
                    </div>
                </div>
                <div class="trace-step">
                    <div class="step-icon">⚡</div>
                    <div class="step-content">
                        <div class="step-name">Generation</div>
                        <div class="step-details">Generated response using context</div>
                        <div class="step-time">+156ms</div>
                    </div>
                </div>
            </div>
        `;
    }

    filterTraces(filterType) {
        let filtered = this.traces;

        switch(filterType) {
            case 'errors':
                filtered = this.traces.filter(t => t.status === 'error');
                break;
            case 'slow':
                filtered = this.traces.filter(t => t.duration_ms > 1000);
                break;
            default:
                filtered = this.traces;
        }

        // Re-render with filtered traces
        const container = document.getElementById('traceList');
        container.innerHTML = filtered.map(trace => `
            <div class="trace-item" onclick="dashboard.selectTrace('${trace.trace_id}')">
                <div class="trace-meta">
                    <span class="trace-id">${trace.trace_id.substring(0, 8)}...</span>
                    <span class="trace-duration">${trace.duration_ms}ms</span>
                </div>
                <div class="trace-query">${trace.query}</div>
                <span class="trace-status status-${trace.status}">${trace.status}</span>
            </div>
        `).join('');
    }

    exportTrace() {
        if (!this.currentTrace) return;

        const dataStr = JSON.stringify(this.currentTrace, null, 2);
        const dataBlob = new Blob([dataStr], {type: 'application/json'});
        const url = URL.createObjectURL(dataBlob);

        const link = document.createElement('a');
        link.href = url;
        link.download = `trace_${this.currentTrace.trace_id}.json`;
        link.click();

        URL.revokeObjectURL(url);
    }

    async loadTelemetryData() {
        // Initialize charts
        this.initCharts();

        // Load metrics data
        try {
            const response = await fetch(`${this.apiBase}/metrics/prometheus`);
            const data = await response.text();
            const metrics = this.parsePrometheusMetrics(data);

            this.updateCharts(metrics);
            this.updateCostBreakdown(metrics);
            this.updatePerformanceTable(metrics);
        } catch (error) {
            console.error('Failed to load telemetry data:', error);
        }
    }

    initCharts() {
        const latencyCtx = document.getElementById('latencyChart')?.getContext('2d');
        const tokenCtx = document.getElementById('tokenChart')?.getContext('2d');

        if (latencyCtx) {
            this.charts.latency = new Chart(latencyCtx, {
                type: 'line',
                data: {
                    labels: ['0s', '10s', '20s', '30s', '40s', '50s', '60s'],
                    datasets: [{
                        label: 'Latency (ms)',
                        data: [45, 52, 48, 61, 55, 49, 47],
                        borderColor: 'var(--accent-cyan)',
                        backgroundColor: 'rgba(22, 211, 238, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: { color: 'var(--border)' }
                        },
                        x: {
                            grid: { color: 'var(--border)' }
                        }
                    }
                }
            });
        }

        if (tokenCtx) {
            this.charts.tokens = new Chart(tokenCtx, {
                type: 'bar',
                data: {
                    labels: ['Encode', 'Retrieve', 'Rerank', 'Generate'],
                    datasets: [{
                        label: 'Tokens Used',
                        data: [512, 1024, 256, 2048],
                        backgroundColor: 'var(--accent-violet)',
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: { color: 'var(--border)' }
                        }
                    }
                }
            });
        }
    }

    updateCharts(metrics) {
        // Update chart data with real metrics
        if (this.charts.latency) {
            this.charts.latency.data.datasets[0].data = [
                metrics.latency_p50 || 45,
                metrics.latency_p75 || 52,
                metrics.latency_p90 || 48,
                metrics.latency_p95 || 61,
                metrics.latency_p99 || 55,
                49, 47
            ];
            this.charts.latency.update();
        }
    }

    updateCostBreakdown(metrics) {
        const container = document.getElementById('costBreakdown');
        const costs = [
            { label: 'OpenAI API', value: '$0.0234' },
            { label: 'Vector Storage', value: '$0.0012' },
            { label: 'Database', value: '$0.0008' },
            { label: 'Compute', value: '$0.0056' }
        ];

        container.innerHTML = costs.map(cost => `
            <div class="cost-item">
                <span class="cost-label">${cost.label}</span>
                <span class="cost-value">${cost.value}</span>
            </div>
        `).join('');
    }

    updatePerformanceTable(metrics) {
        const tbody = document.getElementById('performanceTableBody');
        const operations = [
            { name: 'encode', p50: 23, p95: 45, p99: 89, count: 1250 },
            { name: 'retrieve', p50: 67, p95: 123, p99: 234, count: 1250 },
            { name: 'rerank', p50: 12, p95: 34, p99: 67, count: 980 },
            { name: 'generate', p50: 156, p95: 289, p99: 445, count: 1250 }
        ];

        tbody.innerHTML = operations.map(op => `
            <tr>
                <td>${op.name}</td>
                <td>${op.p50}ms</td>
                <td>${op.p95}ms</td>
                <td>${op.p99}ms</td>
                <td>${op.count}</td>
            </tr>
        `).join('');
    }

    startAutoRefresh() {
        setInterval(() => {
            if (this.currentPage === 'overview') {
                this.loadOverviewData();
            }
        }, 30000); // Refresh every 30 seconds
    }

    refreshAll() {
        this.loadInitialData();
        if (this.currentPage !== 'overview') {
            this.switchPage(this.currentPage);
        }
    }
}

// Global functions for HTML onclick handlers
function showPluginDocs(type) {
    const modal = document.getElementById('pluginModal');
    const title = document.getElementById('modalTitle');
    const content = document.getElementById('modalContent');

    title.textContent = `${type.charAt(0).toUpperCase() + type.slice(1)} Plugin Development`;

    const docs = {
        encoder: `
            <h4>Creating a Custom Encoder</h4>
            <p>Inherit from <code>BaseEncoder</code> and implement the <code>encode</code> method:</p>
            <pre><code>from glassbox_rag.core.encoder import BaseEncoder

class MyEncoder(BaseEncoder):
    async def encode(self, texts: List[str]) -> np.ndarray:
        # Your encoding logic here
        embeddings = self.call_my_api(texts)
        return np.array(embeddings)</code></pre>

            <h4>Registration</h4>
            <p>Add to your config:</p>
            <pre><code>encoding:
  local:
    my_encoder:
      type: "my_encoder"
      api_key: "your-key"</code></pre>
        `,
        vectorstore: `
            <h4>Creating a Custom Vector Store</h4>
            <p>Inherit from <code>VectorStorePlugin</code>:</p>
            <pre><code>from glassbox_rag.plugins.base import VectorStorePlugin

class MyVectorStore(VectorStorePlugin):
    async def add_vectors(self, vectors: np.ndarray, metadata: List[dict]):
        # Store vectors in your system
        pass

    async def search(self, query_vector: np.ndarray, top_k: int) -> List[dict]:
        # Search and return results
        pass</code></pre>
        `,
        database: `
            <h4>Creating a Custom Database</h4>
            <p>Inherit from <code>DatabasePlugin</code>:</p>
            <pre><code>from glassbox_rag.plugins.base import DatabasePlugin

class MyDatabase(DatabasePlugin):
    async def store_document(self, doc_id: str, content: str, metadata: dict):
        # Store document
        pass

    async def get_document(self, doc_id: str) -> Optional[dict]:
        # Retrieve document
        pass</code></pre>
        `,
        multimodal: `
            <h4>Creating a Custom Multimodal Processor</h4>
            <p>Inherit from <code>BaseMultimodalProcessor</code>:</p>
            <pre><code>from glassbox_rag.utils.multimodal import BaseMultimodalProcessor

class MyImageProcessor(BaseMultimodalProcessor):
    async def process(self, content: MultimodalContent) -> List[str]:
        # Extract text from images
        texts = self.ocr_my_image(content.content)
        return texts</code></pre>
        `
    };

    content.innerHTML = docs[type] || '<p>Documentation not available.</p>';
    modal.style.display = 'flex';
}

function closePluginModal() {
    document.getElementById('pluginModal').style.display = 'none';
}

// Initialize dashboard when DOM is loaded
let dashboard;
document.addEventListener('DOMContentLoaded', () => {
    dashboard = new GlassBoxDashboard();
});

// Make dashboard globally available for onclick handlers
window.dashboard = dashboard;