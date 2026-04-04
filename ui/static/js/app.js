/* ═══════════════════════════════════════════════════════════════════
   BusinessHorizonENV v2 Dashboard — Client-side JS
   Real-time simulation via Socket.IO + Chart.js visualizations
   ═══════════════════════════════════════════════════════════════════ */

// ─── State ───
let socket = null;
let isRunning = false;
let currentMaxSteps = 340;
let allEvents = [];
let episodeResults = [];

// Chart instances
let rewardChart = null;
let cumulativeChart = null;
let memoryChart = null;
let episodeRewardChart = null;
let episodeDurationChart = null;

// Chart data buffers (per-episode, reset on episode_start)
let stepLabels = [];
let rewardData = [];
let cumulativeData = [];
let memWorkingData = [];
let memEpisodicData = [];
let memSemanticData = [];

// ═══════════════════════════════════════════════════════════════════
//  Socket.IO Connection
// ═══════════════════════════════════════════════════════════════════

function initSocket() {
    socket = io();

    socket.on('connect', () => {
        updateConnectionStatus(true);
    });

    socket.on('disconnect', () => {
        updateConnectionStatus(false);
    });

    socket.on('connected', (data) => {
        console.log('Connected:', data.message);
    });

    socket.on('run_started', (data) => {
        isRunning = true;
        updateRunButtons();
        setSimStatus('running', `Running ${data.env_id} (seed=${data.seed})`);
    });

    socket.on('episode_start', (data) => {
        currentMaxSteps = data.max_steps;
        resetEpisodeData();
        setSimStatus('running', `Episode ${data.episode} | ${data.env_id} | seed=${data.seed}`);
    });

    socket.on('step_update', (data) => {
        handleStepUpdate(data);
    });

    socket.on('episode_end', (data) => {
        handleEpisodeEnd(data);
    });

    socket.on('run_complete', (data) => {
        isRunning = false;
        updateRunButtons();
        setSimStatus('complete', `Completed ${data.total_episodes} episode(s)`);
    });

    socket.on('run_stopped', (data) => {
        isRunning = false;
        updateRunButtons();
        setSimStatus('idle', 'Stopped');
    });

    socket.on('run_stopping', () => {
        setSimStatus('idle', 'Stopping...');
    });

    socket.on('run_error', (data) => {
        isRunning = false;
        updateRunButtons();
        setSimStatus('error', `Error: ${data.error}`);
    });
}

// ═══════════════════════════════════════════════════════════════════
//  UI Updates
// ═══════════════════════════════════════════════════════════════════

function updateConnectionStatus(connected) {
    const el = document.getElementById('connectionStatus');
    const dot = el.querySelector('.status-dot');
    const text = el.querySelector('span:last-child');
    if (connected) {
        dot.className = 'status-dot connected';
        text.textContent = 'Connected';
    } else {
        dot.className = 'status-dot disconnected';
        text.textContent = 'Disconnected';
    }
}

function updateRunButtons() {
    document.getElementById('btnStart').disabled = isRunning;
    document.getElementById('btnStop').disabled = !isRunning;
}

function setSimStatus(state, text) {
    const indicator = document.querySelector('#simStatus .status-indicator');
    indicator.className = `status-indicator ${state}`;
    document.getElementById('simStatusText').textContent = text;
}

function switchPanel(name) {
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(`panel-${name}`).classList.add('active');
    document.querySelector(`.nav-btn[data-panel="${name}"]`).classList.add('active');
}

function selectEnv(envId) {
    document.querySelectorAll('.env-card').forEach(c => c.classList.remove('selected'));
    document.querySelector(`.env-card[data-env="${envId}"]`).classList.add('selected');
    document.getElementById('cfgEnv').value = envId;
}

// ═══════════════════════════════════════════════════════════════════
//  Simulation Controls
// ═══════════════════════════════════════════════════════════════════

function startRun() {
    if (!socket || !socket.connected) {
        alert('Not connected to server');
        return;
    }

    const config = {
        env_id: document.getElementById('cfgEnv').value,
        episodes: document.getElementById('cfgEpisodes').value,
        seed: document.getElementById('cfgSeed').value,
        beam_width: document.getElementById('cfgBeamWidth').value,
        beam_depth: document.getElementById('cfgBeamDepth').value,
        reset_state: document.getElementById('cfgReset').value === 'true',
    };

    socket.emit('start_run', config);

    // Auto-switch to simulation panel
    switchPanel('simulation');
}

function stopRun() {
    if (socket) {
        socket.emit('stop_run');
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Step Update Handler
// ═══════════════════════════════════════════════════════════════════

function handleStepUpdate(data) {
    const step = data.step;
    const pct = Math.min(100, (step / currentMaxSteps) * 100);

    // Progress bar
    document.getElementById('progressFill').style.width = `${pct}%`;
    document.getElementById('progressLabel').textContent = `Step ${step} / ${currentMaxSteps}`;
    document.getElementById('progressPhase').textContent = `Phase ${data.phase}`;

    // Live metrics
    document.getElementById('liveStep').textContent = step;
    document.getElementById('livePhase').textContent = data.phase;
    document.getElementById('liveAction').textContent = data.action;
    document.getElementById('liveReward').textContent = formatReward(data.shaped_reward);
    document.getElementById('liveReward').style.color = data.shaped_reward >= 0 ? 'var(--accent-green)' : 'var(--accent-red)';
    document.getElementById('liveTotalReward').textContent = formatReward(data.shaped_total);
    document.getElementById('liveTotalReward').style.color = data.shaped_total >= 0 ? 'var(--accent-green)' : 'var(--accent-red)';

    // Chart data (sample every 2 steps to avoid too many points)
    if (step % 2 === 0 || data.done) {
        stepLabels.push(step);
        rewardData.push(data.shaped_reward);
        cumulativeData.push(data.shaped_total);
        memWorkingData.push(data.memory.working);
        memEpisodicData.push(data.memory.episodic);
        memSemanticData.push(data.memory.semantic);
        updateRewardChart();
        updateCumulativeChart();
        updateMemoryChart();
    }

    // Memory panel
    document.getElementById('memWorking').textContent = data.memory.working;
    document.getElementById('memEpisodic').textContent = data.memory.episodic;
    document.getElementById('memSemantic').textContent = data.memory.semantic;
    document.getElementById('memWorkingBar').style.width = `${Math.min(100, (data.memory.working / 50) * 100)}%`;
    document.getElementById('memEpisodicBar').style.width = `${Math.min(100, (data.memory.episodic / 500) * 100)}%`;
    document.getElementById('memSemanticBar').style.width = `${Math.min(100, (data.memory.semantic / 50) * 100)}%`;

    // Compression stats
    if (data.memory.compression) {
        document.getElementById('compDaily').textContent = data.memory.compression.daily_summaries;
        document.getElementById('compMilestone').textContent = data.memory.compression.milestone_summaries;
        document.getElementById('compStrategic').textContent = data.memory.compression.strategic_insights;
    }

    // Goal progress
    updateGoalBars(data.goal_progress);

    // State digest
    updateStateGrid(data.state);

    // Live events
    if (data.events && data.events.length > 0) {
        data.events.forEach(text => {
            addFeedItem(step, text);
            allEvents.push({ step, text, type: data.action });
        });
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Episode End Handler
// ═══════════════════════════════════════════════════════════════════

function handleEpisodeEnd(data) {
    episodeResults.push(data);

    // Quick stats
    document.getElementById('statEpisodes').textContent = episodeResults.length;
    document.getElementById('statSkills').textContent = data.skills_learned;
    document.getElementById('statReplay').textContent = data.replay_size;

    const bestReward = Math.max(...episodeResults.map(r => r.shaped_reward));
    document.getElementById('statBestReward').textContent = formatReward(bestReward);

    // Analytics table
    addResultRow(data);

    // Analytics charts
    updateEpisodeCharts();

    // Update event log panel
    if (data.events_log) {
        data.events_log.forEach(e => {
            addEventEntry(e.step, e.type, e.text);
        });
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Chart Setup & Updates
// ═══════════════════════════════════════════════════════════════════

const chartDefaults = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            display: false,
        },
    },
    scales: {
        x: {
            grid: { color: 'rgba(42,45,62,0.5)' },
            ticks: { color: '#5a5e73', font: { size: 10 } },
        },
        y: {
            grid: { color: 'rgba(42,45,62,0.5)' },
            ticks: { color: '#5a5e73', font: { size: 10 } },
        },
    },
    animation: { duration: 0 },
};

function initCharts() {
    // Reward chart
    const rwCtx = document.getElementById('rewardChart').getContext('2d');
    rewardChart = new Chart(rwCtx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                data: [],
                backgroundColor: [],
                borderRadius: 2,
                barPercentage: 0.8,
            }],
        },
        options: {
            ...chartDefaults,
            plugins: { ...chartDefaults.plugins },
            scales: {
                ...chartDefaults.scales,
                x: { ...chartDefaults.scales.x, display: false },
            },
        },
    });

    // Cumulative chart
    const cumCtx = document.getElementById('cumulativeChart').getContext('2d');
    cumulativeChart = new Chart(cumCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                data: [],
                borderColor: '#4f8ff7',
                backgroundColor: 'rgba(79,143,247,0.1)',
                fill: true,
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.3,
            }],
        },
        options: chartDefaults,
    });

    // Memory chart
    const memCtx = document.getElementById('memoryChart').getContext('2d');
    memoryChart = new Chart(memCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Working',
                    data: [],
                    borderColor: '#fb923c',
                    backgroundColor: 'rgba(251,146,60,0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.3,
                },
                {
                    label: 'Episodic',
                    data: [],
                    borderColor: '#4f8ff7',
                    backgroundColor: 'rgba(79,143,247,0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.3,
                },
                {
                    label: 'Semantic',
                    data: [],
                    borderColor: '#a78bfa',
                    backgroundColor: 'rgba(167,139,250,0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.3,
                },
            ],
        },
        options: {
            ...chartDefaults,
            plugins: {
                legend: {
                    display: true,
                    labels: { color: '#8b8fa3', font: { size: 11 } },
                },
            },
        },
    });

    // Episode reward chart
    const epRwCtx = document.getElementById('episodeRewardChart').getContext('2d');
    episodeRewardChart = new Chart(epRwCtx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                data: [],
                backgroundColor: [],
                borderRadius: 4,
            }],
        },
        options: chartDefaults,
    });

    // Episode duration chart
    const epDurCtx = document.getElementById('episodeDurationChart').getContext('2d');
    episodeDurationChart = new Chart(epDurCtx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                data: [],
                backgroundColor: 'rgba(79,143,247,0.6)',
                borderRadius: 4,
            }],
        },
        options: chartDefaults,
    });
}

function updateRewardChart() {
    rewardChart.data.labels = stepLabels;
    rewardChart.data.datasets[0].data = rewardData;
    rewardChart.data.datasets[0].backgroundColor = rewardData.map(v =>
        v >= 0 ? 'rgba(52,211,153,0.6)' : 'rgba(248,113,113,0.6)'
    );
    rewardChart.update();
}

function updateCumulativeChart() {
    cumulativeChart.data.labels = stepLabels;
    cumulativeChart.data.datasets[0].data = cumulativeData;
    cumulativeChart.update();
}

function updateMemoryChart() {
    memoryChart.data.labels = stepLabels;
    memoryChart.data.datasets[0].data = memWorkingData;
    memoryChart.data.datasets[1].data = memEpisodicData;
    memoryChart.data.datasets[2].data = memSemanticData;
    memoryChart.update();
}

function updateEpisodeCharts() {
    const labels = episodeResults.map((_, i) => `Ep ${i + 1}`);
    const rewards = episodeResults.map(r => r.shaped_reward);
    const durations = episodeResults.map(r => r.elapsed_seconds);

    episodeRewardChart.data.labels = labels;
    episodeRewardChart.data.datasets[0].data = rewards;
    episodeRewardChart.data.datasets[0].backgroundColor = rewards.map(v =>
        v >= 0 ? 'rgba(52,211,153,0.6)' : 'rgba(248,113,113,0.6)'
    );
    episodeRewardChart.update();

    episodeDurationChart.data.labels = labels;
    episodeDurationChart.data.datasets[0].data = durations;
    episodeDurationChart.update();
}

// ═══════════════════════════════════════════════════════════════════
//  Goal Progress Bars
// ═══════════════════════════════════════════════════════════════════

function updateGoalBars(goalProgress) {
    const container = document.getElementById('goalBars');
    container.innerHTML = '';

    if (!goalProgress || Object.keys(goalProgress).length === 0) {
        container.innerHTML = '<p style="color:var(--text-muted); font-size:12px;">No goals active</p>';
        return;
    }

    for (const [name, progress] of Object.entries(goalProgress)) {
        const pct = Math.round(progress * 100);
        const bar = document.createElement('div');
        bar.className = 'goal-bar';
        bar.innerHTML = `
            <div class="goal-bar-header">
                <span class="goal-name">${name}</span>
                <span class="goal-pct">${pct}%</span>
            </div>
            <div class="goal-track">
                <div class="goal-fill" style="width: ${pct}%"></div>
            </div>
        `;
        container.appendChild(bar);
    }
}

// ═══════════════════════════════════════════════════════════════════
//  State Grid
// ═══════════════════════════════════════════════════════════════════

function updateStateGrid(state) {
    const grid = document.getElementById('stateGrid');
    grid.innerHTML = '';

    if (!state) return;

    for (const [key, val] of Object.entries(state)) {
        const item = document.createElement('div');
        item.className = 'state-item';

        const keySpan = document.createElement('span');
        keySpan.className = 'state-key';
        keySpan.textContent = formatKey(key);

        const valSpan = document.createElement('span');
        valSpan.className = 'state-val';
        valSpan.textContent = typeof val === 'number' ? formatNumber(val) : String(val);

        // Color code booleans
        if (val === true) valSpan.style.color = 'var(--accent-green)';
        if (val === false) valSpan.style.color = 'var(--accent-red)';

        item.appendChild(keySpan);
        item.appendChild(valSpan);
        grid.appendChild(item);
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Event Feed & Log
// ═══════════════════════════════════════════════════════════════════

function addFeedItem(step, text) {
    const feed = document.getElementById('liveFeed');
    const item = document.createElement('div');
    item.className = 'feed-item';
    item.innerHTML = `<span class="feed-step">${step}</span><span class="feed-text">${escapeHtml(text)}</span>`;
    feed.insertBefore(item, feed.firstChild);

    // Keep max 60 items
    while (feed.children.length > 60) {
        feed.removeChild(feed.lastChild);
    }
}

function addEventEntry(step, type, text) {
    const log = document.getElementById('eventLog');
    const entry = document.createElement('div');
    entry.className = 'event-entry';
    entry.innerHTML = `
        <span class="e-step">${step}</span>
        <span class="e-type">${type}</span>
        <span class="e-text">${escapeHtml(text)}</span>
    `;
    log.appendChild(entry);
}

function filterEvents() {
    const filter = document.getElementById('eventFilter').value.toLowerCase();
    document.querySelectorAll('#eventLog .event-entry').forEach(entry => {
        const text = entry.textContent.toLowerCase();
        entry.style.display = text.includes(filter) ? '' : 'none';
    });
}

function clearEvents() {
    document.getElementById('eventLog').innerHTML = '';
}

// ═══════════════════════════════════════════════════════════════════
//  Analytics Table
// ═══════════════════════════════════════════════════════════════════

function addResultRow(data) {
    const tbody = document.getElementById('resultsBody');
    const tr = document.createElement('tr');
    const rewardClass = data.shaped_reward >= 0 ? 'reward-positive' : 'reward-negative';
    tr.innerHTML = `
        <td class="mono">${data.episode}</td>
        <td>${data.env_id}</td>
        <td class="mono">${data.total_steps}</td>
        <td class="mono ${rewardClass}">${formatReward(data.shaped_reward)}</td>
        <td class="mono">${data.phase_reached}</td>
        <td>${formatTerminal(data.terminal_reason)}</td>
        <td class="mono">${data.elapsed_seconds}s</td>
        <td class="mono">${data.skills_learned}</td>
    `;
    tbody.appendChild(tr);
}

// ═══════════════════════════════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════════════════════════════

function resetEpisodeData() {
    stepLabels = [];
    rewardData = [];
    cumulativeData = [];
    memWorkingData = [];
    memEpisodicData = [];
    memSemanticData = [];

    document.getElementById('progressFill').style.width = '0%';
    document.getElementById('liveFeed').innerHTML = '';
}

function formatReward(val) {
    return (val >= 0 ? '+' : '') + val.toFixed(1);
}

function formatNumber(val) {
    if (Number.isInteger(val)) return val.toLocaleString();
    return val.toFixed(2);
}

function formatKey(key) {
    return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function formatTerminal(reason) {
    const map = {
        'timeout': '<span style="color:var(--accent-orange)">Timeout</span>',
        'deal_closed': '<span style="color:var(--accent-green)">Deal Closed</span>',
        'program_delivered': '<span style="color:var(--accent-green)">Delivered</span>',
        'migration_complete': '<span style="color:var(--accent-green)">Migration Done</span>',
        'budget_exhausted': '<span style="color:var(--accent-red)">Budget Exhausted</span>',
        'morale_collapsed': '<span style="color:var(--accent-red)">Morale Collapsed</span>',
    };
    return map[reason] || reason;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ═══════════════════════════════════════════════════════════════════
//  Init
// ═══════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    initSocket();
});
