"""
GlassBox TUI — Theme and style constants.

Centralised design tokens matching the GlassBox visual identity.
All Textual CSS and color references flow from here.
"""

from __future__ import annotations

# ── Colour Palette ────────────────────────────────────────────────
INDIGO = "#6366f1"
VIOLET = "#8b5cf6"
CYAN = "#22d3ee"
GREEN = "#34d399"
AMBER = "#fbbf24"
ROSE = "#f43f5e"
ORANGE = "#fb923c"

BG_PRIMARY = "#0a0b0f"
BG_SECONDARY = "#12141c"
BG_CARD = "#181b27"
BG_CARD_HOVER = "#1e2235"
BG_ELEVATED = "#22263a"

BORDER = "#2a2e42"
BORDER_ACTIVE = "#6366f1"

TEXT_PRIMARY = "#e8eaf0"
TEXT_SECONDARY = "#8b8fa3"
TEXT_MUTED = "#5c5f73"

# ── Status Colours ────────────────────────────────────────────────
STATUS_OK = GREEN
STATUS_WARN = AMBER
STATUS_ERROR = ROSE
STATUS_DISABLED = TEXT_MUTED

# ── Sparkline Colours (per-metric) ────────────────────────────────
SPARK_REQUESTS = INDIGO
SPARK_LATENCY = CYAN
SPARK_TOKENS = GREEN
SPARK_COST = AMBER

# ── Box-Drawing Characters ────────────────────────────────────────
BOX_H = "─"
BOX_V = "│"
BOX_TL = "┌"
BOX_TR = "┐"
BOX_BL = "└"
BOX_BR = "┘"
BOX_ARROW = "───▶"
BOX_CROSS = "┼"

# ── Pipeline Step Icons ───────────────────────────────────────────
STEP_ICONS = {
    "input": "📝",
    "encode": "🧠",
    "retrieve": "🔍",
    "rerank": "🔄",
    "generate": "⚡",
    "output": "📤",
    "chunk": "✂️",
    "store": "💾",
    "ingest": "📥",
    "extract": "📑",
}

# ── App-Wide Textual CSS ──────────────────────────────────────────
APP_CSS = """
$primary: #6366f1;
$secondary: #8b5cf6;
$accent: #22d3ee;
$success: #34d399;
$warning: #fbbf24;
$error: #f43f5e;
$surface: #181b27;
$surface-lighten: #22263a;
$panel: #12141c;
$text: #e8eaf0;
$text-muted: #8b8fa3;
$text-dim: #5c5f73;
$boost: #2a2e42;

Screen {
    background: #0a0b0f;
}

/* ── Branded Header ───────────────────────────────────────────── */

Header {
    background: #12141c;
    color: #6366f1;
    dock: top;
    height: 3;
}

/* ── Footer with accent key styling ───────────────────────────── */

Footer {
    background: #12141c;
    color: #8b8fa3;
}

FooterKey .footer-key--key {
    color: #6366f1;
    text-style: bold;
    background: #22263a;
}

FooterKey .footer-key--description {
    color: #8b8fa3;
}

/* ── Tabs with active glow ────────────────────────────────────── */

TabbedContent {
    background: #0a0b0f;
}

TabPane {
    background: #0a0b0f;
    padding: 1 2;
}

ContentSwitcher {
    background: #0a0b0f;
}

Tabs {
    background: #12141c;
    dock: top;
    overflow-x: auto;
}

Tab {
    background: #12141c;
    color: #8b8fa3;
    padding: 1 2;
    min-width: 10;
}

Tab.-active {
    background: #181b27;
    color: #6366f1;
    text-style: bold;
    border-bottom: tall #6366f1;
}

Tab:hover {
    background: #1e2235;
    color: #e8eaf0;
}

/* ── Data Tables ──────────────────────────────────────────────── */

DataTable {
    background: #181b27;
    height: auto;
    max-height: 100%;
    scrollbar-background: #12141c;
    scrollbar-color: #2a2e42;
    scrollbar-color-active: #6366f1;
    scrollbar-color-hover: #8b5cf6;
}

DataTable > .datatable--header {
    background: #22263a;
    color: #8b8fa3;
    text-style: bold;
}

DataTable > .datatable--cursor {
    background: #6366f1 30%;
    color: #e8eaf0;
}

DataTable > .datatable--hover {
    background: #1e2235;
}

/* ── Tree ─────────────────────────────────────────────────────── */

Tree {
    background: #181b27;
    scrollbar-background: #12141c;
    scrollbar-color: #2a2e42;
    scrollbar-color-active: #6366f1;
}

Tree > .tree--cursor {
    background: #6366f1 30%;
    color: #e8eaf0;
}

/* ── Input ────────────────────────────────────────────────────── */

Input {
    background: #22263a;
    border: tall #2a2e42;
    color: #e8eaf0;
}

Input:focus {
    border: tall #6366f1;
}

/* ── Buttons ──────────────────────────────────────────────────── */

Button {
    background: #6366f1;
    color: #e8eaf0;
    border: none;
    min-width: 16;
    height: 3;
}

Button:hover {
    background: #8b5cf6;
}

Button.-secondary {
    background: #22263a;
    border: tall #2a2e42;
    color: #e8eaf0;
}

Button.-secondary:hover {
    background: #1e2235;
    border: tall #6366f1;
}

/* ── Select ───────────────────────────────────────────────────── */

Select {
    background: #22263a;
    border: tall #2a2e42;
    color: #e8eaf0;
}

/* ── Static ───────────────────────────────────────────────────── */

Static {
    background: transparent;
    color: #e8eaf0;
}

/* ── ListView / ListItem ──────────────────────────────────────── */

ListView {
    background: #181b27;
    scrollbar-background: #12141c;
    scrollbar-color: #2a2e42;
}

ListItem {
    background: #181b27;
    color: #e8eaf0;
    padding: 1 2;
}

ListItem:hover {
    background: #1e2235;
}

ListView > ListItem.-highlight {
    background: #6366f1 30%;
}

/* ── RichLog ──────────────────────────────────────────────────── */

RichLog {
    background: #181b27;
    color: #e8eaf0;
    scrollbar-background: #12141c;
    scrollbar-color: #2a2e42;
}

/* ── Sparkline ────────────────────────────────────────────────── */

Sparkline {
    min-height: 2;
}

/* ── Progress Bar ─────────────────────────────────────────────── */

ProgressBar Bar {
    color: #6366f1;
    background: #22263a;
}

/* ── Markdown ─────────────────────────────────────────────────── */

Markdown {
    background: #181b27;
    margin: 1 0;
    padding: 1 2;
}

/* ── Metric Grid ──────────────────────────────────────────────── */

#metric-grid {
    layout: horizontal;
    height: auto;
    min-height: 7;
    margin: 1 0;
}

/* ── Metric Cards ─────────────────────────────────────────────── */

.metric-card {
    width: 1fr;
    height: auto;
    min-height: 5;
    background: #181b27;
    border: round #2a2e42;
    padding: 1 2;
    margin: 0 1;
}

.metric-card:hover {
    border: round #6366f1;
}

.metric-value {
    text-style: bold;
    color: #e8eaf0;
}

.metric-label {
    color: #8b8fa3;
}

.metric-trend-up {
    color: #34d399;
}

.metric-trend-down {
    color: #f43f5e;
}

/* ── Section Titles ───────────────────────────────────────────── */

.section-title {
    color: #6366f1;
    text-style: bold;
    padding: 1 0 0 0;
    margin: 0 0 1 0;
}

/* ── Panels with hover glow ───────────────────────────────────── */

.panel {
    background: #181b27;
    border: round #2a2e42;
    padding: 1 2;
    margin: 0 0 1 0;
    height: auto;
}

.panel:hover {
    border: round #6366f1;
}

/* ── Status Classes ───────────────────────────────────────────── */

.status-ok {
    color: #34d399;
}

.status-warn {
    color: #fbbf24;
}

.status-error {
    color: #f43f5e;
}

.status-disabled {
    color: #5c5f73;
}

/* ── Code View ────────────────────────────────────────────────── */

.code-view {
    background: #12141c;
    border: round #2a2e42;
    padding: 1 2;
    margin: 1 0;
    color: #e8eaf0;
}

/* ── Drag ──────────────────────────────────────────────────────── */

.drag-active {
    background: #6366f1 20%;
    border: round #6366f1;
}

.drag-target {
    border: dashed #8b5cf6;
}

/* ── Empty State ──────────────────────────────────────────────── */

.empty-state {
    color: #5c5f73;
    text-align: center;
    padding: 3 0;
    text-style: italic;
}

/* ── Branded Title ────────────────────────────────────────────── */

.brand-title {
    color: #6366f1;
    text-style: bold;
}

.brand-subtitle {
    color: #8b8fa3;
}

/* ── Debugger Trace Row Styles ────────────────────────────────── */

.trace-ok {
    color: #34d399;
}

.trace-error {
    color: #f43f5e;
}

.trace-slow {
    color: #fbbf24;
}
"""
