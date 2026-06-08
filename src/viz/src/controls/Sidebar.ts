import type {
  ColorMode,
  GeoJSONLineFeature,
  QAAnnotation,
  QAReport,
  ViewMode,
} from '../types/spatial.js';

export type LayerKey = 'pointCloud' | 'features' | 'qa';

type LayerState = Record<LayerKey, boolean>;

type SidebarCallbacks = {
  onLayerToggle: (layer: LayerKey, enabled: boolean) => void;
  onColorMode: (mode: ColorMode) => void;
  onViewToggle: () => void;
  onRunPipeline: () => void;
};

const LAYER_CONFIG: Array<{ key: LayerKey; label: string; color: string; shortcut: string }> = [
  { key: 'pointCloud', label: 'Point Cloud', color: '#e6edf3', shortcut: '1' },
  { key: 'features',   label: 'Features',    color: '#38bdf8', shortcut: '2' },
  { key: 'qa',         label: 'QA Overlays', color: '#f59e0b', shortcut: '3' },
];

export class Sidebar {
  private readonly layerState: LayerState = { pointCloud: true, features: true, qa: true };
  private colorMode: ColorMode = 'intensity';

  private fpsEl!: HTMLElement;
  private selectedEl!: HTMLElement;
  private viewLabelEl!: HTMLElement;
  private qaSectionEl!: HTMLElement;
  private qaMetricsEl!: HTMLElement;
  private layerToggles: Record<LayerKey, HTMLElement> = {} as Record<LayerKey, HTMLElement>;
  private colorBtns: Record<ColorMode, HTMLElement> = {} as Record<ColorMode, HTMLElement>;
  private pointCountEl!: HTMLElement;
  private pipelineBtnEl!: HTMLButtonElement;
  private pipelineStatusEl!: HTMLElement;

  constructor(private readonly callbacks: SidebarCallbacks) {
    this.buildDOM();
  }

  private buildDOM(): void {
    const sidebar = document.getElementById('sidebar')!;
    sidebar.innerHTML = `
      <div class="sb-header">
        <span class="sb-title">HD Map Viewer</span>
      </div>

      <div class="sb-section">
        <div class="sb-section-label">Pipeline</div>
        <button class="sb-run-btn" id="run-pipeline-btn">Run Pipeline</button>
        <div id="pipeline-status" class="sb-pipeline-status"></div>
      </div>

      <div class="sb-section">
        <div class="sb-section-label">Layers</div>
        ${LAYER_CONFIG.map(({ key, label, color, shortcut }) => `
          <div class="sb-layer-row" data-layer="${key}">
            <span class="sb-dot" style="background:${color}"></span>
            <span class="sb-layer-name">${label}</span>
            <span class="sb-key">[${shortcut}]</span>
            <div class="sb-toggle on" id="toggle-${key}"></div>
          </div>
        `).join('')}
      </div>

      <div class="sb-section">
        <div class="sb-section-label">Color mode</div>
        <div class="sb-btn-row">
          <button class="sb-mode-btn active" id="mode-intensity" title="[I]">Intensity</button>
          <button class="sb-mode-btn" id="mode-height" title="[H]">Height</button>
        </div>
        <div class="sb-key-hint">[I] intensity &nbsp;·&nbsp; [H] height</div>
      </div>

      <div class="sb-section">
        <div class="sb-section-label">View</div>
        <button class="sb-view-btn" id="view-toggle">
          <span id="view-label">Perspective</span>
          <span class="sb-key">[V]</span>
        </button>
      </div>

      <div class="sb-section" id="qa-section" style="display:none">
        <div class="sb-section-label">QA report</div>
        <div id="qa-metrics"></div>
      </div>

      <div class="sb-section sb-section-grow">
        <div class="sb-section-label">Selected</div>
        <div id="selected-feature" class="sb-muted">click a feature line</div>
      </div>

      <div class="sb-footer">
        <span class="sb-footer-label">Points</span>
        <span id="point-count" class="sb-stat">—</span>
        <span class="sb-footer-label sb-fps-label">fps</span>
        <span id="fps-value" class="sb-stat">—</span>
      </div>
    `;

    this.bindEvents();
  }

  private bindEvents(): void {
    for (const { key } of LAYER_CONFIG) {
      const toggleEl = document.getElementById(`toggle-${key}`)!;
      this.layerToggles[key] = toggleEl;
      toggleEl.closest('.sb-layer-row')!.addEventListener('click', () => {
        this.setLayer(key, !this.layerState[key]);
      });
    }

    const intensityBtn = document.getElementById('mode-intensity')!;
    const heightBtn = document.getElementById('mode-height')!;
    this.colorBtns.intensity = intensityBtn;
    this.colorBtns.height = heightBtn;
    intensityBtn.addEventListener('click', () => this.setColorMode('intensity'));
    heightBtn.addEventListener('click', () => this.setColorMode('height'));

    document.getElementById('view-toggle')!.addEventListener('click', () => {
      this.callbacks.onViewToggle();
    });

    this.viewLabelEl = document.getElementById('view-label')!;
    this.fpsEl = document.getElementById('fps-value')!;
    this.selectedEl = document.getElementById('selected-feature')!;
    this.qaSectionEl = document.getElementById('qa-section')!;
    this.qaMetricsEl = document.getElementById('qa-metrics')!;
    this.pointCountEl = document.getElementById('point-count')!;
    this.pipelineBtnEl = document.getElementById('run-pipeline-btn') as HTMLButtonElement;
    this.pipelineStatusEl = document.getElementById('pipeline-status')!;

    this.pipelineBtnEl.addEventListener('click', () => {
      this.callbacks.onRunPipeline();
    });
  }

  setLayer(key: LayerKey, enabled: boolean): void {
    this.layerState[key] = enabled;
    const el = this.layerToggles[key];
    el.classList.toggle('on', enabled);
    el.classList.toggle('off', !enabled);
    this.callbacks.onLayerToggle(key, enabled);
  }

  setColorMode(mode: ColorMode): void {
    this.colorMode = mode;
    this.colorBtns.intensity.classList.toggle('active', mode === 'intensity');
    this.colorBtns.height.classList.toggle('active', mode === 'height');
    this.callbacks.onColorMode(mode);
  }

  setViewMode(mode: ViewMode): void {
    this.viewLabelEl.textContent = mode === 'bev' ? 'BEV (top-down)' : 'Perspective';
  }

  updateFPS(fps: number): void {
    this.fpsEl.textContent = fps.toFixed(1);
  }

  updatePointCount(count: number): void {
    this.pointCountEl.textContent = count >= 1000
      ? `${(count / 1000).toFixed(0)}K`
      : String(count);
  }

  showQAReport(report: QAReport): void {
    this.qaSectionEl.style.display = 'block';

    const fmt3 = (v: number) => v.toFixed(3);

    const rows: Array<[string, string, string?]> = [
      ['Completeness', fmt3(report.completeness), report.completeness >= 0.9 ? 'good' : 'warn'],
      ['P50 acc.', `${fmt3(report.positional_accuracy_p50)} m`],
      ['P95 acc.', `${fmt3(report.positional_accuracy_p95)} m`],
      ['FP rate', fmt3(report.false_positive_rate), report.false_positive_rate < 0.1 ? 'good' : 'warn'],
      ['Classification', fmt3(report.classification_accuracy), report.classification_accuracy >= 0.9 ? 'good' : 'warn'],
    ];

    let html = rows
      .map(
        ([label, value, quality]) =>
          `<div class="sb-metric-row">
            <span class="sb-metric-label">${label}</span>
            <span class="sb-metric-value ${quality ?? ''}">${value}</span>
          </div>`,
      )
      .join('');

    const perClass = Object.entries(report.per_class_completeness);
    if (perClass.length > 0) {
      html += `<div class="sb-metric-subhead">Per class</div>`;
      for (const [cls, val] of perClass) {
        html += `<div class="sb-metric-row">
          <span class="sb-metric-label sb-class-label">${cls}</span>
          <span class="sb-metric-value">${fmt3(val)}</span>
        </div>`;
      }
    }

    this.qaMetricsEl.innerHTML = html;
  }

  showSelectedFeature(feature: GeoJSONLineFeature, index: number): void {
    const p = feature.properties ?? {};
    const type = p.feature_type ?? 'unknown';
    const conf = p.confidence != null ? p.confidence.toFixed(3) : '—';
    const src = p.source ?? '—';
    const nPts = feature.geometry.coordinates.length;

    this.selectedEl.className = '';
    this.selectedEl.innerHTML = `
      <div class="sb-metric-row">
        <span class="sb-metric-label">index</span>
        <span class="sb-metric-value">#${index}</span>
      </div>
      <div class="sb-metric-row">
        <span class="sb-metric-label">type</span>
        <span class="sb-metric-value" style="color:#38bdf8">${type}</span>
      </div>
      <div class="sb-metric-row">
        <span class="sb-metric-label">confidence</span>
        <span class="sb-metric-value">${conf}</span>
      </div>
      <div class="sb-metric-row">
        <span class="sb-metric-label">source</span>
        <span class="sb-metric-value">${src}</span>
      </div>
      <div class="sb-metric-row">
        <span class="sb-metric-label">vertices</span>
        <span class="sb-metric-value">${nPts}</span>
      </div>
    `;
  }

  showSelectedAnnotation(annotation: QAAnnotation): void {
    const kindColor = annotation.kind === 'false_positive' ? '#f59e0b' : '#ef4444';
    const kindLabel = annotation.kind === 'false_positive' ? 'False positive' : 'Missed GT';

    this.selectedEl.className = '';
    this.selectedEl.innerHTML = `
      <div class="sb-metric-row">
        <span class="sb-metric-label">kind</span>
        <span class="sb-metric-value" style="color:${kindColor}">${kindLabel}</span>
      </div>
      <div class="sb-metric-row">
        <span class="sb-metric-label">id</span>
        <span class="sb-metric-value">${annotation.id}</span>
      </div>
      ${Object.entries(annotation.metrics)
        .map(
          ([k, v]) =>
            `<div class="sb-metric-row">
              <span class="sb-metric-label">${k}</span>
              <span class="sb-metric-value">${typeof v === 'number' ? v.toFixed(3) : v}</span>
            </div>`,
        )
        .join('')}
    `;
  }

  clearSelection(): void {
    this.selectedEl.className = 'sb-muted';
    this.selectedEl.textContent = 'click a feature line';
  }

  setPipelineStatus(status: 'idle' | 'running' | 'done' | 'error', message?: string): void {
    const btn = this.pipelineBtnEl;
    const statusEl = this.pipelineStatusEl;

    btn.className = 'sb-run-btn';
    btn.disabled = false;
    statusEl.textContent = '';
    statusEl.className = 'sb-pipeline-status';

    switch (status) {
      case 'idle':
        btn.textContent = 'Run Pipeline';
        break;
      case 'running':
        btn.textContent = 'Running…';
        btn.disabled = true;
        btn.classList.add('running');
        statusEl.textContent = 'processing point cloud';
        statusEl.classList.add('muted');
        break;
      case 'done':
        btn.textContent = 'Run Pipeline';
        statusEl.textContent = 'loaded ✓';
        statusEl.classList.add('done');
        break;
      case 'error':
        btn.textContent = 'Retry';
        btn.classList.add('error');
        statusEl.textContent = message ?? 'pipeline failed';
        statusEl.classList.add('err');
        break;
    }
  }

  get currentColorMode(): ColorMode {
    return this.colorMode;
  }
}
