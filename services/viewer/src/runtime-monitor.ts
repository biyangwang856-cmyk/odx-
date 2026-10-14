export interface ResourceTotals {
  requests: number;
  encodedBodyBytes: number;
  decodedBodyBytes: number;
  transferBytes: number;
}

const EMPTY_TOTALS: ResourceTotals = {
  requests: 0,
  encodedBodyBytes: 0,
  decodedBodyBytes: 0,
  transferBytes: 0,
};

export class ResourceLedger {
  readonly base: URL;
  readonly buckets: Map<string, ResourceTotals>;
  entries = 0;
  ignored = 0;

  constructor(base: string, prefixes: readonly string[]) {
    this.base = new URL(base);
    if (
      prefixes.length === 0
      || new Set(prefixes).size !== prefixes.length
      || prefixes.some((prefix) => !prefix.startsWith("/"))
    ) {
      throw new Error("resource prefixes must be unique absolute paths");
    }
    this.buckets = new Map(prefixes.map((prefix) => [prefix, { ...EMPTY_TOTALS }]));
  }

  add(entries: readonly PerformanceResourceTiming[]): void {
    for (const entry of entries) {
      let url: URL;
      try {
        url = new URL(entry.name, this.base);
      } catch {
        this.ignored += 1;
        continue;
      }
      if (url.origin !== this.base.origin) {
        this.ignored += 1;
        continue;
      }
      const matches = [...this.buckets].filter(([prefix]) => url.pathname.startsWith(prefix));
      if (matches.length === 0) continue;
      const sizes = {
        encodedBodyBytes: entry.encodedBodySize ?? 0,
        decodedBodyBytes: entry.decodedBodySize ?? 0,
        transferBytes: entry.transferSize ?? 0,
      };
      if (
        Object.values(sizes).some((value) => !Number.isSafeInteger(value) || value < 0)
        || matches.some(([, bucket]) => Object.entries(sizes).some(
          ([key, value]) => !Number.isSafeInteger(bucket[key as keyof typeof sizes] + value),
        ))
      ) {
        this.ignored += 1;
        continue;
      }
      this.entries += 1;
      for (const [, bucket] of matches) {
        bucket.requests += 1;
        bucket.encodedBodyBytes += sizes.encodedBodyBytes;
        bucket.decodedBodyBytes += sizes.decodedBodyBytes;
        bucket.transferBytes += sizes.transferBytes;
      }
    }
  }

  totals(prefix: string): ResourceTotals {
    const bucket = this.buckets.get(prefix);
    if (!bucket) throw new Error(`unknown resource prefix: ${prefix}`);
    return { ...bucket };
  }
}

export class RefreshCadence {
  readonly activeMs: number;
  readonly idleMs: number;
  readonly idleAfterMs: number;
  private lastActivity = Number.NEGATIVE_INFINITY;

  constructor({ activeMs = 250, idleMs = 1000, idleAfterMs = 2000 } = {}) {
    if (
      ![activeMs, idleMs, idleAfterMs].every((value) => Number.isFinite(value) && value >= 0)
      || activeMs === 0
      || idleMs < activeMs
    ) {
      throw new Error("invalid refresh cadence");
    }
    this.activeMs = activeMs;
    this.idleMs = idleMs;
    this.idleAfterMs = idleAfterMs;
  }

  touch(now: number): void {
    if (!Number.isFinite(now)) throw new Error("invalid activity time");
    this.lastActivity = now;
  }

  idle(now: number): boolean {
    return now - this.lastActivity >= this.idleAfterMs;
  }

  delay(now: number): number {
    return this.idle(now) ? this.idleMs : this.activeMs;
  }
}

interface LoopOptions {
  activeMs?: number;
  idleMs?: number;
  immediate?: boolean;
}

export class AdaptiveLoop {
  private timer: number | undefined;
  private lastRun = Number.NEGATIVE_INFINITY;
  private stopped = false;

  constructor(
    private readonly callback: () => void,
    private readonly monitor: RuntimeMonitor,
    private readonly options: LoopOptions = {},
  ) {
    for (const value of [options.activeMs, options.idleMs]) {
      if (value !== undefined && (!Number.isFinite(value) || value <= 0)) {
        throw new Error("invalid loop delay");
      }
    }
  }

  private delay(): number {
    const now = performance.now();
    return this.monitor.cadence.idle(now)
      ? this.options.idleMs ?? this.monitor.cadence.idleMs
      : this.options.activeMs ?? this.monitor.cadence.activeMs;
  }

  private schedule(delay: number): void {
    this.cancel();
    if (this.stopped || this.monitor.paused) return;
    this.timer = window.setTimeout(() => {
      this.timer = undefined;
      if (this.stopped || this.monitor.paused) return;
      this.lastRun = performance.now();
      try {
        this.callback();
      } finally {
        this.schedule(this.delay());
      }
    }, delay);
  }

  start(): this {
    this.schedule(this.options.immediate === false ? this.delay() : 0);
    return this;
  }

  wake(): void {
    if (this.stopped || this.monitor.paused) {
      this.cancel();
      return;
    }
    const activeMs = this.options.activeMs ?? this.monitor.cadence.activeMs;
    this.schedule(Math.max(0, activeMs - (performance.now() - this.lastRun)));
  }

  cancel(): void {
    if (this.timer !== undefined) window.clearTimeout(this.timer);
    this.timer = undefined;
  }

  stop(): void {
    this.stopped = true;
    this.cancel();
  }
}

interface RuntimeCounters {
  metricDomWrites: number;
  resourceBatches: number;
  fullResourceScans: number;
  hiddenSuspensions: number;
}

export class RuntimeMonitor {
  readonly cadence = new RefreshCadence();
  readonly resources: ResourceLedger;
  readonly counters: RuntimeCounters = {
    metricDomWrites: 0,
    resourceBatches: 0,
    fullResourceScans: 0,
    hiddenSuspensions: 0,
  };
  readonly resourceMode: "observer" | "buffer-fallback";
  private readonly loops = new Set<AdaptiveLoop>();
  private readonly suspendCallbacks = new Set<() => void>();
  private readonly resumeCallbacks = new Set<() => void>();
  private pageSuspended = false;
  private resourceObserver?: PerformanceObserver;
  private bufferCursor = 0;

  constructor(base: string, prefixes: readonly string[]) {
    this.resources = new ResourceLedger(base, prefixes);
    this.cadence.touch(performance.now());
    let resourceMode: "observer" | "buffer-fallback" = "observer";
    try {
      this.resourceObserver = new PerformanceObserver((list) => {
        this.resources.add(list.getEntries() as PerformanceResourceTiming[]);
        this.counters.resourceBatches += 1;
      });
      this.resourceObserver.observe({ type: "resource", buffered: true });
    } catch {
      this.resourceObserver?.disconnect();
      this.resourceObserver = undefined;
      resourceMode = "buffer-fallback";
    }
    this.resourceMode = resourceMode;

    document.addEventListener("pointerdown", () => this.touch(true), { passive: true });
    document.addEventListener("keydown", () => this.touch(true));
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) {
        this.counters.hiddenSuspensions += 1;
        this.suspend();
      } else if (!this.pageSuspended) {
        this.resume();
      }
    });
    window.addEventListener("pagehide", () => {
      this.pageSuspended = true;
      this.suspend();
    });
    window.addEventListener("pageshow", () => {
      this.pageSuspended = false;
      if (!document.hidden) this.resume();
    });

    this.startLoop(() => this.collectFallbackResources());
  }

  get paused(): boolean {
    return this.pageSuspended || document.hidden;
  }

  startLoop(callback: () => void, options: LoopOptions = {}): AdaptiveLoop {
    const loop = new AdaptiveLoop(callback, this, options).start();
    this.loops.add(loop);
    return loop;
  }

  onLifecycle(suspend: () => void, resume: () => void): () => void {
    this.suspendCallbacks.add(suspend);
    this.resumeCallbacks.add(resume);
    return () => {
      this.suspendCallbacks.delete(suspend);
      this.resumeCallbacks.delete(resume);
    };
  }

  touch(wake = false): void {
    this.cadence.touch(performance.now());
    if (wake) for (const loop of this.loops) loop.wake();
  }

  setText(element: Element | null, value: string): void {
    if (element && element.textContent !== value) {
      element.textContent = value;
      this.counters.metricDomWrites += 1;
    }
  }

  snapshot(): object {
    const now = performance.now();
    return {
      version: 56,
      ...this.counters,
      resourceMode: this.resourceMode,
      cumulativeResourceEntries: this.resources.entries,
      ignoredResourceEntries: this.resources.ignored,
      cadenceMs: this.cadence.delay(now),
      idle: this.cadence.idle(now),
      hidden: document.hidden,
      resourceCountersRetainTimingEntries: false,
      limitations: [
        "Cesium totalMemoryUsageInBytes is an engine estimate, not physical VRAM",
        "resource bytes and wire transfer bytes are distinct",
        "fallback mode can lose entries when the browser timing buffer overflows",
      ],
    };
  }

  private collectFallbackResources(): void {
    if (this.resourceMode !== "buffer-fallback") return;
    const entries = performance.getEntriesByType("resource") as PerformanceResourceTiming[];
    this.counters.fullResourceScans += 1;
    if (entries.length < this.bufferCursor) this.bufferCursor = 0;
    this.resources.add(entries.slice(this.bufferCursor));
    this.bufferCursor = entries.length;
  }

  private suspend(): void {
    for (const loop of this.loops) loop.cancel();
    for (const callback of this.suspendCallbacks) callback();
  }

  private resume(): void {
    this.touch();
    for (const callback of this.resumeCallbacks) callback();
    for (const loop of this.loops) loop.wake();
  }
}
