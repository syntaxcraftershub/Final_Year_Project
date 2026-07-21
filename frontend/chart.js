/* Richer dependency-free canvas chart for the live drift view.
   Gradient area fills, shaded trigger zones, glowing head point, smooth
   redraw as points stream in. Fully offline (no CDN). Theme-aware. */

class LiveChart {
  constructor(canvas) {
    this.c = canvas;
    this.ctx = canvas.getContext("2d");
    this.dpr = window.devicePixelRatio || 1;
    this.pad = { l: 44, r: 18, t: 18, b: 30 };
    this.drift = [];
    this.delta = [];
    this.total = 2;
    this.driftHigh = 0.62;
    this.deltaSpike = 0.55;
    this._css = getComputedStyle(document.documentElement);
    this._resize();
    window.addEventListener("resize", () => this._resize());
  }

  _c(name, fallback) {
    const v = this._css.getPropertyValue(name).trim();
    return v || fallback;
  }

  _resize() {
    this._css = getComputedStyle(document.documentElement);
    const cssW = this.c.clientWidth || 1000;
    const cssH = 380;
    this.c.width = cssW * this.dpr;
    this.c.height = cssH * this.dpr;
    this.w = cssW; this.h = cssH;
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    this.draw();
  }

  reset(total, driftHigh, deltaSpike) {
    this.drift = []; this.delta = [];
    this.total = Math.max(2, total || 2);
    this.driftHigh = driftHigh; this.deltaSpike = deltaSpike;
    this.draw();
  }

  push(cumDrift, delta) { this.drift.push(cumDrift); this.delta.push(delta); this.draw(); }

  _x(i) {
    const span = Math.max(1, this.total - 1);
    return this.pad.l + (i / span) * (this.w - this.pad.l - this.pad.r);
  }
  _y(v) {
    const vv = Math.max(0, Math.min(1, v));
    return this.pad.t + (1 - vv) * (this.h - this.pad.t - this.pad.b);
  }

  draw() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.w, this.h);
    const grid = this._c("--border", "#e4e7f0");
    const muted = this._c("--muted", "#6b7280");
    const drift = this._c("--drift", "#3b82f6");
    const delta = this._c("--delta", "#ec4899");
    const thr = this._c("--thr", "#94a3b8");

    // trigger zones (subtle shading above thresholds)
    this._zone(this.deltaSpike, delta, 0.05);
    this._zone(this.driftHigh, drift, 0.05);

    // grid + y labels
    ctx.strokeStyle = grid; ctx.fillStyle = muted; ctx.lineWidth = 1;
    ctx.font = "11px Inter, Segoe UI, sans-serif";
    for (let g = 0; g <= 1.0001; g += 0.25) {
      const y = this._y(g);
      ctx.globalAlpha = 0.6; ctx.beginPath();
      ctx.moveTo(this.pad.l, y); ctx.lineTo(this.w - this.pad.r, y); ctx.stroke();
      ctx.globalAlpha = 1; ctx.fillText(g.toFixed(2), 8, y + 3);
    }

    // threshold dashed lines
    this._dashed(this.driftHigh, thr);
    this._dashed(this.deltaSpike, thr);

    // area + line series
    this._series(this.drift, drift, true);
    this._series(this.delta, delta, false);

    // x label
    ctx.fillStyle = muted;
    ctx.fillText("step →", this.w - this.pad.r - 40, this.h - 9);
  }

  _zone(v, color, alpha) {
    const ctx = this.ctx;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.fillStyle = color;
    ctx.fillRect(this.pad.l, this.pad.t, this.w - this.pad.l - this.pad.r, this._y(v) - this.pad.t);
    ctx.restore();
  }

  _dashed(v, color) {
    const ctx = this.ctx;
    const y = this._y(v);
    ctx.save();
    ctx.setLineDash([4, 6]); ctx.strokeStyle = color; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(this.pad.l, y); ctx.lineTo(this.w - this.pad.r, y); ctx.stroke();
    ctx.restore();
  }

  _series(arr, color, fill) {
    if (!arr.length) return;
    const ctx = this.ctx;

    if (fill) {
      const grad = ctx.createLinearGradient(0, this.pad.t, 0, this.h - this.pad.b);
      grad.addColorStop(0, this._hexA(color, 0.28));
      grad.addColorStop(1, this._hexA(color, 0.02));
      ctx.beginPath();
      ctx.moveTo(this._x(0), this.h - this.pad.b);
      arr.forEach((v, i) => ctx.lineTo(this._x(i), this._y(v)));
      ctx.lineTo(this._x(arr.length - 1), this.h - this.pad.b);
      ctx.closePath(); ctx.fillStyle = grad; ctx.fill();
    }

    ctx.strokeStyle = color; ctx.lineWidth = 2.4; ctx.lineJoin = "round";
    ctx.beginPath();
    arr.forEach((v, i) => { const x = this._x(i), y = this._y(v); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();

    // points
    ctx.fillStyle = color;
    arr.forEach((v, i) => { ctx.beginPath(); ctx.arc(this._x(i), this._y(v), 2.6, 0, 7); ctx.fill(); });

    // glowing head
    const i = arr.length - 1;
    const hx = this._x(i), hy = this._y(arr[i]);
    ctx.save();
    ctx.shadowColor = color; ctx.shadowBlur = 14;
    ctx.fillStyle = color; ctx.beginPath(); ctx.arc(hx, hy, 4.5, 0, 7); ctx.fill();
    ctx.restore();
    ctx.strokeStyle = this._c("--panel", "#fff"); ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(hx, hy, 4.5, 0, 7); ctx.stroke();
  }

  _hexA(hex, a) {
    // accepts #rrggbb; falls back to rgba wrapper for named/other
    const h = hex.replace("#", "");
    if (h.length === 6) {
      const r = parseInt(h.slice(0, 2), 16), g = parseInt(h.slice(2, 4), 16), b = parseInt(h.slice(4, 6), 16);
      return `rgba(${r},${g},${b},${a})`;
    }
    return hex;
  }
}

window.LiveChart = LiveChart;
