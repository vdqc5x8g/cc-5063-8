(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

  const state = {
    yawRad: 35 * Math.PI / 180,
    server: null,
    data: null,
    mode: "B",
    att: { offset: 1.7, lateral: 0, scale: 0.24, angleDeg: 0 },
    picks: [],
    pickMode: false,
    anchors: [],
    anchorMode: false,
    branchStrategy: "auto",
    hoverIndex: null,
    camera: { yaw: 0.52, pitch: 0.30, zoom: 170, target: [0.05, 0.16, -0.05] },
    patternView: { unit: 80 },
    viewDrag: null,
    patternDrag: null,
  };
  let projectTimer = null;
  let yawTimer = null;

  const SURFACE_COLORS = {
    spout: "#2563eb",
    body: "#ea580c",
    plane: "#16a34a",
    curved: "#7c3aed",
    foldback: "#0891b2",
    sphere: "#db2777",
  };
  const colorFor = (sid) => SURFACE_COLORS[sid] || "#475569";

  function hexRgba(hex, alpha) {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${alpha})`;
  }

  async function request(path, body, method = "POST") {
    const options = { method, headers: {} };
    if (body !== undefined) {
      options.headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(body);
    }
    const res = await fetch(path, options);
    const text = await res.text();
    let data;
    try { data = JSON.parse(text); } catch { throw new Error(text.slice(0, 300)); }
    if (!res.ok || data.error) {
      throw new Error(data.message ? `${data.error}: ${data.message}` : (data.error || `HTTP ${res.status}`));
    }
    return data;
  }

  function setStatus(text, isError = false) {
    const el = $("viewStatus");
    el.textContent = text;
    el.classList.toggle("error", isError);
  }

  function setupCanvas(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    const pw = Math.max(1, Math.round(w * dpr));
    const ph = Math.max(1, Math.round(h * dpr));
    if (canvas.width !== pw || canvas.height !== ph) {
      canvas.width = pw;
      canvas.height = ph;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx, w, h };
  }

  function projectPoint(p, cam, w, h) {
    let x = p[0] - cam.target[0];
    let y = p[1] - cam.target[1];
    let z = p[2] - cam.target[2];
    const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
    const x1 = cy * x + sy * z;
    const z1 = -sy * x + cy * z;
    const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    const y2 = cp * y - sp * z1;
    const z2 = sp * y + cp * z1;
    return {
      x: w / 2 + cam.zoom * x1,
      y: h / 2 - cam.zoom * y2,
      depth: z2,
    };
  }

  function screenToRay(sx, sy, w, h) {
    const cam = state.camera;
    const lx = (sx - w / 2) / cam.zoom;
    const ly = -(sy - h / 2) / cam.zoom;
    function unproject(x, y, z) {
      const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
      const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
      const wy = cp * y + sp * z;
      const z1 = -sp * y + cp * z;
      const wx = cy * x - sy * z1;
      const wz = sy * x + cy * z1;
      return [wx + cam.target[0], wy + cam.target[1], wz + cam.target[2]];
    }
    const origin = unproject(lx, ly, 8);
    const through = unproject(lx, ly, -3);
    const direction = through.map((v, i) => v - origin[i]);
    const norm = Math.hypot(direction[0], direction[1], direction[2]) || 1;
    return { origin, direction: direction.map((v) => v / norm) };
  }

  function drawScreenPolyline(ctx, points, options = {}) {
    if (points.length < 2) return;
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y);
    if (options.close) ctx.closePath();
    if (options.fill) {
      ctx.fillStyle = options.fill;
      ctx.fill();
    }
    if (options.stroke) {
      ctx.strokeStyle = options.stroke;
      ctx.lineWidth = options.width || 1;
      ctx.setLineDash(options.dash || []);
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  function drawWorldPolyline(ctx, xyzList, proj, options) {
    const pts = xyzList.filter(Boolean).map(proj);
    drawScreenPolyline(ctx, pts, options);
  }

  function drawMarker(ctx, p, color, text) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#ffffff";
    ctx.stroke();
    if (text) {
      ctx.font = "700 12px ui-monospace, monospace";
      const width = ctx.measureText(text).width + 10;
      ctx.fillStyle = "rgba(15,23,42,.82)";
      ctx.fillRect(p.x + 8, p.y - 18, width, 18);
      ctx.fillStyle = "#fff";
      ctx.fillText(text, p.x + 13, p.y - 5);
    }
  }

  function drawGround(ctx, proj) {
    const y = -0.96;
    ctx.lineWidth = 1;
    ctx.strokeStyle = "rgba(100,116,139,.22)";
    for (let x = -2; x <= 3.001; x += 0.5) {
      const a = proj([x, y, -2]);
      const b = proj([x, y, 2]);
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    }
    for (let z = -2; z <= 2.001; z += 0.5) {
      const a = proj([-2, y, z]);
      const b = proj([3, y, z]);
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    }
  }

  function drawBranchOverlay(ctx, proj, projection) {
    const tracks = projection.tracks || {};
    const hover = state.hoverIndex;
    for (const sid of Object.keys(tracks)) {
      const track = tracks[sid];
      for (const frame of track.frames) {
        const p = projection.points[frame.index];
        if (!p || p.surfaceId !== sid || !p.xyz) continue;
        const chosen = proj(p.xyz);

        // Abandoned candidates: faint hollow ring at their world position.
        if (hover === frame.index) {
          for (const alt of frame.abandoned || []) {
            if (!alt.xyz) continue;
            const q = proj(alt.xyz);
            ctx.beginPath();
            ctx.arc(q.x, q.y, 3.6, 0, Math.PI * 2);
            ctx.strokeStyle = "rgba(100,116,139,.85)";
            ctx.lineWidth = 1.2;
            ctx.setLineDash([2, 2]);
            ctx.stroke();
            ctx.setLineDash([]);
          }
        }

        if (frame.status === "anchored" || frame.locked) {
          ctx.beginPath();
          ctx.arc(chosen.x, chosen.y, 5.2, 0, Math.PI * 2);
          ctx.fillStyle = "rgba(234,179,8,.92)";
          ctx.fill();
          ctx.lineWidth = 1.6;
          ctx.strokeStyle = "#a16207";
          ctx.stroke();
        }
        if (frame.recovery) {
          ctx.beginPath();
          ctx.arc(chosen.x, chosen.y, 6, 0, Math.PI * 2);
          ctx.strokeStyle = "#0891b2";
          ctx.lineWidth = 2;
          ctx.stroke();
        }
        if (frame.status === "branch-switch" || frame.status === "break") {
          const q = chosen;
          ctx.strokeStyle = frame.status === "break" ? "#b91c1c" : "#7c3aed";
          ctx.lineWidth = 2.4;
          ctx.beginPath();
          ctx.moveTo(q.x - 6, q.y - 6); ctx.lineTo(q.x + 6, q.y + 6);
          ctx.moveTo(q.x + 6, q.y - 6); ctx.lineTo(q.x - 6, q.y + 6);
          ctx.stroke();
        }
        if (hover === frame.index) {
          ctx.beginPath();
          ctx.arc(chosen.x, chosen.y, 7.5, 0, Math.PI * 2);
          ctx.strokeStyle = "#0f172a";
          ctx.lineWidth = 1.4;
          ctx.stroke();
        }
      }
    }

    for (const anchor of state.anchors) {
      if (!anchor.xyz) continue;
      const q = proj(anchor.xyz);
      ctx.save();
      ctx.translate(q.x, q.y);
      ctx.beginPath();
      ctx.moveTo(0, -8); ctx.lineTo(7, 0); ctx.lineTo(0, 8); ctx.lineTo(-7, 0);
      ctx.closePath();
      ctx.fillStyle = "#eab308";
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.restore();
    }
  }

  function renderView() {
    const canvas = $("view3d");
    const { ctx, w, h } = setupCanvas(canvas);
    if (!state.server) {
      ctx.fillStyle = "#64748b";
      ctx.font = "15px sans-serif";
      ctx.fillText("正在加载真实曲面网格…", 28, 40);
      return;
    }
    const proj = (p) => projectPoint(p, state.camera, w, h);
    drawGround(ctx, proj);

    const drawables = [];
    for (const surf of state.server.surfaces) {
      const positions = surf.mesh.positions;
      for (const quad of surf.mesh.indices) {
        const sp = quad.map((idx) => proj(positions[idx]));
        drawables.push({
          sid: surf.id,
          sp,
          depth: sp.reduce((a, p) => a + p.depth, 0) / 4,
        });
      }
    }
    drawables.sort((a, b) => a.depth - b.depth);
    for (const d of drawables) {
      ctx.beginPath();
      ctx.moveTo(d.sp[0].x, d.sp[0].y);
      for (let i = 1; i < 4; i++) ctx.lineTo(d.sp[i].x, d.sp[i].y);
      ctx.closePath();
      ctx.fillStyle = hexRgba(colorFor(d.sid), 0.105);
      ctx.fill();
      ctx.strokeStyle = hexRgba(colorFor(d.sid), 0.24);
      ctx.lineWidth = 0.55;
      ctx.stroke();
    }

    const design = state.server.designPoints || [];
    drawWorldPolyline(ctx, design, proj, {
      stroke: "rgba(15,23,42,.62)", width: 1.5, dash: [7, 5],
    });

    for (const pick of state.picks) {
      if (pick.xyz) {
        const p = proj(pick.xyz);
        drawMarker(ctx, p, "#eab308", pick.surfaceId);
      }
    }
    if (state.picks.length >= 2) {
      drawWorldPolyline(ctx, state.picks.map((p) => p.xyz).filter(Boolean), proj, {
        stroke: "#a16207", width: 2, dash: [4, 4],
      });
    }

    const projection = state.data ? state.data.projection : null;
    if (projection) {
      for (const iv of projection.intervals) {
        const xyz = [];
        if (iv.xyzStart) xyz.push(iv.xyzStart);
        for (let j = iv.index0 + 1; j < iv.index1; j++) {
          if (projection.points[j] && projection.points[j].xyz) {
            xyz.push(projection.points[j].xyz);
          }
        }
        if (iv.xyzEnd) xyz.push(iv.xyzEnd);
        drawWorldPolyline(ctx, xyz, proj, {
          stroke: colorFor(iv.surfaceId), width: 2.4,
        });
      }

      for (const p of projection.points) {
        if (p.status !== "oob" || !p.xyz) continue;
        const q = proj(p.xyz);
        ctx.strokeStyle = "#dc2626";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(q.x - 5, q.y - 5); ctx.lineTo(q.x + 5, q.y + 5);
        ctx.moveTo(q.x + 5, q.y - 5); ctx.lineTo(q.x - 5, q.y + 5);
        ctx.stroke();
      }

      // Branch tracking: abandoned candidates, locked anchors, and breaks.
      drawBranchOverlay(ctx, proj, projection);

      for (const run of state.data.decal.runs) {
        drawWorldPolyline(ctx, run.xyz, proj, {
          stroke: hexRgba(colorFor(run.surfaceId), 0.96), width: 4.4,
        });
      }

      for (const letter of state.data.decal.letters) {
        if (!letter.xyz || letter.status !== "attached") continue;
        const p = proj(letter.xyz);
        ctx.font = "900 15px ui-monospace, monospace";
        const tw = ctx.measureText(letter.char).width + 8;
        ctx.fillStyle = hexRgba(colorFor(letter.surfaceId), 0.92);
        ctx.fillRect(p.x + 7, p.y - 20, tw, 19);
        ctx.fillStyle = "#fff";
        ctx.fillText(letter.char, p.x + 11, p.y - 6);
      }
    }
  }

  function renderPattern() {
    const canvas = $("patternCanvas");
    const { ctx, w, h } = setupCanvas(canvas);
    if (!state.server) return;
    const pattern = state.server.pattern;
    const xmin = -0.48, xmax = pattern.width + 0.48;
    const ymin = -0.36, ymax = 1.36;
    const unit = Math.min((w - 24) / (xmax - xmin), (h - 18) / (ymax - ymin));
    state.patternView.unit = unit;
    const X = (x) => w / 2 + unit * (x - (xmin + xmax) / 2);
    const Y = (y) => h / 2 - unit * (y - (ymin + ymax) / 2);

    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 1;
    for (let gy = -0.2; gy <= 1.21; gy += 0.2) {
      ctx.beginPath(); ctx.moveTo(X(xmin), Y(gy)); ctx.lineTo(X(xmax), Y(gy)); ctx.stroke();
    }

    for (const stroke of pattern.strokes) {
      const pts = stroke.points.map(([x, y]) => ({ x: X(x), y: Y(y) }));
      drawScreenPolyline(ctx, pts, {
        stroke: stroke.char ? "#cbd5e1" : "#94a3b8",
        width: stroke.char ? 2.1 : 1.5,
        dash: stroke.char ? [] : [7, 5],
      });
    }

    if (state.data) {
      const decal = state.data.decal;
      for (const rec of decal.clipRecords) {
        ctx.beginPath();
        ctx.arc(X(rec.x), Y(rec.y), 2.1, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(220,38,38,.42)";
        ctx.fill();
      }
      for (const run of decal.runs) {
        const pts = run.points2d.map(([x, y]) => ({ x: X(x), y: Y(y) }));
        drawScreenPolyline(ctx, pts, {
          stroke: colorFor(run.surfaceId), width: 4.2,
        });
      }

      const seam = decal.seam;
      if (seam && seam.hasSeam) {
        const att = decal.attachment;
        const ca = Math.cos(att.angleDeg * Math.PI / 180);
        const sa = Math.sin(att.angleDeg * Math.PI / 180);
        const xr = (seam.seamArc - att.offset) / Math.max(att.scale, 1e-9);
        const line = [];
        for (const yr of [-0.86, 0.86]) {
          const x0 = xr * ca + yr * sa;
          const y0 = -xr * sa + yr * ca;
          line.push({ x: X(pattern.width / 2 + x0), y: Y(y0 + 0.5) });
        }
        drawScreenPolyline(ctx, line, { stroke: "#111827", width: 2.2, dash: [10, 6, 2, 6] });
      }
    }

    ctx.fillStyle = "#64748b";
    ctx.font = "12px sans-serif";
    ctx.fillText("拖动此图样：横向 offset，纵向 lateral", 14, h - 12);
  }

  function heatColor(t) {
    const hue = 218 - clamp(t, 0, 1) * 218;
    return `hsl(${hue},82%,52%)`;
  }

  function renderDistortion() {
    const canvas = $("distCanvas");
    const { ctx, w, h } = setupCanvas(canvas);
    if (!state.server) return;
    const select = $("distSurface");
    const sid = select.value || state.server.surfaces[0].id;
    const surf = state.server.surfaces.find((s) => s.id === sid) || state.server.surfaces[0];
    const grid = surf.distortion;
    const ml = 48, mr = 16, mt = 16, mb = 38;
    const pw = w - ml - mr, ph = h - mt - mb;
    const nx = grid.us.length, ny = grid.vs.length;
    const x = (u) => ml + (u - grid.uDomain[0]) / (grid.uDomain[1] - grid.uDomain[0] || 1) * pw;
    const y = (v) => mt + ph - (v - grid.vDomain[0]) / (grid.vDomain[1] - grid.vDomain[0] || 1) * ph;

    const range = grid.areaMax - grid.areaMin;
    for (let i = 0; i < nx - 1; i++) {
      for (let j = 0; j < ny - 1; j++) {
        const t = range < 1e-12 ? 0.5 : (grid.area[i][j] - grid.areaMin) / range;
        ctx.fillStyle = heatColor(t);
        ctx.fillRect(x(grid.us[i]), y(grid.vs[j + 1]),
          x(grid.us[i + 1]) - x(grid.us[i]) + 0.6,
          y(grid.vs[j]) - y(grid.vs[j + 1]) + 0.6);
      }
    }
    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 1.2;
    ctx.strokeRect(ml, mt, pw, ph);
    ctx.fillStyle = "#334155";
    ctx.font = "11.5px ui-monospace, monospace";
    ctx.fillText(`u ${grid.uDomain[0].toFixed(2)} → ${grid.uDomain[1].toFixed(2)}`, ml, h - 12);
    ctx.save();
    ctx.translate(14, mt + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(`v ${grid.vDomain[0].toFixed(2)} → ${grid.vDomain[1].toFixed(2)}`, -55, 0);
    ctx.restore();

    const dec = state.data ? state.data.decal.distortion : null;
    $("distStats").innerHTML = `
      <h3>${surf.label}</h3>
      <table class="kv">
        <tr><td>面积尺度 min</td><td>${grid.areaMin.toFixed(4)}</td></tr>
        <tr><td>max</td><td>${grid.areaMax.toFixed(4)}</td></tr>
        <tr><td>mean</td><td>${grid.areaMean.toFixed(4)}</td></tr>
        <tr><td>median</td><td>${grid.areaMedian.toFixed(4)}</td></tr>
      </table>
      <h3>贴花有限差分畸变</h3>
      ${dec && dec.available ? `<table class="kv">
        <tr><td>字母面积比</td><td>${dec.areaRatioMin.toFixed(3)}–${dec.areaRatioMax.toFixed(3)}</td></tr>
        <tr><td>平均面积比</td><td>${dec.areaRatioMean.toFixed(3)}</td></tr>
        <tr><td>最大各向异性</td><td>${dec.anisotropyMax.toFixed(3)}</td></tr>
      </table>` : `<p class="tiny">运行投影后显示。</p>`}
      <p class="tiny">热力值为参数度量面积 √(EG−F²)；不是显示网格顶点误差。</p>
    `;
  }

  function fmt(value, digits = 3) {
    return value === null || value === undefined ? "—" : Number(value).toFixed(digits);
  }

  function renderReport() {
    const root = $("report");
    if (!state.data) {
      root.innerHTML = "<h2>求解与裁切报告</h2><p class='tiny'>等待运行真实投影…</p>";
      return;
    }
    const { projection: p, decal } = state.data;
    const seam = decal.seam || {};
    const clipEntries = Object.entries(decal.clipCounts || {});
    const warnings = p.warnings || [];
    root.innerHTML = `
      <h2>求解与裁切报告 · ${state.mode}</h2>
      <div class="report-section chips">
        <span class="badge-pass">${p.intervals.length} 个连续区间</span>
        <span class="badge-pass">${p.switches.length} 次授权换片</span>
        <span class="${p.tieHolds.length ? "badge-fail" : "badge-pass"}">${p.tieHolds.length} 次等距保持</span>
        <span class="${clipEntries.length ? "badge-fail" : "badge-pass"}">${clipEntries.reduce((a, x) => a + x[1], 0)} 个裁切样本</span>
      </div>

      <div class="report-section">
        <h3>字母落点</h3>
        <div class="chips">
          ${decal.letters.map((l) => `<span class="letter-chip ${l.status === "clipped" ? "clipped" : ""}"
            style="${l.status === "attached" ? `background:${colorFor(l.surfaceId)}` : ""}"
            title="${l.status === "clipped" ? l.reason : l.surfaceId}">${l.char}</span>`).join("")}
        </div>
        <table class="kv">
          ${Object.entries(decal.faceLetters).map(([sid, chars]) =>
            `<tr><td><span style="color:${colorFor(sid)};font-weight:800">●</span> ${sid}</td><td class="mono">${chars.join("")}</td></tr>`).join("")}
        </table>
      </div>

      ${seam.hasSeam ? `<div class="report-section">
        <h3>接缝对读/注册</h3>
        <span class="${seam.readable ? "badge-pass" : "badge-fail"}">${seam.readable ? "READABLE" : "NOT READABLE"}</span>
        <table class="kv">
          <tr><td>物理间隙</td><td>${fmt(seam.physicalGap, 4)}</td></tr>
          <tr><td>切向夹角</td><td>${fmt(seam.tangentAngleDeg, 2)}°</td></tr>
          <tr><td>侧架夹角</td><td>${fmt(seam.sideAngleDeg, 2)}°</td></tr>
          <tr><td>注册跨距</td><td>${fmt(seam.registrationGap, 4)}</td></tr>
        </table>
      </div>` : ""}

      <div class="report-section">
        <h3>曲面身份与连续区间</h3>
        <table class="kv">
          ${p.intervals.map((iv) => `<tr>
            <td><strong style="color:${colorFor(iv.surfaceId)}">${iv.surfaceId}</strong><br>
            <span class="muted">${iv.index0}–${iv.index1}</span></td>
            <td>arc ${fmt(iv.arc0, 2)}→${fmt(iv.arc1, 2)}<br>
            <span class="${iv.endReason ? "" : "muted"}">${iv.endReason || "open interval"}</span></td>
          </tr>`).join("")}
        </table>
      </div>

      <div class="report-section">
        <h3>裁切原因（反求器 → 裁切器）</h3>
        ${clipEntries.length ? `<table class="kv">
          ${clipEntries.map(([reason, count]) => `<tr><td class="mono">${reason}</td><td>×${count}</td></tr>`).join("")}
        </table>` : `<p class="tiny" style="color:var(--green)">无裁切，所有密化笔画均在有效参数域内。</p>`}
      </div>

      <div class="report-section">
        <h3>等距、警告与受限搜索</h3>
        <p class="tiny">等距记录：${p.tieHolds.length}；规则：最近距离相等时保持当前曲面身份。</p>
        ${warnings.length ? `<ul class="warning-list">${warnings.slice(0, 8).map((w) => `<li>${w}</li>`).join("")}</ul>`
          : `<p class="tiny muted">无警告。</p>`}
      </div>
    `;
  }

  function renderPicks() {
    $("transitionList").innerHTML = state.picks.map((p, i) => `
      <div class="pick-chip ${p.surfaceId}">
        <span>${i + 1}. ${p.surfaceId}</span>
        <span>u ${p.u.toFixed(3)} · v ${p.v.toFixed(3)}</span>
      </div>`).join("") || `<p class="tiny">尚未设置转接线拾取点。</p>`;
  }

  function renderAnchors() {
    const root = $("anchorList");
    if (!root) return;
    root.innerHTML = state.anchors.length ? state.anchors.map((a, i) => `
      <div class="pick-chip ${a.surfaceId} anchor-chip">
        <span>锚点 ${i + 1} · ${a.surfaceId} @ #${a.index}</span>
        <span>u ${a.u.toFixed(3)} · v ${a.v.toFixed(3)}</span>
      </div>`).join("") : "";
  }

  const STRATEGY_LABELS = {
    auto: "自动（实测择优）",
    neighbor: "邻域延续",
    path: "路径约束",
  };

  function renderBranchReport() {
    const root = $("branchReport");
    if (!root) return;
    const projection = state.data ? state.data.projection : null;
    const summary = projection ? projection.branchSummary : null;
    if (!summary) {
      root.innerHTML = `<h2>分支跟踪</h2>
        <p class="tiny">运行投影后显示：候选数、被放弃的候选、选择依据与断点。</p>`;
      return;
    }
    const sids = Object.keys(summary.surfaces || {});
    const rows = sids.map((sid) => {
      const s = summary.surfaces[sid];
      const stats = s.stats || {};
      const err = s.attachmentError || {};
      const breaks = s.breaks || [];
      const comp = s.strategyComparison || {};
      const totals = comp.totals || {};
      const chosen = s.chosenStrategy || summary.strategy;
      return `
        <tr class="branch-surface-row">
          <td><strong style="color:${colorFor(sid)}">${sid}</strong><br>
            <span class="muted">${STRATEGY_LABELS[chosen] || chosen}</span></td>
          <td class="mono">
            候选 ${stats.ambiguousSamples || 0}/${stats.samples || 0} 样本多解<br>
            换支 ${stats.branchSwitches || 0} · 恢复 ${stats.recoveries || 0} · 锁定 ${stats.lockedSamples || 0}<br>
            断点 ${breaks.length}
          </td>
          <td class="mono">
            内点正交 ${fmt(err.maxInteriorOrthoRel !== undefined ? err.maxInteriorOrthoRel : err.maxOrthoRel, 2)}e0<br>
            重投影 ≤ ${fmt(err.maxSurfaceReprojectionError, 2)}
            ${err.onSurfaceSamples ? `<br><span class="muted">贴面样本 ${err.onSurfaceSamples}</span>` : ""}
            ${err.boundarySamples ? `<br><span class="muted">边界极小 ${err.boundarySamples}</span>` : ""}
          </td>
        </tr>
        <tr class="branch-basis-row">
          <td colspan="3">
            <span class="muted">选择依据（实测比较）：</span>
            ${Object.entries(totals).map(([name, t]) =>
              `<span class="strategy-chip ${name === chosen ? "chosen" : ""}"
                title="总代价 = 距离 + 连续性 + 切换惩罚">${STRATEGY_LABELS[name] || name}
                ${typeof t === "object" ? fmt(t.total, 3) : fmt(t, 3)}</span>`).join("")}
            ${breaks.length ? `<span class="badge-fail">${breaks.length} 处不连续（未用平滑掩盖）</span>` : `<span class="badge-pass">连续</span>`}
          </td>
        </tr>
        ${breaks.map((b) => `<tr class="branch-break-row">
          <td colspan="3">
            <span class="badge-fail">断点 @ #${b.afterIndex}</span>
            <span class="muted">${b.kind} · xyz 跳变 ${fmt(b.xyzJump, 3)} · 参数跳变 ${fmt(b.paramJump, 3)}</span>
          </td></tr>`).join("")}
      `;
    }).join("");

    root.innerHTML = `
      <h2>分支跟踪</h2>
      <p class="tiny">策略 ${STRATEGY_LABELS[summary.strategy] || summary.strategy} ·
        总断点 ${summary.totalBreaks} · 多解样本 ${Object.values(summary.ambiguousSamples || {}).reduce((a, b) => a + b, 0)}。
        悬停曲面轨迹可查看被放弃的候选。</p>
      <table class="kv branch-table">${rows}</table>
      ${state.anchors.length ? `<p class="tiny">当前 ${state.anchors.length} 个分支锚点参与锁定。</p>` : ""}
    `;
  }

  function renderExportLinks(urls) {
    if (!urls) return;
    const stamp = Date.now();
    const labels = { svg: "SVG 图样", json: "JSON", csv: "笔画 CSV", routeCsv: "轨迹 CSV" };
    let html = "";
    for (const edit of ["A", "B"]) {
      for (const kind of ["svg", "json", "csv", "routeCsv"]) {
        html += `<a href="${urls[edit][kind]}?t=${stamp}" target="_blank" rel="noopener">${edit} · ${labels[kind]}</a>`;
      }
    }
    html += `<a href="${urls.reportJson}?t=${stamp}" target="_blank" rel="noopener">report.json</a>`;
    $("exportLinks").innerHTML = html;
  }

  function updateHover(event) {
    const projection = state.data ? state.data.projection : null;
    if (!projection || !state.server) return;
    const canvas = $("view3d");
    const rect = canvas.getBoundingClientRect();
    const w = rect.width, h = rect.height;
    const sx = event.clientX - rect.left;
    const sy = event.clientY - rect.top;
    let best = null;
    for (const p of projection.points) {
      if (!p.xyz || p.status === "oob") continue;
      const q = projectPoint(p.xyz, state.camera, w, h);
      const d = Math.hypot(q.x - sx, q.y - sy);
      if (d < 14 && (!best || d < best.d)) best = { d, index: p.index };
    }
    const next = best ? best.index : null;
    if (next !== state.hoverIndex) {
      state.hoverIndex = next;
      renderView();
      renderHoverInfo();
    }
  }

  function renderHoverInfo() {
    updateBranchOverlay();
  }

  function hoverHtml(projection) {
    if (state.hoverIndex === null) return "";
    const p = projection.points[state.hoverIndex];
    if (!p || !p.branch) return "";
    const b = p.branch;
    const alts = (b.abandoned || []).map((c) =>
      `<span class="cand-chip">u ${c.u.toFixed(3)}, v ${c.v.toFixed(3)} · d ${c.distance.toFixed(4)}${c.oobReason ? ` · ${c.oobReason}` : ""}</span>`
    ).join("");
    return `
      <span class="overlay-line strong">
        样本 #${p.index} · ${b.surfaceId} · ${STRATEGY_LABELS[b.strategy] || b.strategy}
      </span>
      <span class="overlay-line">
        采用 u ${p.u.toFixed(3)}, v ${p.v.toFixed(3)} · 状态 ${b.status}
        ${b.anchored ? " · 锚定" : ""}${b.locked ? " · 锁定" : ""}
        ${b.reason ? ` · ${b.reason}` : ""}
      </span>
      ${alts ? `<span class="overlay-line subdued">放弃候选 ${b.abandoned.length} 个：${alts}</span>` : ""}
    `;
  }

  function renderAll() {
    renderView();
    renderPattern();
    renderDistortion();
    renderReport();
    renderBranchReport();
    renderPicks();
    renderAnchors();
    updateBranchOverlay();
  }

  function updateBranchOverlay() {
    const el = $("branchOverlay");
    if (!el) return;
    const projection = state.data ? state.data.projection : null;
    if (!projection) {
      el.classList.add("hidden");
      return;
    }
    el.classList.remove("hidden");
    const summary = projection.branchSummary || {};
    const totalBreaks = summary.totalBreaks || 0;
    const ambiguous = Object.values(summary.ambiguousSamples || {}).reduce((a, b) => a + b, 0);
    const anomalies = [];
    for (const [sid, s] of Object.entries(summary.surfaces || {})) {
      const err = s.attachmentError || {};
      if ((s.breaks || []).length) anomalies.push(`${sid} 轨迹断点 ×${s.breaks.length}`);
      // Only interior minima must satisfy the normal condition; a verified
      // boundary minimum legitimately leaves a residual along its free direction.
      const interiorOrtho = err.maxInteriorOrthoRel !== undefined
        ? err.maxInteriorOrthoRel : (err.maxOrthoRel || 0);
      if (interiorOrtho > 1e-4) anomalies.push(`${sid} 内点正交残差 ${interiorOrtho.toExponential(1)}`);
      const stats = s.stats || {};
      if ((stats.recoveries || 0) > 0) anomalies.push(`${sid} 丢失后恢复 ×${stats.recoveries}`);
    }
    el.innerHTML = `
      <span class="overlay-line ${totalBreaks ? "bad" : "ok"}">
        分支跟踪 · ${STRATEGY_LABELS[projection.strategy] || projection.strategy} ·
        多解样本 ${ambiguous} · 断点 ${totalBreaks}
      </span>
      ${anomalies.length ? `<span class="overlay-line bad">${anomalies.join("；")}</span>` : ""}
      ${hoverHtml(projection)}
    `;
    el.classList.toggle("with-hover", state.hoverIndex !== null);
  }

  function syncControlValues() {
    const a = state.att;
    $("offsetSlider").value = a.offset;
    $("offsetValue").textContent = a.offset.toFixed(2);
    $("lateralSlider").value = a.lateral;
    $("lateralValue").textContent = a.lateral.toFixed(2);
    $("scaleSlider").value = a.scale;
    $("scaleValue").textContent = a.scale.toFixed(2);
    $("angleSlider").value = a.angleDeg;
    $("angleValue").textContent = `${Math.round(a.angleDeg)}°`;
  }

  async function loadState() {
    const data = await request(`/api/state?yaw=${state.yawRad}`, undefined, "GET");
    state.server = data;
    state.yawRad = data.yawRad;
    state.camera = { ...state.camera, ...data.camera };
    state.att = {
      offset: data.attachment.offset,
      lateral: data.attachment.lateral,
      scale: data.attachment.scale,
      angleDeg: data.attachment.angleDeg,
    };
    state.picks = data.transition.map((p) => ({ ...p }));
    $("yawSlider").value = String(Math.round(data.yawDeg));
    $("yawValue").textContent = `${data.yawDeg.toFixed(1)}°`;
    $("distSurface").innerHTML = data.surfaces
      .map((s) => `<option value="${s.id}">${s.label}</option>`).join("");
    syncControlValues();
    renderPicks();
    renderAll();
  }

  async function runProject(mode) {
    if (mode) state.mode = mode;
    clearTimeout(projectTimer);
    setStatus("正在运行解析最近点求解与贴附…");
    try {
      const payload = await request("/api/project", {
        yawRad: state.yawRad,
        mode: state.mode === "A" ? "spout_only" : "transition",
        attachment: state.att,
        transitionUv: state.picks.map((p) => ({ surfaceId: p.surfaceId, u: p.u, v: p.v })),
        branchStrategy: state.branchStrategy,
        branchAnchors: state.anchors.map((a) => ({
          surfaceId: a.surfaceId, index: a.index, u: a.u, v: a.v,
        })),
        usePreset: false,
      });
      state.data = payload;
      state.att = {
        offset: payload.attachment.offset,
        lateral: payload.attachment.lateral,
        scale: payload.attachment.scale,
        angleDeg: payload.attachment.angleDeg,
      };
      syncControlValues();
      renderAll();
      const intervals = payload.projection.intervals.length;
      setStatus(`已连接真实求解器 · ${intervals} 区间 · ${payload.decal.runs.length} 条曲面笔画`);
    } catch (err) {
      setStatus(`求解失败：${err.message}`, true);
    }
  }

  async function handleCanvasPick(event) {
    const canvas = $("view3d");
    const rect = canvas.getBoundingClientRect();
    const sx = event.clientX - rect.left;
    const sy = event.clientY - rect.top;
    const ray = screenToRay(sx, sy, rect.width, rect.height);
    const info = $("pickInfo");
    info.classList.remove("hidden");
    info.textContent = "正在进行网格种子拾取与解析 UV 复核…";
    try {
      const res = await request("/api/pick", {
        yawRad: state.yawRad,
        origin: ray.origin,
        direction: ray.direction,
      });
      if (!res.hit) {
        info.textContent = "射线未命中曲面，请调整视角后重试。";
        return;
      }
      const p = res.pick;
      state.picks.push({ surfaceId: p.surfaceId, u: p.u, v: p.v, xyz: p.xyz });
      renderPicks();
      renderView();
      info.innerHTML = `已拾取 <strong>${p.surfaceId}</strong>：u=${p.u.toFixed(4)}, v=${p.v.toFixed(4)}<br>
        ${p.source}${p.rayResidual !== undefined ? ` · ray residual ${p.rayResidual.toExponential(2)}` : ""}`;
      projectTimer = setTimeout(() => runProject("B"), 260);
    } catch (err) {
      info.textContent = `拾取失败：${err.message}`;
    }
  }

  async function handleAnchorPick(event) {
    const canvas = $("view3d");
    const rect = canvas.getBoundingClientRect();
    const ray = screenToRay(
      event.clientX - rect.left, event.clientY - rect.top, rect.width, rect.height
    );
    const info = $("pickInfo");
    info.classList.remove("hidden");
    info.textContent = "正在拾取分支锚点：复核 UV 与候选分支…";
    try {
      const res = await request("/api/pick", {
        yawRad: state.yawRad, origin: ray.origin, direction: ray.direction,
      });
      if (!res.hit) {
        info.textContent = "射线未命中曲面，无法放置锚点。";
        return;
      }
      const hit = res.pick;
      const projection = state.data ? state.data.projection : null;
      const track = projection && projection.tracks ? projection.tracks[hit.surfaceId] : null;
      let index = null;
      let branch = { u: hit.u, v: hit.v };
      let which = "拾取 UV";
      if (track && track.frames.length) {
        // Snap to the design sample whose candidate is nearest the clicked point;
        // among that sample's candidates prefer the one nearest the pick UV, so
        // the user can anchor an abandoned branch explicitly.
        let best = null;
        for (const frame of track.frames) {
          for (const cand of frame.candidates.concat([frame.chosen])) {
            if (!cand || !cand.xyz) continue;
            const d = Math.hypot(
              cand.xyz[0] - hit.xyz[0],
              cand.xyz[1] - hit.xyz[1],
              cand.xyz[2] - hit.xyz[2],
            );
            if (!best || d < best.d) best = { d, frame, cand };
          }
        }
        if (best) {
          index = best.frame.index;
          const usable = (c) => c.converged && !(c.oobReasons && c.oobReasons.length);
          const cands = best.frame.candidates.filter(usable);
          const pool = cands.length ? cands : [best.cand];
          const pickCand = pool.reduce((a, b) => {
            const da = Math.abs(a.u - hit.u) + Math.abs(a.v - hit.v);
            const db = Math.abs(b.u - hit.u) + Math.abs(b.v - hit.v);
            return db < da ? b : a;
          });
          branch = { u: pickCand.u, v: pickCand.v };
          which = best.frame.candidates.length > 1
            ? `多解样本（${best.frame.candidates.length} 个候选）`
            : "单候选样本";
        }
      }
      if (index === null) {
        // No projection yet: fall back to nearest design sample by world distance.
        const design = state.server.designPoints || [];
        let bi = 0, bd = Infinity;
        design.forEach((p, i) => {
          const d = Math.hypot(p[0] - hit.xyz[0], p[1] - hit.xyz[1], p[2] - hit.xyz[2]);
          if (d < bd) { bd = d; bi = i; }
        });
        index = bi;
      }
      const existing = state.anchors.findIndex(
        (a) => a.surfaceId === hit.surfaceId && a.index === index
      );
      const record = { surfaceId: hit.surfaceId, index, u: branch.u, v: branch.v, xyz: hit.xyz };
      if (existing >= 0) state.anchors[existing] = record;
      else state.anchors.push(record);
      state.anchors.sort((a, b) => a.index - b.index);
      renderAnchors();
      renderView();
      info.innerHTML = `已添加分支锚点：<strong>${hit.surfaceId}</strong> @ #${index}<br>
        u=${branch.u.toFixed(4)}, v=${branch.v.toFixed(4)} · ${which}`;
      scheduleProject();
    } catch (err) {
      info.textContent = `锚点拾取失败：${err.message}`;
    }
  }

  function bindControls() {
    $("runA").addEventListener("click", () => runProject("A"));
    $("runB").addEventListener("click", () => runProject("B"));
    $("presetLine").addEventListener("click", async () => {
      if (!state.server) return;
      state.picks = state.server.transition.map((p) => ({ ...p }));
      renderPicks();
      runProject("B");
    });
    $("clearPicks").addEventListener("click", () => {
      state.picks = [];
      $("pickInfo").classList.add("hidden");
      renderPicks();
      runProject("B");
    });
    $("pickToggle").addEventListener("click", (e) => {
      state.pickMode = !state.pickMode;
      if (state.pickMode) {
        state.anchorMode = false;
        $("anchorToggle").classList.remove("active");
        $("view3d").classList.remove("anchoring");
      }
      e.currentTarget.classList.toggle("active", state.pickMode);
      $("view3d").classList.toggle("picking", state.pickMode);
    });
    $("anchorToggle").addEventListener("click", (e) => {
      state.anchorMode = !state.anchorMode;
      if (state.anchorMode) {
        state.pickMode = false;
        $("pickToggle").classList.remove("active");
        $("view3d").classList.remove("picking");
      }
      e.currentTarget.classList.toggle("active", state.anchorMode);
      $("view3d").classList.toggle("anchoring", state.anchorMode);
    });
    $("clearAnchors").addEventListener("click", () => {
      state.anchors = [];
      renderAnchors();
      scheduleProject();
    });
    $("branchStrategy").addEventListener("change", (e) => {
      state.branchStrategy = e.target.value;
      scheduleProject();
    });
    $("distSurface").addEventListener("change", renderDistortion);

    const sliderBindings = [
      ["offsetSlider", "offset", "offsetValue", (v) => v.toFixed(2)],
      ["lateralSlider", "lateral", "lateralValue", (v) => v.toFixed(2)],
      ["scaleSlider", "scale", "scaleValue", (v) => v.toFixed(2)],
      ["angleSlider", "angleDeg", "angleValue", (v) => `${Math.round(v)}°`],
    ];
    for (const [inputId, key, outId, fmtFn] of sliderBindings) {
      $(inputId).addEventListener("input", (e) => {
        state.att[key] = parseFloat(e.target.value);
        $(outId).textContent = fmtFn(state.att[key]);
        scheduleProject();
      });
    }

    $("yawSlider").addEventListener("input", (e) => {
      const deg = parseFloat(e.target.value);
      state.yawRad = deg * Math.PI / 180;
      $("yawValue").textContent = `${deg.toFixed(1)}°`;
      clearTimeout(yawTimer);
      yawTimer = setTimeout(async () => {
        try {
          await loadState();
          await runProject(state.mode);
        } catch (err) {
          setStatus(err.message, true);
        }
      }, 260);
    });

    $("diagnoseBtn").addEventListener("click", async () => {
      const panel = $("diagnosticsPanel");
      const content = $("diagnosticsContent");
      panel.classList.remove("hidden");
      content.innerHTML = "<p class='tiny' style='padding:16px'>正在运行平面、弯曲、折返和奇点诊断…</p>";
      try {
        const result = await request("/api/diagnose", {});
        content.innerHTML = `<div class="diag-grid">${result.items.map((item) => `
          <div class="diag-card">
            <h3>${item.label}<span class="${item.pass ? "badge-pass" : "badge-fail"}">${item.pass ? "PASS" : "FAIL"}</span></h3>
            <pre>${JSON.stringify(item.metrics, null, 2)}</pre>
          </div>`).join("")}</div>`;
      } catch (err) {
        content.innerHTML = `<p style="padding:16px;color:#991b1b">诊断失败：${err.message}</p>`;
      }
    });

    $("exportBtn").addEventListener("click", async (e) => {
      const btn = e.currentTarget;
      btn.disabled = true;
      const old = btn.textContent;
      btn.textContent = "正在生成真实 SVG/JSON/CSV…";
      try {
        const result = await request("/api/export", {
          yawRad: state.yawRad,
          attachment: state.att,
        });
        renderExportLinks(result.urls);
        setStatus(`导出完成 · correctness=${result.report.correctnessPass}`);
      } catch (err) {
        setStatus(`导出失败：${err.message}`, true);
      } finally {
        btn.disabled = false;
        btn.textContent = old;
      }
    });
  }

  function scheduleProject() {
    clearTimeout(projectTimer);
    projectTimer = setTimeout(() => runProject(state.mode), 180);
  }

  function bindCanvas() {
    const view = $("view3d");
    view.addEventListener("pointerdown", (e) => {
      view.setPointerCapture(e.pointerId);
      state.viewDrag = {
        x: e.clientX,
        y: e.clientY,
        yaw: state.camera.yaw,
        pitch: state.camera.pitch,
        moved: false,
        pointerId: e.pointerId,
      };
      view.classList.add("dragging");
    });
    view.addEventListener("pointermove", (e) => {
      const drag = state.viewDrag;
      if (!drag) {
        updateHover(e);
        return;
      }
      const dx = e.clientX - drag.x;
      const dy = e.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 4) drag.moved = true;
      state.camera.yaw = drag.yaw + dx * 0.008;
      state.camera.pitch = clamp(drag.pitch + dy * 0.006, -1.25, 1.25);
      renderView();
    });
    view.addEventListener("pointerup", async (e) => {
      const drag = state.viewDrag;
      view.classList.remove("dragging");
      state.viewDrag = null;
      if (drag && !drag.moved) {
        if (state.pickMode) await handleCanvasPick(e);
        else if (state.anchorMode) await handleAnchorPick(e);
      }
    });
    view.addEventListener("pointercancel", () => {
      state.viewDrag = null;
      view.classList.remove("dragging");
    });
    view.addEventListener("wheel", (e) => {
      e.preventDefault();
      state.camera.zoom = clamp(
        state.camera.zoom * Math.exp(-e.deltaY * 0.001), 55, 430
      );
      renderView();
    }, { passive: false });

    const pattern = $("patternCanvas");
    pattern.addEventListener("pointerdown", (e) => {
      pattern.setPointerCapture(e.pointerId);
      state.patternDrag = {
        x: e.clientX,
        y: e.clientY,
        att: { ...state.att },
      };
    });
    pattern.addEventListener("pointermove", (e) => {
      const drag = state.patternDrag;
      if (!drag) return;
      const rect = pattern.getBoundingClientRect();
      const dx = (e.clientX - drag.x) / state.patternView.unit;
      const dy = -(e.clientY - drag.y) / state.patternView.unit;
      state.att.offset = clamp(drag.att.offset + dx * drag.att.scale, 0, 4.2);
      state.att.lateral = clamp(drag.att.lateral + dy * drag.att.scale, -0.65, 0.65);
      $("offsetSlider").value = String(state.att.offset);
      $("lateralSlider").value = String(state.att.lateral);
      $("offsetValue").textContent = state.att.offset.toFixed(2);
      $("lateralValue").textContent = state.att.lateral.toFixed(2);
      scheduleProject();
    });
    const endPatternDrag = () => { state.patternDrag = null; };
    pattern.addEventListener("pointerup", endPatternDrag);
    pattern.addEventListener("pointercancel", endPatternDrag);

    window.addEventListener("resize", () => renderAll());
  }

  bindControls();
  bindCanvas();
  loadState().then(() => runProject("B")).catch((err) => setStatus(err.message, true));
})();
