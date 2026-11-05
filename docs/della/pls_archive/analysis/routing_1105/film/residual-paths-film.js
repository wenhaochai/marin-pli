/* The opening film of blogs/residual-paths-not-routing.html, drawn live in the page like assets/vocab-multi-epoch-film.js: each animation frame redraws the SVG of time t. Source: tmp/pls_routing/film/ in the pls archive. RPFilm.frame(t) returns the SVG markup of time t in 0..T; RPFilm.mount(el) plays the film inside el. */
(function () {
'use strict';
const C = {bg: '#FFFFFF', ink: '#202124', ink2: '#3C4043', grey: '#5F6368', grey2: '#80868B', grid: '#E8EAED', axis: '#DADCE0', blue: '#1967D2', light: '#8AB4F8', tint: '#E8F0FE', red: '#D93025', redTint: '#FCE8E6'};
const FONT = "'Hanken Grotesk', 'Helvetica Neue', Arial, 'PingFang SC', 'Noto Sans CJK SC', 'Microsoft YaHei', sans-serif";
// every number is in tmp/pls_routing/facts.md: held-out loss of the last layer, gains, and HC's learned mixing at the MLP of layer 12 of the 48-layer model (tmp/pls_hcid/a0_l12.json)
const D = {
  depths: [12, 24, 48],
  one: [3.845, 3.745, 3.670], none: [3.784, 3.660, 3.580], learned: [3.768, 3.680, 3.636],
  gainMix: ['+0.016', '−0.020', '−0.056'], gainPath: ['+0.061', '+0.085', '+0.090'],
  A: [[0.94, -0.06, -0.16, 0.09], [-0.13, 0.50, -0.44, 0.18], [-0.12, -0.46, 0.18, -0.06], [-0.07, -0.08, -0.05, 1.06]],
  rows: [0.81, 0.10, -0.47, 0.86]
};
const L = (en, zh) => (typeof document !== 'undefined' && document.documentElement.getAttribute('data-lang') === 'zh') ? zh : en;
// the timeline: builds are quick, and each scene holds on its point (owner 2026-11-01 on the last film: "一张一弛"); the loop ends on the main result
const ORDER = ['title', 'paths', 'mixing', 'board'];
const DUR = {title: 3.2, paths: 6.5, mixing: 8.0, board: 12.5};
const ST = {}; let _acc = 0; ORDER.forEach(k => { ST[k] = _acc; _acc += DUR[k]; });
const T = +_acc.toFixed(3);
const S = ORDER.map(k => [k, ST[k], ST[k] + DUR[k]]);
const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const seg = (t, a, b) => clamp((t - a) / (b - a));
const eo = x => 1 - Math.pow(1 - x, 3);
const eio = x => x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
const lerp = (a, b, p) => a + (b - a) * p;
const mix = (c1, c2, p) => { const h = c => [1, 3, 5].map(i => parseInt(c.slice(i, i + 2), 16)); const a = h(c1), b = h(c2);
  return '#' + a.map((v, i) => Math.round(lerp(v, b[i], p)).toString(16).padStart(2, '0')).join(''); };
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;');
const fmt = n => n.toLocaleString('en-US');
const ctx = document.createElement('canvas').getContext('2d');
const TW = new Map();
const tw = (s, size, w = 500) => { const k = w + "|" + size + "|" + s; let v = TW.get(k); if (v === undefined) { ctx.font = `${w} ${size}px ${FONT}`; v = ctx.measureText(s).width; TW.set(k, v); } return v; };
function tx(x, y, s, o = {}) {
  const a = [`x="${x}"`, `y="${y}"`, `font-family="${FONT}"`, `font-size="${o.s || 22}"`, `font-weight="${o.w || 400}"`, `fill="${o.c || C.ink}"`];
  if (o.a) a.push(`text-anchor="${o.a}"`); if (o.op != null) a.push(`opacity="${o.op}"`);
  return `<text ${a.join(' ')}>${esc(s)}</text>`;
}
const R = (x, y, w, h, o = {}) => `<rect x="${x}" y="${y}" width="${Math.max(0, w)}" height="${Math.max(0, h)}" fill="${o.fill || 'none'}" stroke="${o.st || 'none'}" stroke-width="${o.sw || 1}" rx="${o.rx || 0}"${o.op != null ? ` opacity="${o.op}"` : ''}${o.dash ? ` stroke-dasharray="${o.dash}"` : ''}/>`;
const Ln = (x1, y1, x2, y2, o = {}) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${o.st || C.ink}" stroke-width="${o.sw || 1}"${o.dash ? ` stroke-dasharray="${o.dash}"` : ''}${o.op != null ? ` opacity="${o.op}"` : ''} stroke-linecap="round"/>`;
const P = (pts, o = {}) => pts.length < 2 ? '' : `<polyline points="${pts.map(p => p[0].toFixed(1) + ',' + p[1].toFixed(1)).join(' ')}" fill="none" stroke="${o.st || C.ink}" stroke-width="${o.sw || 2}" stroke-linejoin="round" stroke-linecap="round"${o.op != null ? ` opacity="${o.op}"` : ''}/>`;
const card = (x, y, w, h, op = 1) => R(x, y, w, h, {fill: C.bg, st: C.grid, sw: 2, rx: 14, op});
const G = (op, inner, dy = 0) => op <= 0 ? '' : `<g opacity="${op.toFixed(3)}"${dy ? ` transform="translate(0,${dy.toFixed(1)})"` : ''}>${inner}</g>`;
const SFX = [];
const sfx = (t, k, g) => SFX.push({t: +t.toFixed(3), k, g});
const SC = {};
const head = (title, sub) => tx(96, 104, title, {s: 46, w: 600}) + tx(96, 156, sub, {s: 28, c: C.grey});
function chip(x, y, label, on) {
  const w = tw(label, 24, on ? 600 : 500) + 34, h = 42;
  return [w, R(x, y, w, h, {fill: on ? C.blue : C.bg, st: on ? C.blue : C.axis, sw: 2, rx: 21}) + tx(x + w / 2, y + 29, label, {s: 24, w: on ? 600 : 500, c: on ? '#FFFFFF' : C.grey, a: 'middle'})];
}
// the compared variables stay on screen for the whole loop
function variables(depth, mixing) {
  let o = '';
  const row = (y, name, items, on, all = false) => {
    let x = 1820; const parts = [];
    for (let i = items.length - 1; i >= 0; i--) { const lab = items[i], hit = all || lab === on, w = tw(lab, 24, hit ? 600 : 500) + 34; x -= w; parts.unshift(chip(x, y, lab, hit)[1]); x -= 10; }
    o += parts.join('') + tx(x - 8, y + 29, name, {s: 24, c: C.grey, a: 'end'});
  };
  row(52, L('Layers', '层数'), ['12', '24', '48'], depth);
  row(104, L('Mixing', '混合'), [L('learned', '学到的'), L('none', '没有')], mixing === 'both' ? null : mixing, mixing === 'both');
  return o;
}
const dot = (x, y, r, fill, op = 1) => `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r}" fill="${fill}" opacity="${op.toFixed(3)}"/>`;
const box = (x, y, w, h, label, o = {}) => R(x, y, w, h, {fill: o.fill || C.bg, st: o.st || C.grey, sw: 2.5, rx: 10}) + tx(x + w / 2, y + h / 2 + 11, label, {s: o.s || 32, w: 600, a: 'middle', c: o.c || C.ink});

// title: the claim
SC.title = t => {
  const a = ST.title, p = eo(seg(t, a + 0.1, a + 0.6));
  return G(p, tx(96, 440, L('Deeper Models Gain from Residual Paths,', '更深的模型靠残差通路获益，'), {s: 76, w: 600}) + tx(96, 540, L('Not from Learned Routing', '而不是学到的路由'), {s: 76, w: 600}), 14 * (1 - p)) +
    G(eo(seg(t, a + 0.7, a + 1.2)), tx(96, 630, L('Held-out loss of language models at 12, 24 and 48 layers', '12、24、48 层语言模型的留出损失'), {s: 38, c: C.grey}));
};
sfx(ST.title + 0.15, 'whoosh', 0.3);

// one path against four streams, with writes running through both
const YS = [860, 700, 540, 380];
function column(t, a, xs, xF, title, sub, withMix) {
  const top = 300, bot = 950, cx = xs.reduce((p, v) => p + v, 0) / xs.length, st = eo(seg(t, a, a + 0.4));
  let o = tx(cx, 228, title, {s: 34, w: 600, a: 'middle'}) ;
  xs.forEach(x => { o += Ln(x, bot, x, top, {st: C.ink2, sw: xs.length > 1 ? 5 : 8}); });
  if (xs.length > 1) o += Ln(xs[0], top, xs[xs.length - 1], top, {st: C.ink2, sw: 5});
  o += Ln(cx, top, cx, top - 22, {st: C.ink2, sw: 5});
  o += tx(xs[xs.length - 1] + 24 + (xs.length > 1 ? 0 : 0), top - 4, sub, {s: 26, c: C.grey});
  YS.forEach((y, i) => {
    const isAttn = i % 2 === 0, name = isAttn ? L('Attention', '注意力') : 'MLP';
    o += box(xF, y - 40, 170, 72, name, {s: 28});
    const xr = xs[xs.length - 1];
    o += Ln(xr, y + 18, xF, y + 18, {st: C.axis, sw: 3});
    xs.forEach(x => { o += Ln(x, y - 26, xF, y - 26, {st: C.axis, sw: 2}); });
    // the write runs from the sublayer into every stream; one sublayer after the other
    const per = 0.9, ph = ((t - a) / per - i) % 4; const q = ph >= 0 && ph < 1 ? ph : -1;
    if (q >= 0 && st > 0.99) xs.forEach((x, j) => { o += dot(lerp(xF, x, eo(q)), y - 26, 9, C.blue, 1 - seg(q, 0.85, 1)); });
    if (withMix) { const m = eo(seg(t, a + 2.4, a + 3.0)); o += G(m, R(xs[0] - 20, y - 76, xs[xs.length - 1] - xs[0] + 40, 34, {fill: C.tint, st: C.blue, sw: 2, rx: 8}) + tx(cx, y - 50, 'A', {s: 24, w: 600, a: 'middle', c: C.blue})); }
  });
  return o;
}
SC.paths = t => {
  const a = ST.paths;
  let o = head(L('One running sum, or four streams', '一个累加和，还是四条流'), L('A path lets an earlier output reach a later layer directly.', '通路让前面的输出直接到达后面的层。'));
  o += variables('48', L('learned', '学到的'));
  o += G(eo(seg(t, a + 0.1, a + 0.5)), column(t, a + 0.1, [420], 620, L('Pre-LN: one path', 'Pre-LN：一条通路'), L('running sum', '累加和'), false));
  o += G(eo(seg(t, a + 1.0, a + 1.4)), column(t, a + 1.0, [1140, 1200, 1260, 1320], 1500, L('Hyper-Connections: four streams', 'Hyper-Connections：四条流'), L('stream sum', '流之和'), true));
  o += G(eo(seg(t, a + 3.4, a + 3.9)), tx(1660, 300, L('A mixes', 'A 混合'), {s: 26, c: C.blue, a: 'start'}) + tx(1660, 332, L('the streams', '各条流'), {s: 26, c: C.blue, a: 'start'}));
  return o;
};
sfx(ST.paths + 0.2, 'tick', 0.5); sfx(ST.paths + 1.1, 'tick', 0.5); sfx(ST.paths + 3.5, 'pop', 0.5);

// the mixing matrix: the real one at layer 12, then no mixing
SC.mixing = t => {
  const a = ST.mixing, flip = eio(seg(t, a + 4.6, a + 5.2)), off = flip > 0.5;
  let o = head(L('What the learned mixing does at layer 12', '学到的混合在第 12 层做了什么'), L('Row i says how much of stream i goes into each stream.', '第 i 行表示第 i 条流有多少进入每条流。'));
  o += variables('48', off ? L('none', '没有') : L('learned', '学到的'));
  const x0 = 300, y0 = 290, c = 136, gap = 6;
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
    const ap = eo(seg(t, a + 0.1 + (i + j) * 0.05, a + 0.5 + (i + j) * 0.05));
    const v = lerp(D.A[i][j], i === j ? 1 : 0, flip), mag = clamp(Math.abs(v));
    const fill = v >= 0 ? mix('#FFFFFF', C.blue, mag * 0.75) : mix('#FFFFFF', C.red, mag * 0.75);
    o += G(ap, R(x0 + j * (c + gap), y0 + i * (c + gap), c, c, {fill, st: C.grid, sw: 1.5, rx: 8}) +
      tx(x0 + j * (c + gap) + c / 2, y0 + i * (c + gap) + c / 2 + 12, v.toFixed(2).replace('-', '−'), {s: 34, w: 500, a: 'middle', c: mag > 0.62 ? '#FFFFFF' : C.ink}));
  }
  for (let j = 0; j < 4; j++) o += tx(x0 + j * (c + gap) + c / 2, y0 - 16, L('to ', '到 ') + (j + 1), {s: 24, c: C.grey, a: 'middle'});
  for (let i = 0; i < 4; i++) o += tx(x0 - 18, y0 + i * (c + gap) + c / 2 + 9, L('from ', '从 ') + (i + 1), {s: 24, c: C.grey, a: 'end'});
  // the row sums, then their mean
  const xr = x0 + 4 * (c + gap) + 50, sumIn = eo(seg(t, a + 1.0, a + 1.5));
  o += G(sumIn, tx(xr + 60, y0 - 16, L('row sum', '行和'), {s: 24, c: C.grey, a: 'middle'}));
  for (let i = 0; i < 4; i++) {
    const v = lerp(D.rows[i], 1, flip), neg = v < 0;
    o += G(eo(seg(t, a + 1.1 + i * 0.12, a + 1.4 + i * 0.12)), tx(xr + 60, y0 + i * (c + gap) + c / 2 + 14, v.toFixed(2).replace('-', '−'), {s: 44, w: 600, a: 'middle', c: neg ? C.red : C.ink}));
  }
  const xt = xr + 190, lerpMean = lerp(0.33, 1, flip);
  o += G(eo(seg(t, a + 2.0, a + 2.5)), tx(xt, y0 + 190, lerpMean.toFixed(2), {s: 110, w: 600, c: off ? C.blue : C.red}) +
    tx(xt, y0 + 244, L('mean row sum', '行和的平均'), {s: 28, c: C.grey}) +
    (off ? tx(xt, y0 + 320, L('The stream sum', '流之和'), {s: 34, w: 500}) + tx(xt, y0 + 364, L('keeps its scale.', '保持尺度不变。'), {s: 34, w: 500})
         : tx(xt, y0 + 320, L('This sublayer shrinks', '这个子层把流之和'), {s: 34, w: 500}) + tx(xt, y0 + 364, L('the stream sum to a third.', '缩小到三分之一。'), {s: 34, w: 500})));
  return o;
};
sfx(ST.mixing + 0.15, 'tick', 0.5); [0, 1, 2, 3].forEach(i => sfx(ST.mixing + 1.1 + i * 0.12, 'tick2', 0.35)); sfx(ST.mixing + 2.05, 'pop', 0.7); sfx(ST.mixing + 4.7, 'whoosh', 0.3);

// the board: three depths, three designs; the loop ends here, on the result
const BX = {12: 400, 24: 840, 48: 1280}, BY = v => 250 + (3.90 - v) / (3.90 - 3.55) * 470;
SC.board = t => {
  const a = ST.board, times = [a + 0.5, a + 2.6, a + 4.7];
  const cur = t < times[1] ? 0 : t < times[2] ? 1 : 2;
  let o = head(L('Three depths, same budget', '三个深度，相同预算'), L('Final held-out loss of the last layer, in nats.', '最后一层的最终留出损失，单位 nats。'));
  o += variables(String(D.depths[cur]), 'both');
  // grid
  [3.6, 3.7, 3.8, 3.9].forEach(v => { o += Ln(250, BY(v), 1500, BY(v), {st: C.grid, sw: 2}) + tx(240, BY(v) + 8, v.toFixed(1), {s: 24, c: C.grey, a: 'end'}); });
  D.depths.forEach((d, i) => { o += tx(BX[d], 752, d + L(' layers', ' 层'), {s: 30, w: i === cur ? 600 : 500, c: i === cur ? C.ink : C.grey, a: 'middle'}); });
  const sers = [[D.one, C.grey2, L('One path', '一条通路')], [D.learned, C.blue, L('Four streams, learned mixing', '四条流，学到的混合')], [D.none, C.light, L('Four streams, no mixing', '四条流，没有混合')]];
  sers.forEach(([v, col, name]) => {
    const pts = [];
    D.depths.forEach((d, i) => {
      if (t < times[i]) return;
      if (i > 0) { const p = eo(seg(t, times[i], times[i] + 0.45)); pts.push([lerp(BX[D.depths[i - 1]], BX[d], p), lerp(BY(v[i - 1]), BY(v[i]), p)]); }
      else pts.push([BX[d], BY(v[0])]);
    });
    o += P(pts, {st: col, sw: 6});
    D.depths.forEach((d, i) => { const q = eo(seg(t, times[i] + (i ? 0.4 : 0), times[i] + (i ? 0.6 : 0.2))); if (q > 0) o += dot(BX[d], BY(v[i]), 11 * q, col); });
  });
  // the direct labels, once the lines reach them
  const ln = eo(seg(t, times[2] + 0.5, times[2] + 0.9));
  o += G(ln, tx(1304, BY(D.one[2]) - 16, L('One path', '一条通路'), {s: 26, w: 600, c: C.grey}) + tx(1304, BY(D.learned[2]) + 10, L('Learned mixing', '学到的混合'), {s: 26, w: 600, c: C.blue}) + tx(1304, BY(D.none[2]) + 10, L('No mixing', '没有混合'), {s: 26, w: 600, c: '#4285F4'}));
  // the two gains under each depth, carried from the chart
  const y1 = 830, y2 = 930;
  o += G(eo(seg(t, a + 0.3, a + 0.7)), tx(96, y1 + 8, L('Gain from', '通路带来的'), {s: 28, w: 500, c: C.grey}) + tx(96, y1 + 42, L('paths', '收益'), {s: 28, w: 500, c: C.grey}) +
    tx(96, y2 + 8, L('Gain from', '学到的混合'), {s: 28, w: 500, c: C.grey}) + tx(96, y2 + 42, L('learned mixing', '带来的收益'), {s: 28, w: 500, c: C.grey}));
  o += Ln(96, y1 + 60, 1500, y1 + 60, {st: C.grid, sw: 2});
  D.depths.forEach((d, i) => {
    const q = eo(seg(t, times[i] + 0.6, times[i] + 1.0)), neg = i > 0;
    o += G(q, tx(BX[d], y1 + 34, D.gainPath[i], {s: 52, w: 600, a: 'middle', c: C.ink}), 8 * (1 - q));
    o += G(eo(seg(t, times[i] + 0.9, times[i] + 1.3)), tx(BX[d], y2 + 36, D.gainMix[i], {s: 64, w: 600, a: 'middle', c: neg ? C.red : C.blue}));
  });
  // the claim, on the held result
  o += G(eo(seg(t, a + 7.6, a + 8.2)), R(1560, 300, 300, 420, {fill: C.tint, rx: 16}) +
    tx(1584, 370, L('Paths pay', '通路在每个'), {s: 34, w: 600}) + tx(1584, 414, L('at every', '深度都有'), {s: 34, w: 600}) + tx(1584, 458, L('depth.', '收益。'), {s: 34, w: 600}) +
    tx(1584, 548, L('Learned', '学到的混合'), {s: 34, w: 600, c: C.red}) + tx(1584, 592, L('mixing', '越深'), {s: 34, w: 600, c: C.red}) + tx(1584, 636, L('stops paying.', '越不划算。'), {s: 34, w: 600, c: C.red}));
  return o;
};
[0.5, 2.6, 4.7].forEach(x => { sfx(ST.board + x, 'tick', 0.5); sfx(ST.board + x + 0.6, 'tick2', 0.4); sfx(ST.board + x + 0.9, 'pop', 0.6); });
sfx(ST.board + 7.7, 'chime', 0.6);
const CHAPTER = {title: 'Title', paths: 'One running sum, or four streams', mixing: 'The learned mixing at layer 12', board: 'Three depths'};
const LABEL = {title: ['Claim', '结论'], paths: ['Paths', '通路'], mixing: ['Mixing', '混合'], board: ['Three depths', '三个深度']};

function frame(t) {
  let out = `<rect width="1920" height="1080" fill="${C.bg}"/>`;
  for (const [name, a, b] of S) {
    if (t < a || t > b || (t === b && b !== T)) continue;
    const enter = a === 0 ? 1 : eo(seg(t, a, a + 0.35)), exit = b === T ? 1 : 1 - seg(t, b - 0.3, b);
    out += G(Math.min(enter, exit), SC[name](t), 14 * (1 - enter));
  }
  return out;
}
SFX.sort((a, b) => a.t - b.t);
const fontsReady = () => Promise.all(['400', '500', '600', '700'].map(w => document.fonts.load(`${w} 30px "Hanken Grotesk"`)))
  .catch(() => null).then(() => { TW.clear(); });

// sound effects with Web Audio, the same kinds as render_film.py synthesises for the MP4 (sines and filtered noise)
function audio() {
  const ac = new (window.AudioContext || window.webkitAudioContext)(), comp = ac.createDynamicsCompressor(), master = ac.createGain();
  master.gain.value = 1.3; master.connect(comp); comp.connect(ac.destination);
  const noise = ac.createBuffer(1, ac.sampleRate, ac.sampleRate), nd = noise.getChannelData(0);
  for (let i = 0; i < nd.length; i++) nd[i] = Math.random() * 2 - 1;
  function voice(gain, pan) {
    const g = ac.createGain(); g.gain.value = gain;
    if (ac.createStereoPanner) { const p = ac.createStereoPanner(); p.pan.value = pan; g.connect(p); p.connect(master); } else g.connect(master);
    return g;
  }
  function tone(out, f0, f1, dur, amp, att, tau, when) {
    const o = ac.createOscillator(), e = ac.createGain();
    o.frequency.setValueAtTime(f0, when); if (f1 !== f0) o.frequency.linearRampToValueAtTime(f1, when + dur);
    e.gain.setValueAtTime(0, when); e.gain.linearRampToValueAtTime(amp, when + att);
    if (tau) e.gain.setTargetAtTime(0, when + att, tau); else e.gain.linearRampToValueAtTime(0, when + dur);
    o.connect(e); e.connect(out); o.start(when); o.stop(when + dur + 0.05);
  }
  function hiss(out, centre, dur, amp, when) {
    const s = ac.createBufferSource(), f = ac.createBiquadFilter(), e = ac.createGain();
    s.buffer = noise; f.type = 'bandpass'; f.frequency.value = centre; f.Q.value = 0.9;
    e.gain.setValueAtTime(0, when); e.gain.linearRampToValueAtTime(amp, when + dur * 0.5); e.gain.linearRampToValueAtTime(0, when + dur);
    s.connect(f); f.connect(e); e.connect(out); s.start(when); s.stop(when + dur + 0.05);
  }
  function play(k, g) {
    const when = ac.currentTime + 0.01, out = voice(g, ['chime', 'chimelow', 'whoosh'].includes(k) ? 0 : Math.random() * 0.4 - 0.2);
    if (k === 'tick' || k === 'tick2') {
      const f = k === 'tick2' ? 3300 : 2500, tau = k === 'tick2' ? 0.012 : 0.007;
      tone(out, f, f, 0.08, 0.8, 0.001, tau, when); tone(out, f / 2, f / 2, 0.08, 0.3, 0.001, tau, when);
    } else if (k === 'pop') tone(out, 520, 860, 0.16, 1, 0.004, 0.045, when);
    else if (k === 'chime' || k === 'chimelow') {
      const base = k === 'chimelow' ? 523.25 : 1046.5, tau = k === 'chimelow' ? 0.5 : 0.35, dur = k === 'chimelow' ? 1.6 : 1.2;
      [[1, 1], [1.4983, 0.6], [2, 0.25], [3, 0.08]].forEach(([r, a]) => tone(out, base * r, base * r, dur, 0.6 * a, 0.003, tau, when));
    } else if (k === 'rise' || k === 'fall') {
      const [f0, f1] = k === 'rise' ? [300, 720] : [720, 330];
      tone(out, f0, f1, 0.5, 0.7, 0.25, 0, when); tone(out, 2 * f0, 2 * f1, 0.5, 0.18, 0.25, 0, when);
    } else if (k === 'whoosh') hiss(out, 1400, 0.5, 0.55, when);
    else if (k === 'slide') hiss(out, 700, 0.6, 0.5, when);
  }
  return {ac, play};
}

// the player: an SVG redrawn each animation frame; plays while on screen, pauses when the reader pauses or scrolls away;
// sound off until the reader turns it on; a reader who prefers reduced motion gets the result frame, paused
const STYLE = `.rpf-film{position:relative}.post-feature .rpf-film{margin-bottom:28px}html:not([data-lang]) .rpf-film [lang="zh"]{display:none}.rpf-film svg.rpf-stage{display:block;width:100%;height:auto;aspect-ratio:16/9;background:#fff;border:1px solid var(--rule-2,#e5e5e5);border-radius:8px;cursor:pointer}
.rpf-film .rpf-row{display:flex;align-items:flex-start;gap:10px;margin-top:8px}
.rpf-film .rpf-bar{flex:1;min-width:0;cursor:pointer;display:flex;align-items:flex-start;gap:3px;padding-top:13px}
.rpf-film .rpf-seg{display:flex;flex-direction:column;gap:6px;min-width:4px}
.rpf-film .rpf-bar i{display:block;height:4px;background:var(--rule-2,#E8EAED);border-radius:2px;overflow:hidden;transition:transform .12s}
.rpf-film .rpf-seg:hover i{transform:scaleY(1.75)}
.rpf-film .rpf-bar b{display:block;height:100%;width:0;background:#1967D2}
.rpf-film .rpf-seg em{font-style:normal;font-size:12px;line-height:1.25;color:var(--ink-3,#5F6368);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.rpf-film .rpf-seg.on em{color:var(--ink,#202124);font-weight:600}
@media (max-width:640px){.rpf-film .rpf-seg em{visibility:hidden}.rpf-film .rpf-seg.on em{visibility:visible;overflow:visible}}
.rpf-film .rpf-row button{flex:none;width:30px;height:30px;border-radius:15px;border:1px solid var(--rule-2,#E8EAED);background:var(--paper,#fff);color:var(--ink,#202124);display:grid;place-items:center;cursor:pointer;padding:0}
.rpf-film .rpf-row button:focus-visible,.rpf-film .rpf-bar:focus-visible{outline:2px solid #1967D2;outline-offset:2px}
.rpf-film .rpf-row svg{width:14px;height:14px}`;
const ICON = {play: '<svg viewBox="0 0 16 16"><path d="M4 2.5v11l9-5.5z" fill="currentColor"/></svg>',
  pause: '<svg viewBox="0 0 16 16"><path d="M4 2.5h3v11H4zM9 2.5h3v11H9z" fill="currentColor"/></svg>',
  mute: '<svg viewBox="0 0 16 16"><path d="M2 6h3l4-3v10l-4-3H2z" fill="currentColor"/><path d="M11 6l4 4M15 6l-4 4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>',
  sound: '<svg viewBox="0 0 16 16"><path d="M2 6h3l4-3v10l-4-3H2z" fill="currentColor"/><path d="M11 5.5a3.5 3.5 0 0 1 0 5M12.8 3.6a6 6 0 0 1 0 8.8" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>'};
const POSTER = ST.board + 8.5;   // the board, built: the result

function mount(el) {
  if (!document.getElementById('rpf-film-style')) { const s = document.createElement('style'); s.id = 'rpf-film-style'; s.textContent = STYLE; document.head.appendChild(s); }
  el.classList.add('rpf-film');
  el.innerHTML = '<svg class="rpf-stage" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1920 1080" aria-hidden="true"></svg>' +
    '<div class="rpf-row"><button type="button" class="rpf-play"></button>' +
    '<div class="rpf-bar" role="slider" aria-label="Position" aria-valuemin="0" aria-valuemax="' + T + '" tabindex="0">' +
    S.map(([k, a, b], j) => `<span class="rpf-seg" style="flex:${(b - a).toFixed(2)}" title="${CHAPTER[k]}" data-j="${j}"><i><b></b></i>` +
      `<em><span lang="en">${LABEL[k][0]}</span><span lang="zh">${LABEL[k][1]}</span></em></span>`).join('') + '</div>' +
    '<button type="button" class="rpf-snd"></button></div>';
  // the progress bar is split at the scene changes: one segment per scene, each filled by the time spent in its scene
  const stage = el.querySelector('.rpf-stage'), bar = el.querySelector('.rpf-bar'), segs = [...bar.querySelectorAll('.rpf-seg')], fills = segs.map(s => s.querySelector('b'));
  let onJ = -1;
  const bPlay = el.querySelector('.rpf-play'), bSnd = el.querySelector('.rpf-snd');
  const reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let t = reduced ? POSTER : 0, playing = false, wanted = !reduced, visible = false, last = 0, raf = 0, snd = null, soundOn = false, shown = -1;
  const draw = () => { const q = Math.round(t * 30) / 30; if (q !== shown) { stage.innerHTML = frame(q); shown = q; }
    S.forEach(([, a, b], j) => { fills[j].style.width = (100 * clamp((t - a) / (b - a))).toFixed(2) + '%'; });
    const j = S.findIndex(([, a, b]) => t >= a && t < b);
    if (j !== onJ) { if (onJ >= 0) segs[onJ].classList.remove('on'); if (j >= 0) segs[j].classList.add('on'); onJ = j; }
    bar.setAttribute('aria-valuenow', t.toFixed(1)); };
  const ui = () => { bPlay.innerHTML = playing ? ICON.pause : ICON.play; bPlay.setAttribute('aria-label', playing ? 'Pause' : 'Play');
    bSnd.innerHTML = soundOn ? ICON.sound : ICON.mute; bSnd.setAttribute('aria-label', soundOn ? 'Sound off' : 'Sound on'); bSnd.setAttribute('aria-pressed', String(soundOn)); };
  function tick(now) {
    if (!playing) return;
    const dt = Math.min(0.1, (now - last) / 1000); last = now;
    const t0 = t; t += dt;
    if (soundOn && snd) SFX.forEach(e => { if (e.t > t0 && e.t <= t) snd.play(e.k, e.g); });
    if (t >= T) t -= T;
    draw(); raf = requestAnimationFrame(tick);
  }
  function setPlaying(p) { p = p && visible; if (p === playing) return ui(); playing = p; ui();
    if (playing) { last = performance.now(); raf = requestAnimationFrame(tick); } else cancelAnimationFrame(raf); }
  stage.addEventListener('click', () => { wanted = !playing; setPlaying(wanted); });
  bPlay.addEventListener('click', () => { wanted = !playing; setPlaying(wanted); });
  bSnd.addEventListener('click', () => {
    soundOn = !soundOn;
    if (soundOn && !snd) { try { snd = audio(); } catch (e) { soundOn = false; } }
    if (snd && snd.ac.state === 'suspended') snd.ac.resume();
    ui();
  });
  // a click on a segment's bar seeks within its scene; a click on its name, or in a gap, goes to the start of that scene
  const seek = e => {
    const seg_ = e.target.closest('.rpf-seg'), j = seg_ ? +seg_.dataset.j : segs.findIndex(s => s.getBoundingClientRect().left >= e.clientX);
    if (j < 0) return;
    const [, a, b] = S[j], r = segs[j].getBoundingClientRect(), onBar = seg_ && e.target.closest('i');
    t = Math.max(0, Math.min(T - 0.001, onBar ? a + (b - a) * clamp((e.clientX - r.left) / r.width) : a)); draw();
  };
  bar.addEventListener('click', seek);
  bar.addEventListener('keydown', e => { if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { t = Math.max(0, Math.min(T - 0.001, t + (e.key === 'ArrowRight' ? 5 : -5))); draw(); e.preventDefault(); } });
  fontsReady().then(() => {
    shown = -1; draw(); ui();
    if ('IntersectionObserver' in window) new IntersectionObserver(es => { visible = es[0].isIntersecting; setPlaying(wanted); }, {threshold: 0.35}).observe(el);
    else { visible = true; setPlaying(wanted); }
  });
}

window.RPFilm = {T, SFX, frame, mount, fontsReady};
})();
