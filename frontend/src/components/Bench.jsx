import { motion } from "framer-motion";
import history from "../data/history.json";
import metrics from "../data/metrics.json";
import scatter from "../data/scatter.json";
import { FadeUp, RevealLines } from "./Reveal";

const SUITES = [
  ["koniq_test", "KonIQ-10k", "official test split · 2,015 images"],
  ["livec", "LIVE Challenge", "never seen in training · 1,169 images"],
  ["spaq_test", "SPAQ", "held-out smartphone photos · 2,225"],
  ["kadid_test", "KADID-10k", "held-out reference scenes · synthetic"],
];

const fmt = (v) => (v == null ? "—" : v.toFixed(3).replace(/^0/, ""));

function Curve() {
  if (!history.length) return null;
  const W = 1100;
  const H = 210;
  const ys = history.map((h) => h.val.srcc);
  const lo = Math.floor(Math.min(...ys.slice(1)) * 20) / 20;
  const hi = 1;
  const px = (i) => (i / Math.max(1, history.length - 1)) * W;
  const py = (v) => H - ((Math.max(v, lo) - lo) / (hi - lo)) * H;
  const path = (key) => history.map((h, i) => `${i ? "L" : "M"}${px(i)} ${py(h.val[key])}`).join(" ");
  const best = history.reduce((a, b) => (b.val.srcc + b.val.plcc > a.val.srcc + a.val.plcc ? b : a));
  return (
    <svg viewBox={`-34 -10 ${W + 50} ${H + 40}`} style={{ width: "100%", height: "auto" }} role="img" aria-label="Validation curve">
      {[lo, (lo + hi) / 2, hi].map((v) => (
        <g key={v}>
          <line x1="0" x2={W} y1={py(v)} y2={py(v)} stroke="var(--line)" />
          <text x="-8" y={py(v) + 4} textAnchor="end" fill="var(--mute)" fontFamily="var(--mono)" fontSize="10">{v.toFixed(2)}</text>
        </g>
      ))}
      {[["plcc", "var(--amber)"], ["srcc", "var(--safe)"]].map(([k, c]) => (
        <motion.path key={k} d={path(k)} fill="none" stroke={c} strokeWidth="1.8"
          initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }} viewport={{ once: true, margin: "-15%" }}
          transition={{ duration: 2.2, ease: [0.65, 0, 0.35, 1] }} />
      ))}
      <circle cx={px(best.epoch - 1)} cy={py(best.val.srcc)} r="4" fill="var(--safe)" />
      <text x={px(best.epoch - 1)} y={py(best.val.srcc) + 20} textAnchor="middle" fill="var(--paper)" fontFamily="var(--mono)" fontSize="10">
        best · ep {best.epoch}
      </text>
      <text x={W} y={H + 26} textAnchor="end" fill="var(--mute)" fontFamily="var(--mono)" fontSize="10">epoch {history.length}</text>
      <text x="0" y={H + 26} fill="var(--mute)" fontFamily="var(--mono)" fontSize="10">epoch 1</text>
    </svg>
  );
}

function Scatter() {
  if (!scatter.pred?.length) return null;
  const S = 360;
  const px = (v) => (v / 100) * S;
  return (
    <svg viewBox={`-30 -10 ${S + 44} ${S + 40}`} style={{ width: "100%", height: "auto" }} role="img"
      aria-label="Predicted versus human MOS on KonIQ-10k test">
      {[0, 25, 50, 75, 100].map((v) => (
        <g key={v}>
          <line x1={px(v)} x2={px(v)} y1="0" y2={S} stroke="var(--line)" />
          <line y1={S - px(v)} y2={S - px(v)} x1="0" x2={S} stroke="var(--line)" />
          <text x={px(v)} y={S + 16} textAnchor="middle" fill="var(--mute)" fontFamily="var(--mono)" fontSize="9">{v}</text>
          <text x="-8" y={S - px(v) + 3} textAnchor="end" fill="var(--mute)" fontFamily="var(--mono)" fontSize="9">{v}</text>
        </g>
      ))}
      <line x1="0" y1={S} x2={S} y2="0" stroke="var(--amber)" strokeDasharray="3 5" strokeOpacity=".6" />
      <motion.g initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true, margin: "-15%" }} transition={{ duration: 1.6 }}>
        {scatter.pred.map((p, i) => (
          <circle key={i} cx={px(scatter.mos[i])} cy={S - px(p)} r="1.6" fill="var(--safe)" fillOpacity=".55" />
        ))}
      </motion.g>
      <text x={S} y={S + 32} textAnchor="end" fill="var(--mute)" fontFamily="var(--mono)" fontSize="9.5">HUMAN MOS →</text>
      <text x="0" y="-2" fill="var(--mute)" fontFamily="var(--mono)" fontSize="9.5">↑ LUMINA</text>
    </svg>
  );
}

export default function Bench() {
  const v2 = (s) => metrics[s]?.lumina_v2_tta ?? metrics[s]?.lumina_v2;
  const headline = [
    ["koniq_test", "SRCC on KonIQ-10k test", "Rank agreement with 1.2 million human ratings, on images the model never trained on."],
    ["livec", "SRCC zero-shot, LIVE-C", "A different lab, different cameras, different raters. No fine-tuning."],
    ["spaq_test", "SRCC on SPAQ", "Smartphone photography, the kind you are most likely to upload."],
  ];
  return (
    <section className="bench" id="bench">
      <div className="wrap">
        <FadeUp className="tag mono">§ 04 — Bench</FadeUp>
        <RevealLines className="h2" lines={["Measured against", <>human <em>judgement.</em></>]} />

        <div className="bench-grid">
          {headline.map(([k, label, note], i) => (
            <FadeUp className="big-stat" key={k} delay={i * 0.1}>
              <div className="mono mute">{label}</div>
              <div className="v">{fmt(v2(k)?.srcc)}</div>
              <div className="mono" style={{ marginTop: 10 }}>PLCC <span className="amber">{fmt(v2(k)?.plcc)}</span>
                <span className="mute">  ·  v1 {fmt(metrics[k]?.lumina_v1?.srcc)}</span></div>
              <p>{note}</p>
            </FadeUp>
          ))}

          <FadeUp className="scatter" delay={0.05}>
            <div className="block-title" style={{ marginTop: 40 }}><span className="mono">KonIQ-10k test · 2,015 photographs</span><span className="mono mute">one dot each</span></div>
            <Scatter />
          </FadeUp>

          <FadeUp className="compare" delay={0.1}>
            <div className="block-title" style={{ marginTop: 40 }}><span className="mono">v1 → v2, SRCC</span><span className="mono mute">higher is better</span></div>
            {SUITES.filter(([k]) => metrics[k]).map(([k, name, sub]) => (
              <div className="compare-row" key={k}>
                <div><div style={{ fontFamily: "var(--serif)", fontSize: 22, lineHeight: 1 }}>{name}</div><div className="mono mute" style={{ fontSize: 9.5, marginTop: 4 }}>{sub}</div></div>
                <div className="lines">
                  {[["lumina_v1", "var(--mute-2)"], [metrics[k].lumina_v2_tta ? "lumina_v2_tta" : "lumina_v2", "var(--safe)"]].map(([m, c], j) => (
                    <motion.div key={m} className="ln" style={{ background: c, width: `${Math.max(0, metrics[k][m]?.srcc ?? 0) * 100}%` }}
                      initial={{ scaleX: 0 }} whileInView={{ scaleX: 1 }} viewport={{ once: true }}
                      transition={{ duration: 1.4, ease: [0.16, 1, 0.3, 1], delay: 0.2 + j * 0.15 }} />
                  ))}
                </div>
                <div className="nums"><span className="mute">{fmt(metrics[k].lumina_v1?.srcc)}</span><br />
                  <span className="safe">{fmt((metrics[k].lumina_v2_tta ?? metrics[k].lumina_v2).srcc)}</span></div>
              </div>
            ))}
            <div className="legend mono mute"><span><i style={{ background: "var(--mute-2)" }} />v1 · EfficientNet-B0</span><span><i style={{ background: "var(--safe)" }} />v2 · Lumina</span></div>
            <p className="note">
              v1 was trained on a random 80/20 split of all of KonIQ, so most of the official KonIQ test images were in its
              training set. Its KonIQ number is optimistic; LIVE-C and SPAQ are the fair comparisons. v2 numbers use
              horizontal-flip test-time averaging.
            </p>
          </FadeUp>

          <FadeUp className="curve" delay={0.2}>
            <div className="block-title" style={{ marginTop: 40 }}><span className="mono">Validation, every epoch</span>
              <span className="mono"><span className="safe">SRCC</span> · <span className="amber">PLCC</span></span></div>
            <Curve />
          </FadeUp>
        </div>
      </div>
    </section>
  );
}
