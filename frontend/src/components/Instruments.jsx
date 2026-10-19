import { motion, useSpring, useTransform } from "framer-motion";
import { useEffect } from "react";
import { ATTR_LABEL, DEFECT_LABEL, severity } from "../lib/read";

const EASE = [0.16, 1, 0.3, 1];

/* ─────────────── analog light meter ─────────────── */
const ZONES = [
  [0, 25, "var(--safe)", "bad"], [25, 45, "var(--amber)", "poor"], [45, 65, "var(--paper-2)", "fair"],
  [65, 80, "var(--sage)", "good"], [80, 100, "var(--sage)", "excellent"],
];

function polar(v, r) {
  const a = Math.PI * (1 - v / 100);
  return [Math.cos(a) * r, -Math.sin(a) * r];
}

function arc(v0, v1, r) {
  const [x0, y0] = polar(v0, r);
  const [x1, y1] = polar(v1, r);
  return `M${x0} ${y0} A${r} ${r} 0 0 1 ${x1} ${y1}`;
}

export function Meter({ score }) {
  const target = score ?? 0;
  const spring = useSpring(0, { stiffness: 55, damping: 9, mass: 1.2 });
  useEffect(() => { spring.set(target); }, [target, spring]);
  const rot = useTransform(spring, (v) => -90 + (v / 100) * 180);
  return (
    <div className="meter">
      <svg viewBox="-230 -215 460 240" role="img" aria-label={`Quality ${score ?? "unknown"} out of 100`}>
        {ZONES.map(([a, b, c]) => (
          <path key={a} d={arc(a + 0.6, b - 0.6, 188)} stroke={c} strokeWidth="3" fill="none" opacity={score == null ? 0.25 : 0.9} />
        ))}
        {Array.from({ length: 51 }, (_, i) => {
          const v = i * 2;
          const major = v % 10 === 0;
          const [x0, y0] = polar(v, 170);
          const [x1, y1] = polar(v, major ? 152 : 162);
          return <line key={i} x1={x0} y1={y0} x2={x1} y2={y1} stroke="var(--paper)" strokeOpacity={major ? 0.7 : 0.25} strokeWidth={major ? 1.4 : 1} />;
        })}
        {[0, 20, 40, 60, 80, 100].map((v) => {
          const [x, y] = polar(v, 132);
          return <text key={v} x={x} y={y + 4} textAnchor="middle" fill="var(--mute)" fontFamily="var(--mono)" fontSize="11">{v}</text>;
        })}
        {ZONES.map(([a, b, , label]) => {
          const [x, y] = polar((a + b) / 2, 206);
          return <text key={label} x={x} y={y} textAnchor="middle" fill="var(--mute-2)" fontFamily="var(--mono)" fontSize="9.5" letterSpacing="1.5">{label.toUpperCase()}</text>;
        })}
        <motion.g style={{ rotate: rot, transformOrigin: "0px 0px", transformBox: "view-box" }}>
          <line x1="0" y1="14" x2="0" y2="-176" stroke="var(--safe)" strokeWidth="2" />
          <circle cy="-176" r="3" fill="var(--safe)" />
        </motion.g>
        <circle r="9" fill="var(--ink)" stroke="var(--paper)" strokeOpacity="0.4" />
        <circle r="2.5" fill="var(--paper)" />
      </svg>
    </div>
  );
}

/* ─────────────── defect scales ─────────────── */
const sevColor = (v) => (v >= 0.45 ? "var(--safe)" : v >= 0.25 ? "var(--amber)" : "var(--paper-2)");

export function DefectScales({ defects }) {
  const rows = Object.entries(defects);
  return (
    <div>
      {rows.map(([k, v], i) => (
        <div className="scale-row" key={k}>
          <span className="mono">{DEFECT_LABEL[k]}</span>
          <div className="bar">
            <div className="ticks" />
            <motion.div className="fill" style={{ background: sevColor(v), width: `${v * 100}%` }}
              initial={{ scaleX: 0 }} animate={{ scaleX: 1 }} transition={{ duration: 1.2, ease: EASE, delay: 0.15 + i * 0.07 }} />
            <motion.div className="pin" style={{ background: sevColor(v), boxShadow: v >= 0.45 ? "0 0 10px var(--safe)" : "none" }}
              initial={{ left: "0%" }} animate={{ left: `${v * 100}%` }} transition={{ duration: 1.2, ease: EASE, delay: 0.15 + i * 0.07 }} />
          </div>
          <span className="val" style={{ color: sevColor(v) }} title={severity(v)}>{Math.round(v * 100)}</span>
        </div>
      ))}
    </div>
  );
}

/* ─────────────── attribute radar (SPAQ-trained) ─────────────── */
const AXES = ["sharpness", "noisiness", "contrast", "colorfulness", "brightness"];

export function Radar({ attrs }) {
  const R = 92;
  const pt = (i, v) => {
    const a = -Math.PI / 2 + (i * 2 * Math.PI) / AXES.length;
    return [Math.cos(a) * R * v, Math.sin(a) * R * v];
  };
  const poly = AXES.map((k, i) => pt(i, Math.max(0.04, (attrs?.[k] ?? 0) / 100)).join(",")).join(" ");
  return (
    <svg viewBox="-150 -125 300 245" style={{ width: "100%", height: "auto" }} role="img" aria-label="Perceptual attributes">
      {[0.25, 0.5, 0.75, 1].map((s) => (
        <polygon key={s} points={AXES.map((_, i) => pt(i, s).join(",")).join(" ")} fill="none" stroke="var(--line-2)" strokeWidth="1" />
      ))}
      {AXES.map((k, i) => {
        const [x, y] = pt(i, 1);
        const [lx, ly] = pt(i, 1.24);
        return (
          <g key={k}>
            <line x1="0" y1="0" x2={x} y2={y} stroke="var(--line)" />
            <text x={lx} y={ly + 4} textAnchor="middle" fill="var(--mute)" fontFamily="var(--mono)" fontSize="9.5" letterSpacing="1">
              {ATTR_LABEL[k].toUpperCase()} {attrs ? Math.round(attrs[k]) : ""}
            </text>
          </g>
        );
      })}
      <motion.polygon key={poly} points={poly} fill="var(--safe-soft)" stroke="var(--safe)" strokeWidth="1.5"
        initial={{ scale: 0, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} style={{ transformOrigin: "0px 0px", transformBox: "view-box" }}
        transition={{ duration: 1.1, ease: EASE }} />
    </svg>
  );
}

/* ─────────────── luminance histogram ─────────────── */
export function Histogram({ bins, clipped, crushed }) {
  const W = 300;
  const H = 80;
  const n = bins.length;
  const d = `M0 ${H} ` + bins.map((v, i) => `L${(i / (n - 1)) * W} ${H - v * (H - 4)}`).join(" ") + ` L${W} ${H} Z`;
  return (
    <svg viewBox={`0 0 ${W} ${H + 16}`} style={{ width: "100%", height: "auto" }} role="img" aria-label="Luminance histogram">
      <defs>
        <linearGradient id="hg" x1="0" x2="1">
          <stop offset="0" stopColor="#3b3229" /><stop offset="1" stopColor="#ece4d4" />
        </linearGradient>
      </defs>
      <motion.path d={d} fill="url(#hg)" opacity="0.85" initial={{ opacity: 0 }} animate={{ opacity: 0.85 }} transition={{ duration: 1 }} />
      <line x1="0" x2={W} y1={H} y2={H} stroke="var(--line-2)" />
      <text x="0" y={H + 13} fill={crushed > 0.02 ? "var(--safe)" : "var(--mute)"} fontFamily="var(--mono)" fontSize="9">
        SHADOWS {(crushed * 100).toFixed(1)}%
      </text>
      <text x={W} y={H + 13} textAnchor="end" fill={clipped > 0.02 ? "var(--safe)" : "var(--mute)"} fontFamily="var(--mono)" fontSize="9">
        HIGHLIGHTS {(clipped * 100).toFixed(1)}%
      </text>
    </svg>
  );
}
