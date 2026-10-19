import { motion, useScroll, useSpring, useTransform } from "framer-motion";
import { useLayoutEffect, useRef, useState } from "react";

const INK = "#0d0b09";
const SAFE = "#ff4a2e";
const AMBER = "#f2a33a";

function VizNative() {
  return (
    <svg viewBox="0 0 400 170">
      <g>
        <rect x="10" y="40" width="96" height="96" fill="none" stroke={INK} strokeOpacity=".35" strokeDasharray="3 4" />
        <rect x="10" y="40" width="96" height="96" fill={INK} fillOpacity=".06" />
        <text x="58" y="158" textAnchor="middle" fontFamily="var(--mono)" fontSize="10" fill={INK} opacity=".55">v1 · 224² squash</text>
      </g>
      <path d="M126 88 H160" stroke={INK} strokeOpacity=".4" markerEnd="url(#ar)" />
      <defs><marker id="ar" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="8" markerHeight="8" orient="auto"><path d="M0 0 L8 4 L0 8" fill="none" stroke={INK} /></marker></defs>
      <rect x="176" y="22" width="208" height="132" fill="none" stroke={SAFE} strokeWidth="1.5" />
      {Array.from({ length: 8 }, (_, i) => <line key={i} x1={176 + (i + 1) * 26} x2={176 + (i + 1) * 26} y1="22" y2="154" stroke={SAFE} strokeOpacity=".15" />)}
      {Array.from({ length: 5 }, (_, i) => <line key={i} y1={22 + (i + 1) * 22} y2={22 + (i + 1) * 22} x1="176" x2="384" stroke={SAFE} strokeOpacity=".15" />)}
      <text x="280" y="168" textAnchor="middle" fontFamily="var(--mono)" fontSize="10" fill={SAFE}>v2 · 512×384 native aspect</text>
    </svg>
  );
}

function VizScales() {
  const levels = [[12, 9, 8], [8, 6, 16], [4, 3, 32]];
  return (
    <svg viewBox="0 0 400 186">
      {levels.map(([c, r, s], k) => (
        <g key={k} transform={`translate(${16 + k * 122} ${20 + k * 10}) skewY(-8)`}>
          {Array.from({ length: c * r }, (_, i) => (
            <rect key={i} x={(i % c) * (96 / c)} y={Math.floor(i / c) * (72 / r)} width={96 / c - 1} height={72 / r - 1}
              fill={INK} fillOpacity={0.08 + ((i * 37) % 11) / 30} />
          ))}
          <text x="0" y="104" fontFamily="var(--mono)" fontSize="10" fill={INK} opacity=".6">stride {s}</text>
        </g>
      ))}
      <path d="M60 140 C150 160 250 160 340 140" stroke={SAFE} fill="none" strokeDasharray="4 4" />
      <text x="200" y="178" textAnchor="middle" fontFamily="var(--mono)" fontSize="10" fill={SAFE}>fused on the 1/16 grid</text>
    </svg>
  );
}

function VizField() {
  const W = 16;
  const H = 8;
  return (
    <svg viewBox="0 0 400 190">
      {Array.from({ length: W * H }, (_, i) => {
        const x = i % W;
        const y = Math.floor(i / W);
        const blur = Math.max(0, 1 - Math.hypot(x - 3, y - 6) / 5);
        const q = 0.85 - blur * 0.75 - (y < 2 ? 0.05 : 0);
        return <rect key={i} x={8 + x * 24} y={8 + y * 21} width="23" height="20" fill={q < 0.5 ? SAFE : INK} fillOpacity={q < 0.5 ? 0.85 - q : 0.05 + q * 0.12} />;
      })}
      <text x="8" y="186" fontFamily="var(--mono)" fontSize="10" fill={INK} opacity=".6">q(x, y) — local quality, MOS units</text>
    </svg>
  );
}

function VizFormula() {
  return (
    <svg viewBox="0 0 400 150">
      <text x="200" y="70" textAnchor="middle" fontFamily="var(--serif)" fontStyle="italic" fontSize="44" fill={INK}>
        S = Σ w·q ⁄ Σ w
      </text>
      <circle cx="120" cy="112" r="20" fill={AMBER} fillOpacity=".5" />
      <circle cx="160" cy="118" r="10" fill={AMBER} fillOpacity=".3" />
      <circle cx="270" cy="110" r="14" fill={AMBER} fillOpacity=".4" />
      <text x="200" y="146" textAnchor="middle" fontFamily="var(--mono)" fontSize="10" fill={INK} opacity=".6">w(x, y) — learned viewer attention</text>
    </svg>
  );
}

const PANELS = [
  { n: "01", title: "Read the frame as it is.", body: "v1 squashed every photo into a 224-pixel square, smearing the very blur and noise it was meant to judge. v2 reads images at their native aspect on a 512×384 canvas, and is fully convolutional so any shape works.", viz: <VizNative /> },
  { n: "02", title: "Three depths of vision.", body: "A CLIP-pretrained ConvNeXt-B sees texture at stride 8, structure at 16 and composition at 32. The three are projected and fused on one grid, so fine grain and global framing inform the same decision.", viz: <VizScales /> },
  { n: "03", title: "A field, not a number.", body: "Instead of regressing one score, the model paints a quality value for every region. Soft focus in a corner, noise in a shadow, banding in the sky: each shows up where it lives.", viz: <VizField /> },
  { n: "04", title: "Weighted like a viewer.", body: "A second head learns where people actually look. The final grade is the attention-weighted mean of the field, so a blurred background behind a sharp subject is forgiven, while a soft face is not.", viz: <VizFormula /> },
  { n: "05", title: "Taught by people, and by damage.", body: "KonIQ-10k opinions set the scale. KonIQ++ votes and KADID-10k distortions teach the six defect families. SPAQ adds human-rated attributes. Every step, synthetic degradations of real photos must also rank below their originals.", viz: null },
];

export default function Method() {
  const ref = useRef(null);
  const trackRef = useRef(null);
  const [dist, setDist] = useState(0);
  useLayoutEffect(() => {
    const measure = () => setDist(Math.max(0, (trackRef.current?.scrollWidth ?? 0) - window.innerWidth));
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, []);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start start", "end end"] });
  const p = useSpring(scrollYProgress, { stiffness: 120, damping: 30, mass: 0.3 });
  const x = useTransform(p, [0.04, 0.96], [0, -dist]);

  return (
    <section className="method" id="method" ref={ref}>
      <div className="method-stage">
        <motion.div className="method-track" ref={trackRef} style={{ x }}>
          <div className="panel intro">
            <div className="tag mono">§ 03 — Method</div>
            <h2 className="h2" style={{ color: INK }}>How Lumina <em style={{ color: SAFE }}>sees.</em></h2>
            <p style={{ fontSize: 18, maxWidth: "40ch" }}>
              A 92-million-parameter model with four heads (quality, attention, defects, attributes) trained jointly on three
              human-opinion databases. Keep scrolling.
            </p>
          </div>
          {PANELS.map((pnl) => (
            <article className="panel" key={pnl.n}>
              <div className="num">{pnl.n}</div>
              <div style={{ display: "grid", gap: 14 }}>
                <h3>{pnl.title}</h3>
                <p>{pnl.body}</p>
              </div>
              <div className="viz">{pnl.viz || <TrainingMix />}</div>
            </article>
          ))}
        </motion.div>
        <div className="method-progress"><motion.span style={{ scaleX: p }} /></div>
      </div>
    </section>
  );
}

function TrainingMix() {
  const rows = [["KonIQ-10k", "7,058", "MOS · defects"], ["SPAQ", "8,900", "MOS · attributes"], ["KADID-10k", "8,190", "MOS · distortions"], ["Synthetic", "∞", "rank constraint"]];
  return (
    <div style={{ display: "grid", gap: 0 }}>
      {rows.map(([a, b, c]) => (
        <div key={a} style={{ display: "grid", gridTemplateColumns: "1fr auto", padding: "10px 0", borderBottom: "1px solid rgba(13,11,9,.15)" }}>
          <span style={{ fontFamily: "var(--serif)", fontSize: 26 }}>{a} <span className="mono" style={{ color: "#6c6253" }}>{c}</span></span>
          <span className="mono" style={{ alignSelf: "center" }}>{b}</span>
        </div>
      ))}
    </div>
  );
}
