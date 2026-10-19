import { useRef, useState } from "react";
import { motion, useMotionValueEvent, useScroll, useSpring, useTransform } from "framer-motion";

const BLADES = 9;
const R = 500; // lens radius in SVG units
const F_STOPS = ["22", "16", "11", "8", "5.6", "4", "2.8", "2", "1.4"];

/** A camera iris: half-plane blades pivoted into a pinwheel, inside an ink lens surround. `v` ∈ [0,1]. */
function Iris({ v }) {
  const r = R * (0.12 + v * 0.98);
  const twist = 36 - 28 * v;
  const spin = 70 * v;
  const scale = 1 + Math.pow(Math.max(0, v - 0.55) / 0.45, 2.2) * 3.2;
  const blade = `M0 ${-2 * R} L${3 * R} ${-2 * R} L${3 * R} ${2 * R} L0 ${2 * R} Z`;
  return (
    <svg viewBox={`${-R - 60} ${-R - 60} ${2 * R + 120} ${2 * R + 120}`} aria-hidden="true"
      style={{ transform: `scale(${scale})`, overflow: "visible" }}>
      <defs>
        <clipPath id="barrel"><circle r={R} /></clipPath>
        <linearGradient id="bladeFill" x1="0" x2="1">
          <stop offset="0" stopColor="#241d16" />
          <stop offset="0.35" stopColor="#15110d" />
          <stop offset="1" stopColor="#0b0907" />
        </linearGradient>
      </defs>
      {/* ink surround with a lens-shaped hole */}
      <path d={`M-6000 -6000 H6000 V6000 H-6000 Z M${R} 0 A${R} ${R} 0 1 0 ${-R} 0 A${R} ${R} 0 1 0 ${R} 0 Z`}
        fill="var(--ink)" fillRule="evenodd" />
      <g clipPath="url(#barrel)">
        <g transform={`rotate(${spin})`}>
          {Array.from({ length: BLADES }, (_, i) => (
            <g key={i} transform={`rotate(${(360 / BLADES) * i}) translate(${r} 0) rotate(${twist})`}>
              <path d={blade} fill="url(#bladeFill)" stroke="rgba(236,228,212,0.14)" strokeWidth="1.5" />
              <line x1="0" y1={-2 * R} x2="0" y2={2 * R} stroke="rgba(242,163,58,0.22)" strokeWidth="1.2" />
            </g>
          ))}
        </g>
      </g>
      <circle r={R} fill="none" stroke="rgba(236,228,212,0.25)" strokeWidth="1.4" />
      <circle r={R + 26} fill="none" stroke="rgba(236,228,212,0.08)" strokeWidth="1" strokeDasharray="2 10" />
      <g transform={`rotate(${-spin * 0.4})`}>
        {Array.from({ length: 90 }, (_, i) => (
          <line key={i} x1={R + 36} x2={R + (i % 9 === 0 ? 56 : 44)} transform={`rotate(${i * 4})`}
            stroke="rgba(236,228,212,0.28)" strokeWidth="1" />
        ))}
        {F_STOPS.map((f, i) => (
          <text key={f} transform={`rotate(${-150 + i * 12}) translate(0 ${-R - 72}) rotate(0)`} textAnchor="middle"
            fill="rgba(236,228,212,0.4)" fontFamily="var(--mono)" fontSize="15">{f}</text>
        ))}
      </g>
    </svg>
  );
}

export default function Hero({ photo }) {
  const ref = useRef(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start start", "end end"] });
  const p = useSpring(scrollYProgress, { stiffness: 90, damping: 24, mass: 0.4 });
  const open = useTransform(p, [0, 0.8], [0.1, 1]);
  const photoScale = useTransform(p, [0, 1], [1.25, 1]);
  const copyY = useTransform(p, [0, 1], ["0%", "-40%"]);
  const copyOpacity = useTransform(p, [0.5, 0.78], [1, 0]);
  const [v, setV] = useState(open.get());
  useMotionValueEvent(open, "change", setV);
  const fstop = F_STOPS[Math.min(F_STOPS.length - 1, Math.max(0, Math.round(v * (F_STOPS.length - 1))))];

  return (
    <section className="hero" ref={ref} id="top">
      <div className="hero-stage">
        <motion.div className="hero-photo" style={{ backgroundImage: `url(${photo})`, scale: photoScale }} />
        <div className="hero-iris"><Iris v={v} /></div>

        <motion.div className="hero-left mono mute" style={{ opacity: copyOpacity }}>
          No-reference perceptual image quality, learned from 24,000 human-rated images across KonIQ-10k, SPAQ and KADID-10k.
        </motion.div>
        <motion.div className="hero-meta mono mute" style={{ opacity: copyOpacity }}>
          <span>aperture</span>
          <span className="fstop">ƒ/{fstop}</span>
          <span>ConvNeXt-B · CLIP · v2.0</span>
        </motion.div>

        <motion.div className="hero-copy" style={{ y: copyY, opacity: copyOpacity }}>
          <h1 className="display">
            <motion.span style={{ display: "block" }} initial={{ y: 40, opacity: 0 }} animate={{ y: 0, opacity: 1 }}
              transition={{ duration: 1.4, ease: [0.16, 1, 0.3, 1], delay: 0.2 }}>
              Every photograph
            </motion.span>
            <motion.span style={{ display: "block" }} initial={{ y: 40, opacity: 0 }} animate={{ y: 0, opacity: 1 }}
              transition={{ duration: 1.4, ease: [0.16, 1, 0.3, 1], delay: 0.35 }}>
              carries a <em>grade.</em>
            </motion.span>
          </h1>
          <div className="scroll-cue mono mute"><span />develop</div>
        </motion.div>
      </div>
    </section>
  );
}
