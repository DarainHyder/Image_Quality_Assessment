import { AnimatePresence, motion } from "framer-motion";
import { useMemo, useRef, useState } from "react";
import { analyze, fileFromUrl } from "../lib/api";
import { DEFECT_LABEL, scoreColor } from "../lib/read";
import { SAMPLES } from "./LightTable";
import { FadeUp, RevealLines } from "./Reveal";

const MAX = 12;

/* Hand-drawn grease-pencil marks: wobbly ellipse for the keeper, a cross for the reject. */
function Circle() {
  return (
    <svg className="mark" viewBox="0 0 100 100" preserveAspectRatio="none">
      <motion.path d="M50 6 C78 5 97 22 96 50 C95 79 74 95 47 94 C20 93 3 76 4 49 C5 24 24 8 54 9 C70 10 80 15 86 21"
        fill="none" stroke="var(--amber)" strokeWidth="1.6" vectorEffect="non-scaling-stroke" strokeLinecap="round"
        initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1.3, ease: "easeInOut", delay: 0.3 }} />
    </svg>
  );
}
function Cross() {
  return (
    <svg className="mark" viewBox="0 0 100 100" preserveAspectRatio="none">
      {["M14 16 C40 40 62 62 88 86", "M86 14 C62 40 40 60 12 88"].map((d, i) => (
        <motion.path key={d} d={d} fill="none" stroke="var(--safe)" strokeWidth="1.8" vectorEffect="non-scaling-stroke" strokeLinecap="round"
          initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 0.5, delay: 0.4 + i * 0.35 }} />
      ))}
    </svg>
  );
}

export default function ContactSheet() {
  const [frames, setFrames] = useState([]);
  const [busy, setBusy] = useState(false);
  const inputRef = useRef(null);

  async function load(files) {
    const list = Array.from(files).filter((f) => f.type.startsWith("image/")).slice(0, MAX);
    if (!list.length) return;
    frames.forEach((f) => URL.revokeObjectURL(f.url));
    const fresh = list.map((file, i) => ({ id: `${Date.now()}-${i}`, file, url: URL.createObjectURL(file), r: null, err: null }));
    setFrames(fresh);
    setBusy(true);
    for (const fr of fresh) {  // sequential: the free CPU Space has two cores
      try {
        const r = await analyze(fr.file);
        setFrames((cur) => cur.map((c) => (c.id === fr.id ? { ...c, r } : c)));
      } catch (e) {
        setFrames((cur) => cur.map((c) => (c.id === fr.id ? { ...c, err: e.message } : c)));
      }
    }
    setBusy(false);
  }

  const ranked = useMemo(() => {
    const done = frames.filter((f) => f.r);
    if (busy || done.length < frames.length) return frames;
    return [...frames].sort((a, b) => (b.r?.score ?? -1) - (a.r?.score ?? -1));
  }, [frames, busy]);
  const settled = !busy && frames.length > 1 && frames.every((f) => f.r || f.err);

  return (
    <section className="sheet" id="contact-sheet">
      <div className="wrap">
        <FadeUp className="tag mono">§ 02 — The contact sheet</FadeUp>
        <div className="table-head" style={{ marginBottom: 0 }}>
          <RevealLines className="h2" lines={["Shot a burst?", <>Let it pick the <em>keeper.</em></>]} />
          <FadeUp delay={0.15}>
            <p className="lede">
              Drop up to {MAX} frames of the same moment. Each is scored independently, then the strip re-orders itself
              best-to-worst. The keeper gets circled; the reject gets crossed out.
            </p>
          </FadeUp>
        </div>
        <div className="sheet-drop">
          <button className="btn" onClick={() => inputRef.current?.click()} disabled={busy}>
            {busy ? "Developing…" : "Load frames"} <span aria-hidden>→</span>
          </button>
          <button className="btn ghost" disabled={busy}
            onClick={async () => load(await Promise.all(SAMPLES.map((s) => fileFromUrl(s.src, s.src.split("/").pop()))))}>
            Use the sample roll
          </button>
          <input ref={inputRef} type="file" accept="image/*" multiple hidden onChange={(e) => load(e.target.files)} />
        </div>
      </div>

      <div className="strip"
        onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); load(e.dataTransfer.files); }}>
        <div className="strip-inner">
          {!frames.length && (
            <p className="mono mute" style={{ paddingInline: 8 }}>
              ◂ empty roll — drag frames onto the strip ▸
            </p>
          )}
          <AnimatePresence>
            {ranked.map((f, i) => {
              const top = f.r && Object.entries(f.r.defects).sort((a, b) => b[1] - a[1])[0];
              return (
                <motion.figure layout key={f.id} className="frame-cell" style={{ margin: 0 }}
                  initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                  transition={{ layout: { duration: 0.9, ease: [0.16, 1, 0.3, 1] }, duration: 0.6, delay: i * 0.05 }}>
                  <div className="shot">
                    <img src={f.url} alt="" style={{ filter: f.r ? "none" : "grayscale(1) brightness(0.5)", transition: "filter 1.2s" }} />
                    {!f.r && !f.err && <div className="scanline" />}
                  </div>
                  {settled && i === 0 && <Circle />}
                  {settled && i === ranked.length - 1 && <Cross />}
                  <figcaption className="meta">
                    <span>{String(i + 1).padStart(2, "0")}{settled ? (i === 0 ? " · keeper" : "") : ""}</span>
                    <span style={{ color: f.r ? scoreColor(f.r.score) : "var(--mute)" }}>
                      {f.err ? "error" : f.r ? f.r.score.toFixed(1) : "…"}
                    </span>
                  </figcaption>
                  {top && top[1] > 0.3 && <div className="mono mute" style={{ marginTop: 4 }}>{DEFECT_LABEL[top[0]]} {Math.round(top[1] * 100)}</div>}
                </motion.figure>
              );
            })}
          </AnimatePresence>
        </div>
      </div>
    </section>
  );
}
