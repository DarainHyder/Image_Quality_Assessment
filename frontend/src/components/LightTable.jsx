import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { analyze, fileFromUrl } from "../lib/api";
import { containedBox, paintMap } from "../lib/heat";
import { DEFECT_LABEL, fmtExposure, scoreColor, verdict } from "../lib/read";
import { DefectScales, Histogram, Meter, Radar } from "./Instruments";
import { FadeUp, RevealLines } from "./Reveal";

export const SAMPLES = [
  { src: "/samples/lake.jpg", label: "pristine" },
  { src: "/samples/peaks-motion.jpg", label: "shake" },
  { src: "/samples/portrait-iso.jpg", label: "high iso" },
  { src: "/samples/road-jpeg.jpg", label: "jpeg q6" },
  { src: "/samples/valley-dark.jpg", label: "under" },
];

const LOG = ["exposing print", "multi-scale ConvNeXt pass", "reading quality field", "weighting attention", "fixing verdict"];

function Developing() {
  const [n, setN] = useState(1);
  useEffect(() => {
    const t = setInterval(() => setN((v) => Math.min(LOG.length, v + 1)), 650);
    return () => clearInterval(t);
  }, []);
  return (
    <div className="log">
      {LOG.slice(0, n).map((l, i) => (
        <div key={l}><b>{String(i + 1).padStart(2, "0")}</b> — {l}{i === n - 1 ? "…" : ""}</div>
      ))}
    </div>
  );
}

export default function LightTable() {
  const [preview, setPreview] = useState(null);
  const [name, setName] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [mode, setMode] = useState("quality");
  const [over, setOver] = useState(false);
  const [ratio, setRatio] = useState(4 / 3);
  const inputRef = useRef(null);
  const imgRef = useRef(null);
  const canvasRef = useRef(null);
  const abortRef = useRef(null);

  const run = useCallback(async (file) => {
    if (!file || !file.type.startsWith("image/")) return;
    abortRef.current?.abort();
    const ctl = new AbortController();
    abortRef.current = ctl;
    setPreview((old) => { if (old) URL.revokeObjectURL(old); return URL.createObjectURL(file); });
    setName(file.name);
    setResult(null);
    setError(null);
    setLoading(true);
    try {
      const r = await analyze(file, { signal: ctl.signal });
      setResult(r);
      const worst = Object.entries(r.defects).sort((a, b) => b[1] - a[1])[0];
      setMode(worst[1] > 0.35 ? worst[0] : "quality");
    } catch (e) {
      if (e.name !== "AbortError") setError(e.message);
    } finally {
      if (abortRef.current === ctl) setLoading(false);
    }
  }, []);

  const repaint = useCallback(() => {
    const img = imgRef.current;
    const cv = canvasRef.current;
    if (!img || !cv || !result || !img.naturalWidth) return;
    if (mode === "print") { cv.style.opacity = 0; return; }
    const grid = mode === "quality" ? result.maps.quality : mode === "attention" ? result.maps.attention : result.maps.defects[mode];
    paintMap(cv, grid, mode === "quality" || mode === "attention" ? mode : "defect", containedBox(img));
    cv.style.opacity = 1;
  }, [mode, result]);

  useLayoutEffect(() => { repaint(); }, [repaint]);
  useEffect(() => {
    const ro = new ResizeObserver(() => repaint());
    if (imgRef.current) ro.observe(imgRef.current);
    return () => ro.disconnect();
  }, [repaint, preview]);

  const onDrop = (e) => {
    e.preventDefault();
    setOver(false);
    run(e.dataTransfer.files?.[0]);
  };

  const t = result?.technical;
  const ex = t?.exif || {};
  const rebate = [ex.make && ex.model && !ex.model.startsWith(ex.make) ? `${ex.make} ${ex.model}` : ex.model || ex.make,
    ex.focal_mm && `${Math.round(ex.focal_mm)}mm`, ex.f_number && `ƒ/${ex.f_number}`, fmtExposure(ex.exposure_s),
    ex.iso && `ISO ${ex.iso}`].filter(Boolean);

  return (
    <section className="table" id="light-table">
      <div className="wrap">
        <div className="table-head">
          <div>
            <FadeUp className="tag mono">§ 01 — The light table</FadeUp>
            <RevealLines className="h2" lines={["Lay down a frame.", <>Read it like an <em>editor</em> would.</>]} />
          </div>
          <FadeUp delay={0.2}>
            <p className="lede">
              Drop any photograph. Lumina scores it the way a panel of human viewers would, then shows you where the
              grade comes from: a per-region quality field, where viewers look, and which defects are dragging it down.
            </p>
          </FadeUp>
        </div>

        <div className="table-grid">
          <FadeUp className="easel" y={60}>
            <div className="print">
              <span className="tape a" /><span className="tape b" />
              <div className="frame" style={{ aspectRatio: ratio }}
                onDragOver={(e) => { e.preventDefault(); setOver(true); }}
                onDragLeave={() => setOver(false)} onDrop={onDrop}>
                {preview ? (
                  <>
                    <img ref={imgRef} key={preview} src={preview} alt={name} className={result ? "develop" : ""}
                      style={result ? undefined : { filter: "grayscale(1) brightness(0.55)" }}
                      onLoad={(e) => { const t = e.currentTarget; setRatio(Math.min(1.8, Math.max(0.75, t.naturalWidth / t.naturalHeight))); repaint(); }} />
                    <canvas ref={canvasRef} className="heat" style={{ opacity: 0 }} />
                    {loading && <><div className="safelight" /><div className="scanline" /></>}
                  </>
                ) : (
                  <div className={`drop ${over ? "over" : ""}`} onClick={() => inputRef.current?.click()} role="button"
                    tabIndex={0} onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}>
                    <span className="reg tl" /><span className="reg tr" /><span className="reg bl" /><span className="reg br" />
                    <div>
                      <div className="cross"><i /></div>
                      <h3>Drop a frame here</h3>
                      <div className="mono mute">jpeg · png · webp · heic-exported — up to 20 MB</div>
                    </div>
                  </div>
                )}
              </div>
              <div className="print-caption">
                <span>{name ? name.slice(0, 34) : "untitled — no. 000"}</span>
                {result && <b style={{ color: "#231e18" }}>{result.score.toFixed(1)}</b>}
              </div>
            </div>
            <input ref={inputRef} type="file" accept="image/*" hidden onChange={(e) => run(e.target.files?.[0])} />

            <div className="modes" role="group" aria-label="Overlay">
              {[["print", "Print"], ["quality", "Quality field"], ["attention", "Attention"],
                ...Object.keys(DEFECT_LABEL).map((k) => [k, DEFECT_LABEL[k]])].map(([k, label]) => (
                <button key={k} className="mode" aria-pressed={mode === k} disabled={!result} onClick={() => setMode(k)}>{label}</button>
              ))}
            </div>

            <div className="samples">
              <button className="btn ghost" onClick={() => inputRef.current?.click()}>Choose file</button>
              <span className="mono mute" style={{ marginInline: 8 }}>or try</span>
              {SAMPLES.map((s) => (
                <button key={s.src} title={s.label} onClick={async () => run(await fileFromUrl(s.src, s.src.split("/").pop()))}>
                  <img src={s.src} alt={s.label} loading="lazy" /><span>{s.label}</span>
                </button>
              ))}
            </div>
          </FadeUp>

          <div className="readout">
            <FadeUp>
              <Meter score={result?.score ?? null} />
              <div className="score-line">
                <div className="score-num" style={{ color: result ? scoreColor(result.score) : "var(--mute-2)" }}>
                  {result ? result.score.toFixed(1) : "—"}<small>/100</small>
                </div>
                <div className="rating" style={{ color: result ? scoreColor(result.score) : "var(--mute-2)" }}>
                  {result ? result.rating : loading ? "developing" : "awaiting print"}
                </div>
              </div>
            </FadeUp>

            <AnimatePresence mode="wait">
              {loading && (
                <motion.div key="log" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}><Developing /></motion.div>
              )}
              {error && !loading && (
                <motion.p key="err" className="err" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>{error}</motion.p>
              )}
              {result && !loading && (
                <motion.div key={name + result.score} className="readout" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}>
                  <p className="verdict">{verdict(result)}</p>

                  <div>
                    <div className="block-title"><span className="mono">Defect signature</span><span className="mono mute">0 — 100</span></div>
                    <DefectScales defects={result.defects} />
                  </div>

                  <div className="twocol">
                    <div>
                      <div className="block-title"><span className="mono">Perceived</span><span className="mono mute">spaq-calibrated</span></div>
                      <Radar attrs={result.attributes} />
                    </div>
                    <div>
                      <div className="block-title"><span className="mono">Luminance</span><span className="mono mute">{t.luma_mean.toFixed(0)} avg</span></div>
                      <Histogram bins={t.histogram} clipped={t.clipped_highlights} crushed={t.crushed_shadows} />
                      <div className="swatches">
                        {t.palette.map((p) => <span key={p.hex} style={{ background: p.hex, width: `${p.share * 100}%` }} title={p.hex} />)}
                      </div>
                    </div>
                  </div>

                  <div>
                    <div className="rebate">▸ {rebate.length ? rebate.join("  ·  ") : "no exif — stripped or synthetic"} ▸ {t.width}×{t.height}</div>
                    <dl className="facts">
                      <div><dt className="mono">Megapixels</dt><dd>{t.megapixels}</dd></div>
                      <div><dt className="mono">JPEG q</dt><dd>{t.jpeg_quality ?? "—"}</dd></div>
                      <div><dt className="mono">Laplacian σ²</dt><dd>{Math.round(t.sharpness)}</dd></div>
                      <div><dt className="mono">Noise σ</dt><dd>{t.noise_sigma.toFixed(2)}</dd></div>
                      <div><dt className="mono">Colourfulness</dt><dd>{t.colorfulness.toFixed(1)}</dd></div>
                      <div><dt className="mono">Inference</dt><dd>{Math.round(result.inference_ms)} ms</dd></div>
                    </dl>
                  </div>
                </motion.div>
              )}
              {!result && !loading && !error && (
                <motion.p key="idle" className="mute" initial={{ opacity: 0 }} animate={{ opacity: 1 }} style={{ maxWidth: "40ch" }}>
                  The needle stays at rest until a print is on the table. Nothing is stored — images are scored in memory and discarded.
                </motion.p>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </section>
  );
}
