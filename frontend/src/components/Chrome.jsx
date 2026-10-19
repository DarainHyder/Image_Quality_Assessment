import { motion, useScroll, useTransform } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { API_URL, GITHUB_URL, MODEL_URL, SPACE_URL, ping } from "../lib/api";
import { FadeUp, RevealLines } from "./Reveal";

export function Nav() {
  const [status, setStatus] = useState("waking");
  useEffect(() => {
    let stop = false;
    const ctl = new AbortController();
    const tick = async (n) => {
      try {
        const r = await ping(ctl.signal);
        if (!stop) setStatus(r.model_loaded ? "online" : "waking");
        if (!r.model_loaded && n < 30 && !stop) setTimeout(() => tick(n + 1), 6000);
      } catch {
        if (stop) return;
        setStatus(n < 20 ? "waking" : "offline");
        if (n < 20) setTimeout(() => tick(n + 1), 8000);
      }
    };
    tick(0);
    return () => { stop = true; ctl.abort(); };
  }, []);
  return (
    <nav className="nav">
      <a href="#top" className="brand">Lumina<sup>v2</sup></a>
      <ul className="mono">
        <li className="opt"><a className="link" href="#light-table">Light table</a></li>
        <li className="opt"><a className="link" href="#contact-sheet">Contact sheet</a></li>
        <li className="opt"><a className="link" href="#method">Method</a></li>
        <li className="opt"><a className="link" href="#bench">Bench</a></li>
        <li><a className="link" href={GITHUB_URL} target="_blank" rel="noreferrer">GitHub ↗</a></li>
        <li><span className="status" data-s={status} title={API_URL}><i />{status === "online" ? "live" : status}</span></li>
      </ul>
    </nav>
  );
}

const CURL = (
  <>
    <span className="c"># score a photo — multipart upload, JSON back</span>{"\n"}
    <span className="k">curl</span> -s -X POST {API_URL}/analyze \{"\n"}
    {"     "}-F <span className="s">"file=@frame.jpg"</span> | jq <span className="s">'{"{"}score, rating, percentile, defects{"}"}'</span>{"\n\n"}
    <span className="c"># →</span>{"\n"}
    {"{"}{"\n"}
    {"  "}<span className="s">"score"</span>: 71.84, <span className="s">"rating"</span>: <span className="s">"good"</span>, <span className="s">"percentile"</span>: 79,{"\n"}
    {"  "}<span className="s">"defects"</span>: {"{"} <span className="s">"blur"</span>: 0.08, <span className="s">"noise"</span>: 0.21, … {"}"}{"\n"}
    {"}"}
  </>
);

export function Api() {
  return (
    <section className="api" id="api">
      <div className="wrap">
        <FadeUp className="tag mono">§ 05 — Instrument API</FadeUp>
        <RevealLines className="h2" lines={["Wire it into", <>your <em>pipeline.</em></>]} />
        <div className="api-grid">
          <FadeUp>
            <ul className="endpoints">
              <li><b>POST</b><code>/analyze</code><span>Score, percentile, quality &amp; attention maps, six defect maps, SPAQ attributes, EXIF and forensics.</span></li>
              <li><b>POST</b><code>/predict</code><span>v1-compatible: <code>quality_score</code> plus rating and defects.</span></li>
              <li><b>GET</b><code>/</code><span>Model card, version and benchmark numbers.</span></li>
              <li><b>GET</b><code>/docs</code><span>Interactive OpenAPI explorer.</span></li>
            </ul>
            <p className="note">
              Served on CPU with ONNX Runtime (int8), no GPU or PyTorch at inference. Weights and model card live on the
              Hugging Face Hub at <a href={MODEL_URL} target="_blank" rel="noreferrer">sawabedarain/lumina-iqa</a>.
            </p>
          </FadeUp>
          <FadeUp delay={0.15}><pre className="code">{CURL}</pre></FadeUp>
        </div>
      </div>
    </section>
  );
}

export function Footer() {
  const ref = useRef(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start end", "end end"] });
  const x = useTransform(scrollYProgress, [0, 1], ["0%", "-38%"]);
  const words = ["Develop", <em key="a">carefully.</em>, "Expose", <em key="b">honestly.</em>, "Keep the", <em key="c">sharp ones.</em>];
  return (
    <footer className="footer" ref={ref}>
      <motion.div className="marquee" style={{ x }}>
        {[...words, ...words].map((w, i) => <span key={i}>{w}</span>)}
      </motion.div>
      <div className="wrap">
        <div className="foot-row mono mute">
          <span>Lumina IQA 2.0 · built by <a href="https://github.com/DarainHyder" target="_blank" rel="noreferrer">Darain Hyder</a></span>
          <span style={{ display: "flex", gap: 22 }}>
            <a href={GITHUB_URL} target="_blank" rel="noreferrer">Source</a>
            <a href={SPACE_URL} target="_blank" rel="noreferrer">Space</a>
            <a href={MODEL_URL} target="_blank" rel="noreferrer">Model</a>
            <a href={`${API_URL}/docs`} target="_blank" rel="noreferrer">API docs</a>
          </span>
          <span>KonIQ-10k · KonIQ++ · SPAQ · KADID-10k · LIVE-C</span>
        </div>
      </div>
    </footer>
  );
}
