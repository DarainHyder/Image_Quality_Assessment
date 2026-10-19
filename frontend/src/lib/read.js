/** Human language for model outputs. */

export const DEFECT_LABEL = {
  blur: "Blur", noise: "Noise", artifacts: "Artifacts", exposure: "Exposure", contrast: "Contrast", color: "Colour",
};

export const ATTR_LABEL = {
  sharpness: "Sharpness", noisiness: "Clean", contrast: "Contrast", colorfulness: "Colour", brightness: "Light",
};

export function severity(v) {
  if (v < 0.12) return "none";
  if (v < 0.25) return "trace";
  if (v < 0.45) return "mild";
  if (v < 0.65) return "moderate";
  return "severe";
}

export function scoreColor(s) {
  if (s >= 65) return "var(--sage)";
  if (s >= 45) return "var(--paper)";
  if (s >= 25) return "var(--amber)";
  return "var(--safe)";
}

const DEFECT_NOTE = {
  blur: "detail is soft — focus or camera shake is costing sharpness",
  noise: "grain / sensor noise is visible in flat areas",
  artifacts: "compression blocks or processing artifacts are showing",
  exposure: "the exposure is off — highlights or shadows are losing information",
  contrast: "tonal range looks flat",
  color: "colour rendition looks shifted or over-processed",
};

export function verdict(r) {
  const top = Object.entries(r.defects).sort((a, b) => b[1] - a[1]);
  const [name, val] = top[0];
  const pct = r.percentile != null ? `Better than ${Math.round(r.percentile)}% of the 7,058 KonIQ reference photographs. ` : "";
  if (r.score >= 70 && val < 0.3) return `${pct}A keeper — clean, well-resolved and nothing the eye snags on.`;
  if (val < 0.2) return `${pct}No single defect dominates; the score reflects overall rendering and content.`;
  return `${pct}Main issue: ${DEFECT_NOTE[name]}.`;
}

export function fmtExposure(s) {
  if (!s) return null;
  return s >= 1 ? `${s}s` : `1/${Math.round(1 / s)}s`;
}
