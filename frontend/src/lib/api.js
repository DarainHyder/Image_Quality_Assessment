export const API_URL = (import.meta.env.VITE_API_URL || "https://sawabedarain-lumina-iqa.hf.space").replace(/\/$/, "");
export const GITHUB_URL = "https://github.com/DarainHyder/Image_Quality_Assessment";
export const SPACE_URL = "https://huggingface.co/spaces/sawabedarain/Lumina-IQA";
export const MODEL_URL = "https://huggingface.co/sawabedarain/lumina-iqa";

const MAX_UPLOAD = 8 * 1024 * 1024;

/** Very large files are downscaled client-side before upload (the model reads 512 px anyway). */
async function shrinkIfHuge(file) {
  if (file.size <= MAX_UPLOAD) return file;
  const bmp = await createImageBitmap(file);
  const s = Math.min(1, 2400 / Math.max(bmp.width, bmp.height));
  const c = document.createElement("canvas");
  c.width = Math.round(bmp.width * s);
  c.height = Math.round(bmp.height * s);
  c.getContext("2d").drawImage(bmp, 0, 0, c.width, c.height);
  const blob = await new Promise((r) => c.toBlob(r, "image/jpeg", 0.95));
  return new File([blob], file.name.replace(/\.\w+$/, "") + ".jpg", { type: "image/jpeg" });
}

export async function analyze(file, { signal } = {}) {
  const body = new FormData();
  body.append("file", await shrinkIfHuge(file));
  let res;
  try {
    res = await fetch(`${API_URL}/analyze`, { method: "POST", body, signal });
  } catch (e) {
    if (e.name === "AbortError") throw e;
    throw new Error("The instrument is unreachable — the Hugging Face Space may be waking up. Try again in a minute.");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* non-JSON error page */ }
    throw new Error(res.status === 503 ? "Model is still loading on the server — retry shortly." : detail);
  }
  return res.json();
}

export async function ping(signal) {
  const res = await fetch(`${API_URL}/`, { signal });
  if (!res.ok) throw new Error(String(res.status));
  return res.json();
}

export async function fileFromUrl(url, name) {
  const blob = await (await fetch(url)).blob();
  return new File([blob], name, { type: blob.type || "image/jpeg" });
}
