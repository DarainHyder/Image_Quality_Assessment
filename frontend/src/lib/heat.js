/**
 * Paint a coarse model map (h x w grid) onto a canvas sized to the displayed image.
 * The grid is drawn into a tiny canvas and upscaled with bilinear smoothing, so it
 * reads as a soft exposure mask rather than blocky tiles.
 */
const PALETTES = {
  // low quality -> safelight red, high quality -> transparent
  quality: (v) => {
    const t = Math.min(1, Math.max(0, (60 - v) / 36));
    return [255, 60 + 40 * (1 - t), 30, Math.round(175 * t)];
  },
  attention: (v) => [242, 163, 58, Math.round(190 * Math.pow(v, 1.2))],
  defect: (v) => {
    const t = Math.min(1, Math.max(0, (v - 0.2) / 0.6));
    return [255, 74, 46, Math.round(185 * t)];
  },
};

export function paintMap(canvas, grid, kind, box) {
  if (!canvas || !grid) return;
  const h = grid.length;
  const w = grid[0].length;
  const small = document.createElement("canvas");
  small.width = w;
  small.height = h;
  const sctx = small.getContext("2d");
  const img = sctx.createImageData(w, h);
  const pal = PALETTES[kind] || PALETTES.defect;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const [r, g, b, a] = pal(grid[y][x]);
      const i = (y * w + x) * 4;
      img.data[i] = r; img.data[i + 1] = g; img.data[i + 2] = b; img.data[i + 3] = a;
    }
  }
  sctx.putImageData(img, 0, 0);
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(box.width * dpr);
  canvas.height = Math.round(box.height * dpr);
  Object.assign(canvas.style, { left: `${box.left}px`, top: `${box.top}px`, width: `${box.width}px`, height: `${box.height}px` });
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.filter = `blur(${Math.max(2, canvas.width / w / 3)}px)`;
  ctx.drawImage(small, 0, 0, canvas.width, canvas.height);
}

/** Rectangle actually occupied by an object-fit: contain image inside its element. */
export function containedBox(imgEl) {
  const cw = imgEl.clientWidth;
  const ch = imgEl.clientHeight;
  const ir = imgEl.naturalWidth / imgEl.naturalHeight;
  const cr = cw / ch;
  const width = ir > cr ? cw : ch * ir;
  const height = ir > cr ? cw / ir : ch;
  return { width, height, left: (cw - width) / 2, top: (ch - height) / 2 };
}
