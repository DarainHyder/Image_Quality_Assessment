"""Shared constants for training, export and serving."""

BACKBONE = "convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384"

# Native training canvas (W, H) — KonIQ-10k / KADID-10k resolution, aspect ratio preserved.
CANVAS = (512, 384)

DEFECTS = ("blur", "noise", "artifacts", "exposure", "contrast", "color")

# Score sources; each gets its own affine calibration head (KonIQ is the canonical scale).
SOURCES = ("koniq", "kadid", "spaq")

# Human-rated perceptual attributes (SPAQ), predicted on a 0-100 scale.
ATTRS = ("brightness", "colorfulness", "contrast", "noisiness", "sharpness")

# KADID-10k distortion id (1..25) -> defect family.
KADID_DEFECT = {
    1: "blur", 2: "blur", 3: "blur",                                   # gaussian / lens / motion blur
    4: "color", 5: "color", 6: "color", 7: "color", 8: "color",        # diffusion, shift, quantisation, saturation
    9: "artifacts", 10: "artifacts",                                   # JPEG2000, JPEG
    11: "noise", 12: "noise", 13: "noise", 14: "noise",                # white / colour / impulse / multiplicative
    15: "blur",                                                        # over-denoising smears detail
    16: "exposure", 17: "exposure", 18: "exposure",                    # brighten, darken, mean shift
    19: "artifacts", 20: "artifacts", 21: "artifacts", 22: "artifacts", 23: "artifacts",  # spatial distortions
    24: "artifacts",                                                   # over-sharpening halos
    25: "contrast",
}
