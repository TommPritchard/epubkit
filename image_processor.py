"""
Image processor for Xteink EPUB Optimizer (X4, X3).
Handles: baseline JPEG conversion, Smart Hybrid cover fitting,
chromatic-aware grayscale conversion, local adaptive contrast,
serpentine halo-free 4-level dithering, Light Novel mode.

Device profiles (display orientation, portrait):
  X4 (SSD1677 controller):
    - Display: 480x800, 4-level grayscale (black, dark gray, light gray, white)
  X3 (SSD1677 controller):
    - Display: 528x792, same 4-level grayscale hardware
    - Note: stock X3 firmware does not render EPUB images at all; image
      output targets CrossPoint-family firmware
  Both:
    - Max image: 1024x1024
    - RAM: 380KB — smaller images = faster rendering
"""

import io
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

from PIL import Image, ImageEnhance, ImageOps, ImageDraw, ImageFont, ImageFilter


DEVICE_PROFILES = {
    'x4': {
        'width': 480,
        'height': 800,
        'gray_levels': [0, 85, 170, 255],
        'label': 'Xteink X4',
    },
    'x3': {
        'width': 528,
        'height': 792,
        'gray_levels': [0, 85, 170, 255],
        'label': 'Xteink X3',
    },
}
DEFAULT_DEVICE = 'x4'

X4_WIDTH = DEVICE_PROFILES['x4']['width']
X4_HEIGHT = DEVICE_PROFILES['x4']['height']

MAX_IMAGE_DIMENSION = 1024
EINK_PALETTE_LEVELS = DEVICE_PROFILES['x4']['gray_levels']
SUPPORTED_EXTENSIONS = {'.png', '.gif', '.webp', '.bmp', '.jpeg', '.jpg', '.tif', '.tiff'}


@dataclass
class ImageOptions:
    grayscale: bool = True
    contrast_boost: bool = True
    contrast_factor: float = 1.5
    quality: int = 70
    max_width: int = X4_WIDTH
    max_height: int = X4_HEIGHT
    eink_quantize: bool = True
    gray_levels: list = field(default_factory=lambda: list(EINK_PALETTE_LEVELS))
    light_novel_mode: bool = False
    light_novel_rotate_left: bool = True
    is_cover: bool = False

    @classmethod
    def for_device(cls, device: str, **overrides) -> 'ImageOptions':
        """Build options from a device profile ('x4' or 'x3')."""
        profile = DEVICE_PROFILES.get(device, DEVICE_PROFILES[DEFAULT_DEVICE])
        defaults = {
            'max_width': profile['width'],
            'max_height': profile['height'],
            'gray_levels': list(profile['gray_levels']),
        }
        defaults.update(overrides)
        return cls(**defaults)


@dataclass
class ImageResult:
    output_bytes: bytes
    new_filename: str
    original_size: int
    new_size: int
    was_converted: bool
    details: str


def should_process(filename: str) -> bool:
    """Check if a file is a processable image based on extension."""
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


def is_progressive_jpeg(image_bytes: bytes) -> bool:
    """Check if JPEG data is progressive/interlaced."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.format != 'JPEG':
            return False
        return bool(img.info.get('progressive', False) or img.info.get('progression', False))
    except Exception:
        return False


def _handle_transparency(img: Image.Image) -> Image.Image:
    """Composite transparent images onto white background."""
    if img.mode in ('RGBA', 'LA', 'PA'):
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'PA':
            img = img.convert('RGBA')
        background.paste(img, mask=img.split()[-1])
        return background
    if img.mode == 'P':
        if 'transparency' in img.info:
            img = img.convert('RGBA')
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            return background
        return img.convert('RGB')
    return img


def _handle_light_novel(img: Image.Image, rotate_left: bool) -> List[Image.Image]:
    """Light Novel mode: rotate landscape images or split double-page spreads."""
    width, height = img.size
    if width <= height:
        return [img]

    aspect = width / height
    if aspect > 1.8:
        mid = width // 2
        right_half = img.crop((mid, 0, width, height))
        left_half = img.crop((0, 0, mid, height))
        return [right_half, left_half]
    else:
        angle = 90 if rotate_left else -90
        return [img.rotate(angle, expand=True)]


def _to_eink_grayscale(img: Image.Image) -> Image.Image:
    """
    Convert RGB/RGBA/P image to grayscale ('L') using chromatic-aware blend:
    80% ITU-R 601-2 perceptual luma + 20% HSV Value (max(R, G, B)).
    Preserves saturated reds, warm skin tones, and gold titles.
    """
    if img.mode == 'L':
        return img.copy()

    img = _handle_transparency(img)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    r_band, g_band, b_band = img.split()
    r_px = list(r_band.getdata())
    g_px = list(g_band.getdata())
    b_px = list(b_band.getdata())

    out_px = [
        min(
            255,
            int(
                (0.299 * r + 0.587 * g + 0.114 * b) * 0.80
                + (r if r >= g and r >= b else (g if g >= b else b)) * 0.20
                + 0.5
            ),
        )
        for r, g, b in zip(r_px, g_px, b_px)
    ]

    gray = Image.new('L', img.size)
    gray.putdata(out_px)
    return gray


def _analyze_border_uniformity(img: Image.Image, is_top: bool) -> Tuple[bool, Tuple[int, int, int]]:
    """
    Measure if a border is a uniform / solid color (e.g. Kazuo Ishiguro's Faber covers, Penguin classics, etc.)
    Returns (is_uniform, (r, g, b)).
    """
    w, h = img.size
    sample_y = range(0, min(h, 4)) if is_top else range(max(0, h - 4), h)
    rgb_img = img.convert('RGB')
    px = rgb_img.load()
    r_sum, g_sum, b_sum, sq_sum, cnt = 0, 0, 0, 0, 0
    step = max(1, w // 100)
    for y in sample_y:
        for x in range(0, w, step):
            r, g, b = px[x, y]
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            r_sum += r
            g_sum += g
            b_sum += b
            sq_sum += lum * lum
            cnt += 1
    if cnt == 0:
        return True, (128, 128, 128)
    mean_lum = (0.299 * r_sum + 0.587 * g_sum + 0.114 * b_sum) / cnt
    var_lum = (sq_sum / cnt) - (mean_lum * mean_lum)
    std_lum = math.sqrt(max(0.0, var_lum))
    is_uniform = std_lum < 28.0
    avg_color = (int(r_sum / cnt), int(g_sum / cnt), int(b_sum / cnt))
    return is_uniform, avg_color


def _fit_cover_smart_hybrid(img: Image.Image, target_w: int = 480, target_h: int = 800) -> Image.Image:
    """
    Smart Hybrid full-screen cover fitting (480x800 for X4, 528x792 for X3):
    1. Zero Distortion: Aspect ratio is strictly preserved (0% vertical or horizontal stretch).
       Squares remain squares, circles remain circles, faces and figures are never distorted.
    2. Zero Cropping: The complete cover artwork, title, and author remain 100% visible.
    3. Seamless Padding:
       - If top/bottom border is uniform/plain (any color: yellow, cream, pastel, orange, black, etc.),
         the border color extends to fill the top/bottom pad with zero visible seam.
       - If photographic or painted artwork reaches the edges, an ambient Gaussian-blurred backdrop
         fills the pad naturally behind the crisp, centered cover.
    """
    sw, sh = img.size
    if sw <= 0 or sh <= 0:
        return img
    if sw == target_w and sh == target_h:
        return img

    target_ratio = float(target_w) / float(target_h)
    src_ratio = float(sw) / float(sh)

    if abs(src_ratio - target_ratio) <= 0.015:
        return img.resize((target_w, target_h), Image.Resampling.LANCZOS)

    img_rgb = img.convert('RGB')

    if src_ratio > target_ratio:
        # Cover is wider than screen aspect (e.g. 2:3 = 0.667 vs 3:5 = 0.600):
        # Scale width to target_w, natural height is smaller than target_h
        natural_h = int(round(target_w / src_ratio))
        deficit_h = target_h - natural_h
        pad_top = deficit_h // 2
        pad_bot = deficit_h - pad_top

        top_uniform, top_col = _analyze_border_uniformity(img_rgb, is_top=True)
        bot_uniform, bot_col = _analyze_border_uniformity(img_rgb, is_top=False)

        scaled_cover = img_rgb.resize((target_w, natural_h), Image.Resampling.LANCZOS)

        if top_uniform and bot_uniform:
            canvas = Image.new('RGB', (target_w, target_h), top_col)
            if pad_bot > 0:
                bot_slice = Image.new('RGB', (target_w, pad_bot + 2), bot_col)
                canvas.paste(bot_slice, (0, pad_top + natural_h - 2))
            canvas.paste(scaled_cover, (0, pad_top))
            return canvas
        else:
            # Ambient mirror/blurred backdrop for complex photographic/painting art
            bg_scale = max(float(target_w) / sw, float(target_h) / sh)
            bg_w = int(round(sw * bg_scale))
            bg_h = int(round(sh * bg_scale))
            bg = img_rgb.resize((bg_w, bg_h), Image.Resampling.LANCZOS)
            crop_x = (bg_w - target_w) // 2
            crop_y = (bg_h - target_h) // 2
            bg = bg.crop((crop_x, crop_y, crop_x + target_w, crop_y + target_h))
            bg = bg.filter(ImageFilter.GaussianBlur(radius=28))

            if top_uniform:
                top_slice = Image.new('RGB', (target_w, pad_top + 1), top_col)
                bg.paste(top_slice, (0, 0))
            if bot_uniform:
                bot_slice = Image.new('RGB', (target_w, pad_bot + 1), bot_col)
                bg.paste(bot_slice, (0, pad_top + natural_h - 1))

            bg.paste(scaled_cover, (0, pad_top))
            return bg
    else:
        # Cover is taller than screen: scale height to target_h, pad left/right
        natural_w = int(round(target_h * src_ratio))
        deficit_w = target_w - natural_w
        pad_left = deficit_w // 2

        scaled_cover = img_rgb.resize((natural_w, target_h), Image.Resampling.LANCZOS)
        bg_scale = max(float(target_w) / sw, float(target_h) / sh)
        bg_w = int(round(sw * bg_scale))
        bg_h = int(round(sh * bg_scale))
        bg = img_rgb.resize((bg_w, bg_h), Image.Resampling.LANCZOS)
        crop_x = (bg_w - target_w) // 2
        crop_y = (bg_h - target_h) // 2
        bg = bg.crop((crop_x, crop_y, crop_x + target_w, crop_y + target_h))
        bg = bg.filter(ImageFilter.GaussianBlur(radius=28))
        bg.paste(scaled_cover, (pad_left, 0))
        return bg


def _enhance_for_eink(gray_img: Image.Image, options: ImageOptions) -> Tuple[Image.Image, list]:
    """
    Adaptive pre-processing for 4-level e-ink:
    1. 0.5% percentile stretch + bimodal line-art detection
    2. Dead-zone clamping ([0..7] -> 0, [248..255] -> 255) to eliminate background speckle
    3. Adaptive shadow-lift curve + highlight roll-up
    4. Local Adaptive Contrast (Gaussian local dodge & burn)
    5. Fine UnsharpMask prior to quantization
    """
    w, h = gray_img.size
    total_pixels = w * h
    if total_pixels == 0:
        return gray_img, []

    hist = gray_img.histogram()
    clip_pixels = max(1, int(total_pixels * 0.005))

    low = 0
    acc = 0
    for i, count in enumerate(hist):
        acc += count
        if acc > clip_pixels:
            low = i
            break

    high = 255
    acc = 0
    for i in range(255, -1, -1):
        acc += hist[i]
        if acc > clip_pixels:
            high = i
            break

    if high <= low + 10:
        low, high = 0, 255

    extreme_pixels = sum(hist[:32]) + sum(hist[224:])
    is_bimodal_line_art = (extreme_pixels / total_pixels) > 0.82

    dark_pixels = sum(hist[:90])
    dark_ratio = dark_pixels / total_pixels

    if is_bimodal_line_art:
        shadow_lift = 0.0
    elif dark_ratio > 0.45:
        shadow_lift = 0.36
    elif dark_ratio > 0.25:
        shadow_lift = 0.25
    else:
        shadow_lift = 0.15

    user_contrast = float(getattr(options, 'contrast_factor', 1.0) or 1.0)
    span = max(1, high - low)

    lut = []
    for i in range(256):
        v = (i - low) * 255.0 / span
        if v <= 7.0:
            lut.append(0)
            continue
        if v >= 248.0:
            lut.append(255)
            continue

        u = v / 255.0
        if shadow_lift > 0.0:
            u = u + shadow_lift * u * ((1.0 - u) ** 1.4)

        if not is_bimodal_line_art:
            s_strength = 0.045 * user_contrast
            u = u - s_strength * math.sin(2.0 * math.pi * u)
            if u > 0.78:
                u = 0.78 + (u - 0.78) * 1.12

        out_v = int(round(max(0.0, min(1.0, u)) * 255.0))
        lut.append(out_v)

    mapped_img = gray_img.point(lut)
    mapped_px = list(mapped_img.getdata())

    if is_bimodal_line_art:
        sharpened = mapped_img.filter(
            ImageFilter.UnsharpMask(radius=1.0, percent=80, threshold=4)
        )
        return sharpened, mapped_px

    local_mean_img = mapped_img.filter(ImageFilter.GaussianBlur(radius=16))
    mean_px = list(local_mean_img.getdata())

    local_enhanced_px = [0] * total_pixels
    for i in range(total_pixels):
        m = mapped_px[i]
        if m <= 6 or m >= 249:
            local_enhanced_px[i] = m
            continue
        mean_val = mean_px[i]
        diff = m - mean_val
        if mean_val < 35:
            tone_weight = max(0.0, (mean_val - 10.0) / 25.0)
        elif mean_val > 200:
            tone_weight = max(0.0, (245.0 - mean_val) / 45.0)
        else:
            tone_weight = 1.0
        boosted = m + 0.50 * tone_weight * diff
        local_enhanced_px[i] = min(255, max(0, int(round(boosted))))

    local_img = Image.new('L', (w, h))
    local_img.putdata(local_enhanced_px)

    sharpened = local_img.filter(
        ImageFilter.UnsharpMask(radius=1.1, percent=85, threshold=3)
    )
    return sharpened, mapped_px


def _quantize_4level_serpentine(sharpened_img: Image.Image, mapped_px: list) -> Image.Image:
    """
    Serpentine Edge-Aware Damped 4-Level [0, 85, 170, 255] Dithering:
    - Alternates scan direction on even/odd rows to eliminate diagonal worms.
    - Damped 14/16 error diffusion for crisp e-paper contrast.
    - Attenuates error propagation across sharp edges (> 64 jump) by 75% to eliminate text haloing.
    - Pins dead-zone pure blacks (0) and pure whites (255) so backgrounds stay clean.
    """
    w, h = sharpened_img.size
    src_px = list(sharpened_img.getdata())
    work = [float(v) for v in src_px]

    for y in range(h):
        reverse = (y % 2 == 1)
        if reverse:
            x_iter = range(w - 1, -1, -1)
            x_step = -1
        else:
            x_iter = range(0, w)
            x_step = 1

        row_offset = y * w
        has_next_y = (y + 1) < h
        next_row_offset = (y + 1) * w

        for x in x_iter:
            idx = row_offset + x
            orig_mapped = mapped_px[idx]
            if orig_mapped == 0:
                work[idx] = 0.0
                continue
            if orig_mapped == 255:
                work[idx] = 255.0
                continue

            old_val = min(255.0, max(0.0, work[idx]))
            if old_val < 43.0:
                new_val = 0.0
            elif old_val < 128.0:
                new_val = 85.0
            elif old_val < 213.0:
                new_val = 170.0
            else:
                new_val = 255.0
            work[idx] = new_val

            err = old_val - new_val
            if err > 40.0:
                err = 40.0
            elif err < -40.0:
                err = -40.0

            src_here = src_px[idx]

            nx = x + x_step
            if 0 <= nx < w:
                n_idx = row_offset + nx
                gate = 0.25 if abs(src_px[n_idx] - src_here) > 64 else 1.0
                work[n_idx] += err * 0.375 * gate

            if has_next_y:
                bx = x - x_step
                if 0 <= bx < w:
                    b_idx = next_row_offset + bx
                    gate = 0.25 if abs(src_px[b_idx] - src_here) > 64 else 1.0
                    work[b_idx] += err * 0.125 * gate

                d_idx = next_row_offset + x
                gate = 0.25 if abs(src_px[d_idx] - src_here) > 64 else 1.0
                work[d_idx] += err * 0.25 * gate

                if 0 <= nx < w:
                    f_idx = next_row_offset + nx
                    gate = 0.25 if abs(src_px[f_idx] - src_here) > 64 else 1.0
                    work[f_idx] += err * 0.125 * gate

    out_img = Image.new('L', (w, h))
    out_img.putdata([int(round(min(255.0, max(0.0, v)))) for v in work])
    return out_img


def _quantize_to_levels(img: Image.Image, levels: list) -> Image.Image:
    """Generic fallback quantization to arbitrary palette levels."""
    palette_img = Image.new('P', (1, 1))
    palette = []
    for level in levels:
        palette.extend([level, level, level])
    palette.extend([0, 0, 0] * (256 - len(levels)))
    palette_img.putpalette(palette)

    rgb = img.convert('RGB')
    quantized = rgb.quantize(
        colors=len(levels),
        palette=palette_img,
        dither=Image.Dither.FLOYDSTEINBERG
    )
    return quantized.convert('L')


def process_image(image_bytes: bytes, filename: str, options: ImageOptions = None) -> List[ImageResult]:
    """
    Process a single image for e-ink device optimization (Xteink X4, X3).
    Returns a list of ImageResult.
    """
    if options is None:
        options = ImageOptions()

    original_size = len(image_bytes)
    original_ext = Path(filename).suffix.lower()
    stem = Path(filename).stem

    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        return [ImageResult(
            output_bytes=image_bytes,
            new_filename=filename,
            original_size=original_size,
            new_size=original_size,
            was_converted=False,
            details=f"Skipped (corrupt: {e})"
        )]

    if getattr(img, 'is_animated', False):
        img.seek(0)

    if img.mode == 'CMYK':
        img = img.convert('RGB')
    if img.mode == '1':
        img = img.convert('L')

    img = _handle_transparency(img)

    if options.light_novel_mode and not options.is_cover:
        images = _handle_light_novel(img, options.light_novel_rotate_left)
    else:
        images = [img]

    results = []
    for i, current_img in enumerate(images):
        details_parts = []

        if original_ext not in ('.jpg', '.jpeg'):
            details_parts.append(f"{original_ext.upper().strip('.')}->JPEG")

        orig_w, orig_h = current_img.size

        # Clamp hard limit (1024x1024)
        if orig_w > MAX_IMAGE_DIMENSION or orig_h > MAX_IMAGE_DIMENSION:
            current_img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)
            clamped_w, clamped_h = current_img.size
            details_parts.append(f"clamped {orig_w}x{orig_h}->{clamped_w}x{clamped_h}")
            orig_w, orig_h = clamped_w, clamped_h

        # Size handling: Smart Hybrid (0% stretch, 0% crop) for covers; proportional thumbnail for interior art
        if options.is_cover:
            current_img = _fit_cover_smart_hybrid(current_img, options.max_width, options.max_height)
            details_parts.append(f"smart-hybrid {current_img.size[0]}x{current_img.size[1]}")
        elif orig_w > options.max_width or orig_h > options.max_height:
            current_img.thumbnail((options.max_width, options.max_height), Image.Resampling.LANCZOS)
            new_w, new_h = current_img.size
            details_parts.append(f"resized {orig_w}x{orig_h}->{new_w}x{new_h}")

        if options.grayscale:
            if current_img.mode != 'L':
                current_img = _to_eink_grayscale(current_img)

            sharpened_img, mapped_px = _enhance_for_eink(current_img, options)

            if options.eink_quantize:
                if list(options.gray_levels) == [0, 85, 170, 255]:
                    current_img = _quantize_4level_serpentine(sharpened_img, mapped_px)
                    details_parts.append("4-level serpentine dithered")
                else:
                    current_img = _quantize_to_levels(sharpened_img, options.gray_levels)
                    details_parts.append(f"{len(options.gray_levels)}-level dithered")
            else:
                current_img = sharpened_img
                details_parts.append("grayscale")

            current_img = current_img.convert('RGB')
        elif options.contrast_boost:
            enhancer = ImageEnhance.Contrast(current_img)
            current_img = enhancer.enhance(options.contrast_factor)
            details_parts.append(f"contrast {options.contrast_factor}x")

        effective_quality = max(options.quality, 92) if options.is_cover else max(options.quality, 85)
        buffer = io.BytesIO()
        current_img.save(
            buffer,
            format='JPEG',
            quality=effective_quality,
            progressive=False,
            optimize=True,
            subsampling=0
        )
        output_bytes = buffer.getvalue()

        if len(images) > 1:
            new_filename = f"{stem}_part{i + 1}.jpg"
            details_parts.insert(0, f"split part {i + 1}/{len(images)}")
        else:
            new_filename = f"{stem}.jpg"

        results.append(ImageResult(
            output_bytes=output_bytes,
            new_filename=new_filename,
            original_size=original_size if i == 0 else 0,
            new_size=len(output_bytes),
            was_converted=True,
            details=", ".join(details_parts) if details_parts else "baseline JPEG"
        ))

    return results


def generate_cover_image(title: str, author: str,
                         width: int = X4_WIDTH, height: int = X4_HEIGHT,
                         gray_levels: Optional[List[int]] = None) -> bytes:
    """Generate a clean cover image from title and author text."""
    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    title_size = 36
    author_size = 24

    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", title_size)
        author_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", author_size)
    except (OSError, IOError):
        try:
            title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", title_size)
            author_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", author_size)
        except (OSError, IOError):
            title_font = ImageFont.load_default()
            author_font = ImageFont.load_default()

    border = 20
    draw.rectangle(
        [border, border, width - border, height - border],
        outline=(180, 180, 180),
        width=2
    )

    padding = 40
    max_text_width = width - (padding * 2)

    def wrap_text(text, font, max_w):
        words = text.split()
        lines = []
        current_line = ""
        for word in words:
            test = f"{current_line} {word}".strip()
            bbox = draw.textbbox((0, 0), test, font=font)
            if bbox[2] - bbox[0] <= max_w:
                current_line = test
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        return lines

    title_lines = wrap_text(title, title_font, max_text_width)
    title_y = height // 3
    for line in title_lines:
        bbox = draw.textbbox((0, 0), line, font=title_font)
        line_w = bbox[2] - bbox[0]
        x = (width - line_w) // 2
        draw.text((x, title_y), line, fill=(30, 30, 30), font=title_font)
        title_y += bbox[3] - bbox[1] + 8

    if author:
        author_lines = wrap_text(author, author_font, max_text_width)
        author_y = title_y + 40
        for line in author_lines:
            bbox = draw.textbbox((0, 0), line, font=author_font)
            line_w = bbox[2] - bbox[0]
            x = (width - line_w) // 2
            draw.text((x, author_y), line, fill=(100, 100, 100), font=author_font)
            author_y += bbox[3] - bbox[1] + 6

    if gray_levels:
        gray_img = _to_eink_grayscale(img)
        if list(gray_levels) == [0, 85, 170, 255]:
            img = _quantize_4level_serpentine(gray_img, list(gray_img.getdata())).convert('RGB')
        else:
            img = _quantize_to_levels(gray_img, gray_levels).convert('RGB')

    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=92, progressive=False, optimize=True, subsampling=0)
    return buffer.getvalue()
