from PIL import Image, ImageFilter
import io
import math
from typing import Tuple, Dict


class ImageProcessor:
    @staticmethod
    def _to_eink_grayscale(img: Image.Image) -> Image.Image:
        """
        Convert RGB/RGBA/P image to grayscale ('L') using a chromatic-aware blend:
        80% ITU-R 601-2 perceptual luma + 20% HSV Value (max(R, G, B)).
        Preserves saturated reds, golds, and warm skin tones against dark shadows.
        """
        if img.mode == 'L':
            return img.copy()

        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            rgba = img.convert('RGBA')
            bg = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, rgba).convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        r_band, g_band, b_band = img.split()
        r_px = r_band.getdata()
        g_px = g_band.getdata()
        b_px = b_band.getdata()

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

    @staticmethod
    def _fit_cover_smart_hybrid(
        gray_img: Image.Image, target_w: int = 480, target_h: int = 800
    ) -> Image.Image:
        """
        Smart Hybrid 480x800 full-screen cover fitting for Xteink X4 (3:5 aspect ratio):
        1. Scans the outer 3.5% left & right margins for high-contrast typography/strokes
           so edge-to-edge titles (e.g. 'LARRY KRAMER / FAGGOTS') are never clipped.
        2. Checks whether the top or bottom border is a uniform solid background (std < 10)
           that can be seamlessly extended by up to 22px with zero visible seam.
        3. Absorbs the remaining small aspect difference via smooth Lanczos scaling to
           produce an exact (target_w, target_h) full-screen cover image.
        """
        sw, sh = gray_img.size
        if sw <= 0 or sh <= 0:
            return gray_img
        if sw == target_w and sh == target_h:
            return gray_img

        target_ratio = float(target_w) / float(target_h)
        src_ratio = float(sw) / float(sh)

        if abs(src_ratio - target_ratio) <= 0.012:
            return gray_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

        px = list(gray_img.getdata())

        if src_ratio > target_ratio:
            # 1. Measure safe stroke-free horizontal margin on left and right (up to 3.5% each side)
            max_margin_px = max(1, int(round(sw * 0.035)))
            ref_offset = min(sw // 6, max_margin_px * 3)
            y_start = int(sh * 0.04)
            y_end = int(sh * 0.96)
            y_step = max(1, sh // 250)
            hit_threshold = max(2, (y_end - y_start) // (y_step * 45))

            safe_left = max_margin_px
            for x in range(2, max_margin_px):
                hits = 0
                for y in range(y_start, y_end, y_step):
                    bg = px[y * sw + ref_offset]
                    if abs(px[y * sw + x] - bg) > 65:
                        hits += 1
                if hits > hit_threshold:
                    safe_left = max(0, x - 4)
                    break

            safe_right = max_margin_px
            for x in range(2, max_margin_px):
                rx = sw - 1 - x
                hits = 0
                for y in range(y_start, y_end, y_step):
                    bg = px[y * sw + (sw - 1 - ref_offset)]
                    if abs(px[y * sw + rx] - bg) > 65:
                        hits += 1
                if hits > hit_threshold:
                    safe_right = max(0, x - 4)
                    break

            # Cap horizontal trim to what is actually needed when combined with ~5% vertical stretch
            ideal_scaled_w = max(target_w, int(round((target_h * src_ratio) / 1.05)))
            max_needed_sym_crop = int(round(sw * ((ideal_scaled_w - target_w) / float(2 * ideal_scaled_w))))
            safe_sym_crop = min(safe_left, safe_right, max(max_margin_px, max_needed_sym_crop))
            crop_w = max(16, sw - 2 * safe_sym_crop)

            # 2. Check if top or bottom border is a uniform solid dark/light background
            top_vals = []
            bot_vals = []
            top_row = min(sh - 1, 2)
            bot_row = max(0, sh - 3)
            for x in range(safe_sym_crop + 4, sw - safe_sym_crop - 4, max(1, crop_w // 120)):
                top_vals.append(px[top_row * sw + x])
                bot_vals.append(px[bot_row * sw + x])

            def _mean_std(vals):
                if not vals:
                    return 128.0, 999.0
                m = sum(vals) / float(len(vals))
                var = sum((v - m) ** 2 for v in vals) / float(len(vals))
                return m, math.sqrt(max(0.0, var))

            top_mean, top_std = _mean_std(top_vals)
            bot_mean, bot_std = _mean_std(bot_vals)

            top_is_uniform = (top_std < 10.0) and (top_mean <= 22.0 or top_mean >= 235.0)
            bot_is_uniform = (bot_std < 10.0) and (bot_mean <= 22.0 or bot_mean >= 235.0)

            cropped_ratio = float(crop_w) / float(sh)
            natural_h = int(round(target_w / cropped_ratio))
            deficit_h = max(0, target_h - natural_h)

            pad_top = 0
            pad_bot = 0
            if deficit_h > 0:
                if top_is_uniform and bot_is_uniform:
                    pad_top = min(20, deficit_h // 3)
                    pad_bot = min(20, deficit_h // 3)
                elif top_is_uniform:
                    pad_top = min(22, deficit_h // 3)
                elif bot_is_uniform:
                    pad_bot = min(22, deficit_h // 3)

            draw_h = max(16, target_h - pad_top - pad_bot)
            cropped_img = gray_img.crop((safe_sym_crop, 0, safe_sym_crop + crop_w, sh))
            scaled_body = cropped_img.resize((target_w, draw_h), Image.Resampling.LANCZOS)

            if pad_top == 0 and pad_bot == 0:
                return scaled_body

            canvas = Image.new('L', (target_w, target_h), 0)
            if pad_top > 0:
                top_fill = 0 if top_mean <= 22.0 else 255
                canvas.paste(top_fill, (0, 0, target_w, pad_top + 2))
            if pad_bot > 0:
                bot_fill = 0 if bot_mean <= 22.0 else 255
                canvas.paste(bot_fill, (0, pad_top + draw_h - 2, target_w, target_h))
            canvas.paste(scaled_body, (0, pad_top))
            return canvas

        else:
            # Narrower than 3:5: apply subtle horizontal stretch (up to 5%) + center vertical trim
            scaled_h = max(target_h, int(round((target_w / src_ratio) / 1.05)))
            resized = gray_img.resize((target_w, scaled_h), Image.Resampling.LANCZOS)
            crop_y = max(0, (scaled_h - target_h) // 2)
            return resized.crop((0, crop_y, target_w, crop_y + target_h))

    @staticmethod
    def _enhance_for_eink(gray_img: Image.Image, options: Dict) -> Tuple[Image.Image, list]:
        """
        Adaptive pre-processing for 4-level e-ink screens:
        1. 0.5% to 99.5% percentile histogram normalization.
        2. Bimodal detection (line art / text / diagrams vs continuous-tone artwork/covers).
        3. Dead-zone clamping ([0..7] -> 0, [248..255] -> 255) to eliminate background speckle.
        4. Adaptive shadow-lift tone curve + highlight roll-up for typography.
        5. Local Adaptive Contrast (radius=16 broad Gaussian local dodge & burn) so facial
           features, drapery, and shadow details pop on a 4-shade display without washing
           out flat backgrounds.
        6. Fine UnsharpMask prior to 4-level quantization.
        Returns (sharpened_image, deadzone_mapped_pixels).
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

        user_contrast = float(options.get('contrast_boost', 1.0))
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

        # Local Adaptive Contrast (Broad Radius=16 Gaussian Local Dodge & Burn)
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

    @staticmethod
    def _quantize_4level_serpentine(
        sharpened_img: Image.Image, mapped_px: list
    ) -> Image.Image:
        """
        Serpentine Edge-Aware Damped 4-Level [0, 85, 170, 255] Dithering:
        - Alternates scan direction on even/odd rows to eliminate diagonal Floyd-Steinberg worms.
        - Uses damped 14/16 ([6, 2, 4, 2]/16) error diffusion for crisp e-paper midtone contrast.
        - Attenuates error propagation across sharp contrast boundaries (> 64 luminance jump)
          by 75% so bright typography on dark backgrounds has zero gray haloing.
        - Pins dead-zone pure blacks (0) and pure whites (255) so solid backgrounds stay clean.
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

                # Forward neighbor: (x + x_step, y) -> 6/16 (0.375)
                nx = x + x_step
                if 0 <= nx < w:
                    n_idx = row_offset + nx
                    gate = 0.25 if abs(src_px[n_idx] - src_here) > 64 else 1.0
                    work[n_idx] += err * 0.375 * gate

                if has_next_y:
                    # Back-diagonal: (x - x_step, y + 1) -> 2/16 (0.125)
                    bx = x - x_step
                    if 0 <= bx < w:
                        b_idx = next_row_offset + bx
                        gate = 0.25 if abs(src_px[b_idx] - src_here) > 64 else 1.0
                        work[b_idx] += err * 0.125 * gate

                    # Down: (x, y + 1) -> 4/16 (0.25)
                    d_idx = next_row_offset + x
                    gate = 0.25 if abs(src_px[d_idx] - src_here) > 64 else 1.0
                    work[d_idx] += err * 0.25 * gate

                    # Forward-diagonal: (x + x_step, y + 1) -> 2/16 (0.125)
                    if 0 <= nx < w:
                        f_idx = next_row_offset + nx
                        gate = 0.25 if abs(src_px[f_idx] - src_here) > 64 else 1.0
                        work[f_idx] += err * 0.125 * gate

        out_img = Image.new('L', (w, h))
        out_img.putdata([int(round(min(255.0, max(0.0, v)))) for v in work])
        return out_img

    @staticmethod
    def optimize_image(image_bytes: bytes, options: Dict) -> Tuple[bytes, str]:
        """
        Optimize image for e-ink display (Xteink X4: 480x800, 4-level grayscale):
        1. Chromatic-aware grayscale conversion (preserves warm reds/skin tones)
        2. Smart Hybrid 480x800 full-screen fitting for covers (with edge-typography protection),
           or proportional Lanczos resize for interior book illustrations
        3. Adaptive shadow-lift + Local Adaptive Contrast (radius=16 dodge & burn) + fine UnsharpMask
        4. Serpentine edge-aware damped 4-level [0, 85, 170, 255] dithering
        5. High-precision subsampling=0 JPEG output so DCT blocks do not smear dither dots
        Returns: (optimized_bytes, new_media_type)
        """
        try:
            img = Image.open(io.BytesIO(image_bytes))

            max_w = int(options.get('max_width', 480))
            max_h = int(options.get('max_height', 800))
            quality = int(options.get('quality', 75))
            is_cover = bool(options.get('is_cover', False))

            if options.get('grayscale', True):
                img = ImageProcessor._to_eink_grayscale(img)
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            if is_cover and img.mode == 'L':
                img = ImageProcessor._fit_cover_smart_hybrid(img, max_w, max_h)
            else:
                w, h = img.size
                if w > max_w or h > max_h:
                    scale = min(max_w / w, max_h / h)
                    new_w = max(1, int(round(w * scale)))
                    new_h = max(1, int(round(h * scale)))
                    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            if options.get('grayscale', True):
                if img.mode != 'L':
                    img = ImageProcessor._to_eink_grayscale(img)
                sharpened_img, mapped_px = ImageProcessor._enhance_for_eink(img, options)
                if options.get('quantize_4level', True):
                    img = ImageProcessor._quantize_4level_serpentine(sharpened_img, mapped_px)
                else:
                    img = sharpened_img

            effective_quality = max(quality, 92) if is_cover else max(quality, 85)

            output = io.BytesIO()
            img.save(
                output,
                format='JPEG',
                quality=effective_quality,
                subsampling=0,
                optimize=True,
            )
            return output.getvalue(), 'image/jpeg'

        except Exception as e:
            print(f"Image optimization failed, keeping original: {str(e)}")
            return image_bytes, None
