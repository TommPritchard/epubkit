"""
Main EPUB processing pipeline for Xteink Optimizer (X4, X3).
Orchestrates all processing steps and generates validation reports.
"""

import base64
import gc
import os
import posixpath
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from dataclasses import dataclass, field, replace
from typing import Callable, Optional
from urllib.parse import unquote

from lxml import etree

from image_processor import (
    ImageOptions, process_image, should_process, generate_cover_image,
    DEVICE_PROFILES, DEFAULT_DEVICE
)
from metadata_handler import (
    extract_metadata, update_metadata, strip_store_metadata, format_filename
)
from html_cleaner import (
    repair_html, remove_unused_css, collect_used_selectors,
    remove_embedded_fonts_from_css, find_font_files, normalize_whitespace,
    add_chapter_page_breaks, strip_unnecessary_attributes, xml_parser
)
from text_cleaner import clean_text_content, TextCleanOptions, TextCleanReport
from epub_packager import (
    extract_epub, package_epub, remove_os_artifacts, has_drm, find_opf_path
)
from epub_structure import (
    build_rename_map, update_opf, update_opf_remove_fonts,
    update_xhtml_references, update_css_references,
    fix_svg_covers, fix_toc, find_content_files, add_image_to_opf,
    sanitize_package_manifest_and_files, _find_element, _is_element,
)
from matter_trimmer import (
    trim_epub_matter, count_words, extract_clean_text,
    parse_xhtml_safe, get_body, is_cover_stem, local_name,
)
from chapter_splitter import split_oversized_chapters
from house_style import (
    apply_house_style, ensure_cover_entry,
    HOUSE_STYLE_FILENAME, COVER_PAGE_FILENAME, TITLE_PAGE_FILENAME,
)


@dataclass
class ProcessingOptions:
    """All user-configurable processing options."""
    device: str = DEFAULT_DEVICE  # 'x4' (480x800) or 'x3' (528x792), both 4-level gray
    grayscale: bool = True
    contrast_boost: bool = True
    contrast_factor: float = 1.5  # Higher default for low-bit-depth displays
    quality: int = 70
    eink_quantize: bool = True  # Quantize to device gray levels
    remove_fonts: bool = True
    remove_unused_css: bool = True
    light_novel_mode: bool = False
    light_novel_rotate_left: bool = True
    generate_missing_cover: bool = True
    clean_metadata: bool = True
    text_cleanup: bool = True
    normalize_quotes: bool = True
    # Story-First Standardization Engine options
    standardize_layout: bool = True
    trim_matter: bool = True
    split_chapters: bool = True
    normalize_scene_breaks: bool = True
    preserve_typography: bool = True
    preset: str = 'story'  # 'story' (Pure Story), 'nonfiction' (Keep Notes & Intro), or 'classic' (Original X4 Optimizer)
    filename_format: str = 'author-title'
    filename_template: str = ''
    original_filename: str = ''
    # Metadata edits (applied if non-empty)
    metadata_edits: dict = field(default_factory=dict)


@dataclass
class ProcessingReport:
    """Validation report of everything that was done."""
    success: bool = True
    error: str = ''
    original_size: int = 0
    optimized_size: int = 0
    output_filename: str = ''
    # Word & structure metrics
    word_count_raw: int = 0
    word_count_story: int = 0
    chapter_count: int = 0
    reading_minutes: int = 0
    files_trimmed: int = 0
    words_trimmed: int = 0
    chapters_split: int = 0
    scene_breaks_normalized: int = 0
    dead_footnotes_removed: int = 0
    orphan_images_removed: int = 0
    # Counts
    images_converted: int = 0
    images_total: int = 0
    image_formats: dict = field(default_factory=dict)  # e.g. {"PNG→JPEG": 5, "baseline JPEG": 3}
    fonts_removed: int = 0
    css_rules_removed: int = 0
    svg_covers_fixed: int = 0
    toc_status: str = ''
    metadata_items_stripped: int = 0
    whitespace_cleaned: int = 0
    attrs_stripped: int = 0
    text_fixes_total: int = 0
    text_cleanup_summary: str = ''
    os_artifacts_removed: int = 0
    cover_generated: bool = False
    # Details & source warnings
    image_details: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def summary(self) -> str:
        """Generate a human-readable summary of all changes."""
        parts = []

        if self.word_count_story > 0:
            hrs, mins = divmod(self.reading_minutes, 60)
            time_str = f"~{hrs}h {mins:02d}m" if hrs else f"~{mins}m"
            parts.append(f"Story: {self.word_count_story:,} words ({self.chapter_count} ch, {time_str})")

        if self.files_trimmed > 0:
            parts.append(f"Trimmed {self.files_trimmed} non-story files ({self.words_trimmed:,} words)")

        if self.chapters_split > 0:
            parts.append(f"Split into {self.chapters_split} chapters")

        if self.scene_breaks_normalized > 0:
            parts.append(f"Standardized {self.scene_breaks_normalized} scene breaks")

        if self.dead_footnotes_removed > 0:
            parts.append(f"Cleaned {self.dead_footnotes_removed} dead footnote markers")

        if self.images_converted > 0:
            fmt_parts = [f"{v} {k}" for k, v in self.image_formats.items()]
            parts.append(f"Converted {self.images_converted}/{self.images_total} images ({', '.join(fmt_parts)})")

        if self.fonts_removed > 0:
            parts.append(f"Removed {self.fonts_removed} embedded fonts")

        if self.css_rules_removed > 0:
            parts.append(f"Stripped {self.css_rules_removed} unused CSS rules")

        if self.svg_covers_fixed > 0:
            parts.append(f"Fixed {self.svg_covers_fixed} SVG cover wrappers")

        if self.cover_generated:
            parts.append("Generated missing cover image")

        if self.toc_status:
            parts.append(f"TOC: {self.toc_status}")

        if self.metadata_items_stripped > 0:
            parts.append(f"Stripped {self.metadata_items_stripped} store metadata entries")

        if self.whitespace_cleaned > 0:
            parts.append(f"Cleaned {self.whitespace_cleaned} empty elements")

        if self.attrs_stripped > 0:
            parts.append(f"Stripped {self.attrs_stripped} unnecessary attributes")

        if self.text_fixes_total > 0:
            parts.append(f"Text cleanup: {self.text_cleanup_summary}")

        if self.os_artifacts_removed > 0:
            parts.append(f"Removed {self.os_artifacts_removed} OS artifacts")

        if self.original_size > 0 and self.optimized_size > 0:
            reduction = (1 - self.optimized_size / self.original_size) * 100
            parts.append(f"Size: {_fmt_size(self.original_size)} → {_fmt_size(self.optimized_size)} ({reduction:.1f}% reduction)")

        return "; ".join(parts) if parts else "No changes needed"


ProgressCallback = Callable[[int, str], None]


def process_epub(input_path: str, output_path: str,
                 options: ProcessingOptions = None,
                 progress: ProgressCallback = None) -> ProcessingReport:
    """
    Main processing pipeline. Takes an EPUB, optimizes it, writes output.
    """
    if options is None:
        options = ProcessingOptions()
    if options.preset == 'classic':
        # Copy so caller's ProcessingOptions instance is not mutated;
        # Classic mode runs the original pre-Story-First X4 optimizer.
        options = replace(
            options,
            standardize_layout=False,
            trim_matter=False,
            split_chapters=False,
            normalize_scene_breaks=False,
            preserve_typography=False,
        )

    report = ProcessingReport()
    report.original_size = os.path.getsize(input_path)

    def _progress(pct: int, msg: str):
        if progress:
            progress(pct, msg)

    # Create temp working directory
    work_dir = tempfile.mkdtemp(prefix='epub_opt_')

    try:
        # Step 1: Check for DRM
        _progress(2, "Checking for DRM...")
        if has_drm(input_path):
            report.success = False
            report.error = "This EPUB is DRM-protected. Please remove DRM first (e.g. using DeDRM with Calibre)."
            return report

        # Step 2: Extract
        _progress(5, "Extracting EPUB...")
        extract_epub(input_path, work_dir)

        # Step 3: Find OPF
        _progress(8, "Parsing structure...")
        opf_rel_path = find_opf_path(work_dir)
        opf_path = os.path.join(work_dir, opf_rel_path)
        opf_tree = etree.parse(opf_path, xml_parser())

        # Step 4: Extract metadata
        _progress(10, "Reading metadata...")
        metadata = extract_metadata(opf_tree)

        # Step 5: Apply metadata edits
        if options.metadata_edits:
            update_metadata(opf_tree, options.metadata_edits)
            opf_tree.write(opf_path, xml_declaration=True, encoding='utf-8', pretty_print=True)
            # Re-read metadata for filename generation
            opf_tree = etree.parse(opf_path, xml_parser())
            metadata = extract_metadata(opf_tree)

        # Step 6a: Trim non-story front/back matter & prune orphaned images BEFORE image processing
        cover_xhtml_rel = None
        if options.trim_matter:
            _progress(11, "Trimming non-story front/back matter...")
            trim_rep = trim_epub_matter(
                work_dir, opf_path,
                preset=options.preset,
                original_filename=options.original_filename,
            )
            report.word_count_raw = trim_rep.raw_words
            report.files_trimmed = trim_rep.files_trimmed
            report.words_trimmed = trim_rep.trimmed_words
            report.dead_footnotes_removed = trim_rep.dead_footnotes_removed
            report.orphan_images_removed += trim_rep.orphan_images_removed
            report.warnings.extend(trim_rep.warnings)
            cover_xhtml_rel = trim_rep.cover_xhtml_rel
        else:
            report.word_count_raw = _count_spine_words(opf_path)

        # Step 6b: Split oversized multi-chapter XHTML files & trim glued trailing promos
        if options.split_chapters or options.trim_matter:
            _progress(13, "Checking chapter splits...")
            split_rep = split_oversized_chapters(
                work_dir, opf_path,
                cover_xhtml_rel=cover_xhtml_rel,
                split_enabled=options.split_chapters,
            )
            report.chapters_split = split_rep.chapters_created
            report.words_trimmed += split_rep.trailing_promo_trimmed_words

            if options.trim_matter and split_rep.files_split > 0:
                _progress(14, "Trimming front/back matter from split chapters...")
                trim_rep2 = trim_epub_matter(
                    work_dir, opf_path,
                    preset=options.preset,
                    original_filename=options.original_filename,
                )
                report.files_trimmed += trim_rep2.files_trimmed
                report.words_trimmed += trim_rep2.trimmed_words
                report.dead_footnotes_removed += trim_rep2.dead_footnotes_removed
                report.orphan_images_removed += trim_rep2.orphan_images_removed
                cover_xhtml_rel = trim_rep2.cover_xhtml_rel or cover_xhtml_rel
                if trim_rep2.files_trimmed > 0:
                    _renumber_trimmed_section_splits(opf_path)

        # Step 6c: Apply X4 Bookerly House Style & normalize scene breaks (incl. ornament images -> <hr>)
        # Runs BEFORE image processing so ornament & deleted images are pruned before Pillow runs!
        if options.standardize_layout or options.normalize_scene_breaks:
            _progress(15, "Applying X4 Bookerly House Style...")
            hs_rep = apply_house_style(
                work_dir, opf_path,
                standardize_layout=options.standardize_layout,
                normalize_scene_breaks=options.normalize_scene_breaks,
                cover_xhtml_rel=cover_xhtml_rel,
                ensure_title_page=options.standardize_layout,
            )
            report.scene_breaks_normalized = hs_rep.scene_breaks_normalized
            report.orphan_images_removed += hs_rep.orphan_images_removed

        # Step 6d: Find surviving content files AFTER trimming/splitting/ornament pruning
        content_files = find_content_files(work_dir, opf_path)

        # Step 7: Process images (20-60%)
        _progress(18, "Processing images...")
        if options.standardize_layout and not metadata.get('cover_href'):
            # Re-read OPF in case ensure_cover_entry discovered & registered a hash-named cover image
            metadata = extract_metadata(etree.parse(opf_path, xml_parser()))
        opf_dir_norm = os.path.normpath(str(Path(opf_path).parent))
        cover_abs_norm = (
            os.path.normpath(os.path.join(opf_dir_norm, metadata['cover_href']))
            if metadata.get('cover_href')
            else None
        )

        image_files = content_files['images']
        report.images_total = len(image_files)
        processed_images = {}  # old_rel_path -> new_filename

        for i, img_path in enumerate(image_files):
            pct = 15 + int(45 * (i / max(len(image_files), 1)))
            _progress(pct, f"Processing image {i + 1}/{len(image_files)}...")

            if not os.path.exists(img_path):
                continue

            with open(img_path, 'rb') as f:
                img_bytes = f.read()

            if not should_process(img_path):
                continue

            img_norm = os.path.normpath(img_path)
            is_cover_img = (
                (cover_abs_norm is not None and img_norm == cover_abs_norm)
                or ('cover' in Path(img_path).stem.lower())
            )

            image_options = ImageOptions.for_device(
                options.device,
                grayscale=options.grayscale,
                contrast_boost=options.contrast_boost,
                contrast_factor=options.contrast_factor,
                quality=options.quality,
                eink_quantize=options.eink_quantize,
                light_novel_mode=options.light_novel_mode,
                light_novel_rotate_left=options.light_novel_rotate_left,
                is_cover=is_cover_img,
            )

            results = process_image(img_bytes, Path(img_path).name, image_options)

            for j, result in enumerate(results):
                if result.was_converted:
                    report.images_converted += 1

                    # Track format changes
                    detail_key = result.details.split(',')[0].strip() if result.details else 'processed'
                    report.image_formats[detail_key] = report.image_formats.get(detail_key, 0) + 1

                    # Write new file
                    new_path = Path(img_path).parent / result.new_filename
                    with open(str(new_path), 'wb') as f:
                        f.write(result.output_bytes)

                    # Remove old file if name changed
                    if result.new_filename != Path(img_path).name:
                        if os.path.exists(img_path):
                            os.unlink(img_path)

                    # Track for reference updates (POSIX slashes so OPF hrefs match on Windows)
                    old_rel = os.path.relpath(img_path, Path(opf_path).parent).replace('\\', '/')
                    processed_images[old_rel] = result.new_filename

                    report.image_details.append(result.details)

        # Step 8: Fix SVG covers (62%)
        _progress(62, "Fixing SVG covers...")
        report.svg_covers_fixed = fix_svg_covers(work_dir, opf_path)

        # Step 9: Generate missing cover (64%)
        if options.generate_missing_cover:
            _progress(64, "Checking cover...")
            opf_tree = etree.parse(opf_path, xml_parser())
            meta = extract_metadata(opf_tree)
            if not meta['cover_href']:
                title = options.metadata_edits.get('title', meta['title']) or 'Untitled'
                author = options.metadata_edits.get('author', meta['author']) or ''
                profile = DEVICE_PROFILES.get(options.device, DEVICE_PROFILES[DEFAULT_DEVICE])
                cover_bytes = generate_cover_image(
                    title, author,
                    width=profile['width'], height=profile['height'],
                    gray_levels=profile['gray_levels'] if options.eink_quantize else None,
                )
                opf_dir = str(Path(opf_path).parent)
                # Respect exact on-disk casing (Windows is case-insensitive so os.path.exists('images') matches 'Images')
                existing_img_dir = next(
                    (
                        d for d in os.listdir(opf_dir)
                        if d.lower() == 'images' and os.path.isdir(os.path.join(opf_dir, d))
                    ),
                    None,
                )
                images_dir = os.path.join(opf_dir, existing_img_dir or 'images')
                os.makedirs(images_dir, exist_ok=True)

                cover_path = os.path.join(images_dir, 'cover_generated.jpg')
                with open(cover_path, 'wb') as f:
                    f.write(cover_bytes)

                # Add to OPF
                cover_href = os.path.relpath(cover_path, opf_dir).replace('\\', '/')
                add_image_to_opf(opf_path, cover_href, 'cover-image-generated')
                report.cover_generated = True
                if options.standardize_layout:
                    ensure_cover_entry(
                        opf_path,
                        title=title,
                        author=author,
                        lang=meta.get('language', ''),
                        ensure_title_page=True,
                        cover_xhtml_rel=cover_xhtml_rel,
                    )
                    content_files = find_content_files(work_dir, opf_path)

        # Step 10: Update references (68%)
        _progress(68, "Updating references...")
        rename_map = build_rename_map(
            work_dir,
            {k: v for k, v in processed_images.items()}
        )
        if rename_map:
            update_opf(opf_path, rename_map)
            for xhtml_path in content_files['xhtml']:
                if os.path.exists(xhtml_path):
                    update_xhtml_references(xhtml_path, rename_map)
            for css_path in content_files['css']:
                if os.path.exists(css_path):
                    update_css_references(css_path, rename_map)

        # Step 11-15 (Fused single-pass XHTML cleanup): Repair, strip attrs, normalize whitespace,
        # clean text, and collect used CSS selectors in ONE disk read + write per file!
        _progress(70, "Repairing, normalizing & cleaning XHTML...")
        all_classes = set()
        all_ids = set()
        all_elements = set()
        text_opts = TextCleanOptions(
            normalize_quotes=options.normalize_quotes,
            preserve_smart_typography=options.preserve_typography,
        ) if options.text_cleanup else None
        aggregate_text_report = TextCleanReport()

        for xhtml_path in content_files['xhtml']:
            if not os.path.exists(xhtml_path):
                continue
            with open(xhtml_path, 'rb') as f:
                orig_bytes = f.read()

            cur = repair_html(orig_bytes)
            cur, stripped = strip_unnecessary_attributes(cur)
            report.attrs_stripped += stripped

            cur, ws_removed = normalize_whitespace(cur)
            report.whitespace_cleaned += ws_removed
            if not options.standardize_layout:
                cur = add_chapter_page_breaks(cur)

            if text_opts is not None:
                cur, text_report = clean_text_content(cur, text_opts)
                if text_report.total_fixes > 0:
                    aggregate_text_report.merge(text_report)

            if options.remove_unused_css:
                classes, ids, elements = collect_used_selectors(cur)
                all_classes.update(classes)
                all_ids.update(ids)
                all_elements.update(elements)

            if cur != orig_bytes:
                with open(xhtml_path, 'wb') as f:
                    f.write(cur)

        if options.text_cleanup:
            report.text_fixes_total = aggregate_text_report.total_fixes
            report.text_cleanup_summary = aggregate_text_report.summary()

        # Step 12: Remove unused CSS (78%)
        if options.remove_unused_css:
            _progress(78, "Removing unused CSS...")
            for css_path in content_files['css']:
                if os.path.exists(css_path):
                    if Path(css_path).name == HOUSE_STYLE_FILENAME:
                        continue
                    with open(css_path, 'r', encoding='utf-8', errors='ignore') as f:
                        css_text = f.read()
                    cleaned, removed = remove_unused_css(css_text, all_classes, all_ids, all_elements)
                    report.css_rules_removed += removed
                    if removed > 0:
                        with open(css_path, 'w', encoding='utf-8') as f:
                            f.write(cleaned)

        # Step 13: Remove embedded fonts (82%)
        if options.remove_fonts:
            _progress(82, "Removing embedded fonts...")
            font_files = content_files['fonts']

            if font_files:
                # Remove @font-face from CSS
                for css_path in content_files['css']:
                    if os.path.exists(css_path):
                        with open(css_path, 'r', encoding='utf-8', errors='ignore') as f:
                            css_text = f.read()
                        cleaned, removed = remove_embedded_fonts_from_css(css_text)
                        report.fonts_removed += removed
                        if removed > 0:
                            with open(css_path, 'w', encoding='utf-8') as f:
                                f.write(cleaned)

                # Remove font files
                for font_path in font_files:
                    if os.path.exists(font_path):
                        os.unlink(font_path)
                        report.fonts_removed += 1

                # Remove from OPF manifest
                update_opf_remove_fonts(opf_path, font_files)

        # Step 16: Clean metadata (88%)
        if options.clean_metadata:
            _progress(87, "Cleaning metadata...")
            opf_tree = etree.parse(opf_path, xml_parser())
            report.metadata_items_stripped = strip_store_metadata(opf_tree)
            if report.metadata_items_stripped > 0:
                opf_tree.write(opf_path, xml_declaration=True, encoding='utf-8', pretty_print=True)

        # Step 17: Fix / Rebuild Dual TOC (90%)
        _progress(90, "Rebuilding TOC...")
        rebuild_toc = options.trim_matter or options.split_chapters or options.standardize_layout
        toc_fixed, toc_msg = fix_toc(work_dir, opf_path, force_rebuild=rebuild_toc)
        report.toc_status = toc_msg

        # Step 17a: Package hygiene (dedup manifest IDs/hrefs, prune missing files & unmanifested clutter)
        sanitize_package_manifest_and_files(work_dir, opf_path)

        # Step 17b: Compute final story word count, chapter count & reading time (92%)
        _progress(92, "Counting story words...")
        story_words, ch_count = _count_story_metrics(opf_path)
        report.word_count_story = story_words
        report.chapter_count = ch_count
        report.reading_minutes = max(1, round(story_words / 250)) if story_words > 0 else 0

        # Step 18: Clean OS artifacts (93%)
        _progress(93, "Cleaning up...")
        report.os_artifacts_removed = remove_os_artifacts(work_dir)

        # Step 19: Repackage (95%)
        _progress(95, "Repackaging EPUB...")
        package_epub(work_dir, output_path)

        # Step 20: Generate output filename
        opf_tree = etree.parse(opf_path, xml_parser())
        final_metadata = extract_metadata(opf_tree)
        title = options.metadata_edits.get('title', final_metadata['title']) or final_metadata['title']
        author = options.metadata_edits.get('author', final_metadata['author']) or final_metadata['author']
        report.output_filename = format_filename(
            title,
            author,
            filename_format=options.filename_format,
            template=options.filename_template,
            original_filename=options.original_filename,
            year=final_metadata['year'],
            series=final_metadata['series'],
            series_index=final_metadata['series_index'],
            language=final_metadata['language'],
        )

        # Done
        report.optimized_size = os.path.getsize(output_path)
        report.success = True
        _progress(100, "Complete")

    except Exception as e:
        report.success = False
        report.error = str(e)
        _progress(100, f"Error: {e}")

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        gc.collect()

    return report


def _find_zip_entry(namelist: list[str], target_rel: str) -> Optional[str]:
    norm = posixpath.normpath(target_rel.replace('\\', '/')).lstrip('./')
    if norm in namelist:
        return norm
    low_map = {n.lower(): n for n in namelist}
    return low_map.get(norm.lower())


def extract_epub_metadata(epub_path: str) -> dict:
    """
    Quick metadata extraction directly from the ZIP archive without disk extraction.
    Used for the preview step in the web UI (Render 512 MB RAM & I/O safe).
    Returns: {title, author, series, series_index, language, cover_id, cover_href, cover_data, has_drm}
    """
    result = {
        'title': '',
        'author': '',
        'series': '',
        'series_index': '',
        'language': '',
        'cover_data': None,  # Base64-encoded cover image
        'has_drm': False,
        'is_kepub': Path(epub_path).name.lower().endswith('.kepub.epub'),
    }

    if result['is_kepub']:
        return result

    result['has_drm'] = has_drm(epub_path)
    if result['has_drm']:
        return result

    try:
        with zipfile.ZipFile(epub_path, 'r') as zf:
            names = zf.namelist()
            opf_rel: Optional[str] = None
            container_name = _find_zip_entry(names, 'META-INF/container.xml')
            if container_name:
                c_root = etree.fromstring(zf.read(container_name), xml_parser())
                for el in c_root.iter():
                    if isinstance(el.tag, str) and (el.tag.endswith('}rootfile') or el.tag == 'rootfile'):
                        cand = (el.get('full-path') or '').strip()
                        if cand:
                            opf_rel = _find_zip_entry(names, unquote(cand))
                            break
            if not opf_rel:
                opf_rel = next((n for n in names if n.lower().endswith('.opf')), None)
            if not opf_rel:
                return result

            opf_root = etree.fromstring(zf.read(opf_rel), xml_parser())
            metadata = extract_metadata(etree.ElementTree(opf_root))
            result.update(metadata)

            # Extract cover image bytes directly (guarded at <= 5 MB)
            if metadata['cover_href']:
                opf_dir = posixpath.dirname(opf_rel)
                clean_cov = unquote(metadata['cover_href']).replace('\\', '/')
                cov_rel = posixpath.join(opf_dir, clean_cov) if opf_dir else clean_cov
                cov_entry = _find_zip_entry(names, cov_rel)
                if cov_entry and zf.getinfo(cov_entry).file_size <= 5 * 1024 * 1024:
                    result['cover_data'] = base64.b64encode(zf.read(cov_entry)).decode('ascii')

    except Exception:
        pass

    return result


def _renumber_trimmed_section_splits(opf_path: str) -> None:
    """Re-sequence surviving fallback 'Section N' split titles from Section 1 without retaining DOM trees in RAM."""
    try:
        opf_tree = etree.parse(opf_path, xml_parser())
    except Exception:
        return
    opf_root = opf_tree.getroot()
    opf_dir = str(Path(opf_path).parent)
    manifest = _find_element(opf_root, 'manifest')
    spine = _find_element(opf_root, 'spine')
    if manifest is None or spine is None:
        return

    id_to_href = {
        it.get('id', ''): posixpath.normpath(unquote(it.get('href', '')).replace('\\', '/'))
        for it in manifest if _is_element(it)
    }

    groups: dict[str, list[tuple[str, int]]] = {}
    for itemref in spine:
        if not _is_element(itemref):
            continue
        rel = id_to_href.get(itemref.get('idref', ''), '')
        m_ch = re.match(r'^(.+)_ch\d+$', Path(rel).stem, re.I)
        if not rel or not m_ch:
            continue
        stem_key = posixpath.join(posixpath.dirname(rel), m_ch.group(1))
        abs_p = os.path.join(opf_dir, rel)
        if not os.path.exists(abs_p):
            continue
        root, _ = parse_xhtml_safe(Path(abs_p).read_bytes())
        title_el = next(
            (el for el in root.iter() if _is_element(el) and local_name(el.tag) == 'title'),
            None,
        )
        t_txt = ' '.join((title_el.text or '').split()) if title_el is not None else ''
        m_sec = re.match(r'^Section\s+(\d+)$', t_txt, re.I)
        sec_num = int(m_sec.group(1)) if (m_sec and title_el is not None) else 0
        groups.setdefault(stem_key, []).append((abs_p, sec_num))

    for entries in groups.values():
        if not entries or entries[0][1] <= 1 or not all(sec > 0 for _, sec in entries):
            continue
        for seq, (abs_p, cur_sec) in enumerate(entries, 1):
            if cur_sec == seq:
                continue
            root, _ = parse_xhtml_safe(Path(abs_p).read_bytes())
            title_el = next(
                (el for el in root.iter() if _is_element(el) and local_name(el.tag) == 'title'),
                None,
            )
            if title_el is not None:
                title_el.text = f"Section {seq}"
                Path(abs_p).write_bytes(
                    etree.tostring(root, xml_declaration=True, encoding='utf-8', pretty_print=True)
                )


def _count_spine_words(opf_path: str) -> int:
    """Count total words across all spine XHTML files."""
    total, _ = _count_story_metrics(opf_path, exclude_entry_pages=False)
    return total


def _count_story_metrics(opf_path: str, exclude_entry_pages: bool = True) -> tuple[int, int]:
    """Stream spine XHTML files one at a time and return (story_words, chapter_count)."""
    from urllib.parse import unquote
    try:
        opf_tree = etree.parse(opf_path, xml_parser())
    except Exception:
        return 0, 0
    root = opf_tree.getroot()
    opf_dir = str(Path(opf_path).parent)
    manifest = _find_element(root, 'manifest')
    spine = _find_element(root, 'spine')
    if manifest is None or spine is None:
        return 0, 0

    id_to_href = {
        it.get('id', ''): unquote(it.get('href', ''))
        for it in manifest if _is_element(it)
    }

    total_words = 0
    chapter_count = 0
    for idx, itemref in enumerate(spine):
        if not _is_element(itemref):
            continue
        href = id_to_href.get(itemref.get('idref', ''), '')
        if not href:
            continue
        bname = Path(href).name
        if exclude_entry_pages and bname in (COVER_PAGE_FILENAME, TITLE_PAGE_FILENAME):
            continue
        abs_p = os.path.join(opf_dir, href)
        if not os.path.exists(abs_p):
            continue
        doc, _ = parse_xhtml_safe(Path(abs_p).read_bytes())
        w = count_words(extract_clean_text(get_body(doc)))
        if exclude_entry_pages and idx == 0 and w < 20 and is_cover_stem(href):
            continue
        total_words += w
        chapter_count += 1

    return total_words, chapter_count


def _fmt_size(size_bytes: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
