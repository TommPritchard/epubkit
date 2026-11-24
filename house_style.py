"""
X4 Bookerly House Style & Scene-Break Standardizer for epubkit.
Applies the locked Landscape X4 Bookerly specification:
  - Margins: 0 on EPUB body (defers cleanly to CrossInk's device margin setting)
  - Paragraphs: Classic indent (1.25em, 0 gap), flush-left on first para after heading/break
  - Alignment: Justified, no hyphenation (hyphens: none)
  - Line Height: 1.38
  - Chapter Headers: Number-First Classic (compact landscape top spacing)
  - Scene Breaks: Centered 35% width, 1px solid black horizontal rule (with spacer-ratio & illustration guards)
  - Book Entry: Cover -> Standardized Minimal Title Page -> Story
  - Prose Hyperlinks & <html lang> normalization
"""
from __future__ import annotations

import os
import posixpath
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

from lxml import etree
from PIL import Image

from html_cleaner import xml_parser
from epub_structure import (
    NS_OPF, NS_XHTML, NS_SVG, NS_XLINK,
    _is_element, _find_element,
)
from matter_trimmer import (
    local_name,
    norm_match_text,
    extract_clean_text,
    count_words,
    parse_xhtml_safe,
    get_body,
    is_cover_stem,
    is_footnote_link,
    _collect_doc_img_refs,
    _normalize_rel,
    _remove_element_keep_tail,
    _unwrap_element,
    _prune_unreferenced_images,
)

HOUSE_STYLE_FILENAME = "x4_house_style.css"
COVER_PAGE_FILENAME = "x4_cover.xhtml"
TITLE_PAGE_FILENAME = "x4_title.xhtml"

HOUSE_STYLE_CSS = """/* Xteink X4 Bookerly House Style (Landscape-tuned, 4-level grayscale) */
html, body {
  margin: 0 !important;
  padding: 0 !important;
  background-color: transparent !important;
  color: #000000 !important;
  font-family: "Bookerly", serif !important;
  line-height: 1.38 !important;
  text-align: justify !important;
  hyphens: none !important;
  -webkit-hyphens: none !important;
  -epub-hyphens: none !important;
  adobe-hyphenate: none !important;
  overflow-wrap: break-word !important;
  word-wrap: break-word !important;
  widows: 1;
  orphans: 1;
}

p {
  margin: 0 !important;
  padding: 0 !important;
  text-indent: 1.25em !important;
  text-align: justify !important;
  line-height: 1.38 !important;
  color: #000000 !important;
  hyphens: none !important;
  -webkit-hyphens: none !important;
  -epub-hyphens: none !important;
  adobe-hyphenate: none !important;
  overflow-wrap: break-word !important;
  word-wrap: break-word !important;
}

h1 + p, h2 + p, h3 + p, hr + p,
.x4-chapter-header + p, .x4-scene-break + p,
p.x4-first, p.first, p.no-indent, p.noindent {
  text-indent: 0 !important;
}

body > div:first-child, body > section:first-child {
  margin-top: 0 !important;
  padding-top: 0 !important;
}

section, article, main {
  margin-left: 0 !important;
  margin-right: 0 !important;
  padding-left: 0 !important;
  padding-right: 0 !important;
}

h1, h2, h3, h4 {
  font-family: "Bookerly", serif !important;
  color: #000000 !important;
  text-align: center !important;
  font-weight: normal !important;
  line-height: 1.2 !important;
  text-indent: 0 !important;
  margin: 1.2em 0 0.7em 0 !important;
  page-break-after: avoid;
  break-after: avoid;
  hyphens: none !important;
  -webkit-hyphens: none !important;
}

.x4-chapter-header {
  text-align: center !important;
  margin: 1.4em 0 1.0em 0 !important;
  padding: 0 !important;
  page-break-after: avoid;
  break-after: avoid;
}

.x4-ch-num {
  font-size: 2.0em !important;
  font-weight: normal !important;
  line-height: 1.05 !important;
  text-align: center !important;
  text-indent: 0 !important;
  margin: 0 !important;
}

.x4-ch-title {
  font-size: 1.05em !important;
  font-style: italic !important;
  font-weight: normal !important;
  line-height: 1.25 !important;
  text-align: center !important;
  text-indent: 0 !important;
  margin: 0.3em 0 0 0 !important;
}

.x4-ch-label {
  font-size: 1.45em !important;
  font-weight: normal !important;
  line-height: 1.2 !important;
  text-align: center !important;
  text-indent: 0 !important;
  margin: 0 !important;
}

hr, hr.x4-scene-break {
  border: 0 !important;
  border-top: 1px solid #000000 !important;
  background: transparent !important;
  width: 35% !important;
  height: 0 !important;
  margin-top: 1.15em !important;
  margin-bottom: 1.15em !important;
  margin-left: 32.5% !important;
  margin-right: 32.5% !important;
  padding: 0 !important;
  display: block !important;
}

blockquote {
  margin: 0.9em 1.4em !important;
  text-indent: 0 !important;
  font-style: italic;
}

blockquote p {
  text-indent: 0 !important;
  margin-bottom: 0.4em !important;
}

img {
  max-width: 100% !important;
  height: auto !important;
  display: block;
  margin: 0.8em auto;
}

.x4-img-wrap, figure {
  text-align: center !important;
  text-indent: 0 !important;
  margin: 0.8em 0 !important;
}

.x4-img-wrap img, figure img {
  display: inline-block !important;
  max-width: 100% !important;
  height: auto !important;
  margin: 0 auto !important;
}

.x4-cover-wrap {
  margin: 0 !important;
  padding: 0 !important;
  width: 100% !important;
  text-align: center !important;
}

.x4-cover-img {
  width: 100% !important;
  max-width: 100% !important;
  height: auto !important;
  margin: 0 auto !important;
  display: block !important;
}

.x4-title-page {
  text-align: center !important;
  margin: 22% 8% 0 8% !important;
  padding: 0 !important;
}

.x4-title-book {
  font-size: 1.7em !important;
  font-weight: normal !important;
  line-height: 1.22 !important;
  text-align: center !important;
  text-indent: 0 !important;
  margin: 0 0 0.7em 0 !important;
  page-break-before: auto !important;
  break-before: auto !important;
}

hr.x4-title-rule, .x4-title-rule {
  border: 0 !important;
  border-top: 1px solid #000000 !important;
  width: 16% !important;
  height: 0 !important;
  font-size: 0 !important;
  line-height: 0 !important;
  overflow: hidden !important;
  margin-top: 0 !important;
  margin-bottom: 0.7em !important;
  margin-left: 42% !important;
  margin-right: 42% !important;
  display: block !important;
}

.x4-title-author {
  font-size: 1.05em !important;
  font-weight: normal !important;
  letter-spacing: 0.04em;
  text-align: center !important;
  text-indent: 0 !important;
  margin: 0 !important;
}

span.x4-dropcap-flat {
  float: none !important;
  font-size: inherit !important;
  line-height: inherit !important;
  margin: 0 !important;
  padding: 0 !important;
  display: inline !important;
}

table {
  width: 100% !important;
  max-width: 100% !important;
  border-collapse: collapse !important;
  margin: 0.9em 0 !important;
  font-size: 0.9em !important;
  table-layout: auto !important;
}

th, td {
  padding: 0.3em 0.4em !important;
  border-bottom: 1px solid #888888 !important;
  text-align: left !important;
  vertical-align: top !important;
  max-width: 100% !important;
  white-space: normal !important;
  overflow-wrap: break-word !important;
  word-wrap: break-word !important;
  text-indent: 0 !important;
}

td p, th p {
  text-indent: 0 !important;
  text-align: left !important;
}
"""

SCENE_SYMBOL_RE = re.compile(
    r"^[\s*•·▪▫◦●○◆◇■□▲△★☆✦✧❖❦❧§†‡~∼≈—–\-_=#·\.•]{1,25}$"
)
SCENE_CLASS_RE = re.compile(
    r"(space[-_]?break|scene[-_]?break|section[-_]?break|asterism|ornament|dinkus|transition|para[-_]?orn|etoiles|sep[-_]?para|\bbreak\b|\bsep\b|\bsb\b)",
    re.I,
)
ORNAMENT_NAME_RE = re.compile(
    r"(ornament|flourish|dingbat|dinkus|asterism|fleuron|divider|sep|separator|break|stars|swirl|vignette|rule)",
    re.I,
)
PIRATE_WATERMARK_RE = re.compile(r"(oceanofpdf\.com|z-library|pdfdrive)", re.I)

_WORD_NUMS_PAT = (
    r"one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"twenty[- ]?\w+|thirty[- ]?\w+|forty[- ]?\w+|fifty[- ]?\w+"
)
_STRICT_ROMAN_RE = re.compile(
    r"^(?=[IVXLCDM]+$)M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})$"
)
CHAPTER_LABEL_ONLY_RE = re.compile(r"^(?:chapter|kapitel|ch\.?)$", re.I)
DIVISION_LABEL_ONLY_RE = re.compile(r"^(?:part|book|act|section)$", re.I)
CHAPTER_NUM_TITLE_RE = re.compile(
    rf"^(?:chapter|kapitel|ch\.?)\s+([0-9]{{1,3}}|[IVXLCDM]{{1,8}}|{_WORD_NUMS_PAT})(?:\s*[:.\-—]\s*(.+))?$",
    re.I,
)
DIVISION_NUM_TITLE_RE = re.compile(
    rf"^((?:part|book|act|section)\s+(?:[0-9]{{1,3}}|[IVXLCDM]{{1,8}}|{_WORD_NUMS_PAT}))(?:\s*[:.\-—]\s*(.+))?$",
    re.I,
)
BARE_WORD_NUM_RE = re.compile(rf"^([0-9]{{1,3}}|{_WORD_NUMS_PAT})(?:\s*[:.\-—]\s*(.+))?$", re.I)
NON_SUBTITLE_HEADING_RE = re.compile(
    r"^(?:chapter|kapitel|ch\.?|part|book|act|section|prologue|epilogue|epigraph|contents)\b",
    re.I,
)
HEADING_CLASS_RE = re.compile(
    r"\b(?:cn|ct|chapnum|chaptitle|ch[-_]?nos?|chnum|bmtitle|h2f|chapter[-_]?(?:num(?:ber)?|no|title|head(?:ing)?)|para-ct|para-cn)\b",
    re.I,
)
_EXPLICIT_P_HEADING_RE = re.compile(
    r"^(?:chapter|kapitel|ch\.?|part|book|act|section|prologue|epilogue|epigraph)\b",
    re.I,
)

HOSTILE_CSS_PROPS = (
    "font-family",
    "font-size",
    "color",
    "background-color",
    "background",
    "margin-left",
    "margin-right",
    "padding-left",
    "padding-right",
    "hyphens",
    "-webkit-hyphens",
    "-epub-hyphens",
    "adobe-hyphenate",
)


@dataclass
class HouseStyleReport:
    scene_breaks_normalized: int = 0
    ornament_images_replaced: int = 0
    fake_spacers_stripped: int = 0
    prose_links_unwrapped: int = 0
    dropcaps_flattened: int = 0
    lang_attrs_added: int = 0
    css_files_scrubbed: int = 0
    orphan_images_removed: int = 0
    title_page_created: bool = False
    cover_page_ensured: bool = False


_AT_PAGE_RE = re.compile(r"@page\b[^{]*\{[^}]*\}", re.I | re.S)
_HOSTILE_CSS_RE = re.compile(
    r"(?:^|[;\s{])(" + "|".join(re.escape(p) for p in HOSTILE_CSS_PROPS) + r")\s*:[^;{}]*;?",
    re.I,
)
_FLOAT_CSS_RE = re.compile(r"float\s*:\s*(left|right)\s*;?", re.I)


def _scrub_css_declarations(css_text: str) -> str:
    """Strip @page rules and hostile font, color, side-margin, hyphenation, and float-dropcap rules from CSS."""
    cleaned = _AT_PAGE_RE.sub("", css_text)
    cleaned = _HOSTILE_CSS_RE.sub(lambda m: "{" if m.group(0).lstrip().startswith("{") else " ", cleaned)
    cleaned = _FLOAT_CSS_RE.sub("", cleaned)
    return cleaned


_TABLE_TAGS = ("table", "colgroup", "col", "tr", "td", "th")
_LAYOUT_WIDTH_TAGS = _TABLE_TAGS + ("p", "div", "section", "blockquote", "span")
_FIXED_WIDTH_RE = re.compile(r"^\s*[\d.]+\s*(?:px|pt|pc|cm|mm|in)\s*$", re.I)


def _scrub_inline_style(style_str: str, tag_name: str = "") -> str:
    if not style_str:
        return ""
    parts = [p.strip() for p in style_str.split(";") if p.strip()]
    kept = []
    is_table_el = tag_name in _TABLE_TAGS
    is_layout_el = not tag_name or tag_name in _LAYOUT_WIDTH_TAGS
    for part in parts:
        if ":" not in part:
            continue
        prop = part.split(":", 1)[0].strip().lower()
        val = part.split(":", 1)[1].strip().lower()
        if prop in HOSTILE_CSS_PROPS:
            continue
        if prop == "float" and val in ("left", "right"):
            continue
        if prop in ("text-indent", "line-height"):
            continue
        if is_table_el and prop in ("width", "min-width", "max-width", "white-space"):
            continue
        if is_layout_el and prop in ("width", "min-width") and _FIXED_WIDTH_RE.match(val):
            continue
        kept.append(part)
    return "; ".join(kept)


def _detect_ornament_images(
    opf_dir: str,
    story_rels: list[str],
) -> set[str]:
    """
    Identify image paths that are scene-break ornaments (not real book illustrations).
    An image is classified as an ornament if:
      - Its filename or alt matches ORNAMENT_NAME_RE, OR
      - It appears >= 3 times across story chapters AND its pixel height is <= 120px (or width/height tiny).
    """
    img_counts: dict[str, int] = {}
    alt_hits: set[str] = set()

    for rel in story_rels:
        abs_p = os.path.join(opf_dir, rel)
        if not os.path.exists(abs_p):
            continue
        rel_dir = posixpath.dirname(rel)
        root, _ = parse_xhtml_safe(Path(abs_p).read_bytes())
        body = get_body(root)
        for el in body.iter():
            if local_name(el.tag) == "img" and el.get("src"):
                img_rel = _normalize_rel(rel_dir, el.get("src"))
                img_counts[img_rel] = img_counts.get(img_rel, 0) + 1
                alt = (el.get("alt") or "").strip()
                if ORNAMENT_NAME_RE.search(Path(img_rel).stem) or (alt and ORNAMENT_NAME_RE.search(alt)):
                    alt_hits.add(img_rel)

    ornaments: set[str] = set(alt_hits)
    for img_rel, count in img_counts.items():
        abs_img = os.path.join(opf_dir, img_rel)
        if os.path.exists(abs_img):
            try:
                with Image.open(abs_img) as im:
                    w, h = im.size
                # Thin horizontal divider lines (e.g. 300x1, 109x1 rule images) or repeated small ornaments
                if h <= 8 or (h <= 16 and w >= h * 6) or (count >= 2 and (h <= 140 or (w <= 220 and h <= 180))):
                    ornaments.add(img_rel)
            except Exception:
                pass
    return ornaments


def _make_hr(ns_prefix: str) -> etree._Element:
    hr = etree.Element(f"{ns_prefix}hr" if ns_prefix else "hr")
    hr.set("class", "x4-scene-break")
    return hr


def _replace_with_hr(el: etree._Element, ns_prefix: str) -> etree._Element:
    parent = el.getparent()
    hr = _make_hr(ns_prefix)
    if parent is not None:
        idx = list(parent).index(el)
        hr.tail = el.tail
        parent.remove(el)
        parent.insert(idx, hr)
    return hr


def _normalize_scene_breaks_in_body(
    body: etree._Element,
    rel_dir: str,
    ornament_rels: set[str],
    normalize_breaks: bool,
    report: HouseStyleReport,
) -> None:
    ns_prefix = body.tag.split("}")[0] + "}" if "}" in str(body.tag) else ""

    # Count <p> vs empty spacer <p> to detect fake double-spacing (Book #100 The Palm House!)
    all_ps = [el for el in body.iter() if local_name(el.tag) == "p"]
    spacer_ps: list[etree._Element] = []
    for p in all_ps:
        raw = "".join(p.itertext())
        stripped = raw.replace("\u00a0", "").strip()
        has_img = any(local_name(c.tag) in ("img", "image", "svg") for c in p.iter())
        if not stripped and not has_img:
            if "\u00a0" in raw or any(local_name(c.tag) == "br" for c in p) or len(p) == 0:
                spacer_ps.append(p)

    spacer_ratio = (len(spacer_ps) / max(len(all_ps), 1)) if all_ps else 0.0
    spacers_are_fake_double_spacing = (len(spacer_ps) >= 10 and spacer_ratio > 0.25)

    # 1. Strip fake double-spacing spacers cleanly (always, so CSS margin:0 works!)
    if spacers_are_fake_double_spacing:
        for p in spacer_ps:
            parent = p.getparent()
            if parent is not None:
                parent.remove(p)
                report.fake_spacers_stripped += 1

    if not normalize_breaks:
        return

    # 2. Replace <hr>, CSS break classes, symbol <p>, ornament-image <p>/<div>, and real spacer <p>
    spacer_set = set() if spacers_are_fake_double_spacing else set(spacer_ps)
    prev_was_hr = False

    # Snapshot top-to-bottom elements
    for el in list(body.iter()):
        if el.getparent() is None:
            continue
        ln = local_name(el.tag)
        cls = (el.get("class") or "").lower()

        if ln == "hr":
            el.set("class", "x4-scene-break")
            for attr in ("style", "width", "size", "noshade", "align"):
                if attr in el.attrib:
                    del el.attrib[attr]
            if prev_was_hr:
                el.getparent().remove(el)
            else:
                prev_was_hr = True
                report.scene_breaks_normalized += 1
            continue

        if ln in ("p", "div", "h1", "h2", "h3", "h4"):
            txt = extract_clean_text(el)
            imgs = [c for c in el.iter() if local_name(c.tag) == "img"]

            # Ornament image paragraph/div/heading (check first so <div class="ornament"><img/></div> collapses!)
            if not txt and len(imgs) == 1:
                src = imgs[0].get("src") or ""
                img_rel = _normalize_rel(rel_dir, src) if src else ""
                if img_rel in ornament_rels:
                    if prev_was_hr:
                        el.getparent().remove(el)
                    else:
                        _replace_with_hr(el, ns_prefix)
                        prev_was_hr = True
                        report.ornament_images_replaced += 1
                        report.scene_breaks_normalized += 1
                    continue
            if ln in ("h1", "h2", "h3", "h4"):
                if txt or imgs:
                    prev_was_hr = False
                continue

            # Explicit CSS scene-break class (skip our own chapter header classes!)
            if "x4-ch" not in cls and SCENE_CLASS_RE.search(cls) and len(txt) <= 30 and not imgs:
                if prev_was_hr:
                    el.getparent().remove(el)
                else:
                    _replace_with_hr(el, ns_prefix)
                    prev_was_hr = True
                    report.scene_breaks_normalized += 1
                continue

            if ln == "p":
                if el in spacer_set:
                    if prev_was_hr:
                        el.getparent().remove(el)
                    else:
                        _replace_with_hr(el, ns_prefix)
                        prev_was_hr = True
                        report.scene_breaks_normalized += 1
                    continue
                if txt and len(txt) <= 25 and not imgs and SCENE_SYMBOL_RE.match(txt):
                    if prev_was_hr:
                        el.getparent().remove(el)
                    else:
                        _replace_with_hr(el, ns_prefix)
                        prev_was_hr = True
                        report.scene_breaks_normalized += 1
                    continue
            elif ln == "div" and "x4-ch" not in cls and txt and len(txt) <= 25 and not imgs:
                # Leaf <div> containing only scene-break symbols like <div class="transition">—</div>
                has_block_child = any(
                    _is_element(c) and local_name(c.tag) in ("p", "div", "hr", "h1", "h2", "h3", "h4", "table", "blockquote")
                    for c in el
                )
                if not has_block_child and SCENE_SYMBOL_RE.match(txt):
                    if prev_was_hr:
                        el.getparent().remove(el)
                    else:
                        _replace_with_hr(el, ns_prefix)
                        prev_was_hr = True
                        report.scene_breaks_normalized += 1
                    continue

            if txt or imgs:
                prev_was_hr = False

    # 3. Remove stray scene-break <hr>s:
    #    - inside <header>/<hgroup>
    #    - leading (before first body content) or trailing (after last body content)
    #    - immediately adjacent (before or after) to a chapter header or another <hr>
    def _in_header_ancestor(node: etree._Element) -> bool:
        anc = node.getparent()
        while anc is not None and anc is not body:
            if local_name(anc.tag) in ("header", "hgroup"):
                return True
            anc = anc.getparent()
        return False

    linear: list[tuple[str, etree._Element]] = []
    for el in body.iter():
        if not _is_element(el):
            continue
        t = local_name(el.tag)
        if t == "hr":
            linear.append(("hr", el))
        elif (t == "div" and "x4-chapter-header" in (el.get("class") or "")) or t in ("h1", "h2", "h3", "h4"):
            linear.append(("hdr", el))
        elif t in ("p", "blockquote", "table", "ul", "ol", "img", "image", "svg"):
            # Skip elements nested inside a chapter header
            anc = el.getparent()
            in_hdr = False
            while anc is not None and anc is not body:
                if local_name(anc.tag) == "div" and "x4-chapter-header" in (anc.get("class") or ""):
                    in_hdr = True
                    break
                anc = anc.getparent()
            if in_hdr:
                continue
            if t in ("img", "image", "svg") or extract_clean_text(el):
                linear.append(("content", el))

    # Iteratively prune invalid <hr> entries so collapsing exposed edges works cleanly
    changed = True
    while changed:
        changed = False
        first_content_idx = next((i for i, (k, _) in enumerate(linear) if k == "content"), None)
        last_content_idx = next((i for i in range(len(linear) - 1, -1, -1) if linear[i][0] == "content"), None)
        new_linear: list[tuple[str, etree._Element]] = []
        for idx_l, (kind, el) in enumerate(linear):
            if kind == "hr":
                prev_k = new_linear[-1][0] if new_linear else None
                next_k = linear[idx_l + 1][0] if idx_l + 1 < len(linear) else None
                remove_hr = (
                    _in_header_ancestor(el)
                    or first_content_idx is None
                    or idx_l < first_content_idx
                    or (last_content_idx is not None and idx_l > last_content_idx)
                    or prev_k in ("hr", "hdr")
                    or next_k == "hdr"
                )
                if remove_hr:
                    parent = el.getparent()
                    if parent is not None:
                        parent.remove(el)
                    changed = True
                    continue
            new_linear.append((kind, el))
        linear = new_linear

    # Mark first <p> after any heading or <hr> with class="x4-first" for reliable flush-left indent
    prev_triggers_noindent = True
    for el in body.iter():
        ln = local_name(el.tag)
        if ln in ("h1", "h2", "h3", "hr"):
            prev_triggers_noindent = True
        elif ln == "div" and "x4-chapter-header" in (el.get("class") or ""):
            prev_triggers_noindent = True
        elif ln == "p":
            if prev_triggers_noindent:
                cur_cls = (el.get("class") or "").split()
                if "x4-first" not in cur_cls:
                    cur_cls.append("x4-first")
                    el.set("class", " ".join(cur_cls))
                prev_triggers_noindent = False
            else:
                prev_triggers_noindent = False


def _heading_lines(el: etree._Element) -> list[str]:
    """Extract cleaned lines from a heading element, treating <br/> as line breaks."""
    buf: list[str] = []

    def walk(node: etree._Element) -> None:
        if node.text:
            buf.append(node.text)
        for child in node:
            if _is_element(child) and local_name(child.tag) == "br":
                buf.append("\n")
            elif _is_element(child):
                walk(child)
            if child.tail:
                buf.append(child.tail)

    walk(el)
    raw = "".join(buf)
    lines = [norm_match_text(ln) for ln in raw.splitlines()]
    return [ln for ln in lines if ln]


def _parse_bare_number(txt: str) -> tuple[str, str] | None:
    stripped = txt.strip()
    m_brk = re.match(r"^[\[\<\(]\s*([0-9]{1,3}|[IVXLCDM]{1,8})\s*[\]\>\)]$", stripped)
    if m_brk:
        inner = m_brk.group(1)
        if inner.isdigit() or _STRICT_ROMAN_RE.match(inner):
            return inner, ""
    m = BARE_WORD_NUM_RE.match(txt)
    if m:
        return m.group(1), (m.group(2) or "").strip()
    parts = re.split(r"\s*[:.\-—]\s*", txt, maxsplit=1)
    head = parts[0].strip()
    if _STRICT_ROMAN_RE.match(head):
        return head, (parts[1].strip() if len(parts) > 1 else "")
    return None


def _has_non_ornament_img(el: etree._Element, rel_dir: str, ornament_rels: set[str]) -> bool:
    for c in el.iter():
        if _is_element(c) and local_name(c.tag) == "img":
            src = c.get("src") or ""
            img_rel = _normalize_rel(rel_dir, src) if src else ""
            if img_rel not in ornament_rels:
                return True
    return False


def _collect_empty_anchors(el: etree._Element) -> list[etree._Element]:
    anchors: list[etree._Element] = []
    for c in el.iter():
        if not _is_element(c) or c is el:
            continue
        if local_name(c.tag) in ("span", "a") and c.get("id") and not (c.text or "").strip() and len(c) == 0:
            anc = etree.Element(c.tag)
            anc.set("id", c.get("id"))
            anchors.append(anc)
    return anchors


def _is_empty_anchor_el(el: etree._Element) -> bool:
    return (
        local_name(el.tag) in ("a", "span")
        and not extract_clean_text(el)
        and not any(local_name(c.tag) == "img" for c in el.iter() if _is_element(c))
    )


def _format_number_first_headings(
    body: etree._Element,
    rel_dir: str = "",
    ornament_rels: set[str] | None = None,
) -> None:
    """
    Standardize single or multi-element chapter headers into uniform Number-First Classic blocks:
      <div class="x4-chapter-header">
        [<div class="x4-ch-num">4</div> | <div class="x4-ch-label">Prologue</div>]
        [<div class="x4-ch-title">The Haunting</div>]
      </div>
    Handles:
      - 'Chapter 4' / 'Chapter 19: The Haunting'
      - Split '<h1>CHAPTER</h1><h1>4</h1>' (The Secret History, stripping inline rule images)
      - Split '<h1>Chapter 19</h1><h2>The Haunting</h2>' (The Wager)
      - Bare number headings '<h3>2</h3>', '<h2>IV</h2>' (Fingersmith)
      - Named section headings '<h1>PROLOGUE</h1>', '<h1>EPILOGUE</h1>', '<h1>BOOK I</h1>'
    """
    orn_set = ornament_rels or set()
    ns_prefix = body.tag.split("}")[0] + "}" if "}" in str(body.tag) else ""

    def is_heading_candidate(el: etree._Element, saw_prose: bool, is_first_block: bool = False, next_sib: etree._Element | None = None) -> bool:
        if not _is_element(el) or el.getparent() is None:
            return False
        ln = local_name(el.tag)
        # Fast-path reject before walking descendants or extracting text (avoids O(N) work on body <p>s!)
        if ln not in ("h1", "h2", "h3", "h4"):
            if not (ln == "p" and not saw_prose):
                return False
        for desc in el.iter():
            if desc is not el and _is_element(desc) and local_name(desc.tag) in ("p", "div", "table", "ul", "ol", "blockquote"):
                return False
        if len(list(el.iter())) > 15:
            return False
        if _has_non_ornament_img(el, rel_dir, orn_set):
            return False
        lines = _heading_lines(el)
        if not lines:
            return False
        full_txt = " ".join(lines)
        if len(full_txt) > 100:
            return False
        if ln in ("h1", "h2", "h3", "h4"):
            return True
        cls = el.get("class") or ""
        is_prose_sentence = bool(re.search(r'[.!?…"”\'’]$', full_txt)) or len(full_txt.split()) > 8
        has_heading_cls = bool(HEADING_CLASS_RE.search(cls))
        if has_heading_cls:
            is_explicit = bool(
                extract_num_sub(lines)
                or CHAPTER_LABEL_ONLY_RE.match(full_txt)
                or DIVISION_LABEL_ONLY_RE.match(full_txt)
                or _EXPLICIT_P_HEADING_RE.match(full_txt)
            )
            if is_explicit or not is_prose_sentence:
                return True
        else:
            # Explicit 'Chapter N', 'Part I', 'Prologue', 'Epilogue' on <p>
            explicit_non_bare = bool(
                CHAPTER_NUM_TITLE_RE.match(full_txt)
                or DIVISION_NUM_TITLE_RE.match(full_txt)
                or CHAPTER_LABEL_ONLY_RE.match(full_txt)
                or DIVISION_LABEL_ONLY_RE.match(full_txt)
                or _EXPLICIT_P_HEADING_RE.match(full_txt)
            )
            if explicit_non_bare and len(full_txt.split()) <= 6 and not is_prose_sentence:
                return True
            # First-block standalone word-number ('One', 'Nine'), Roman numeral ('IV'), or 1-99 chapter number
            # (e.g. Calibre <p class="calibre_"><b>Nine</b></p> in Notes on a Scandal), while rejecting 3-digit PDF page numbers like '151'!
            if is_first_block and not is_prose_sentence:
                bn = _parse_bare_number(full_txt)
                if bn and not bn[1]:
                    if bn[0].isdigit():
                        num_val = int(bn[0])
                        if 1 <= num_val <= 99:
                            has_emph = any(
                                _is_element(c) and (
                                    local_name(c.tag) in ("b", "strong", "i", "em")
                                    or "bold" in (c.get("class") or "").lower()
                                )
                                for c in el.iter()
                            )
                            next_is_prose = (
                                next_sib is not None
                                and _is_element(next_sib)
                                and local_name(next_sib.tag) == "p"
                                and len(extract_clean_text(next_sib).split()) >= 10
                            )
                            if has_emph or next_is_prose:
                                return True
                    else:
                        return True
        return False

    def is_ornament_only_heading_el(el: etree._Element) -> bool:
        if not _is_element(el) or extract_clean_text(el):
            return False
        imgs = [c for c in el.iter() if _is_element(c) and local_name(c.tag) == "img"]
        if not imgs:
            return False
        return all(_normalize_rel(rel_dir, c.get("src") or "") in orn_set for c in imgs)

    def extract_num_sub(lines: list[str]) -> tuple[str, str, str] | None:
        """Returns (num_part, label_part, sub_part) or None."""
        t1 = " ".join(lines)
        if len(lines) == 2:
            a, b = lines[0], lines[1]
            if CHAPTER_LABEL_ONLY_RE.match(a):
                bn = _parse_bare_number(b)
                if bn:
                    return bn[0], "", bn[1]
            m1 = CHAPTER_NUM_TITLE_RE.match(a)
            if m1 and not m1.group(2) and not NON_SUBTITLE_HEADING_RE.match(b):
                return m1.group(1), "", b
            bn = _parse_bare_number(a)
            if bn and not bn[1] and not NON_SUBTITLE_HEADING_RE.match(b) and not _parse_bare_number(b):
                return bn[0], "", b
            if DIVISION_LABEL_ONLY_RE.match(a):
                bn = _parse_bare_number(b)
                if bn:
                    return "", f"{a.title()} {bn[0]}", bn[1]

        m = CHAPTER_NUM_TITLE_RE.match(t1)
        if m:
            return m.group(1), "", (m.group(2) or "").strip()
        bn = _parse_bare_number(t1)
        if bn:
            return bn[0], "", bn[1]
        dm = DIVISION_NUM_TITLE_RE.match(t1)
        if dm:
            return "", dm.group(1), (dm.group(2) or "").strip()
        return None

    def process_container(container: etree._Element) -> None:
        saw_prose = False
        seen_nonempty = False
        children = list(container)
        i = 0
        while i < len(children):
            ch = children[i]
            if not _is_element(ch):
                i += 1
                continue

            ln = local_name(ch.tag)
            cls = ch.get("class") or ""
            if ln in ("div", "section", "header", "hgroup", "article", "main") and "x4-chapter-header" not in cls:
                has_block_child = any(
                    _is_element(c) and local_name(c.tag) in ("p", "div", "h1", "h2", "h3", "h4", "section", "header", "hgroup", "article", "main")
                    for c in ch
                )
                if has_block_child:
                    process_container(ch)
                    if not saw_prose and len(extract_clean_text(ch).split()) > 15:
                        saw_prose = True
                    seen_nonempty = True
                    i += 1
                    continue

            # Strip empty whitespace-only heading placeholders like <h2>&nbsp;</h2>
            if (
                ln in ("h1", "h2", "h3", "h4")
                and not _heading_lines(ch)
                and not any(_is_element(c) and local_name(c.tag) in ("img", "image", "svg") for c in ch.iter())
            ):
                anc_id = ch.get("id")
                insert_at = container.index(ch)
                tail = ch.tail
                container.remove(ch)
                if anc_id:
                    a_el = etree.Element(f"{ns_prefix}a" if ns_prefix else "a")
                    a_el.set("id", anc_id)
                    a_el.tail = tail
                    container.insert(insert_at, a_el)
                    i = insert_at + 1
                else:
                    i = insert_at
                children = list(container)
                continue

            next_sib = next((c for c in children[i + 1 :] if _is_element(c) and not _is_empty_anchor_el(c)), None)
            if not is_heading_candidate(ch, saw_prose, is_first_block=(not seen_nonempty), next_sib=next_sib):
                if not saw_prose and ln == "p":
                    ptxt = extract_clean_text(ch)
                    if ptxt:
                        seen_nonempty = True
                        if len(ptxt.split()) > 15:
                            saw_prose = True
                i += 1
                continue
            seen_nonempty = True

            lines = _heading_lines(ch)
            t1 = " ".join(lines)
            consumed = [ch]
            num_part, label_part, sub_part = "", "", ""

            # Check single/multiline pattern first
            parsed = extract_num_sub(lines)
            if parsed:
                num_part, label_part, sub_part = parsed
            elif CHAPTER_LABEL_ONLY_RE.match(t1) or DIVISION_LABEL_ONLY_RE.match(t1):
                # Peek next heading candidate (skipping up to 2 empty anchors)
                j = i + 1
                while j < len(children) and j <= i + 3 and _is_empty_anchor_el(children[j]):
                    j += 1
                if j < len(children) and is_heading_candidate(children[j], saw_prose):
                    nlines = _heading_lines(children[j])
                    bn = _parse_bare_number(" ".join(nlines)) if len(nlines) == 1 else _parse_bare_number(nlines[0])
                    if bn:
                        consumed.extend(children[i + 1 : j + 1])
                        if CHAPTER_LABEL_ONLY_RE.match(t1):
                            num_part = bn[0]
                            sub_part = bn[1] or (nlines[1] if len(nlines) == 2 else "")
                        else:
                            label_part = f"{t1.title()} {bn[0]}"
                            sub_part = bn[1] or (nlines[1] if len(nlines) == 2 else "")
                    else:
                        label_part = t1
                else:
                    label_part = t1
            else:
                label_part = t1

            # Peek following heading siblings for subtitle(s), skipping empty anchors and ornament-only divider headings
            sub_lines: list[str] = [sub_part] if sub_part else []
            allow_sub_peek = bool(num_part) or bool(label_part)
            if allow_sub_peek:
                for _step in range(2):
                    if sub_lines and _step == 0:
                        continue
                    last_idx = i + len(consumed) - 1
                    j = last_idx + 1
                    while (
                        j < len(children)
                        and j <= last_idx + 3
                        and (_is_empty_anchor_el(children[j]) or is_ornament_only_heading_el(children[j]))
                    ):
                        j += 1
                    if j < len(children) and is_heading_candidate(children[j], saw_prose):
                        nxt_txt = " ".join(_heading_lines(children[j]))
                        if (
                            nxt_txt
                            and len(nxt_txt) <= 100
                            and not NON_SUBTITLE_HEADING_RE.match(nxt_txt)
                            and not _parse_bare_number(nxt_txt)
                        ):
                            sub_lines.append(nxt_txt)
                            consumed.extend(children[last_idx + 1 : j + 1])
                            continue
                    break

            if not (num_part or label_part):
                i += 1
                continue

            header_div = etree.Element(f"{ns_prefix}div" if ns_prefix else "div")
            header_div.set("class", "x4-chapter-header")
            first_id = ch.get("id")
            if first_id:
                header_div.set("id", first_id)

            for c_el in consumed:
                for anc in _collect_empty_anchors(c_el):
                    header_div.append(anc)

            if num_part:
                num_div = etree.SubElement(header_div, f"{ns_prefix}div" if ns_prefix else "div")
                num_div.set("class", "x4-ch-num")
                num_div.text = num_part
            elif label_part:
                lbl_div = etree.SubElement(header_div, f"{ns_prefix}div" if ns_prefix else "div")
                lbl_div.set("class", "x4-ch-label")
                lbl_div.text = label_part

            if sub_lines:
                sub_div = etree.SubElement(header_div, f"{ns_prefix}div" if ns_prefix else "div")
                sub_div.set("class", "x4-ch-title")
                sub_div.text = sub_lines[0]
                for extra_line in sub_lines[1:]:
                    br = etree.SubElement(sub_div, f"{ns_prefix}br" if ns_prefix else "br")
                    br.tail = f" {extra_line}"

            header_div.tail = consumed[-1].tail
            insert_at = container.index(ch)
            for c_el in consumed:
                container.remove(c_el)
            container.insert(insert_at, header_div)
            children = list(container)
            i = insert_at + 1

    process_container(body)


def _find_cover_image_href(opf_root: etree._Element) -> str | None:
    manifest = _find_element(opf_root, "manifest")
    if manifest is None:
        return None
    for item in manifest:
        if _is_element(item) and "cover-image" in (item.get("properties") or "").split():
            return unquote(item.get("href", ""))
    for meta in opf_root.iter():
        if local_name(meta.tag) == "meta" and meta.get("name") == "cover":
            cid = meta.get("content", "")
            for item in manifest:
                if _is_element(item) and item.get("id") == cid:
                    return unquote(item.get("href", ""))
    for item in manifest:
        if _is_element(item) and (item.get("media-type") or "").startswith("image/"):
            href = unquote(item.get("href", ""))
            if "cover" in Path(href).stem.lower():
                return href
    return None


def _build_cover_xhtml(cover_img_rel_from_cover: str, title: str, css_href: str = HOUSE_STYLE_FILENAME) -> bytes:
    safe_title = (title or "Cover").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml">\n'
        f'<head><title>{safe_title}</title>'
        f'<link rel="stylesheet" type="text/css" href="{css_href}"/></head>\n'
        '<body class="x4-cover-wrap">'
        f'<div class="x4-cover-wrap"><img class="x4-cover-img" src="{cover_img_rel_from_cover}" alt="Cover" style="width:100%;max-width:100%;height:auto;display:block;margin:0 auto;"/></div>'
        '</body>\n</html>\n'
    )
    return html.encode("utf-8")


def _build_title_xhtml(title: str, author: str, lang: str) -> bytes:
    def esc(s: str) -> str:
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    lang_attr = f' lang="{esc(lang)}" xml:lang="{esc(lang)}"' if lang else ""
    author_html = f'<p class="x4-title-author">{esc(author)}</p>' if author else ""
    html = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f'<html xmlns="http://www.w3.org/1999/xhtml"{lang_attr}>\n'
        f'<head><title>{esc(title or "Title Page")}</title>'
        f'<link rel="stylesheet" type="text/css" href="{HOUSE_STYLE_FILENAME}"/></head>\n'
        '<body>\n'
        '<div class="x4-title-page">\n'
        f'  <h1 class="x4-title-book">{esc(title or "Untitled")}</h1>\n'
        '  <div class="x4-title-rule">&#160;</div>\n'
        f'  {author_html}\n'
        '</div>\n'
        '</body>\n</html>\n'
    )
    return html.encode("utf-8")


def ensure_cover_entry(
    opf_path: str,
    title: str = "",
    author: str = "",
    lang: str = "",
    ensure_title_page: bool = True,
    cover_xhtml_rel: str | None = None,
) -> tuple[bool, bool]:
    """
    Guarantee clean Cover -> Standardized Minimal Title Page -> Story ordering at the front of spine.
    Safe to call again after `generate_missing_cover` runs!
    Returns (cover_ensured, title_created).
    """
    opf_tree = etree.parse(opf_path, xml_parser())
    opf_root = opf_tree.getroot()
    opf_dir = str(Path(opf_path).parent)
    manifest = _find_element(opf_root, "manifest")
    spine = _find_element(opf_root, "spine")
    if manifest is None or spine is None:
        return False, False

    id_to_item = {it.get("id", ""): it for it in manifest if _is_element(it)}
    cover_img_href = _find_cover_image_href(opf_root)

    cover_ensured = False
    title_created = False

    # Check whether spine[0] is already a cover page
    spine_els = [s for s in spine if _is_element(s)]
    has_cover_page = False
    existing_cover_href = None
    has_title_page = False

    for s in spine_els[:2]:
        item = id_to_item.get(s.get("idref", ""))
        if item is not None:
            href = unquote(item.get("href", ""))
            norm_href = posixpath.normpath(href.replace("\\", "/"))
            if (
                (cover_xhtml_rel and norm_href == cover_xhtml_rel)
                or Path(href).name == COVER_PAGE_FILENAME
                or is_cover_stem(href)
            ):
                has_cover_page = True
                if not existing_cover_href:
                    existing_cover_href = norm_href
            if Path(href).name == TITLE_PAGE_FILENAME:
                has_title_page = True

    insert_pos = 0
    if cover_img_href and not has_cover_page:
        cover_abs = os.path.join(opf_dir, COVER_PAGE_FILENAME)
        Path(cover_abs).write_bytes(_build_cover_xhtml(cover_img_href, title))

        cid = "x4-cover-page"
        if cid not in id_to_item:
            c_item = etree.Element(f"{{{NS_OPF}}}item")
            c_item.set("id", cid)
            c_item.set("href", COVER_PAGE_FILENAME)
            c_item.set("media-type", "application/xhtml+xml")
            manifest.append(c_item)
        c_ref = etree.Element(f"{{{NS_OPF}}}itemref")
        c_ref.set("idref", cid)
        spine.insert(0, c_ref)
        insert_pos = 1
        cover_ensured = True
    elif has_cover_page:
        # Standardize existing cover XHTML so every book renders full-width centered (no tiny stamp covers!)
        if existing_cover_href:
            resolved_cov_img = cover_img_href
            existing_abs = os.path.join(opf_dir, existing_cover_href)
            cov_dir = posixpath.dirname(existing_cover_href)
            if not resolved_cov_img and os.path.exists(existing_abs):
                # Fallback for hash-named cover images (e.g. Wolfskin) not marked cover-image in OPF
                try:
                    c_root, _ = parse_xhtml_safe(Path(existing_abs).read_bytes())
                    for el in c_root.iter():
                        eln = local_name(el.tag)
                        if eln == "img" and el.get("src"):
                            resolved_cov_img = _normalize_rel(cov_dir, el.get("src"))
                            break
                        if eln == "image":
                            h_v = el.get(f"{{{NS_XLINK}}}href") or el.get("href") or ""
                            if h_v:
                                resolved_cov_img = _normalize_rel(cov_dir, h_v)
                                break
                except Exception:
                    pass
            if resolved_cov_img:
                # Register resolved cover image in OPF if missing so pruning & generate_missing_cover keep it
                cov_item = next(
                    (
                        it for it in manifest
                        if _is_element(it)
                        and posixpath.normpath(unquote(it.get("href", "")).replace("\\", "/")) == resolved_cov_img
                    ),
                    None,
                )
                if cov_item is not None:
                    props = (cov_item.get("properties") or "").split()
                    if "cover-image" not in props:
                        cov_item.set("properties", " ".join(props + ["cover-image"]).strip())
                        cover_ensured = True
                    meta_el = _find_element(opf_root, "metadata")
                    has_cover_meta = any(
                        _is_element(m) and local_name(m.tag) == "meta" and m.get("name") == "cover"
                        for m in opf_root.iter()
                    )
                    if meta_el is not None and not has_cover_meta and cov_item.get("id"):
                        m_el = etree.SubElement(meta_el, f"{{{NS_OPF}}}meta")
                        m_el.set("name", "cover")
                        m_el.set("content", cov_item.get("id", ""))
                        cover_ensured = True

                rel_img = posixpath.relpath(resolved_cov_img, cov_dir) if cov_dir else resolved_cov_img
                rel_css = posixpath.relpath(HOUSE_STYLE_FILENAME, cov_dir) if cov_dir else HOUSE_STYLE_FILENAME
                Path(existing_abs).parent.mkdir(parents=True, exist_ok=True)
                Path(existing_abs).write_bytes(_build_cover_xhtml(rel_img, title, css_href=rel_css))
        insert_pos = 1

    if ensure_title_page and not has_title_page and (title or author):
        title_abs = os.path.join(opf_dir, TITLE_PAGE_FILENAME)
        Path(title_abs).write_bytes(_build_title_xhtml(title, author, lang))

        tid = "x4-title-page"
        if tid not in id_to_item:
            t_item = etree.Element(f"{{{NS_OPF}}}item")
            t_item.set("id", tid)
            t_item.set("href", TITLE_PAGE_FILENAME)
            t_item.set("media-type", "application/xhtml+xml")
            manifest.append(t_item)
        t_ref = etree.Element(f"{{{NS_OPF}}}itemref")
        t_ref.set("idref", tid)
        spine.insert(insert_pos, t_ref)
        title_created = True

    if cover_ensured or title_created:
        opf_tree.write(opf_path, xml_declaration=True, encoding="utf-8", pretty_print=True)

    return cover_ensured, title_created


_SCHEMELESS_URL_RE = re.compile(
    r"^(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)+\.(?:com|org|net|edu|gov|io|co\.uk|de)(?:/.*|#.*)?$",
    re.I,
)


def _cleanup_dead_internal_links(
    opf_dir: str,
    candidate_rels: list[str],
    all_manifest_rels: set[str],
    ids_by_rel: dict[str, set[str]],
) -> int:
    """
    Remove/unwrap dead internal links (e.g. fragments trimmed during chapter splitting
    or missing source anchors) and scheme-less domain URLs. Streams one file at a time.
    """
    cleaned = 0
    for rel in candidate_rels:
        abs_p = os.path.join(opf_dir, rel)
        if not os.path.exists(abs_p):
            continue
        rel_dir = posixpath.dirname(rel)
        root, _ = parse_xhtml_safe(Path(abs_p).read_bytes())
        body = get_body(root)
        modified = False

        for a in list(body.iter()):
            if not _is_element(a) or local_name(a.tag) != "a":
                continue
            href = (a.get("href") or "").strip()
            if not href or href.lower().startswith(("http://", "https://", "mailto:", "javascript:")):
                continue

            norm_full = _normalize_rel(rel_dir, href)
            if _SCHEMELESS_URL_RE.match(href) and norm_full not in all_manifest_rels:
                _unwrap_element(a)
                cleaned += 1
                modified = True
                continue

            file_part, frag = (href.split("#", 1) + [""])[:2]
            tgt_rel = _normalize_rel(rel_dir, file_part) if file_part else rel
            if not tgt_rel:
                continue

            missing_file = tgt_rel not in all_manifest_rels
            missing_frag = bool(frag) and (tgt_rel in ids_by_rel) and (frag not in ids_by_rel[tgt_rel])
            if not (missing_file or missing_frag):
                continue

            reason_hint = "notes" if re.search(r"(?:^|_|-)(?:edn|fn|note|endnote)", frag, re.I) else ""
            if is_footnote_link(a, reason_hint):
                parent = a.getparent()
                _remove_element_keep_tail(a)
                if parent is not None and local_name(parent.tag) == "sup":
                    if not (parent.text or "").strip() and len(parent) == 0:
                        _remove_element_keep_tail(parent)
            else:
                _unwrap_element(a)
            cleaned += 1
            modified = True

        if modified:
            Path(abs_p).write_bytes(
                etree.tostring(root, xml_declaration=True, encoding="utf-8", pretty_print=True)
            )
    return cleaned


def apply_house_style(
    epub_dir: str,
    opf_path: str,
    standardize_layout: bool = True,
    normalize_scene_breaks: bool = True,
    cover_xhtml_rel: str | None = None,
    ensure_title_page: bool = True,
) -> HouseStyleReport:
    """
    Apply X4 Bookerly House Style CSS, normalize scene breaks (including ornament images
    BEFORE image processing), flatten dropcaps, unwrap prose URLs, sync <html lang>,
    build minimal Title Page, and prune newly orphaned ornament images.
    """
    report = HouseStyleReport()
    opf_tree = etree.parse(opf_path, xml_parser())
    opf_root = opf_tree.getroot()
    opf_dir = str(Path(opf_path).parent)

    manifest = _find_element(opf_root, "manifest")
    spine = _find_element(opf_root, "spine")
    if manifest is None or spine is None:
        return report

    title, author, lang = "", "", ""
    for el in opf_root.iter():
        ln = local_name(el.tag)
        if ln == "title" and not title and (el.text or "").strip():
            title = el.text.strip()
        elif ln == "creator" and not author and (el.text or "").strip():
            author = el.text.strip()
        elif ln == "language" and not lang and (el.text or "").strip():
            lang = el.text.strip()

    id_to_item = {it.get("id", ""): it for it in manifest if _is_element(it)}

    # Inject x4_house_style.css into OPF root & manifest
    if standardize_layout:
        css_abs = os.path.join(opf_dir, HOUSE_STYLE_FILENAME)
        Path(css_abs).write_text(HOUSE_STYLE_CSS, encoding="utf-8")

        if not any(unquote(it.get("href", "")) == HOUSE_STYLE_FILENAME for it in id_to_item.values()):
            css_item = etree.Element(f"{{{NS_OPF}}}item")
            css_item.set("id", "x4-house-style")
            css_item.set("href", HOUSE_STYLE_FILENAME)
            css_item.set("media-type", "text/css")
            manifest.append(css_item)

        # Scrub hostile rules from existing publisher CSS files
        for item in id_to_item.values():
            if (item.get("media-type") or "").lower() == "text/css":
                href = unquote(item.get("href", ""))
                if href == HOUSE_STYLE_FILENAME:
                    continue
                c_abs = os.path.join(opf_dir, href)
                if os.path.exists(c_abs):
                    raw_css = Path(c_abs).read_text(encoding="utf-8", errors="ignore")
                    scrubbed = _scrub_css_declarations(raw_css)
                    if scrubbed != raw_css:
                        Path(c_abs).write_text(scrubbed, encoding="utf-8")
                        report.css_files_scrubbed += 1

    # Gather story spine files
    spine_rels: list[str] = []
    for itemref in spine:
        if not _is_element(itemref):
            continue
        it = id_to_item.get(itemref.get("idref", ""))
        if it is not None:
            rel = posixpath.normpath(unquote(it.get("href", "")).replace("\\", "/"))
            if os.path.exists(os.path.join(opf_dir, rel)):
                spine_rels.append(rel)

    story_rels = [r for r in spine_rels if r != cover_xhtml_rel]
    ornament_rels = _detect_ornament_images(opf_dir, story_rels) if normalize_scene_breaks else set()

    # Stream one story XHTML file at a time (Render 512 MB RAM safe)
    surviving_img_refs: set[str] = set()
    emptied_rels: list[str] = []
    ids_by_rel: dict[str, set[str]] = {}
    candidate_link_rels: list[str] = []
    for rel in story_rels:
        abs_p = os.path.join(opf_dir, rel)
        rel_dir = posixpath.dirname(rel)
        root, _ = parse_xhtml_safe(Path(abs_p).read_bytes())
        body = get_body(root)
        file_has_internal_link = False

        # Normalize <html lang="..."> from OPF metadata
        if lang and not (root.get("lang") or root.get("{http://www.w3.org/XML/1998/namespace}lang")):
            root.set("lang", lang)
            root.set("{http://www.w3.org/XML/1998/namespace}lang", lang)
            report.lang_attrs_added += 1

        # Unwrap external prose hyperlinks & strip OceanofPDF / Z-Lib watermarks
        for a in list(body.iter()):
            if local_name(a.tag) != "a":
                continue
            href = (a.get("href") or "").strip()
            if href and re.match(r"^(https?://|mailto:|www\.)", href, re.I):
                txt = extract_clean_text(a)
                if PIRATE_WATERMARK_RE.search(txt) or PIRATE_WATERMARK_RE.search(href):
                    parent = a.getparent()
                    if parent is not None and local_name(parent.tag) == "p" and len(extract_clean_text(parent)) <= len(txt) + 25:
                        if parent.getparent() is not None:
                            parent.getparent().remove(parent)
                    elif parent is not None:
                        parent.remove(a)
                else:
                    _unwrap_element(a)
                report.prose_links_unwrapped += 1
            elif href:
                file_has_internal_link = True

        if standardize_layout:
            # Scrub inline styles, strip fixed table widths, & flatten floated dropcaps
            for el in body.iter():
                if not _is_element(el):
                    continue
                ln_el = local_name(el.tag)
                if ln_el in _TABLE_TAGS and "width" in el.attrib:
                    del el.attrib["width"]
                st = el.get("style")
                if st:
                    new_st = _scrub_inline_style(st, tag_name=ln_el)
                    if new_st:
                        el.set("style", new_st)
                    else:
                        del el.attrib["style"]
                cls = (el.get("class") or "").lower()
                if ln_el == "span" and any(k in cls for k in ("dropcap", "drop-cap", "initial", "firstletter", "init-cap", "initcap", "lettrine")):
                    el.set("class", "x4-dropcap-flat")
                    report.dropcaps_flattened += 1

            # Ensure <link> to x4_house_style.css is last stylesheet in <head>, and scrub inline <style> tags (incl. @page!)
            head = None
            for ch in root:
                if local_name(ch.tag) == "head":
                    head = ch
                    break
            if head is None:
                ns_p = root.tag.split("}")[0] + "}" if "}" in str(root.tag) else ""
                head = etree.Element(f"{ns_p}head" if ns_p else "head")
                root.insert(0, head)
            else:
                for st_el in list(head):
                    if _is_element(st_el) and local_name(st_el.tag) == "style" and st_el.text:
                        scrubbed_st = _scrub_css_declarations(st_el.text).strip()
                        if scrubbed_st:
                            st_el.text = scrubbed_st
                        else:
                            head.remove(st_el)

            css_href = posixpath.relpath(HOUSE_STYLE_FILENAME, rel_dir) if rel_dir else HOUSE_STYLE_FILENAME
            already_linked = any(
                local_name(el.tag) == "link" and (el.get("href") or "").endswith(HOUSE_STYLE_FILENAME)
                for el in head
            )
            if not already_linked:
                ns_p = head.tag.split("}")[0] + "}" if "}" in str(head.tag) else ""
                link_el = etree.SubElement(head, f"{ns_p}link" if ns_p else "link")
                link_el.set("rel", "stylesheet")
                link_el.set("type", "text/css")
                link_el.set("href", css_href)

            _format_number_first_headings(body, rel_dir, ornament_rels)

        # Scene break normalization & fake-spacer removal
        _normalize_scene_breaks_in_body(
            body, rel_dir, ornament_rels,
            normalize_breaks=normalize_scene_breaks,
            report=report,
        )

        # Strip broken <img> tags pointing to files already missing in the archive & center valid inline images
        if standardize_layout:
            for img in list(body.iter()):
                if not _is_element(img) or local_name(img.tag) != "img":
                    continue
                src = (img.get("src") or "").strip()
                if not src or src.startswith(("data:", "http://", "https://")):
                    continue
                tgt_img_rel = _normalize_rel(rel_dir, src)
                if not tgt_img_rel or not os.path.exists(os.path.join(opf_dir, tgt_img_rel)):
                    parent = img.getparent()
                    _remove_element_keep_tail(img)
                    if parent is not None and local_name(parent.tag) in ("p", "div"):
                        if not (parent.text or "").strip() and len(parent) == 0:
                            _remove_element_keep_tail(parent)

            for el in body.iter():
                if not _is_element(el) or local_name(el.tag) not in ("p", "div"):
                    continue
                cls_tokens = (el.get("class") or "").split()
                if "x4-img-wrap" in cls_tokens or "x4-cover-wrap" in cls_tokens or "x4-chapter-header" in cls_tokens:
                    continue
                has_inner_img_block = any(
                    _is_element(c) and local_name(c.tag) in ("p", "div") and c is not el
                    and any(_is_element(g) and local_name(g.tag) in ("img", "image") for g in c.iter())
                    for c in el.iter()
                )
                if has_inner_img_block:
                    continue
                has_img = any(_is_element(c) and local_name(c.tag) in ("img", "image") for c in el.iter())
                if has_img and count_words(extract_clean_text(el)) <= 15:
                    el.set("class", " ".join(cls_tokens + ["x4-img-wrap"]).strip())

        # If cleaning left a completely blank story chapter (0 text & 0 images), mark for removal
        has_any_text = bool(extract_clean_text(body))
        has_any_media = any(_is_element(c) and local_name(c.tag) in ("img", "image", "svg") for c in body.iter())
        if not has_any_text and not has_any_media and len(story_rels) - len(emptied_rels) > 1:
            emptied_rels.append(rel)
            continue

        if ornament_rels:
            surviving_img_refs.update(_collect_doc_img_refs(root, rel_dir))

        if standardize_layout:
            ids_set: set[str] = set()
            for el in root.iter():
                if not _is_element(el):
                    continue
                eid = el.get("id")
                if eid:
                    ids_set.add(eid)
                if local_name(el.tag) == "a":
                    ename = el.get("name")
                    if ename:
                        ids_set.add(ename)
            ids_by_rel[rel] = ids_set
            if file_has_internal_link:
                candidate_link_rels.append(rel)

        Path(abs_p).write_bytes(
            etree.tostring(root, xml_declaration=True, encoding="utf-8", pretty_print=True)
        )

    # Remove newly emptied blank story files from OPF spine & manifest
    if emptied_rels:
        emptied_set = set(emptied_rels)
        for itemref in list(spine):
            if not _is_element(itemref):
                continue
            it = id_to_item.get(itemref.get("idref", ""))
            if it is not None:
                r = posixpath.normpath(unquote(it.get("href", "")).replace("\\", "/"))
                if r in emptied_set:
                    spine.remove(itemref)
                    if it.getparent() is not None:
                        it.getparent().remove(it)
                    abs_e = os.path.join(opf_dir, r)
                    if os.path.exists(abs_e):
                        os.unlink(abs_e)

    # Save OPF before ensuring Cover -> Title Page -> Story
    opf_tree.write(opf_path, xml_declaration=True, encoding="utf-8", pretty_print=True)

    # Clean dead internal #fragment / missing-target links and scheme-less URLs across finalized spine
    if standardize_layout and candidate_link_rels:
        kept_set = {r for r in story_rels if r not in set(emptied_rels)}
        manifest_rels = {
            posixpath.normpath(unquote(it.get("href", "")).replace("\\", "/"))
            for it in manifest
            if _is_element(it) and it.get("href")
        }
        filtered_candidates = [r for r in candidate_link_rels if r in kept_set]
        report.prose_links_unwrapped += _cleanup_dead_internal_links(
            opf_dir, filtered_candidates, manifest_rels, ids_by_rel
        )

    # Ensure clean Cover -> Minimal Title Page -> Story entry
    cov_ok, title_ok = ensure_cover_entry(
        opf_path, title=title, author=author, lang=lang,
        ensure_title_page=ensure_title_page and standardize_layout,
        cover_xhtml_rel=cover_xhtml_rel,
    )
    report.cover_page_ensured = cov_ok
    report.title_page_created = title_ok

    # Prune newly orphaned ornament images BEFORE Pillow image processing runs!
    if ornament_rels:
        opf_tree = etree.parse(opf_path, xml_parser())
        opf_root = opf_tree.getroot()
        manifest = _find_element(opf_root, "manifest")
        spine = _find_element(opf_root, "spine")
        id_to_item = {it.get("id", ""): it for it in manifest if _is_element(it)}
        all_kept = [
            posixpath.normpath(unquote(id_to_item[s.get("idref", "")].get("href", "")).replace("\\", "/"))
            for s in spine
            if _is_element(s) and s.get("idref", "") in id_to_item
        ]
        cover_img = _find_cover_image_href(opf_root)
        report.orphan_images_removed = _prune_unreferenced_images(
            opf_root, opf_dir, all_kept, cover_img, precollected_img_refs=surviving_img_refs
        )
        if report.orphan_images_removed > 0:
            opf_tree.write(opf_path, xml_declaration=True, encoding="utf-8", pretty_print=True)

    return report
