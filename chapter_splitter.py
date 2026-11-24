"""
Chapter Auto-Splitter for Xteink X4 Story-First Edition Engine.
Detects oversized or multi-chapter XHTML files, trims glued-on trailing promo sections,
cleanly splits DOM trees at chapter boundaries (preserving arbitrary wrapper nesting),
updates OPF manifest/spine, and remaps internal #fragment links.
"""
from __future__ import annotations

import copy
import os
import posixpath
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

from lxml import etree

from html_cleaner import xml_parser
from epub_structure import NS_OPF, NS_XHTML, _is_element, _find_element
from matter_trimmer import (
    TRASH_TITLE_PATTERNS,
    local_name,
    norm_match_text,
    extract_clean_text,
    count_words,
    parse_xhtml_safe,
    get_body,
    _normalize_rel,
)

CHAPTER_HEADING_RE = re.compile(
    r"^(chapter|kapitel|part|teil|book|buch|prologue|prolog|epilogue|epilog|act|canto|interlude)\b"
    r"|^[IVXLCDM]{1,7}\.?$"
    r"|^\d{1,3}\.?$"
    r"|^(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)$",
    re.I,
)

EXPLICIT_CHAPTER_PREFIX_RE = re.compile(
    r"^(chapter|kapitel|part|teil|book|buch|prologue|prolog|epilogue|epilog)\b",
    re.I,
)

GIANT_CHAPTER_WORD_THRESHOLD = 30000
FALLBACK_CHUNK_TARGET_WORDS = 18000

_PAGE_BREAK_STYLE_RE = re.compile(
    r"(?:page-break-before\s*:\s*always|break-before\s*:\s*(?:always|page))",
    re.I,
)
_CSS_RULE_PB_RE = re.compile(
    r"\.([A-Za-z0-9_-]+)[^{]*\{[^}]*(?:page-break-before\s*:\s*always|break-before\s*:\s*(?:always|page))",
    re.I,
)
_GENERIC_STEM_RE = re.compile(
    r"^(?:part|chapter|chap|ch|sec|section|split|text|index|body|book|file)[_-]?\d+$|^.+split_\d+$",
    re.I,
)


@dataclass
class SplitReport:
    files_split: int = 0
    chapters_created: int = 0
    trailing_promo_trimmed_words: int = 0
    links_remapped: int = 0
    details: list[str] = field(default_factory=list)


def _is_trash_heading(label: str) -> bool:
    clean = label.replace("\u200c", "").replace("\u200b", "").strip()
    clean = norm_match_text(clean)
    for part in clean.split(" / "):
        for _, pat in TRASH_TITLE_PATTERNS:
            if pat.search(part.strip()):
                return True
    return False


def _detect_split_points(
    body: etree._Element,
    allow_generic_headings: bool,
) -> tuple[list[tuple[etree._Element, str]], etree._Element | None]:
    """
    Return (story_splits, trailing_trash_el).
    story_splits: list of (heading_element, merged_title) in document order.
    trailing_trash_el: first glued-on non-story heading (e.g. 'Become a subscriber...') to trim, if any.
    """
    for heading_tags in (("h1", "h2"), ("h3",)):
        candidates: list[tuple[int, etree._Element, str]] = []
        words_so_far = 0

        for el in body.iter():
            ln = local_name(el.tag)
            if ln in ("script", "style"):
                continue

            is_heading = False
            if ln in heading_tags:
                txt = extract_clean_text(el)
                if txt and len(txt) <= 100:
                    if heading_tags == ("h3",):
                        if CHAPTER_HEADING_RE.search(norm_match_text(txt)):
                            is_heading = True
                    else:
                        is_heading = True

            if is_heading:
                txt = extract_clean_text(el)
                if candidates and (words_so_far - candidates[-1][0]) < 120:
                    prev_pos, prev_el, prev_txt = candidates[-1]
                    candidates[-1] = (prev_pos, prev_el, f"{prev_txt} — {txt}")
                else:
                    candidates.append((words_so_far, el, txt))
            elif ln in ("p", "li", "blockquote"):
                if not any(local_name(c.tag) in ("p", "div") for c in el):
                    words_so_far += count_words(extract_clean_text(el))

        if not candidates:
            continue

        # Filter candidates separated by sufficient prose
        min_gap = 500 if heading_tags == ("h3",) else 150
        spaced: list[tuple[int, etree._Element, str]] = [candidates[0]]
        for pos, el, lbl in candidates[1:]:
            if pos - spaced[-1][0] >= min_gap:
                spaced.append((pos, el, lbl))

        # Separate trailing glued-on promo/backmatter headings (only near the end: >= 65% of words)
        trailing_trash_el: etree._Element | None = None
        story_list: list[tuple[int, etree._Element, str]] = []
        for pos, el, lbl in spaced:
            if pos >= words_so_far * 0.65 and _is_trash_heading(lbl) and trailing_trash_el is None:
                trailing_trash_el = el
            elif trailing_trash_el is None:
                if not _is_trash_heading(lbl):
                    story_list.append((pos, el, lbl))

        # Drop final heading if it has almost no content after it (< 50 words)
        if len(story_list) >= 2 and (words_so_far - story_list[-1][0]) < 50:
            story_list.pop()

        if len(story_list) >= 2:
            if not allow_generic_headings:
                if not any(EXPLICIT_CHAPTER_PREFIX_RE.search(norm_match_text(lbl)) for _, _, lbl in story_list):
                    story_list = []
            if len(story_list) >= 2:
                return ([(el, lbl) for _, el, lbl in story_list], trailing_trash_el)

        if trailing_trash_el is not None:
            return ([], trailing_trash_el)

    return ([], None)


def _contains_el(ancestor: etree._Element, target: etree._Element) -> bool:
    if ancestor is target:
        return True
    for desc in ancestor.iter():
        if desc is target:
            return True
    return False


def _slice_subtree(
    node: etree._Element,
    split_el: etree._Element,
) -> tuple[etree._Element | None, etree._Element | None]:
    """
    Partition `node` cleanly into (left_tree, right_tree) right BEFORE `split_el`.
    Handles arbitrary nesting of <div>/<section> wrappers without losing text or attributes.
    """
    if node is split_el:
        return (None, copy.deepcopy(node))

    left = etree.Element(node.tag, attrib=dict(node.attrib), nsmap=node.nsmap)
    right = etree.Element(node.tag, attrib=dict(node.attrib), nsmap=node.nsmap)

    if node.text:
        left.text = node.text

    found = False
    for child in node:
        if not found and _contains_el(child, split_el):
            found = True
            c_left, c_right = _slice_subtree(child, split_el)
            if c_left is not None:
                c_left.tail = None
                left.append(c_left)
            if c_right is not None:
                c_right.tail = child.tail
                right.append(c_right)
        elif not found:
            left.append(copy.deepcopy(child))
        else:
            right.append(copy.deepcopy(child))

    # Avoid keeping empty shells
    left_has_content = bool((left.text or "").strip()) or len(left) > 0
    right_has_content = bool((right.text or "").strip()) or len(right) > 0
    return (left if left_has_content else None, right if right_has_content else None)


def _clone_doc_with_body(template_root: etree._Element, new_body: etree._Element, title_text: str) -> etree._Element:
    """Clone <head> from template_root and attach `new_body` with updated <title>."""
    new_root = etree.Element(template_root.tag, attrib=dict(template_root.attrib), nsmap=template_root.nsmap)
    for child in template_root:
        if local_name(child.tag) == "body":
            new_root.append(new_body)
        else:
            cloned = copy.deepcopy(child)
            if local_name(cloned.tag) == "head":
                for hchild in cloned:
                    if local_name(hchild.tag) == "title":
                        hchild.text = title_text
                        break
            new_root.append(cloned)
    return new_root


def _get_spine_text_items(opf_root: etree._Element, opf_dir: str) -> list[tuple[etree._Element, str, str]]:
    manifest = _find_element(opf_root, "manifest")
    spine = _find_element(opf_root, "spine")
    if manifest is None or spine is None:
        return []
    id_to_item = {
        item.get("id", ""): item
        for item in manifest
        if _is_element(item)
    }
    res = []
    for itemref in spine:
        if not _is_element(itemref):
            continue
        idref = itemref.get("idref", "")
        item = id_to_item.get(idref)
        if item is None:
            continue
        rel = posixpath.normpath(unquote(item.get("href", "")).replace("\\", "/"))
        if os.path.exists(os.path.join(opf_dir, rel)):
            res.append((itemref, idref, rel))
    return res


def _is_toc_sparse(opf_root: etree._Element, opf_dir: str, story_count: int) -> bool:
    """Return True if existing NCX/Nav has <= 1 entry or covers < 50% of story files."""
    manifest = _find_element(opf_root, "manifest")
    if manifest is None:
        return True
    ncx_count = 0
    nav_count = 0
    for item in manifest:
        if not _is_element(item):
            continue
        href = unquote(item.get("href", ""))
        props = (item.get("properties") or "").split()
        mtype = (item.get("media-type") or "").lower()
        abs_p = os.path.join(opf_dir, href)
        if not os.path.exists(abs_p):
            continue
        if mtype == "application/x-dtbncx+xml":
            try:
                t, _ = parse_xhtml_safe(Path(abs_p).read_bytes())
                ncx_count = sum(1 for el in t.iter() if local_name(el.tag) == "navpoint")
            except Exception:
                pass
        elif "nav" in props:
            try:
                t, _ = parse_xhtml_safe(Path(abs_p).read_bytes())
                for nav_el in t.iter():
                    if local_name(nav_el.tag) == "nav":
                        if any(k.endswith("type") and "toc" in v.lower() for k, v in nav_el.attrib.items()):
                            nav_count = sum(1 for a in nav_el.iter() if local_name(a.tag) == "a")
                            break
            except Exception:
                pass
    return (ncx_count <= 1 and nav_count <= 1) or (ncx_count < story_count * 0.5 and nav_count < story_count * 0.5)


def _collect_toc_fragment_anchors(opf_root: etree._Element, opf_dir: str) -> dict[str, list[tuple[str, str]]]:
    """Map normalized story XHTML relative paths -> [(fragment_id, label), ...] from NCX / EPUB3 Nav."""
    manifest = _find_element(opf_root, "manifest")
    if manifest is None:
        return {}

    ncx_anchors: dict[str, list[tuple[str, str]]] = {}
    nav_anchors: dict[str, list[tuple[str, str]]] = {}

    def _add_anchor(bucket: dict[str, list[tuple[str, str]]], toc_dir: str, toc_rel: str, raw_src: str, label: str):
        if "#" not in raw_src or not label:
            return
        file_part, frag = raw_src.split("#", 1)
        frag = unquote(frag).strip()
        if not frag:
            return
        clean_file = unquote(file_part).replace("\\", "/").strip()
        target_rel = _normalize_rel(toc_dir, clean_file) if clean_file else toc_rel
        bucket.setdefault(target_rel, []).append((frag, label))

    for item in manifest:
        if not _is_element(item):
            continue
        raw_href = unquote(item.get("href", "")).replace("\\", "/")
        toc_rel = posixpath.normpath(raw_href)
        toc_dir = posixpath.dirname(toc_rel)
        props = (item.get("properties") or "").split()
        mtype = (item.get("media-type") or "").lower()
        abs_p = os.path.join(opf_dir, toc_rel)
        if not os.path.exists(abs_p):
            continue

        if mtype == "application/x-dtbncx+xml":
            try:
                t, _ = parse_xhtml_safe(Path(abs_p).read_bytes())
                for np in t.iter():
                    if local_name(np.tag) != "navpoint":
                        continue
                    lbl = ""
                    src = ""
                    for child in np:
                        cn = local_name(child.tag)
                        if cn == "navlabel":
                            lbl = extract_clean_text(child)
                        elif cn == "content":
                            src = (child.get("src") or "").strip()
                    _add_anchor(ncx_anchors, toc_dir, toc_rel, src, lbl)
            except Exception:
                pass
        elif "nav" in props:
            try:
                t, _ = parse_xhtml_safe(Path(abs_p).read_bytes())
                for nav_el in t.iter():
                    if local_name(nav_el.tag) == "nav":
                        if any(k.endswith("type") and "toc" in v.lower() for k, v in nav_el.attrib.items()):
                            for a in nav_el.iter():
                                if local_name(a.tag) == "a":
                                    _add_anchor(nav_anchors, toc_dir, toc_rel, (a.get("href") or "").strip(), extract_clean_text(a))
                            break
            except Exception:
                pass

    return ncx_anchors if ncx_anchors else nav_anchors


def _collect_page_break_classes(root: etree._Element, opf_dir: str, rel: str) -> set[str]:
    """Find CSS class names declaring page-break-before: always / break-before: page."""
    pb_classes: set[str] = set()
    css_texts: list[str] = []
    rel_dir = posixpath.dirname(rel)

    for el in root.iter():
        ln = local_name(el.tag)
        if ln == "style" and el.text:
            css_texts.append(el.text)
        elif ln == "link":
            link_rel = (el.get("rel") or "").lower()
            href = (el.get("href") or "").strip()
            if href and ("stylesheet" in link_rel or href.lower().endswith(".css")):
                css_rel = _normalize_rel(rel_dir, unquote(href).replace("\\", "/"))
                css_abs = os.path.join(opf_dir, css_rel)
                if os.path.exists(css_abs):
                    try:
                        css_texts.append(Path(css_abs).read_text(encoding="utf-8", errors="ignore"))
                    except Exception:
                        pass

    for css in css_texts:
        for m in _CSS_RULE_PB_RE.finditer(css):
            pb_classes.add(m.group(1))
    return pb_classes


def _index_body_elements(
    body: etree._Element,
    pb_classes: set[str],
) -> tuple[
    int,
    int,
    dict[str, tuple[int, int, etree._Element]],
    list[tuple[int, int, etree._Element]],
    list[tuple[int, etree._Element]],
    list[tuple[int, etree._Element]],
]:
    """
    Single O(N) iterwalk over `body` recording:
      - total_words, total_media
      - id/anchor -> (doc_idx, word_pos, el)
      - page-break elements -> [(word_pos, media_count, el), ...]
      - <hr> elements -> [(word_pos, el), ...]
      - non-empty leaf prose block elements -> [(word_pos, el), ...]
    """
    words = 0
    media_count = 0
    doc_idx = 0
    id_map: dict[str, tuple[int, int, etree._Element]] = {}
    page_breaks: list[tuple[int, int, etree._Element]] = []
    hr_breaks: list[tuple[int, etree._Element]] = []
    leaf_blocks: list[tuple[int, etree._Element]] = []
    leaf_start: dict[etree._Element, int] = {}
    skip_stack: list[bool] = []
    block_tags = ("p", "div", "blockquote", "section")

    for event, el in etree.iterwalk(body, events=("start", "end")):
        if not _is_element(el):
            continue
        ln = local_name(el.tag)
        if event == "start":
            parent_skipped = bool(skip_stack and skip_stack[-1])
            skipped = parent_skipped or ln in ("script", "style")
            skip_stack.append(skipped)
            if skipped:
                continue

            if el is not body:
                cur_idx = doc_idx
                doc_idx += 1

                el_id = el.get("id") or (el.get("name") if ln == "a" else None)
                if el_id and el_id not in id_map:
                    id_map[el_id] = (cur_idx, words, el)

                style_attr = el.get("style") or ""
                classes = (el.get("class") or "").split()
                if (style_attr and _PAGE_BREAK_STYLE_RE.search(style_attr)) or (
                    pb_classes and any(c in pb_classes for c in classes)
                ):
                    page_breaks.append((words, media_count, el))

                if ln in ("img", "image", "svg"):
                    media_count += 1

                if ln == "hr":
                    hr_breaks.append((words, el))

                if ln in block_tags and not any(
                    _is_element(c) and local_name(c.tag) in block_tags for c in el
                ):
                    leaf_start[el] = words

            if el.text:
                words += count_words(el.text)
        else:
            was_skipped = skip_stack.pop()
            if not was_skipped and el in leaf_start:
                start_w = leaf_start.pop(el)
                if words > start_w:
                    leaf_blocks.append((start_w, el))
            parent_skipped = bool(skip_stack and skip_stack[-1])
            if el is not body and not parent_skipped and el.tail:
                words += count_words(el.tail)

    return words, media_count, id_map, page_breaks, hr_breaks, leaf_blocks


def _next_meaningful_sibling_or_self(el: etree._Element) -> etree._Element:
    """When breaking after an <hr>, start the new chunk at the following sibling if present."""
    nxt = el.getnext()
    while nxt is not None:
        if _is_element(nxt) and local_name(nxt.tag) not in ("script", "style"):
            return nxt
        nxt = nxt.getnext()
    return el


def _detect_fallback_splits(
    body: etree._Element,
    pb_classes: set[str],
    toc_anchors: list[tuple[str, str]],
    book_title: str = "",
) -> tuple[list[tuple[etree._Element, str]], etree._Element | None, bool]:
    """
    3-tier fallback splitter for giant (>= 30k words) headingless XHTML files:
      Tier 1: NCX/Nav #fragment anchors resolving inside this file
      Tier 2: CSS page-break-before: always / break-before: page, or spaced <hr> breaks
      Tier 3: Balanced ~18k-word paragraph chunking
    Returns (splits, trailing_trash_el, is_section_fallback).
    """
    total_words, total_media, id_map, page_breaks, hr_breaks, leaf_blocks = _index_body_elements(body, pb_classes)
    if total_words < GIANT_CHAPTER_WORD_THRESHOLD:
        return ([], None, False)

    # --- Tier 1: TOC #fragment anchors ---
    if toc_anchors:
        matched: list[tuple[int, int, etree._Element, str]] = []
        seen_els: set[etree._Element] = set()
        for frag, raw_lbl in toc_anchors:
            hit = id_map.get(frag)
            if hit is None:
                continue
            doc_idx, pos, el = hit
            if el is body or el in seen_els:
                continue
            lbl = " ".join(raw_lbl.split())
            if not lbl:
                continue
            seen_els.add(el)
            matched.append((doc_idx, pos, el, lbl))

        matched.sort(key=lambda t: t[0])

        trailing_trash_el: etree._Element | None = None
        kept: list[tuple[int, etree._Element, str]] = []
        norm_book = norm_match_text(book_title) if book_title else ""

        for _, pos, el, lbl in matched:
            if _is_trash_heading(lbl):
                if pos >= total_words * 0.65 and trailing_trash_el is None:
                    trailing_trash_el = el
                    break
                continue
            if not kept:
                kept.append((pos, el, lbl))
                continue
            prev_pos, prev_el, prev_lbl = kept[-1]
            if pos - prev_pos < 150:
                if prev_pos < 100 and norm_book and norm_match_text(prev_lbl).startswith(norm_book[:24]):
                    kept[-1] = (prev_pos, prev_el, lbl)
                elif len(prev_lbl) + len(lbl) <= 90 and norm_match_text(prev_lbl) != norm_match_text(lbl):
                    kept[-1] = (prev_pos, prev_el, f"{prev_lbl} — {lbl}")
                continue
            kept.append((pos, el, lbl))

        if len(kept) >= 2 and (total_words - kept[-1][0]) < 50:
            kept.pop()

        if len(kept) >= 2:
            return ([(el, lbl) for _, el, lbl in kept], trailing_trash_el, False)
        if trailing_trash_el is not None:
            return ([], trailing_trash_el, False)

    # --- Tier 2a: Explicit CSS page breaks ---
    if len(page_breaks) >= 2:
        min_gap = 150
        if len(page_breaks) > 80 and (total_words / len(page_breaks)) < 400:
            min_gap = 1500
        kept_pb: list[tuple[int, int, etree._Element]] = []
        for pos, media, el in page_breaks:
            if el is body:
                continue
            rem_words = total_words - pos
            rem_media = total_media - media
            if rem_words == 0 and rem_media == 0:
                continue
            in_boundary = (pos < 2500) or (rem_words < 2500)
            if not in_boundary and rem_words < 50 and rem_media == 0:
                continue
            if not kept_pb:
                if pos > 0 or media > 0 or len(page_breaks) > 2:
                    kept_pb.append((pos, media, el))
                continue
            prev_pos, prev_media, _ = kept_pb[-1]
            if in_boundary:
                # Separate tiny front/back matter pages (half-title, copyright, dedication, TOC, back ads)
                if pos > prev_pos or media > prev_media:
                    kept_pb.append((pos, media, el))
            elif pos - prev_pos >= min_gap:
                kept_pb.append((pos, media, el))
        if len(kept_pb) >= 2 or (len(kept_pb) == 1 and kept_pb[0][0] >= 3000 and total_words - kept_pb[0][0] >= 3000):
            return ([(el, f"Section {i}") for i, (_, _, el) in enumerate(kept_pb, 1)], None, True)

    # --- Tier 2b: Spaced <hr> breaks ---
    if hr_breaks:
        min_hr_gap = 2500 if len(hr_breaks) <= 25 else max(4000, total_words // 25)
        kept_hr: list[tuple[int, etree._Element]] = []
        last_pos = 0
        for pos, hr_el in hr_breaks:
            if pos - last_pos >= min_hr_gap and (total_words - pos) >= 1500:
                split_target = _next_meaningful_sibling_or_self(hr_el)
                if split_target is not body:
                    kept_hr.append((pos, split_target))
                    last_pos = pos
        if kept_hr:
            return ([(el, f"Section {i}") for i, (_, el) in enumerate(kept_hr, 1)], None, True)

    # --- Tier 3: Continuous prose balanced chunking ---
    if leaf_blocks:
        n_chunks = max(2, round(total_words / FALLBACK_CHUNK_TARGET_WORDS))
        step = total_words / n_chunks
        chosen: list[tuple[int, etree._Element]] = []
        last_chosen_pos = 0

        for k in range(1, n_chunks):
            target = k * step
            best: tuple[int, etree._Element] | None = None
            best_dist = float("inf")
            for pos, el in leaf_blocks:
                if el is body:
                    continue
                if pos - last_chosen_pos < 3000 or (total_words - pos) < 3000:
                    continue
                dist = abs(pos - target)
                if dist < best_dist:
                    best_dist = dist
                    best = (pos, el)
                elif pos > target and dist > best_dist:
                    break
            if best is not None:
                chosen.append(best)
                last_chosen_pos = best[0]

        if chosen:
            return ([(el, f"Section {i}") for i, (_, el) in enumerate(chosen, 1)], None, True)

    return ([], None, False)


def split_oversized_chapters(
    epub_dir: str,
    opf_path: str,
    cover_xhtml_rel: str | None = None,
    split_enabled: bool = True,
) -> SplitReport:
    """
    Inspect kept story spine files one at a time:
      1. Always trim glued-on trailing promo sections (e.g. 'Become a subscriber...').
      2. When `split_enabled=True`, split multi-chapter giant XHTML files into 1 file per chapter.
      3. Update OPF manifest/spine and remap internal #fragment links across the EPUB.
    """
    report = SplitReport()
    opf_tree = etree.parse(opf_path, xml_parser())
    opf_root = opf_tree.getroot()
    opf_dir = str(Path(opf_path).parent)

    manifest = _find_element(opf_root, "manifest")
    spine = _find_element(opf_root, "spine")
    if manifest is None or spine is None:
        return report

    spine_items = _get_spine_text_items(opf_root, opf_dir)
    story_items = [(ir, iid, rel) for ir, iid, rel in spine_items if rel != cover_xhtml_rel]
    if not story_items:
        return report

    broken_toc = _is_toc_sparse(opf_root, opf_dir, len(story_items))
    allow_generic = (len(story_items) <= 3) or broken_toc
    toc_anchors_by_rel = _collect_toc_fragment_anchors(opf_root, opf_dir)
    book_title = ""
    for el in opf_root.iter():
        if local_name(el.tag) == "title" and el.text and el.text.strip():
            book_title = el.text.strip()
            break

    id_to_item = {item.get("id", ""): item for item in manifest if _is_element(item)}
    existing_ids = set(id_to_item.keys())

    # Maps old_rel -> {frag_id: new_rel} for internal link remapping
    frag_remap: dict[str, dict[str, str]] = {}
    default_remap: dict[str, str] = {}

    for itemref, idref, rel in story_items:
        abs_path = os.path.join(opf_dir, rel)
        raw_bytes = Path(abs_path).read_bytes()
        root, _ = parse_xhtml_safe(raw_bytes)
        body = get_body(root)
        words = count_words(extract_clean_text(body))
        if words < 800:
            continue

        pb_classes = _collect_page_break_classes(root, opf_dir, rel) if (split_enabled and words >= GIANT_CHAPTER_WORD_THRESHOLD) else set()
        toc_anchors_for_file = toc_anchors_by_rel.get(rel, [])

        def _resolve_splits(target_body: etree._Element, target_words: int):
            splits, trash_el = _detect_split_points(target_body, allow_generic_headings=allow_generic)
            is_fb = False
            is_sec_fb = False
            if split_enabled and len(splits) < 2 and target_words >= GIANT_CHAPTER_WORD_THRESHOLD:
                fb_splits, fb_trash, is_sec_fb = _detect_fallback_splits(
                    target_body,
                    pb_classes,
                    toc_anchors_for_file,
                    book_title=book_title,
                )
                if fb_splits:
                    splits = fb_splits
                    is_fb = True
                if trash_el is None and fb_trash is not None:
                    trash_el = fb_trash
            return splits, trash_el, is_fb, is_sec_fb

        story_splits, trailing_trash_el, is_fallback, is_section_fallback = _resolve_splits(body, words)
        min_splits = 1 if is_fallback else 2

        # Case A: Trim glued-on trailing promo only (no chapter split needed)
        if trailing_trash_el is not None and (not split_enabled or len(story_splits) < min_splits):
            left_body, right_body = _slice_subtree(body, trailing_trash_el)
            if left_body is not None:
                trimmed_w = count_words(extract_clean_text(right_body)) if right_body is not None else 0
                if trimmed_w > 0 and trimmed_w < words * 0.35:
                    new_doc = _clone_doc_with_body(root, left_body, Path(rel).stem)
                    Path(abs_path).write_bytes(
                        etree.tostring(new_doc, xml_declaration=True, encoding="utf-8", pretty_print=True)
                    )
                    report.trailing_promo_trimmed_words += trimmed_w
                    report.details.append(f"Trimmed {trimmed_w}w trailing promo from {rel}")
            continue

        if not split_enabled or len(story_splits) < min_splits:
            continue

        # Case B: Split body into multiple 1-chapter XHTML files
        work_body = body
        work_words = words
        if trailing_trash_el is not None:
            left_b, right_b = _slice_subtree(work_body, trailing_trash_el)
            if left_b is not None:
                tw = count_words(extract_clean_text(right_b)) if right_b is not None else 0
                if tw < words * 0.35:
                    work_body = left_b
                    work_words = max(0, words - tw)
                    report.trailing_promo_trimmed_words += tw

        if work_body is not body:
            story_splits, _, is_fallback, is_section_fallback = _resolve_splits(work_body, work_words)
            min_splits = 1 if is_fallback else 2
            if len(story_splits) < min_splits:
                continue

        # Tag split headings with temporary markers so deep-copied left trees stay trackable
        for s_idx, (s_el, _) in enumerate(story_splits):
            s_el.set("_x4_split_idx", str(s_idx))

        chunks: list[tuple[etree._Element, str]] = []
        remaining = work_body
        for s_idx in range(len(story_splits) - 1, -1, -1):
            split_title = story_splits[s_idx][1]
            target_el = next(
                (el for el in remaining.iter() if _is_element(el) and el.get("_x4_split_idx") == str(s_idx)),
                None,
            )
            if target_el is None:
                continue
            left_b, right_b = _slice_subtree(remaining, target_el)
            if right_b is not None:
                for el in right_b.iter():
                    if _is_element(el) and "_x4_split_idx" in el.attrib:
                        del el.attrib["_x4_split_idx"]
                chunks.append((right_b, split_title))
            remaining = left_b if left_b is not None else etree.Element(work_body.tag)

        for el in remaining.iter():
            if _is_element(el) and "_x4_split_idx" in el.attrib:
                del el.attrib["_x4_split_idx"]

        chunks.reverse()

        # Check if there is meaningful leading content before the first split heading
        lead_words = count_words(extract_clean_text(remaining))
        if lead_words >= 30:
            lead_title = "Opening" if _GENERIC_STEM_RE.match(Path(rel).stem) else Path(rel).stem
            chunks.insert(0, (remaining, lead_title))
        elif lead_words > 0 and chunks:
            # Prepend tiny lead-in children onto first chapter body so nothing is lost
            first_body, first_title = chunks[0]
            for idx_c, ch in enumerate(list(remaining)):
                remaining.remove(ch)
                first_body.insert(idx_c, ch)

        if is_section_fallback and chunks:
            chunks = [(b, f"Section {idx}") for idx, (b, _) in enumerate(chunks, 1)]

        if len(chunks) < 2:
            continue

        # Write split XHTML files alongside original file
        orig_item = id_to_item[idref]
        orig_dir_rel = posixpath.dirname(rel)
        orig_stem = Path(rel).stem
        orig_ext = Path(rel).suffix or ".xhtml"
        mtype = orig_item.get("media-type", "application/xhtml+xml")

        spine_idx = list(spine).index(itemref)
        manifest_idx = list(manifest).index(orig_item)

        new_entries: list[tuple[etree._Element, etree._Element, str]] = []
        id_map_for_file: dict[str, str] = {}

        for ci, (chunk_body, chunk_title) in enumerate(chunks, 1):
            new_filename = f"{orig_stem}_ch{ci:03d}{orig_ext}"
            new_rel = posixpath.join(orig_dir_rel, new_filename) if orig_dir_rel else new_filename
            new_abs = os.path.join(opf_dir, new_rel)

            # Guarantee unique manifest id
            new_id = f"{idref}_ch{ci:03d}"
            while new_id in existing_ids:
                new_id += "_x"
            existing_ids.add(new_id)

            for el in chunk_body.iter():
                el_id = el.get("id")
                if el_id:
                    id_map_for_file[el_id] = new_rel

            doc_root = _clone_doc_with_body(root, chunk_body, chunk_title)
            Path(new_abs).write_bytes(
                etree.tostring(doc_root, xml_declaration=True, encoding="utf-8", pretty_print=True)
            )

            new_item = etree.Element(f"{{{NS_OPF}}}item")
            new_item.set("id", new_id)
            new_item.set("href", new_rel)
            new_item.set("media-type", mtype)

            new_itemref = etree.Element(f"{{{NS_OPF}}}itemref")
            new_itemref.set("idref", new_id)
            if itemref.get("linear"):
                new_itemref.set("linear", itemref.get("linear"))

            new_entries.append((new_item, new_itemref, new_rel))

        # Replace original item and itemref in OPF
        manifest.remove(orig_item)
        spine.remove(itemref)
        for offset, (ni, nir, _) in enumerate(new_entries):
            manifest.insert(manifest_idx + offset, ni)
            spine.insert(spine_idx + offset, nir)

        if os.path.exists(abs_path):
            os.unlink(abs_path)

        frag_remap[rel] = id_map_for_file
        default_remap[rel] = new_entries[0][2]
        report.files_split += 1
        report.chapters_created += len(new_entries)
        report.details.append(f"Split {rel} ({words:,}w) into {len(new_entries)} chapters")

    if report.files_split > 0:
        opf_tree.write(opf_path, xml_declaration=True, encoding="utf-8", pretty_print=True)
        # Remap internal links across all surviving spine files
        updated_spine = _get_spine_text_items(opf_root, opf_dir)
        for _, _, srel in updated_spine:
            s_abs = os.path.join(opf_dir, srel)
            s_dir = posixpath.dirname(srel)
            root, _ = parse_xhtml_safe(Path(s_abs).read_bytes())
            changed = False
            for a in root.iter():
                if local_name(a.tag) != "a":
                    continue
                href = (a.get("href") or "").strip()
                if not href or re.match(r"^(https?://|mailto:|www\.)", href, re.I):
                    continue
                if "#" in href:
                    file_part, frag = href.split("#", 1)
                else:
                    file_part, frag = href, ""

                # Check if pointing to a split file (either externally or via moved #frag)
                target_old = _normalize_rel(s_dir, file_part) if file_part else ""
                if target_old in default_remap:
                    new_target = frag_remap[target_old].get(frag, default_remap[target_old])
                    new_href = posixpath.relpath(new_target, s_dir) if s_dir else new_target
                    if frag:
                        new_href = f"{new_href}#{frag}"
                    a.set("href", new_href)
                    report.links_remapped += 1
                    changed = True
                elif not file_part and frag:
                    # Same-file fragment link inside a split chunk: check if frag belongs to another chunk!
                    for old_src, id_map in frag_remap.items():
                        if frag in id_map and srel != id_map[frag] and srel.startswith(
                            posixpath.join(posixpath.dirname(old_src), Path(old_src).stem + "_ch")
                            if posixpath.dirname(old_src) else Path(old_src).stem + "_ch"
                        ):
                            new_rel_href = posixpath.relpath(id_map[frag], s_dir) if s_dir else id_map[frag]
                            a.set("href", f"{new_rel_href}#{frag}")
                            report.links_remapped += 1
                            changed = True
                            break
            if changed:
                Path(s_abs).write_bytes(
                    etree.tostring(root, xml_declaration=True, encoding="utf-8", pretty_print=True)
                )

    return report
