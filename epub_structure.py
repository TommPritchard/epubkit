"""
EPUB structure handler for Xteink X4 EPUB Optimizer.
Handles: OPF/NCX/XHTML reference updates, SVG cover fix, TOC repair/regeneration.
"""

import html as _html_mod
import os
import posixpath
import re
from pathlib import Path
from urllib.parse import unquote, quote

from lxml import etree

from html_cleaner import recovery_parser, xml_parser

NAMESPACES = {
    'opf': 'http://www.idpf.org/2007/opf',
    'dc': 'http://purl.org/dc/elements/1.1/',
    'ncx': 'http://www.daisy.org/z3986/2005/ncx/',
    'xhtml': 'http://www.w3.org/1999/xhtml',
    'epub': 'http://www.idpf.org/2007/ops',
    'svg': 'http://www.w3.org/2000/svg',
    'xlink': 'http://www.w3.org/1999/xlink',
    'container': 'urn:oasis:names:tc:opendocument:xmlns:container',
}

NS_OPF = 'http://www.idpf.org/2007/opf'
NS_XHTML = 'http://www.w3.org/1999/xhtml'
NS_SVG = 'http://www.w3.org/2000/svg'
NS_XLINK = 'http://www.w3.org/1999/xlink'
NS_NCX = 'http://www.daisy.org/z3986/2005/ncx/'
NS_EPUB = 'http://www.idpf.org/2007/ops'


def _is_element(node):
    """Check if a node is a real element (not a comment or PI)."""
    return isinstance(node.tag, str)


def _find_element(root, local_name):
    """Find an element by local name, trying namespaced then unnamespaced."""
    # Try with OPF namespace first
    el = root.find(f'.//{{{NS_OPF}}}{local_name}')
    if el is not None:
        return el
    # Try without namespace (some EPUBs omit it)
    el = root.find(f'.//{local_name}')
    if el is not None:
        return el
    # Try wildcard namespace match
    for child in root.iter():
        tag = child.tag if isinstance(child.tag, str) else ''
        if tag.endswith('}' + local_name) or tag == local_name:
            return child
    return None


def build_rename_map(epub_dir: str, processed_images: dict) -> dict:
    """
    Build a mapping of old image paths to new paths.
    processed_images: {old_relative_path: new_filename}
    Returns: {old_path: new_path} using POSIX forward slashes.
    """
    rename_map = {}
    for old_path, new_filename in processed_images.items():
        old_norm = posixpath.normpath(str(old_path).replace('\\', '/'))
        old_dir = posixpath.dirname(old_norm)
        new_norm = posixpath.join(old_dir, new_filename) if old_dir else new_filename
        if old_norm != new_norm:
            rename_map[old_norm] = new_norm
    return rename_map


def update_opf(opf_path: str, rename_map: dict) -> None:
    """Update manifest entries in OPF when images are renamed."""
    tree = etree.parse(opf_path)
    root = tree.getroot()

    manifest = _find_element(root, 'manifest')
    if manifest is None:
        return

    norm_rename = {
        posixpath.normpath(str(k).replace('\\', '/')): posixpath.normpath(str(v).replace('\\', '/'))
        for k, v in rename_map.items()
    }

    for item in manifest:
        if not _is_element(item):
            continue
        href = item.get('href', '')
        if not href:
            continue
        decoded_href = posixpath.normpath(unquote(href).replace('\\', '/'))
        new_path = norm_rename.get(decoded_href)
        if new_path:
            item.set('href', quote(new_path, safe='/:@'))
            item.set('media-type', 'image/jpeg')

    tree.write(opf_path, xml_declaration=True, encoding='utf-8', pretty_print=True)


def update_opf_remove_fonts(opf_path: str, font_files: list[str]) -> int:
    """Remove font file entries from OPF manifest. Returns count removed."""
    tree = etree.parse(opf_path)
    root = tree.getroot()

    manifest = _find_element(root, 'manifest')
    if manifest is None:
        return 0

    removed = 0
    font_basenames = {Path(f).name for f in font_files}

    to_remove = []
    for item in manifest:
        if not _is_element(item):
            continue
        href = unquote(item.get('href', ''))
        if Path(href).name in font_basenames:
            to_remove.append(item)

    for item in to_remove:
        manifest.remove(item)
        removed += 1

    if removed > 0:
        tree.write(opf_path, xml_declaration=True, encoding='utf-8', pretty_print=True)

    return removed


def add_image_to_opf(opf_path: str, image_href: str, image_id: str) -> None:
    """Add a new image entry to the OPF manifest."""
    tree = etree.parse(opf_path)
    root = tree.getroot()

    manifest = _find_element(root, 'manifest')
    if manifest is None:
        return

    item = etree.SubElement(manifest, f'{{{NS_OPF}}}item')
    item.set('id', image_id)
    item.set('href', image_href)
    item.set('media-type', 'image/jpeg')

    tree.write(opf_path, xml_declaration=True, encoding='utf-8', pretty_print=True)


def update_xhtml_references(xhtml_path: str, rename_map: dict) -> int:
    """
    Update image references in an XHTML file.
    Returns count of updated references.
    """
    try:
        tree = etree.parse(xhtml_path)
    except etree.XMLSyntaxError:
        with open(xhtml_path, 'rb') as fh:
            parser = recovery_parser(fh.read())
        tree = etree.parse(xhtml_path, parser)

    root = tree.getroot()
    updated = 0

    # Update <img src="...">
    for img in root.iter():
        if not _is_element(img):
            continue
        tag = img.tag.split('}')[-1] if '}' in str(img.tag) else str(img.tag)

        if tag == 'img':
            src = img.get('src', '')
            new_src = _resolve_reference(src, rename_map)
            if new_src != src:
                img.set('src', new_src)
                updated += 1

        elif tag == 'image':
            # SVG <image xlink:href="...">
            href = img.get(f'{{{NS_XLINK}}}href', '') or img.get('href', '')
            new_href = _resolve_reference(href, rename_map)
            if new_href != href:
                if img.get(f'{{{NS_XLINK}}}href') is not None:
                    img.set(f'{{{NS_XLINK}}}href', new_href)
                else:
                    img.set('href', new_href)
                updated += 1

    # Update inline style background-image references
    for el in root.iter():
        if not _is_element(el):
            continue
        style = el.get('style') or ''
        if 'url(' in style:
            new_style = _update_css_urls(style, rename_map)
            if new_style != style:
                el.set('style', new_style)
                updated += 1

    if updated > 0:
        tree.write(xhtml_path, xml_declaration=True, encoding='utf-8', pretty_print=True)

    return updated


def update_css_references(css_path: str, rename_map: dict) -> int:
    """Update url() references in a CSS file. Returns count of updates."""
    with open(css_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    new_content = _update_css_urls(content, rename_map)
    updated = 1 if new_content != content else 0

    if updated:
        with open(css_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

    return updated


def _update_css_urls(css_text: str, rename_map: dict) -> str:
    """Replace url() references in CSS text."""
    def replacer(match):
        url = match.group(1).strip("'\"")
        decoded = unquote(url)
        for old, new in rename_map.items():
            old_name = Path(old).name
            new_name = Path(new).name
            decoded_name = Path(decoded).name
            if decoded_name == old_name:
                return f'url({decoded.replace(old_name, new_name)})'
        return match.group(0)

    return re.sub(r"url\(([^)]+)\)", replacer, css_text)


def _resolve_reference(ref: str, rename_map: dict) -> str:
    """Try to match a reference against the rename map."""
    decoded = unquote(ref)
    ref_name = Path(decoded).name

    for old_path, new_path in rename_map.items():
        old_name = Path(old_path).name
        if ref_name == old_name:
            return decoded.replace(old_name, Path(new_path).name)

    return ref


def fix_svg_covers(epub_dir: str, opf_path: str) -> int:
    """
    Find XHTML files that wrap cover images in SVG and replace with simple <img> tags.
    Returns count of fixed covers.
    """
    tree = etree.parse(opf_path)
    root = tree.getroot()
    fixed = 0

    # Find spine items
    spine = _find_element(root, 'spine')
    manifest = _find_element(root, 'manifest')
    if spine is None or manifest is None:
        return 0

    opf_dir = str(Path(opf_path).parent)

    # Build id->href map from manifest
    id_to_href = {}
    for item in manifest:
        if not _is_element(item):
            continue
        id_to_href[item.get('id', '')] = item.get('href', '')

    # Check first few spine items for SVG cover wrappers
    spine_items = [s for s in spine if _is_element(s)]
    for itemref in spine_items[:3]:
        idref = itemref.get('idref', '')
        href = id_to_href.get(idref, '')
        if not href:
            continue

        xhtml_path = os.path.join(opf_dir, unquote(href))
        if not os.path.exists(xhtml_path):
            continue

        try:
            doc_tree = etree.parse(xhtml_path)
        except Exception:
            continue

        doc_root = doc_tree.getroot()

        # Look for SVG elements containing a single <image>
        svgs = doc_root.findall(f'.//{{{NS_SVG}}}svg')
        if not svgs:
            svgs = doc_root.findall('.//svg')

        for svg in svgs:
            images = svg.findall(f'{{{NS_SVG}}}image')
            if not images:
                images = svg.findall('image')

            if len(images) == 1:
                image = images[0]
                img_href = (image.get(f'{{{NS_XLINK}}}href', '') or
                           image.get('href', ''))

                if not img_href:
                    continue

                # Replace SVG with simple <img>
                parent = svg.getparent()
                if parent is None:
                    continue

                # Determine namespace
                ns_prefix = ''
                if '}' in str(parent.tag):
                    ns_prefix = parent.tag.split('}')[0] + '}'

                img_el = etree.Element(f'{ns_prefix}img' if ns_prefix else 'img')
                img_el.set('src', img_href)
                img_el.set('alt', 'Cover')
                img_el.set('style', 'max-width:100%;max-height:100%;display:block;margin:auto')

                # Replace SVG with img
                idx = list(parent).index(svg)
                parent.remove(svg)
                parent.insert(idx, img_el)
                fixed += 1

        if fixed > 0:
            doc_tree.write(xhtml_path, xml_declaration=True, encoding='utf-8', pretty_print=True)

    return fixed


def fix_toc(epub_dir: str, opf_path: str, force_rebuild: bool = False) -> tuple[bool, str]:
    """
    Check and repair/regenerate synced EPUB 2 (toc.ncx) and EPUB 3 (nav.xhtml) TOCs.
    Rebuilds when TOC is missing, broken, sparse (<= 1 story entry or < 50% spine coverage),
    or when force_rebuild=True (e.g. after matter trimming / chapter splitting).
    Returns (was_fixed, description).
    """
    try:
        tree = etree.parse(opf_path, xml_parser())
    except Exception:
        tree = etree.parse(opf_path, recovery_parser())
    root = tree.getroot()
    opf_dir = str(Path(opf_path).parent)

    manifest = _find_element(root, 'manifest')
    spine = _find_element(root, 'spine')
    if manifest is None or spine is None:
        return False, "No manifest or spine found"

    # Extract book title for docTitle / nav heading
    book_title = 'Table of Contents'
    for el in root.iter():
        tag = el.tag.split('}')[-1].lower() if isinstance(el.tag, str) else ''
        if tag == 'title' and el.text and el.text.strip():
            book_title = el.text.strip()
            break

    spine_idrefs = {
        itemref.get('idref', '')
        for itemref in spine
        if _is_element(itemref) and itemref.get('idref')
    }

    ncx_href = None
    ncx_id = None
    nav_candidates = []
    for item in manifest:
        if not _is_element(item):
            continue
        iid = item.get('id', '')
        media_type = (item.get('media-type') or '').lower()
        props = (item.get('properties') or '').split()
        href = item.get('href', '')
        if media_type == 'application/x-dtbncx+xml':
            ncx_href = href
            ncx_id = iid
        is_xhtml = media_type in ('application/xhtml+xml', 'text/html')
        bname = Path(unquote(href)).name.lower() if href else ''
        if 'nav' in props or (
            is_xhtml
            and iid not in spine_idrefs
            and (iid.lower() == 'nav' or bname in ('nav.xhtml', 'nav.html'))
        ):
            nav_candidates.append(item)

    nav_item = next(
        (it for it in nav_candidates if 'nav' in (it.get('properties') or '').split()),
        nav_candidates[0] if nav_candidates else None,
    )

    id_to_href = {}
    for item in manifest:
        if _is_element(item):
            id_to_href[item.get('id', '')] = item.get('href', '')

    spine_hrefs = []
    for itemref in spine:
        if not _is_element(itemref):
            continue
        idref = itemref.get('idref', '')
        href = id_to_href.get(idref, '')
        if href:
            spine_hrefs.append((idref, href))

    if not spine_hrefs:
        return False, "Empty spine"

    # Check existing NCX completeness unless force_rebuild
    if not force_rebuild and ncx_href:
        ncx_path = os.path.join(opf_dir, unquote(ncx_href))
        if os.path.exists(ncx_path):
            try:
                ncx_tree = etree.parse(ncx_path)
                nav_map = ncx_tree.getroot().find(f'.//{{{NS_NCX}}}navMap')
                if nav_map is not None:
                    nav_points = nav_map.findall(f'{{{NS_NCX}}}navPoint')
                    # Reject 1-entry or severely sparse TOCs (< 50% of spine)
                    if len(nav_points) > 1 and len(nav_points) >= max(2, len(spine_hrefs) * 0.5):
                        broken = _check_ncx_references(nav_points, opf_dir, ncx_path)
                        if not broken:
                            return False, "TOC is valid"
                        fixed_count = _fix_ncx_references(nav_points, opf_dir, ncx_path, spine_hrefs)
                        if fixed_count == len(broken):
                            ncx_tree.write(ncx_path, xml_declaration=True, encoding='utf-8', pretty_print=True)
                            return True, f"Fixed {fixed_count} broken TOC references"
            except Exception:
                pass

    # Extract chapters and rebuild synced toc.ncx + nav.xhtml
    chapters = _extract_chapter_info(opf_dir, spine_hrefs, book_title=book_title)
    # Exclude cover & generated title card from TOC if story chapters exist
    toc_chapters = [
        ch for idx, ch in enumerate(chapters)
        if Path(unquote(ch['href'])).name not in ('x4_cover.xhtml', 'x4_title.xhtml')
        and not (idx == 0 and ch.get('is_cover'))
    ]
    if not toc_chapters:
        toc_chapters = chapters

    if ncx_href:
        ncx_path = os.path.join(opf_dir, unquote(ncx_href))
    else:
        ncx_path = os.path.join(opf_dir, 'toc.ncx')
        ncx_href = 'toc.ncx'

    _generate_ncx(ncx_path, toc_chapters, ncx_href, opf_dir=opf_dir, doc_title=book_title)

    if ncx_id is None:
        item = etree.SubElement(manifest, f'{{{NS_OPF}}}item')
        item.set('id', 'ncx')
        item.set('href', ncx_href)
        item.set('media-type', 'application/x-dtbncx+xml')
        spine.set('toc', 'ncx')

    # Generate synced EPUB 3 nav.xhtml (update existing in-place and remove stale duplicate nav files)
    nav_href = unquote(nav_item.get('href', '')) if nav_item is not None else 'nav.xhtml'
    if not nav_href:
        nav_href = 'nav.xhtml'
    nav_path = os.path.join(opf_dir, nav_href)
    _generate_nav_xhtml(nav_path, toc_chapters, nav_href, opf_dir=opf_dir, doc_title=book_title)

    if nav_item is None:
        n_item = etree.SubElement(manifest, f'{{{NS_OPF}}}item')
        n_item.set('id', 'x4-nav')
        n_item.set('href', nav_href)
        n_item.set('media-type', 'application/xhtml+xml')
        n_item.set('properties', 'nav')
    else:
        props_list = (nav_item.get('properties') or '').split()
        if 'nav' not in props_list:
            props_list.append('nav')
            nav_item.set('properties', ' '.join(props_list))

    for dup_item in nav_candidates:
        if dup_item is nav_item:
            continue
        dup_id = dup_item.get('id', '')
        if dup_id in spine_idrefs:
            continue
        dup_href = unquote(dup_item.get('href', ''))
        dup_abs = os.path.join(opf_dir, dup_href) if dup_href else ''
        if dup_item.getparent() is not None:
            dup_item.getparent().remove(dup_item)
        if dup_abs and os.path.normpath(dup_abs) != os.path.normpath(nav_path) and os.path.exists(dup_abs):
            try:
                os.unlink(dup_abs)
            except OSError:
                pass

    tree.write(opf_path, xml_declaration=True, encoding='utf-8', pretty_print=True)
    return True, f"Rebuilt dual TOC (NCX+Nav) with {len(toc_chapters)} entries"


def _check_ncx_references(nav_points, opf_dir: str, ncx_path: str) -> list:
    """Check if NCX navPoint references point to existing files."""
    broken = []
    ncx_dir = str(Path(ncx_path).parent)

    for np in nav_points:
        content = np.find(f'{{{NS_NCX}}}content')
        if content is not None:
            src = content.get('src', '')
            src_path = src.split('#')[0]  # Remove fragment
            full_path = os.path.join(ncx_dir, unquote(src_path))
            if src_path and not os.path.exists(full_path):
                broken.append(np)

    return broken


def _fix_ncx_references(nav_points, opf_dir: str, ncx_path: str, spine_hrefs: list) -> int:
    """Match broken NCX references to existing spine items by basename or position."""
    ncx_dir = str(Path(ncx_path).parent)
    spine_rel_from_ncx = []
    for _, href in spine_hrefs:
        abs_h = os.path.join(opf_dir, unquote(href))
        spine_rel_from_ncx.append(os.path.relpath(abs_h, ncx_dir).replace('\\', '/'))

    by_basename = {Path(h).name.lower(): h for h in spine_rel_from_ncx}
    fixed = 0
    for idx, np in enumerate(nav_points):
        content = np.find(f'{{{NS_NCX}}}content')
        if content is None:
            continue
        src = content.get('src', '')
        src_path, _, frag = src.partition('#')
        full_path = os.path.join(ncx_dir, unquote(src_path))
        if src_path and os.path.exists(full_path):
            continue
        bname = Path(unquote(src_path)).name.lower()
        new_rel = by_basename.get(bname)
        if not new_rel and idx < len(spine_rel_from_ncx):
            new_rel = spine_rel_from_ncx[idx]
        if new_rel:
            content.set('src', f"{new_rel}#{frag}" if frag else new_rel)
            fixed += 1
    return fixed


_GENERIC_TITLE_RE = re.compile(
    r"^(index_split_\d+|part\d+|ch\d+|chapter|chapter[\s_-]*run[\s_-]*on|continued(?:\b.*)?|"
    r"(?:unnamed|untitled|document|text|body)(?:[-_\s]*\d+)?|"
    r"(?:section|book)(?:[-_]*\d+)?|unknown|calibre|cover|title)$",
    re.I,
)


def _has_alnum(s: str) -> bool:
    return any(ch.isalnum() for ch in s)


_VALID_SECTION_STEMS = frozenset({
    "prologue", "prolog", "epilogue", "epilog", "epigraph", "intermezzo",
    "coda", "prelude", "afterword", "foreword", "preface", "introduction",
})
_SLUG_TITLE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*[-_]\d+$", re.I)


def _is_slug_title(t_clean: str, stem: str) -> bool:
    low = t_clean.strip().lower()
    if not low:
        return True
    if low == stem and low not in _VALID_SECTION_STEMS:
        return True
    if " " not in low and ("_" in low or _SLUG_TITLE_RE.match(low)):
        return True
    return False


def _extract_chapter_info(opf_dir: str, spine_hrefs: list, book_title: str = '') -> list[dict]:
    """Extract clean chapter titles from spine XHTML files."""
    chapters = []
    chapter_idx = 0
    norm_book_title = ' '.join(book_title.lower().split())

    for idx, (idref, href) in enumerate(spine_hrefs):
        clean_href = unquote(href)
        xhtml_path = os.path.join(opf_dir, clean_href)
        stem = Path(clean_href).stem.lower()
        title = ""
        is_cover = False
        from_title_tag = False
        from_fallback = False

        if os.path.exists(xhtml_path):
            try:
                raw_bytes = Path(xhtml_path).read_bytes()
                root = etree.fromstring(raw_bytes, recovery_parser(raw_bytes))

                # Detect pure cover page so TOC starts on Chapter 1
                body_el = None
                for el in root.iter():
                    if isinstance(el.tag, str) and el.tag.split('}')[-1].lower() == 'body':
                        body_el = el
                        break
                body_words = len(' '.join(''.join(body_el.itertext()).split()).split()) if body_el is not None else 0
                has_img = any(isinstance(el.tag, str) and el.tag.split('}')[-1].lower() in ('img', 'image') for el in (body_el.iter() if body_el is not None else ()))
                if idx == 0 and body_words <= 15 and (has_img or any(k in stem for k in ('cover', 'cvi', 'titlepage', 'cubierta'))):
                    is_cover = True

                # 1. Prefer Number-First header (.x4-ch-num + .x4-ch-title, or .x4-ch-label) inside first .x4-chapter-header
                num_txt = ""
                sub_txt = ""
                lbl_txt = ""
                first_hdr = None
                for el in root.iter():
                    if isinstance(el.tag, str) and 'x4-chapter-header' in (el.get('class') or '').split():
                        first_hdr = el
                        break
                if first_hdr is not None:
                    for el in first_hdr.iter():
                        if not isinstance(el.tag, str):
                            continue
                        cls = (el.get('class') or '').split()
                        if 'x4-ch-num' in cls and not num_txt:
                            num_txt = ' '.join(''.join(el.itertext()).split())
                        elif 'x4-ch-title' in cls and not sub_txt:
                            sub_txt = ' '.join(''.join(el.itertext()).split())
                        elif 'x4-ch-label' in cls and not lbl_txt:
                            lbl_txt = ' '.join(''.join(el.itertext()).split())
                if num_txt and _has_alnum(num_txt):
                    is_ch_num = num_txt.isdigit() and int(num_txt) < 500
                    prefix = f"Chapter {num_txt}" if is_ch_num else num_txt
                    title = f"{prefix}: {sub_txt}" if sub_txt else prefix
                elif lbl_txt and _has_alnum(lbl_txt):
                    title = f"{lbl_txt}: {sub_txt}" if sub_txt else lbl_txt

                # 2. Try first heading(s) before prose (joins split <h1>CHAPTER</h1><h1>1</h1>)
                if not title:
                    heading_parts = []
                    search_root = body_el if body_el is not None else root
                    for el in search_root.iter():
                        if not isinstance(el.tag, str):
                            continue
                        ltag = el.tag.split('}')[-1].lower()
                        if ltag in ('h1', 'h2', 'h3'):
                            ht = ' '.join(''.join(el.itertext()).split())
                            if ht and _has_alnum(ht):
                                heading_parts.append(ht)
                                if len(heading_parts) >= 2:
                                    break
                        elif ltag == 'p' and heading_parts:
                            if len(' '.join(''.join(el.itertext()).split())) > 15:
                                break
                    if heading_parts:
                        joined = ' — '.join(heading_parts) if len(heading_parts) > 1 and len(heading_parts[0]) > 10 else ' '.join(heading_parts)
                        title = joined[:90]

                # 2b. Detect Epigraph / Prologue / Epilogue from filename stem, container id, or epub:type
                # Runs before <head><title> so publisher-pasted book titles (e.g. swapped OPF title) never clobber Epigraph
                if not title and not is_cover and Path(clean_href).name not in ('x4_cover.xhtml', 'x4_title.xhtml'):
                    hints = [stem]
                    if body_el is not None:
                        for el in [body_el] + [c for c in list(body_el)[:3] if _is_element(c)]:
                            if el.get('id'):
                                hints.append(el.get('id', ''))
                            for ak, av in el.attrib.items():
                                if ak.endswith('type'):
                                    hints.append(av)
                    hint_blob = " ".join(hints).lower()
                    is_epi_series = bool(re.search(r'(?:^|[/_.\s-])epi[_-](?:sup|ch|sec|part)?[_-]?\d+', stem, re.I))
                    if body_words <= 300 and not is_epi_series and re.search(r'(?:^|[/_.\s-])(epi|epig|epigraph|motto)(?:[._\s-]|$)', hint_blob):
                        title = "Epigraph"
                    elif re.search(r'(?:^|[/_.\s-])(prol|prl|prologue|prolog)(?:[._\s-]|$)', hint_blob):
                        title = "Prologue"
                    elif re.search(r'(?:^|[/_.\s-])(epil|epl|epilogue|epilog)(?:[._\s-]|$)', hint_blob):
                        title = "Epilogue"

                # 3. Try <title> tag ONLY if non-generic AND distinct from the book's own title
                if not title:
                    for el in root.iter():
                        if isinstance(el.tag, str) and el.tag.split('}')[-1].lower() == 'title':
                            t = _html_mod.unescape(' '.join(''.join(el.itertext()).split()))
                            if norm_book_title:
                                t_clean = re.sub(
                                    rf"(?:,\s*|—\s*|-\s*|\|\s*){re.escape(norm_book_title)}$",
                                    "",
                                    t,
                                    flags=re.I,
                                ).strip()
                            else:
                                t_clean = t
                            t_low = re.sub(r"[^a-z0-9\s]+", "", t_clean.lower()).strip()
                            bt_low = re.sub(r"[^a-z0-9\s]+", "", norm_book_title).strip()
                            # Reject '<Author>: <Book Title>' patterns (e.g. 'Virginia Woolf: Between the Acts')
                            author_prefixed_title = False
                            if bt_low and len(bt_low) >= 5:
                                segs = re.split(r"\s*(?::|—|–|-|\||,)\s*", t_clean)
                                if len(segs) >= 2:
                                    tail_low = re.sub(r"[^a-z0-9\s]+", "", segs[-1].lower()).strip()
                                    head_seg = segs[0].strip()
                                    if tail_low == bt_low and not re.match(
                                        r"^(?:chapter|kapitel|part|teil|book|buch|prologue|epilogue|section|act|canto|\d+)\b",
                                        head_seg,
                                        re.I,
                                    ):
                                        author_prefixed_title = True
                            if (
                                t_clean
                                and _has_alnum(t_clean)
                                and not author_prefixed_title
                                and not _GENERIC_TITLE_RE.match(t_clean)
                                and not _is_slug_title(t_clean, stem)
                                and t_low != bt_low
                                and not (bt_low and (bt_low.startswith(t_low) or t_low.startswith(bt_low)) and min(len(t_low), len(bt_low)) > 4)
                            ):
                                title = t_clean[:90]
                                from_title_tag = True
                            break

                # 4. Fallback to clean sequential Chapter N
                if not title and not is_cover and Path(clean_href).name not in ('x4_cover.xhtml', 'x4_title.xhtml'):
                    chapter_idx += 1
                    title = f"Chapter {chapter_idx}"
                    from_fallback = True
            except Exception:
                if not is_cover:
                    chapter_idx += 1
                    title = f"Chapter {chapter_idx}"
                    from_fallback = True
        else:
            chapter_idx += 1
            title = f"Chapter {chapter_idx}"
            from_fallback = True

        chapters.append({
            'href': href,
            'title': title or 'Cover',
            'id': idref,
            'is_cover': is_cover,
            '_from_title_tag': from_title_tag,
            '_from_fallback': from_fallback,
        })

    # Post-pass 1: demote repeated template labels to _from_fallback:
    #   - any <title>-sourced label repeating >=3x (>=2x if containing book title)
    #   - any repeated running header containing the book title >=3x (e.g. 'Virginia Woolf: Between the Acts')
    bt_clean = re.sub(r"[^a-z0-9\s]+", "", norm_book_title).strip()
    title_counts: dict[str, int] = {}
    all_label_counts: dict[str, int] = {}
    for ch in chapters:
        k = ch['title'].strip().lower()
        all_label_counts[k] = all_label_counts.get(k, 0) + 1
        if ch.get('_from_title_tag'):
            title_counts[k] = title_counts.get(k, 0) + 1
    repeated_keys: set[str] = set()
    for k, v in title_counts.items():
        k_clean = re.sub(r"[^a-z0-9\s]+", "", k).strip()
        threshold = 2 if (bt_clean and len(bt_clean) >= 5 and bt_clean in k_clean) else 3
        if v >= threshold:
            repeated_keys.add(k)
    if bt_clean and len(bt_clean) >= 5:
        for k, v in all_label_counts.items():
            if v >= 3:
                k_clean = re.sub(r"[^a-z0-9\s]+", "", k).strip()
                if bt_clean in k_clean:
                    repeated_keys.add(k)
    if repeated_keys:
        for ch in chapters:
            if ch['title'].strip().lower() in repeated_keys and (
                ch.get('_from_title_tag') or (bt_clean and bt_clean in re.sub(r"[^a-z0-9\s]+", "", ch['title'].lower()))
            ):
                ch['_from_fallback'] = True
                ch['title'] = ''

    # Unified sequential numbering for fallback entries:
    # - Before the first explicit story chapter (e.g. Prologue, Part One, Book I, Chapter 1, I, ONE) -> 'Opening'
    # - Trailing headingless continuations after an explicit chapter (when multiple explicit chapters exist) -> '<prev> (cont.)'
    # - Otherwise -> sequential 'Chapter 1..N'
    # Consecutive duplicate fallback labels are disambiguated ('Opening 2', '(cont. 2)').
    explicit_ch_nums: set[int] = set()
    explicit_story_indices: list[int] = []
    for idx_c, ch in enumerate(chapters):
        bname = Path(unquote(ch['href'])).name
        if not ch.get('_from_fallback') and not ch.get('is_cover') and bname not in ('x4_cover.xhtml', 'x4_title.xhtml'):
            explicit_story_indices.append(idx_c)
            m_ex = re.match(r"^chapter\s+(\d+)\b", ch['title'].strip(), re.I)
            if m_ex:
                explicit_ch_nums.add(int(m_ex.group(1)))

    first_explicit_idx = explicit_story_indices[0] if explicit_story_indices else None
    has_opening_context = (
        first_explicit_idx is not None
        and (
            len(explicit_story_indices) >= 2
            or (
                first_explicit_idx <= 3
                and not chapters[first_explicit_idx]['title'].strip().lower().startswith(('epilogue', 'afterword'))
            )
        )
    )
    use_cont_for_tail = bool(explicit_ch_nums) or len(explicit_story_indices) >= 2

    seq = 0
    opening_seq = 0
    last_explicit_title = ''
    cont_count = 0
    for idx_c, ch in enumerate(chapters):
        bname = Path(unquote(ch['href'])).name
        if not ch.get('_from_fallback'):
            if not ch.get('is_cover') and bname not in ('x4_cover.xhtml', 'x4_title.xhtml'):
                last_explicit_title = ch['title']
                cont_count = 0
            continue

        is_before_first = (
            has_opening_context
            and first_explicit_idx is not None
            and idx_c < first_explicit_idx
        )
        is_before_ch1 = (
            idx_c + 1 < len(chapters)
            and not chapters[idx_c + 1].get('_from_fallback')
            and chapters[idx_c + 1]['title'].strip().lower().startswith('chapter 1')
        )

        no_story_prev = (not last_explicit_title) or (last_explicit_title.strip().lower() == 'epigraph')
        if (no_story_prev and (explicit_ch_nums or is_before_ch1)) or is_before_first:
            opening_seq += 1
            ch['title'] = 'Opening' if opening_seq == 1 else f'Opening {opening_seq}'
        elif use_cont_for_tail and last_explicit_title and last_explicit_title.strip().lower() != 'epigraph':
            cont_count += 1
            ch['title'] = f"{last_explicit_title} (cont.)" if cont_count == 1 else f"{last_explicit_title} (cont. {cont_count})"
        else:
            seq += 1
            ch['title'] = f"Chapter {seq}"

    # Renumber resetting 'Section 1..N' runs across multiple split files (e.g. Section 1, 2, 1, 2 -> 1..4)
    sec_indices = [
        (i, int(m.group(1)))
        for i, ch in enumerate(chapters)
        if not ch.get('is_cover')
        and (m := re.match(r'^Section\s+(\d+)$', ch['title'].strip(), re.I)) is not None
    ]
    if len(sec_indices) >= 2 and any(
        sec_indices[j][1] <= sec_indices[j - 1][1]
        for j in range(1, len(sec_indices))
    ):
        for new_n, (ci, _) in enumerate(sec_indices, 1):
            chapters[ci]['title'] = f"Section {new_n}"

    for ch in chapters:
        ch.pop('_from_title_tag', None)
        ch.pop('_from_fallback', None)

    return chapters


def _generate_ncx(
    ncx_path: str,
    chapters: list[dict],
    ncx_href: str,
    opf_dir: str = '',
    doc_title: str = 'Book',
) -> None:
    """Generate a synced EPUB 2 NCX file relative to ncx_path."""
    ncx_dir = os.path.dirname(os.path.abspath(ncx_path))
    base_opf_dir = os.path.abspath(opf_dir) if opf_dir else ncx_dir

    ncx = etree.Element(f'{{{NS_NCX}}}ncx', nsmap={None: NS_NCX})
    ncx.set('version', '2005-1')

    head = etree.SubElement(ncx, f'{{{NS_NCX}}}head')
    meta = etree.SubElement(head, f'{{{NS_NCX}}}meta')
    meta.set('name', 'dtb:depth')
    meta.set('content', '1')

    doc_title_el = etree.SubElement(ncx, f'{{{NS_NCX}}}docTitle')
    doc_text = etree.SubElement(doc_title_el, f'{{{NS_NCX}}}text')
    doc_text.text = doc_title or (chapters[0]['title'] if chapters else 'Book')

    nav_map = etree.SubElement(ncx, f'{{{NS_NCX}}}navMap')

    for i, chapter in enumerate(chapters):
        nav_point = etree.SubElement(nav_map, f'{{{NS_NCX}}}navPoint')
        nav_point.set('id', f'navPoint-{i + 1}')
        nav_point.set('playOrder', str(i + 1))

        nav_label = etree.SubElement(nav_point, f'{{{NS_NCX}}}navLabel')
        text = etree.SubElement(nav_label, f'{{{NS_NCX}}}text')
        text.text = chapter['title']

        abs_target = os.path.abspath(os.path.join(base_opf_dir, unquote(chapter['href'])))
        rel_src = os.path.relpath(abs_target, ncx_dir).replace('\\', '/')
        content = etree.SubElement(nav_point, f'{{{NS_NCX}}}content')
        content.set('src', quote(rel_src, safe='/:@#'))

    Path(ncx_path).parent.mkdir(parents=True, exist_ok=True)
    tree = etree.ElementTree(ncx)
    tree.write(ncx_path, xml_declaration=True, encoding='utf-8', pretty_print=True)


def _generate_nav_xhtml(
    nav_path: str,
    chapters: list[dict],
    nav_href: str,
    opf_dir: str = '',
    doc_title: str = 'Contents',
) -> None:
    """Generate a synced EPUB 3 nav.xhtml document relative to nav_path."""
    nav_dir = os.path.dirname(os.path.abspath(nav_path))
    base_opf_dir = os.path.abspath(opf_dir) if opf_dir else nav_dir

    html = etree.Element(f'{{{NS_XHTML}}}html', nsmap={None: NS_XHTML, 'epub': NS_EPUB})
    head = etree.SubElement(html, f'{{{NS_XHTML}}}head')
    title_el = etree.SubElement(head, f'{{{NS_XHTML}}}title')
    title_el.text = doc_title or 'Contents'

    body = etree.SubElement(html, f'{{{NS_XHTML}}}body')
    nav = etree.SubElement(body, f'{{{NS_XHTML}}}nav')
    nav.set(f'{{{NS_EPUB}}}type', 'toc')
    nav.set('id', 'toc')

    h1 = etree.SubElement(nav, f'{{{NS_XHTML}}}h1')
    h1.text = 'Contents'
    ol = etree.SubElement(nav, f'{{{NS_XHTML}}}ol')

    for ch in chapters:
        li = etree.SubElement(ol, f'{{{NS_XHTML}}}li')
        a = etree.SubElement(li, f'{{{NS_XHTML}}}a')
        abs_target = os.path.abspath(os.path.join(base_opf_dir, unquote(ch['href'])))
        rel_href = os.path.relpath(abs_target, nav_dir).replace('\\', '/')
        a.set('href', quote(rel_href, safe='/:@#'))
        a.text = ch['title']

    Path(nav_path).parent.mkdir(parents=True, exist_ok=True)
    etree.ElementTree(html).write(nav_path, xml_declaration=True, encoding='utf-8', pretty_print=True)


def find_content_files(epub_dir: str, opf_path: str) -> dict:
    """
    Find all content files referenced in the OPF manifest.
    Returns dict with keys: xhtml, css, images, fonts, ncx, other
    """
    tree = etree.parse(opf_path)
    root = tree.getroot()
    opf_dir = str(Path(opf_path).parent)

    files = {
        'xhtml': [],
        'css': [],
        'images': [],
        'fonts': [],
        'ncx': [],
        'other': [],
    }

    manifest = _find_element(root, 'manifest')
    if manifest is None:
        return files

    for item in manifest:
        if not _is_element(item):
            continue
        href = unquote(item.get('href', ''))
        media_type = item.get('media-type', '').lower()
        full_path = os.path.join(opf_dir, href)

        if media_type in ('application/xhtml+xml', 'text/html'):
            files['xhtml'].append(full_path)
        elif media_type == 'text/css':
            files['css'].append(full_path)
        elif media_type.startswith('image/'):
            files['images'].append(full_path)
        elif media_type == 'application/x-dtbncx+xml':
            files['ncx'].append(full_path)
        elif media_type in ('application/font-woff', 'application/font-woff2',
                           'font/woff', 'font/woff2', 'font/ttf', 'font/otf',
                           'application/vnd.ms-opentype', 'application/x-font-ttf'):
            files['fonts'].append(full_path)
        else:
            ext = Path(href).suffix.lower()
            if ext in ('.ttf', '.otf', '.woff', '.woff2'):
                files['fonts'].append(full_path)
            else:
                files['other'].append(full_path)

    return files


_EXT_MEDIA_TYPES = {
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.gif': 'image/gif',
    '.svg': 'image/svg+xml',
    '.webp': 'image/webp',
    '.css': 'text/css',
    '.xhtml': 'application/xhtml+xml',
    '.html': 'application/xhtml+xml',
    '.htm': 'application/xhtml+xml',
    '.ncx': 'application/x-dtbncx+xml',
    '.ttf': 'font/ttf',
    '.otf': 'font/otf',
    '.woff': 'font/woff',
    '.woff2': 'font/woff2',
}


def _resolve_manifest_abs(opf_dir: str, href: str) -> tuple[str, str]:
    """Return (decoded_abs, raw_abs) resolved from opf_dir."""
    dec = unquote(href or '').replace('\\', '/').strip()
    raw = (href or '').replace('\\', '/').strip()
    abs_dec = os.path.abspath(os.path.join(opf_dir, dec)) if dec else ''
    abs_raw = os.path.abspath(os.path.join(opf_dir, raw)) if raw else ''
    return abs_dec, abs_raw


def _extract_css_urls(css_text: str) -> list[str]:
    urls: list[str] = []
    for m in re.findall(r"url\(([^)]+)\)", css_text):
        u = m.strip("'\" ").split('#', 1)[0].strip()
        if u and not u.lower().startswith(('http://', 'https://', 'data:', 'mailto:')):
            urls.append(u)
    return urls


def sanitize_package_manifest_and_files(work_dir: str, opf_path: str) -> dict:
    """
    Ensure strict EPUB 3 OPF package hygiene before final ZIP packaging:
      1) Merge duplicate manifest hrefs (preserving properties & remapping spine/cover refs)
      2) Disambiguate any remaining duplicate manifest IDs
      3) Remove manifest items missing on disk (and strip broken stylesheet <link>s)
      4) Auto-register any unmanifested local asset actually referenced by surviving XHTML/CSS,
         and delete unmanifested orphan/store clutter outside META-INF & mimetype.
    """
    stats = {
        'dups_removed': 0,
        'missing_pruned': 0,
        'unmanifested_removed': 0,
        'unmanifested_registered': 0,
    }
    try:
        tree = etree.parse(opf_path, xml_parser())
    except Exception:
        tree = etree.parse(opf_path, recovery_parser())
    root = tree.getroot()
    manifest = _find_element(root, 'manifest')
    spine = _find_element(root, 'spine')
    if manifest is None:
        return stats

    work_abs = os.path.abspath(work_dir)
    opf_abs = os.path.abspath(opf_path)
    opf_dir = os.path.dirname(opf_abs)

    # 1. Merge duplicate manifest hrefs
    seen_href: dict[str, etree._Element] = {}
    id_remap: dict[str, str] = {}
    for item in list(manifest):
        if not _is_element(item):
            continue
        href = item.get('href', '')
        iid = item.get('id', '')
        abs_dec, _ = _resolve_manifest_abs(opf_dir, href)
        key = os.path.normcase(abs_dec) if abs_dec else posixpath.normpath(unquote(href).replace('\\', '/'))
        if key in seen_href:
            winner = seen_href[key]
            w_props = (winner.get('properties') or '').split()
            d_props = (item.get('properties') or '').split()
            for p in d_props:
                if p not in w_props:
                    w_props.append(p)
            if w_props:
                winner.set('properties', ' '.join(w_props))
            if not winner.get('media-type') and item.get('media-type'):
                winner.set('media-type', item.get('media-type', ''))
            win_id = winner.get('id', '')
            if iid and win_id and iid != win_id:
                id_remap[iid] = win_id
            manifest.remove(item)
            stats['dups_removed'] += 1
        else:
            seen_href[key] = item

    # Remap <spine toc>, <spine itemref>, and deduplicate <meta name="cover">
    if spine is not None:
        toc_id = spine.get('toc', '')
        if toc_id in id_remap:
            spine.set('toc', id_remap[toc_id])
        for ir in spine:
            if _is_element(ir):
                idref = ir.get('idref', '')
                if idref in id_remap:
                    ir.set('idref', id_remap[idref])

    seen_cover_meta = False
    for meta in list(root.iter()):
        if not _is_element(meta):
            continue
        ltag = meta.tag.split('}')[-1].lower()
        if ltag == 'meta' and meta.get('name') == 'cover':
            cval = meta.get('content', '')
            if cval in id_remap:
                meta.set('content', id_remap[cval])
                cval = meta.get('content', '')
            if seen_cover_meta or not cval:
                if meta.getparent() is not None:
                    meta.getparent().remove(meta)
            else:
                seen_cover_meta = True

    # 2. Disambiguate any remaining duplicate manifest IDs (different hrefs sharing an id)
    kept_items = [it for it in manifest if _is_element(it)]
    used_ids: set[str] = set()
    for it in kept_items:
        iid = it.get('id', '')
        if iid:
            if iid not in used_ids:
                used_ids.add(iid)
    seen_ids_pass: set[str] = set()
    for it in kept_items:
        iid = it.get('id', '')
        if not iid or iid in seen_ids_pass:
            base = iid or 'item'
            n = 2
            cand = f'{base}_{n}'
            while cand in used_ids:
                n += 1
                cand = f'{base}_{n}'
            it.set('id', cand)
            used_ids.add(cand)
            seen_ids_pass.add(cand)
            stats['dups_removed'] += 1
        else:
            seen_ids_pass.add(iid)

    # 3. Prune manifest items whose files do not exist on disk
    pruned_ids: set[str] = set()
    missing_css_abs: set[str] = set()
    for it in list(manifest):
        if not _is_element(it):
            continue
        href = it.get('href', '')
        mtype = (it.get('media-type') or '').lower()
        abs_dec, abs_raw = _resolve_manifest_abs(opf_dir, href)
        exists = bool(
            (abs_dec and os.path.isfile(abs_dec))
            or (abs_raw and os.path.isfile(abs_raw))
        )
        if not exists:
            iid = it.get('id', '')
            if iid:
                pruned_ids.add(iid)
            if abs_dec and (mtype == 'text/css' or href.lower().endswith('.css')):
                missing_css_abs.add(os.path.normcase(abs_dec))
            manifest.remove(it)
            stats['missing_pruned'] += 1

    if spine is not None and pruned_ids:
        for ir in list(spine):
            if _is_element(ir) and ir.get('idref', '') in pruned_ids:
                spine.remove(ir)

    # 4. Inspect surviving XHTML & CSS to strip dead stylesheet <link>s & collect local references
    allowed_abs: set[str] = {
        os.path.normcase(opf_abs),
        os.path.normcase(os.path.join(work_abs, 'mimetype')),
    }
    referenced_abs: set[str] = set()
    surviving_xhtml: list[str] = []
    surviving_css: list[str] = []

    for it in manifest:
        if not _is_element(it):
            continue
        href = it.get('href', '')
        mtype = (it.get('media-type') or '').lower()
        abs_dec, abs_raw = _resolve_manifest_abs(opf_dir, href)
        chosen = abs_dec if (abs_dec and os.path.isfile(abs_dec)) else abs_raw
        if chosen and os.path.isfile(chosen):
            allowed_abs.add(os.path.normcase(chosen))
            if abs_raw and os.path.isfile(abs_raw):
                allowed_abs.add(os.path.normcase(abs_raw))
            if mtype in ('application/xhtml+xml', 'text/html'):
                surviving_xhtml.append(chosen)
            elif mtype == 'text/css':
                surviving_css.append(chosen)

    for xpath in surviving_xhtml:
        xdir = os.path.dirname(xpath)
        try:
            raw_bytes = Path(xpath).read_bytes()
            try:
                xroot = etree.fromstring(raw_bytes, xml_parser())
            except Exception:
                xroot = etree.fromstring(raw_bytes, recovery_parser(raw_bytes))
        except Exception:
            continue
        modified = False
        for el in list(xroot.iter()):
            if not _is_element(el):
                continue
            ltag = el.tag.split('}')[-1].lower()
            if ltag == 'link' and 'stylesheet' in (el.get('rel') or '').lower():
                lhref = (el.get('href') or '').strip()
                if lhref and not lhref.lower().startswith(('http://', 'https://', 'data:')):
                    ldec, lraw = _resolve_manifest_abs(xdir, lhref)
                    if (
                        (ldec and os.path.normcase(ldec) in missing_css_abs)
                        or (not os.path.isfile(ldec) and not os.path.isfile(lraw))
                    ):
                        if el.getparent() is not None:
                            el.getparent().remove(el)
                            modified = True
                            continue
                    for cand in (ldec, lraw):
                        if cand and os.path.isfile(cand):
                            referenced_abs.add(os.path.normcase(cand))
            elif ltag in ('img', 'script', 'source', 'audio', 'video', 'track'):
                for attr in ('src', 'poster'):
                    val = (el.get(attr) or '').strip()
                    if val and not val.lower().startswith(('http://', 'https://', 'data:', '#')):
                        ldec, lraw = _resolve_manifest_abs(xdir, val.split('#', 1)[0])
                        for cand in (ldec, lraw):
                            if cand and os.path.isfile(cand):
                                referenced_abs.add(os.path.normcase(cand))
            elif ltag in ('image', 'use'):
                val = (el.get(f'{{{NS_XLINK}}}href') or el.get('href') or '').strip()
                if val and not val.lower().startswith(('http://', 'https://', 'data:', '#')):
                    ldec, lraw = _resolve_manifest_abs(xdir, val.split('#', 1)[0])
                    for cand in (ldec, lraw):
                        if cand and os.path.isfile(cand):
                            referenced_abs.add(os.path.normcase(cand))
            style = el.get('style') or ''
            if 'url(' in style:
                for u in _extract_css_urls(style):
                    ldec, lraw = _resolve_manifest_abs(xdir, u)
                    for cand in (ldec, lraw):
                        if cand and os.path.isfile(cand):
                            referenced_abs.add(os.path.normcase(cand))
        if modified:
            Path(xpath).write_bytes(
                etree.tostring(xroot, xml_declaration=True, encoding='utf-8', pretty_print=True)
            )

    for cpath in surviving_css:
        cdir = os.path.dirname(cpath)
        try:
            ctext = Path(cpath).read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        for u in _extract_css_urls(ctext):
            ldec, lraw = _resolve_manifest_abs(cdir, u)
            for cand in (ldec, lraw):
                if cand and os.path.isfile(cand):
                    referenced_abs.add(os.path.normcase(cand))

    # 5. Walk work_dir (skip META-INF): manifest referenced files, delete unmanifested clutter
    asset_seq = 1
    for dirpath, dirnames, filenames in os.walk(work_abs):
        dirnames[:] = [d for d in dirnames if d.upper() != 'META-INF']
        for fname in filenames:
            abs_f = os.path.abspath(os.path.join(dirpath, fname))
            nc_f = os.path.normcase(abs_f)
            if nc_f in allowed_abs:
                continue
            if nc_f in referenced_abs:
                rel_from_opf = os.path.relpath(abs_f, opf_dir).replace('\\', '/')
                ext = Path(fname).suffix.lower()
                mtype = _EXT_MEDIA_TYPES.get(ext, 'application/octet-stream')
                new_id = f'x4-asset-{asset_seq}'
                while new_id in used_ids:
                    asset_seq += 1
                    new_id = f'x4-asset-{asset_seq}'
                used_ids.add(new_id)
                asset_seq += 1
                item = etree.SubElement(manifest, f'{{{NS_OPF}}}item')
                item.set('id', new_id)
                item.set('href', quote(rel_from_opf, safe='/:@'))
                item.set('media-type', mtype)
                allowed_abs.add(nc_f)
                stats['unmanifested_registered'] += 1
            else:
                try:
                    os.unlink(abs_f)
                    stats['unmanifested_removed'] += 1
                except OSError:
                    pass

    # Clean empty directories (excluding work_abs and META-INF)
    meta_abs = os.path.normcase(os.path.join(work_abs, 'META-INF'))
    for dirpath, _, _ in os.walk(work_abs, topdown=False):
        nc_d = os.path.normcase(os.path.abspath(dirpath))
        if nc_d == os.path.normcase(work_abs) or nc_d == meta_abs or nc_d.startswith(meta_abs + os.sep):
            continue
        try:
            if not os.listdir(dirpath):
                os.rmdir(dirpath)
        except OSError:
            pass

    tree.write(opf_path, xml_declaration=True, encoding='utf-8', pretty_print=True)
    return stats
