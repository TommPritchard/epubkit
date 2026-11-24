"""
Comprehensive pytest suite for the Xteink X4 Story-First Edition Engine (Phase C).
Covers: matter trimming (story vs nonfiction), Phase B regression edge cases,
chapter auto-splitting, Bookerly House Style CSS, scene-break & ornament pruning
before image processing, dual NCX+Nav TOC rebuild, smart typography, source
warnings, and SSE report metrics.
"""
import io
import json
import os
import zipfile
from pathlib import Path

import pytest
from lxml import etree
from PIL import Image

from epub_processor import process_epub, ProcessingOptions
from text_cleaner import clean_text_content, TextCleanOptions


def _png_bytes(w=100, h=40, color=(20, 20, 20)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


def _write_test_epub(path: Path, files: dict[str, bytes | str], opf_xml: str, opf_path: str = "OEBPS/content.opf") -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            f'<rootfiles><rootfile full-path="{opf_path}" media-type="application/oebps-package+xml"/></rootfiles></container>',
        )
        zf.writestr(opf_path, opf_xml)
        for rel, data in files.items():
            full = f"OEBPS/{rel}" if not rel.startswith("OEBPS/") else rel
            zf.writestr(full, data)


def _xhtml(body_inner: str, title: str = "Chapter", lang: str = "", extra_head: str = "") -> str:
    lang_attr = f' lang="{lang}" xml:lang="{lang}"' if lang else ""
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        f'<html xmlns="http://www.w3.org/1999/xhtml"{lang_attr}>'
        f'<head><title>{title}</title>{extra_head}</head>'
        f'<body>{body_inner}</body></html>'
    )


def test_story_trimming_cascade_and_edge_protections(tmp_path):
    """
    Tests:
      - Praise, Copyright, Dedication, TOC, Notes, Sequel Preview cascade -> TRASH
      - Epigraph, Prologue, Epilogue -> KEEP
      - Common-prefix filename ('Ill-Go-On-ch1.xhtml') NOT falsely trashed by 'ill'
      - Calibre 'index_split_005.html' NOT falsely trashed by 'index'
      - Short dialogue chapter ('chapter-1-2.xhtml') NOT trashed as front praise
      - Dead footnote <sup><a> deleted; dead TOC <h2><a> unwrapped cleanly
      - Ornament & trashed-page images pruned BEFORE Pillow image processing
    """
    src = tmp_path / "messy.epub"
    dst = tmp_path / "out.epub"

    prose = " ".join(["word"] * 120)
    long_prose = " ".join(["story"] * 600)

    opf = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="uid">urn:uuid:1234</dc:identifier>
    <dc:title>Ill Go On</dc:title>
    <dc:creator>Test Author</dc:creator>
    <dc:language>en-US</dc:language>
  </metadata>
  <manifest>
    <item id="cov" href="cover.jpg" media-type="image/jpeg" properties="cover-image"/>
    <item id="orn" href="ornament.png" media-type="image/png"/>
    <item id="adimg" href="publisher_logo.png" media-type="image/png"/>
    <item id="style" href="pub.css" media-type="text/css"/>
    <item id="praise" href="Ill-Go-On-praise.xhtml" media-type="application/xhtml+xml"/>
    <item id="ded" href="Ill-Go-On-ded.xhtml" media-type="application/xhtml+xml"/>
    <item id="tocp" href="Ill-Go-On-toc.xhtml" media-type="application/xhtml+xml"/>
    <item id="epi" href="Ill-Go-On-epi.xhtml" media-type="application/xhtml+xml"/>
    <item id="prol" href="Ill-Go-On-prol.xhtml" media-type="application/xhtml+xml"/>
    <item id="ch1" href="Ill-Go-On-c01.xhtml" media-type="application/xhtml+xml"/>
    <item id="ch2" href="chapter-1-2.xhtml" media-type="application/xhtml+xml"/>
    <item id="ch3" href="index_split_005.html" media-type="application/xhtml+xml"/>
    <item id="epil" href="Ill-Go-On-epl.xhtml" media-type="application/xhtml+xml"/>
    <item id="prev" href="preview.xhtml" media-type="application/xhtml+xml"/>
    <item id="prech1" href="prech1.xhtml" media-type="application/xhtml+xml"/>
    <item id="notes" href="notes.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="praise"/><itemref idref="ded"/><itemref idref="tocp"/>
    <itemref idref="epi"/><itemref idref="prol"/>
    <itemref idref="ch1"/><itemref idref="ch2"/><itemref idref="ch3"/><itemref idref="epil"/>
    <itemref idref="prev"/><itemref idref="prech1"/><itemref idref="notes"/>
  </spine>
</package>"""

    files = {
        "cover.jpg": _png_bytes(480, 800, (40, 40, 40)),
        "ornament.png": _png_bytes(120, 24, (0, 0, 0)),
        "publisher_logo.png": _png_bytes(200, 80, (10, 10, 10)),
        "pub.css": "body { font-family: Arial; color: #555; margin-left: 40px; hyphens: auto; }",
        "Ill-Go-On-praise.xhtml": _xhtml('<p>“Masterpiece!” — New York Times</p><img src="publisher_logo.png" alt="Logo"/>', "Praise"),
        "Ill-Go-On-ded.xhtml": _xhtml("<p>For my mother.</p>", "Dedication"),
        "Ill-Go-On-toc.xhtml": _xhtml("<p>Contents</p>", "Contents"),
        "Ill-Go-On-epi.xhtml": _xhtml("<p>All happy families are alike.</p>", "Epigraph"),
        "Ill-Go-On-prol.xhtml": _xhtml(f"<h1>Prologue</h1><p>{prose}</p>", "Prologue"),
        "Ill-Go-On-c01.xhtml": _xhtml(
            f'<h1><a href="Ill-Go-On-toc.xhtml">Chapter 1: Dawn</a></h1>'
            f'<p><span class="dropcap" style="float:left">T</span>he sea rose<sup><a href="notes.xhtml#n1">[1]</a></sup> quickly--unlike before...</p>'
            f'<p><img src="ornament.png" alt=""/></p>'
            f'<p>Visit <a href="https://penguin.com">penguin.com</a> or <a href="https://oceanofpdf.com">OceanofPDF.com</a>. {long_prose}</p>',
            "Chapter 1",
        ),
        "chapter-1-2.xhtml": _xhtml('<p>“Wait,” she said. “Don’t go yet.”</p><p>“I have to,” he replied.</p>', "II"),
        "index_split_005.html": _xhtml(f"<h1>Chapter 3</h1><p>{prose}</p>", "Chapter 3"),
        "Ill-Go-On-epl.xhtml": _xhtml(f"<h1>Epilogue</h1><p>{prose}</p>", "Epilogue"),
        "preview.xhtml": _xhtml("<p>Preview of Book Two</p>", "Preview"),
        "prech1.xhtml": _xhtml(f"<h1>ONE</h1><p>{long_prose}</p>", "ONE"),
        "notes.xhtml": _xhtml('<p id="n1">1. A scholarly note.</p>', "Notes"),
    }
    _write_test_epub(src, files, opf)

    opts = ProcessingOptions(preset="story")
    report = process_epub(str(src), str(dst), opts)
    assert report.success, report.error

    # Only cover.jpg survived ornament + logo pruning before Pillow ran!
    assert report.images_total == 1
    assert report.orphan_images_removed >= 2
    assert report.dead_footnotes_removed == 1
    assert report.scene_breaks_normalized >= 1
    assert report.files_trimmed >= 6

    with zipfile.ZipFile(dst) as zf:
        names = set(zf.namelist())
        assert "OEBPS/ornament.png" not in names
        assert "OEBPS/publisher_logo.png" not in names
        assert "OEBPS/prech1.xhtml" not in names
        assert "OEBPS/notes.xhtml" not in names
        assert "OEBPS/Ill-Go-On-epi.xhtml" in names
        assert "OEBPS/Ill-Go-On-prol.xhtml" in names
        assert "OEBPS/Ill-Go-On-c01.xhtml" in names
        assert "OEBPS/chapter-1-2.xhtml" in names
        assert "OEBPS/index_split_005.html" in names
        assert "OEBPS/Ill-Go-On-epl.xhtml" in names
        assert "OEBPS/x4_cover.xhtml" in names
        assert "OEBPS/x4_title.xhtml" in names
        assert "OEBPS/x4_house_style.css" in names
        assert "OEBPS/toc.ncx" in names
        assert "OEBPS/nav.xhtml" in names

        ch1_txt = zf.read("OEBPS/Ill-Go-On-c01.xhtml").decode("utf-8")
        # Dead footnote deleted, TOC heading link unwrapped (text kept!), prose URL unwrapped, OceanofPDF gone
        assert "[1]" not in ch1_txt
        assert "Dawn" in ch1_txt and 'Ill-Go-On-toc.xhtml' not in ch1_txt
        assert "penguin.com" in ch1_txt and "https://penguin.com" not in ch1_txt
        assert "OceanofPDF" not in ch1_txt
        # Smart typography preserved + upgraded, HR scene break injected
        assert "—" in ch1_txt and "…" in ch1_txt
        assert 'x4-scene-break' in ch1_txt
        assert 'lang="en-US"' in ch1_txt

        # Verify House Style CSS settings
        hs_css = zf.read("OEBPS/x4_house_style.css").decode("utf-8")
        assert "hyphens: none" in hs_css
        assert "text-indent: 1.25em" in hs_css
        assert "line-height: 1.38" in hs_css
        assert "width: 35%" in hs_css


def test_nonfiction_mode_keeps_intro_and_notes(tmp_path):
    src = tmp_path / "nf.epub"
    dst = tmp_path / "nf_out.epub"
    prose = " ".join(["history"] * 300)

    opf = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="uid">urn:uuid:nf</dc:identifier>
    <dc:title>History Book</dc:title><dc:creator>Historian</dc:creator><dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="intro" href="intro.xhtml" media-type="application/xhtml+xml"/>
    <item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/>
    <item id="notes" href="notes.xhtml" media-type="application/xhtml+xml"/>
    <item id="cop" href="copyright.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine><itemref idref="cop"/><itemref idref="intro"/><itemref idref="c1"/><itemref idref="notes"/></spine>
</package>"""
    files = {
        "copyright.xhtml": _xhtml("<p>Copyright 2024 ISBN 123456</p>"),
        "intro.xhtml": _xhtml(f"<h1>Introduction</h1><p>{prose}</p>"),
        "c1.xhtml": _xhtml(f'<h1>Chapter 1</h1><p>Fact<sup><a href="notes.xhtml#n1">1</a></sup>. {prose}</p>'),
        "notes.xhtml": _xhtml('<p id="n1">1. Primary source.</p>'),
    }
    _write_test_epub(src, files, opf)

    report = process_epub(str(src), str(dst), ProcessingOptions(preset="nonfiction"))
    assert report.success, report.error
    assert report.dead_footnotes_removed == 0

    with zipfile.ZipFile(dst) as zf:
        names = set(zf.namelist())
        assert "OEBPS/copyright.xhtml" not in names
        assert "OEBPS/intro.xhtml" in names
        assert "OEBPS/notes.xhtml" in names
        c1_txt = zf.read("OEBPS/c1.xhtml").decode("utf-8")
        assert "notes.xhtml#n1" in c1_txt


def test_chapter_splitter_and_trailing_promo_and_dual_toc(tmp_path):
    """
    Tests splitting a single giant XHTML with 3 chapters + glued-on subscriber promo,
    consecutive heading merging (CHAPTER + 1), link remapping, and dual NCX+Nav rebuild.
    """
    src = tmp_path / "giant.epub"
    dst = tmp_path / "split_out.epub"
    body_words = " ".join(["narrative"] * 300)

    opf = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="uid">urn:uuid:giant</dc:identifier>
    <dc:title>Big Novel</dc:title><dc:creator>Writer</dc:creator><dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
    <item id="all" href="book.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine toc="ncx"><itemref idref="all"/></spine>
</package>"""

    # Deliberately sparse 1-entry NCX (like Midwinter Break!)
    ncx = """<?xml version="1.0" encoding="utf-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <head><meta name="dtb:depth" content="1"/></head>
  <docTitle><text>Big Novel</text></docTitle>
  <navMap><navPoint id="n1" playOrder="1"><navLabel><text>Chapters</text></navLabel><content src="book.xhtml"/></navPoint></navMap>
</ncx>"""

    giant_body = (
        f'<div class="wrap">'
        f'<h1>CHAPTER</h1><h1>1</h1><p id="p1">See <a href="#p2">later</a>. {body_words}</p>'
        f'<h1>Chapter 2: The Crossing</h1><p id="p2">{body_words}</p>'
        f'<h1>Chapter 3: Harbor</h1><p>{body_words}</p>'
        f'<h2>Become a subscriber, or give a subscription to a friend</h2><p>Subscribe now for deals! {body_words}</p>'
        f'</div>'
    )
    _write_test_epub(src, {"toc.ncx": ncx, "book.xhtml": _xhtml(giant_body)}, opf)

    report = process_epub(str(src), str(dst), ProcessingOptions())
    assert report.success, report.error
    assert report.chapters_split == 3
    assert report.chapter_count == 3
    assert report.words_trimmed >= 300  # glued subscriber promo trimmed!

    with zipfile.ZipFile(dst) as zf:
        names = set(zf.namelist())
        assert "OEBPS/book_ch001.xhtml" in names
        assert "OEBPS/book_ch002.xhtml" in names
        assert "OEBPS/book_ch003.xhtml" in names
        assert "OEBPS/book.xhtml" not in names

        ch1_txt = zf.read("OEBPS/book_ch001.xhtml").decode("utf-8")
        assert 'href="book_ch002.xhtml#p2"' in ch1_txt  # fragment remapped!
        ch3_txt = zf.read("OEBPS/book_ch003.xhtml").decode("utf-8")
        assert "Become a subscriber" not in ch3_txt

        # Verify synced NCX + Nav both list all 3 split chapters
        ncx_out = zf.read("OEBPS/toc.ncx").decode("utf-8")
        nav_out = zf.read("OEBPS/nav.xhtml").decode("utf-8")
        for target in ("book_ch001.xhtml", "book_ch002.xhtml", "book_ch003.xhtml"):
            assert target in ncx_out
            assert target in nav_out


def test_fake_double_spacing_spacers_are_stripped_not_hr(tmp_path):
    """Books with >25% <p>&nbsp;</p> spacers (like The Palm House) strip them instead of inserting <hr>."""
    src = tmp_path / "palm.epub"
    dst = tmp_path / "palm_out.epub"
    paras = "".join(f"<p>Line {i} of the story goes here.</p><p>\u00a0</p>" for i in range(25))
    opf = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="uid">urn:uuid:palm</dc:identifier>
    <dc:title>Palm</dc:title><dc:creator>Writer</dc:creator><dc:language>en</dc:language>
  </metadata>
  <manifest><item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/></manifest>
  <spine><itemref idref="c1"/></spine>
</package>"""
    _write_test_epub(src, {"c1.xhtml": _xhtml(paras)}, opf)

    report = process_epub(str(src), str(dst), ProcessingOptions())
    assert report.success, report.error
    assert report.scene_breaks_normalized == 0
    with zipfile.ZipFile(dst) as zf:
        c1 = zf.read("OEBPS/c1.xhtml").decode("utf-8")
        assert "<hr" not in c1


def test_corrupt_source_warning_and_sse_report_fields(monkeypatch, tmp_path):
    """Corrupted placeholder chapters still process cleanly AND emit report.warnings + word counts in SSE."""
    pytest.importorskip("httpx", reason="TestClient needs httpx")
    from fastapi.testclient import TestClient
    import app as app_module

    monkeypatch.setattr(app_module, "CONVERTER_API_KEY", "secret-key")
    monkeypatch.setattr(app_module, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(app_module, "OUTPUT_DIR", tmp_path / "outputs")
    app_module.tasks.clear()

    src = tmp_path / "Beachy - Gay Berlin.epub"
    prose = " ".join(["chapter"] * 400)
    opf = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="uid">urn:uuid:corrupt</dc:identifier>
    <dc:title>Gay Berlin</dc:title><dc:creator>Robert Beachy</dc:creator><dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/>
    <item id="c2" href="c2.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine><itemref idref="c1"/><itemref idref="c2"/></spine>
</package>"""
    _write_test_epub(
        src,
        {
            "c1.xhtml": _xhtml(f"<h1>Chapter 1</h1><p>{prose}</p>"),
            "c2.xhtml": _xhtml("<p>sorry, something went wrong loading your content. please try again.</p>"),
        },
        opf,
    )

    with TestClient(app_module.app) as client:
        with open(src, "rb") as fh:
            up = client.post(
                "/upload",
                headers={"X-API-Key": "secret-key"},
                files={"files": ("Beachy - Gay Berlin.epub", fh, "application/epub+zip")},
            )
        assert up.status_code == 200
        task_id = up.json()["files"][0]["task_id"]

        sse = client.get(f"/process/{task_id}", headers={"X-API-Key": "secret-key"})
        assert sse.status_code == 200
        events = [
            json.loads(line[6:])
            for line in sse.text.splitlines()
            if line.startswith("data: ")
        ]
        final = events[-1]
        assert final["status"] == "done"
        rep = final["report"]
        assert rep["word_count_story"] > 350
        assert rep["chapter_count"] == 2
        assert rep["reading_minutes"] >= 1
        assert any("placeholder error text" in w for w in rep["warnings"])


def test_text_cleaner_preserves_smart_typography_and_strips_junk():
    raw = '<html xmlns="http://www.w3.org/1999/xhtml"><body><p>“Hello\u00ad\u200bworld”--wait...</p></body></html>'.encode("utf-8")
    cleaned, rep = clean_text_content(raw, TextCleanOptions(preserve_smart_typography=True))
    txt = cleaned.decode("utf-8")
    assert "“Helloworld”—wait…" in txt
    assert "\u00ad" not in txt
    assert "\u200b" not in txt
    assert rep.total_fixes >= 4


def test_toc_fallback_ignores_book_title_and_avoids_duplicate_calibre_cover(tmp_path):
    """
    Regression test for Phase D findings:
      - Midwinter Break: <title>Book Title</title> must not clobber headings or repeat as TOC labels,
        and consecutive heading tags (e.g. <h1>Chapter</h1><h2>1</h2>) join into 'Chapter 1'.
      - Fingersmith: existing calibre-titlepage.xhtml cover wrapper avoids inserting a second x4_cover.xhtml,
        and spine[0] cover wrapper is excluded from rebuilt TOC.
    """
    src = tmp_path / "calibre_cover.epub"
    dst = tmp_path / "calibre_cover_out.epub"
    prose = " ".join(["story"] * 300)

    opf = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/opf" version="2.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="uid">urn:uuid:calibre-cov</dc:identifier>
    <dc:title>My Great Novel</dc:title><dc:creator>Author Name</dc:creator><dc:language>en</dc:language>
    <meta name="cover" content="covimg"/>
  </metadata>
  <manifest>
    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
    <item id="covimg" href="cover.jpg" media-type="image/jpeg"/>
    <item id="tp" href="calibre-titlepage.xhtml" media-type="application/xhtml+xml"/>
    <item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
    <item id="ch2" href="ch2.xhtml" media-type="application/xhtml+xml"/>
    <item id="ch3" href="ch3.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine toc="ncx">
    <itemref idref="tp"/><itemref idref="ch1"/><itemref idref="ch2"/><itemref idref="ch3"/>
  </spine>
</package>"""

    # Intentionally sparse 1-entry NCX pointing to ch1
    ncx = """<?xml version="1.0" encoding="utf-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <head><meta name="dtb:depth" content="1"/></head>
  <docTitle><text>My Great Novel</text></docTitle>
  <navMap><navPoint id="n1" playOrder="1"><navLabel><text>Chapters</text></navLabel><content src="ch1.xhtml"/></navPoint></navMap>
</ncx>"""

    files = {
        "toc.ncx": ncx,
        "cover.jpg": _png_bytes(400, 600),
        "calibre-titlepage.xhtml": _xhtml('<div class="cover"><img src="cover.jpg" alt="Cover"/></div>', title="Cover"),
        "ch1.xhtml": _xhtml(f"<h1>Chapter</h1><h2>1</h2><p>{prose}</p>", title="My Great Novel"),
        "ch2.xhtml": _xhtml(f"<h2>Chapter 2: The Harbor</h2><p>{prose}</p>", title="My Great Novel"),
        # Chapter whose <title> matches book_title and has no heading -> must NOT use 'My Great Novel' in TOC
        "ch3.xhtml": _xhtml(f"<p>{prose}</p>", title="My Great Novel"),
    }
    _write_test_epub(src, files, opf)

    report = process_epub(str(src), str(dst), ProcessingOptions())
    assert report.success, report.error

    with zipfile.ZipFile(dst) as zf:
        names = set(zf.namelist())
        assert "OEBPS/x4_cover.xhtml" not in names
        assert "OEBPS/calibre-titlepage.xhtml" in names
        assert "OEBPS/x4_title.xhtml" in names

        ncx_out = zf.read("OEBPS/toc.ncx").decode("utf-8")
        nav_out = zf.read("OEBPS/nav.xhtml").decode("utf-8")
        assert "calibre-titlepage.xhtml" not in ncx_out
        assert "calibre-titlepage.xhtml" not in nav_out
        assert "Chapter 1" in ncx_out and "Chapter 1" in nav_out
        assert "Chapter 2: The Harbor" in ncx_out and "Chapter 2: The Harbor" in nav_out
        # ch3 had only <title>My Great Novel</title> so navLabel must fall back cleanly, not repeat book title
        ncx_root = etree.fromstring(ncx_out.encode("utf-8"))
        labels = [el.text for el in ncx_root.xpath("//*[local-name()='navLabel']/*[local-name()='text']")]
        assert "My Great Novel" not in labels


def test_uniform_number_first_chapter_headers(tmp_path):
    """
    All publisher heading variants must collapse into identical Number-First Classic
    .x4-chapter-header markup (Secret History split+line-img, Wager num+h2 subtitle, Fingersmith bare h3).
    """
    src = tmp_path / "headers.epub"
    dst = tmp_path / "headers_out.epub"
    prose = " ".join(["narrative"] * 300)

    opf = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/opf" version="3.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="uid">urn:uuid:hdrs</dc:identifier>
    <dc:title>Header Test</dc:title><dc:creator>Author</dc:creator><dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="rule" href="rule.png" media-type="image/png"/>
    <item id="lineimg" href="line.png" media-type="image/png"/>
    <item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/>
    <item id="c2" href="c2.xhtml" media-type="application/xhtml+xml"/>
    <item id="c3" href="c3.xhtml" media-type="application/xhtml+xml"/>
    <item id="c4" href="c4.xhtml" media-type="application/xhtml+xml"/>
    <item id="c5" href="c5.xhtml" media-type="application/xhtml+xml"/>
    <item id="c6" href="c6.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="c1"/><itemref idref="c2"/><itemref idref="c3"/><itemref idref="c4"/><itemref idref="c5"/><itemref idref="c6"/>
  </spine>
</package>"""

    files = {
        "rule.png": _png_bytes(300, 1),  # 1px divider line inside Secret History-style heading
        "line.png": _png_bytes(109, 1),  # 1px divider element between Into Thin Air headings
        "c1.xhtml": _xhtml(
            f'<h1 class="chapter" id="c04"><span id="p1"/>CHAPTER</h1>'
            f'<h1 class="chapter2"><img src="rule.png" alt=""/><br/>4</h1><p>{prose}</p>'
        ),
        "c2.xhtml": _xhtml(
            f'<div class="page_top_padding"><h1>Chapter 19</h1>'
            f'<h2 class="para-ct">The Haunting</h2><p>{prose}</p></div>'
        ),
        "c3.xhtml": _xhtml(f'<h3 class="calibre3">2</h3><p>{prose}</p>'),
        # 'LIVID' must NOT be misparsed as Roman numeral
        "c4.xhtml": _xhtml(f'<h2>LIVID</h2><p>{prose}</p>'),
        # Short opening prose sentence with class="chap" (Midwinter Break) + bare <p>151</p> (Kitchen) must NOT become headings!
        "c5.xhtml": _xhtml(f'<p class="chap">The Rijksmuseum was far enough away to travel by tram.</p><p class="calibre_">151</p><p>{prose}</p>'),
        # Into Thin Air style: ONE + ornament-only <h1> + 2 subtitle lines, plus Wellness <1> test via 2nd chapter? Let's put <1> + title in c6!
        "c6.xhtml": _xhtml(
            f'<h1>&lt;1&gt;</h1><h1>The EdgeRank Algorithm</h1>'
            f'<h1 class="chapter3">TWO</h1><h1 class="chapter4"><img src="line.png" alt=""/></h1>'
            f'<h1 class="chapter5">EVEREST SUMMIT</h1><h1 class="chapter6">MAY 10</h1><p>{prose}</p>'
        ),
    }
    _write_test_epub(src, files, opf)

    report = process_epub(str(src), str(dst), ProcessingOptions())
    assert report.success, report.error

    with zipfile.ZipFile(dst) as zf:
        assert "OEBPS/rule.png" not in zf.namelist()  # 1px line pruned before Pillow!

        c1 = zf.read("OEBPS/c1.xhtml").decode("utf-8")
        assert '<div class="x4-chapter-header" id="c04">' in c1
        assert '<div class="x4-ch-num">4</div>' in c1
        assert "CHAPTER" not in c1
        assert "rule.png" not in c1
        assert 'x4-first' in c1

        c2 = zf.read("OEBPS/c2.xhtml").decode("utf-8")
        assert '<div class="x4-ch-num">19</div>' in c2
        assert '<div class="x4-ch-title">The Haunting</div>' in c2
        assert "<h2" not in c2

        c3 = zf.read("OEBPS/c3.xhtml").decode("utf-8")
        assert '<div class="x4-ch-num">2</div>' in c3
        assert "<h3" not in c3

        c4 = zf.read("OEBPS/c4.xhtml").decode("utf-8")
        assert '<div class="x4-ch-label">LIVID</div>' in c4

        c5 = zf.read("OEBPS/c5.xhtml").decode("utf-8")
        assert "x4-chapter-header" not in c5
        assert "The Rijksmuseum was far enough away to travel by tram." in c5

        c6 = zf.read("OEBPS/c6.xhtml").decode("utf-8")
        assert '<div class="x4-ch-num">1</div>' in c6
        assert '<div class="x4-ch-title">The EdgeRank Algorithm</div>' in c6
        assert '<div class="x4-ch-num">TWO</div>' in c6
        assert 'EVEREST SUMMIT' in c6 and 'MAY 10' in c6
        assert 'line.png' not in c6
        assert "OEBPS/line.png" not in zf.namelist()

        nav = zf.read("OEBPS/nav.xhtml").decode("utf-8")
        assert "Chapter 4" in nav
        assert "Chapter 19: The Haunting" in nav
        assert "Chapter 2" in nav
        assert "LIVID" in nav
        assert "The Rijksmuseum" not in nav


def test_inline_footnotes_section_and_duplicate_cover_itemref_kept(tmp_path):
    """
    Regression test for Batch 2 findings:
      - Book 295 (Jon Ronson): Chapter with inline <section epub:type="footnotes"> must NEVER be trashed.
      - Book 211 (Barbara Gordon): Duplicate <itemref idref="cov"/> in spine must not leave a dangling itemref.
    """
    src = tmp_path / "inline_fn.epub"
    dst = tmp_path / "inline_fn_out.epub"
    prose = " ".join(["story"] * 500)

    opf = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/opf" version="3.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="uid">urn:uuid:fn295</dc:identifier>
    <dc:title>Inline Footnotes</dc:title><dc:creator>Author</dc:creator><dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="covimg" href="cover.jpg" media-type="image/jpeg" properties="cover-image"/>
    <item id="cov" href="cover.xhtml" media-type="application/xhtml+xml"/>
    <item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="cov"/><itemref idref="cov"/><itemref idref="c1"/>
  </spine>
</package>"""

    ch1 = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">'
        '<head><title>Ch 1</title></head><body>'
        f'<section epub:type="chapter"><h1>Chapter 1</h1><p>{prose}</p></section>'
        '<section epub:type="footnotes"><p>1. Inline note.</p></section>'
        '</body></html>'
    )
    files = {
        "cover.jpg": _png_bytes(400, 600),
        "cover.xhtml": _xhtml('<img src="cover.jpg" alt="Cover"/>', title="Cover"),
        "c1.xhtml": ch1,
    }
    _write_test_epub(src, files, opf)

    report = process_epub(str(src), str(dst), ProcessingOptions())
    assert report.success, report.error
    assert report.word_count_story >= 500

    with zipfile.ZipFile(dst) as zf:
        assert "OEBPS/c1.xhtml" in zf.namelist()
        opf_out = etree.fromstring(zf.read("OEBPS/content.opf"))
        manifest_ids = {el.get("id") for el in opf_out.iter() if el.tag.endswith("item")}
        spine_ids = [el.get("idref") for el in opf_out.iter() if el.tag.endswith("itemref")]
        assert all(sid in manifest_ids for sid in spine_ids)
        assert spine_ids.count("cov") == 1


def test_visual_consistency_at_page_double_hr_header_hr_slug_toc_and_p_num(tmp_path):
    """
    Regression test for all 7 systemic visual/CSS/TOC findings from the 595-book deep audit:
      1. @page rules stripped from external CSS AND inline <style> tags (Kairos, Cold Enough for Snow)
      2. <hr> + <div class="transition">—</div> collapses into a single <hr> (All That Glitters)
      3. Leading/trailing/pre-header <hr> inside <header> stripped (Luster)
      4. First-block <p class="calibre_"><b>Nine</b></p> promoted to .x4-chapter-header (Notes on a Scandal)
      5. Filename-slug <title>au_cold_enough_for_snow_text-6</title> ignored in TOC (Cold Enough for Snow)
      6. Headingless fallback Chapter 1 before explicit Chapter 1 relabeled 'Opening' (All That Glitters)
      7. Blank story chapter after ornament/empty-heading stripping pruned from spine; Init-cap-copy flattened
    """
    src = tmp_path / "visual_fixes.epub"
    dst = tmp_path / "visual_fixes_out.epub"
    prose = " ".join(["story"] * 250)

    opf = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/opf" version="3.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="uid">urn:uuid:vis7</dc:identifier>
    <dc:title>Visual Consistency</dc:title><dc:creator>Author</dc:creator><dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="css" href="idGeneratedStyles.css" media-type="text/css"/>
    <item id="open" href="au_cold_enough_for_snow_text-1.xhtml" media-type="application/xhtml+xml"/>
    <item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/>
    <item id="c2" href="c2.xhtml" media-type="application/xhtml+xml"/>
    <item id="c9" href="c9.xhtml" media-type="application/xhtml+xml"/>
    <item id="blank" href="blank.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="open"/><itemref idref="c1"/><itemref idref="c2"/><itemref idref="c9"/><itemref idref="blank"/>
  </spine>
</package>"""

    files = {
        "idGeneratedStyles.css": "@page { margin: 0 0 0 0; } body { margin: 0; } .p { font-size: 10pt; }",
        "au_cold_enough_for_snow_text-1.xhtml": _xhtml(
            f'<p><span class="Init-cap-copy">O</span>n that Friday in July. {prose}</p>',
            title="au_cold_enough_for_snow_text-1",
            extra_head='<link rel="stylesheet" href="idGeneratedStyles.css"/><style>@page { margin: 0; }</style>',
        ),
        "c1.xhtml": _xhtml(
            f'<header><hr/><h1>1</h1></header>'
            f'<p>{prose}</p><p>\u00a0</p><div class="transition">—</div><p>{prose}</p><hr/>',
            title="Chapter 1",
        ),
        "c2.xhtml": _xhtml(f'<h1>2</h1><p>{prose}</p>', title="Chapter 2"),
        "c9.xhtml": _xhtml(f'<p class="calibre_"><span class="bold">Nine</span></p><p>{prose}</p>', title="Visual Consistency"),
        "blank.xhtml": _xhtml('<h2>\u00a0</h2>', title="Blank"),
    }
    _write_test_epub(src, files, opf)

    report = process_epub(str(src), str(dst), ProcessingOptions())
    assert report.success, report.error

    with zipfile.ZipFile(dst) as zf:
        names = set(zf.namelist())
        assert "OEBPS/blank.xhtml" not in names  # pruned blank chapter!

        css_out = zf.read("OEBPS/idGeneratedStyles.css").decode("utf-8")
        assert "@page" not in css_out.lower()

        open_out = zf.read("OEBPS/au_cold_enough_for_snow_text-1.xhtml").decode("utf-8")
        assert "@page" not in open_out.lower()
        assert "x4-dropcap-flat" in open_out

        c1_out = zf.read("OEBPS/c1.xhtml").decode("utf-8")
        assert c1_out.count("x4-scene-break") == 1  # no pre-header <hr>, no double <hr>+—, no trailing <hr>!
        assert '<div class="transition">' not in c1_out

        c9_out = zf.read("OEBPS/c9.xhtml").decode("utf-8")
        assert '<div class="x4-ch-num">Nine</div>' in c9_out

        ncx_root = etree.fromstring(zf.read("OEBPS/toc.ncx"))
        labels = [el.text for el in ncx_root.xpath("//*[local-name()='navLabel']/*[local-name()='text']")]
        assert "au_cold_enough_for_snow_text-1" not in labels
        assert labels[:4] == ["Opening", "Chapter 1", "Chapter 2", "Nine"]


def test_classic_preset_preserves_original_layout_and_matter(monkeypatch, tmp_path):
    """
    Classic preset runs pre-Story-First X4 optimization:
      - Preserves front matter (Copyright), does not split multi-chapter files
      - Skips x4_house_style.css, x4_title.xhtml, and scene-break normalization
      - Does not mutate caller's ProcessingOptions instance
      - FastAPI SSE accepts ?preset=classic (200) and rejects invalid presets (400)
    """
    pytest.importorskip("httpx", reason="TestClient needs httpx")
    from fastapi.testclient import TestClient
    import app as app_module

    src = tmp_path / "classic.epub"
    dst = tmp_path / "classic_out.epub"
    prose = " ".join(["story"] * 900)

    opf = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/opf" version="2.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="uid">urn:uuid:classic</dc:identifier>
    <dc:title>Classic Mode Book</dc:title><dc:creator>Author</dc:creator><dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
    <item id="copy" href="copyright.xhtml" media-type="application/xhtml+xml"/>
    <item id="body" href="chapters.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine toc="ncx">
    <itemref idref="copy"/><itemref idref="body"/>
  </spine>
</package>"""
    ncx = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">'
        '<head><meta name="dtb:uid" content="urn:uuid:classic"/></head>'
        '<docTitle><text>Classic Mode Book</text></docTitle>'
        '<navMap>'
        '<navPoint id="n1" playOrder="1"><navLabel><text>Copyright</text></navLabel><content src="copyright.xhtml"/></navPoint>'
        '<navPoint id="n2" playOrder="2"><navLabel><text>Chapters</text></navLabel><content src="chapters.xhtml"/></navPoint>'
        '</navMap></ncx>'
    )
    files = {
        "toc.ncx": ncx,
        "copyright.xhtml": _xhtml("<p>Copyright 2026 All rights reserved.</p>", title="Copyright"),
        "chapters.xhtml": _xhtml(
            f'<h1>Chapter 1</h1><p>{prose}</p><p>* * *</p><h1>Chapter 2</h1><p>{prose}</p>',
            title="Chapters",
        ),
    }
    _write_test_epub(src, files, opf)

    opts = ProcessingOptions(preset="classic")
    report = process_epub(str(src), str(dst), opts)
    assert report.success, report.error
    # Caller's ProcessingOptions must remain unmutated
    assert opts.standardize_layout is True
    assert opts.trim_matter is True

    assert report.files_trimmed == 0
    assert report.chapters_split == 0
    assert report.scene_breaks_normalized == 0
    assert report.word_count_story > 1500
    assert report.chapter_count == 2

    with zipfile.ZipFile(dst) as zf:
        names = set(zf.namelist())
        assert "OEBPS/copyright.xhtml" in names
        assert "OEBPS/chapters.xhtml" in names
        assert "OEBPS/x4_house_style.css" not in names
        assert "OEBPS/x4_title.xhtml" not in names
        ch_out = zf.read("OEBPS/chapters.xhtml").decode("utf-8")
        assert "* * *" in ch_out
        assert "x4-scene-break" not in ch_out

    monkeypatch.setattr(app_module, "CONVERTER_API_KEY", "secret-key")
    monkeypatch.setattr(app_module, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(app_module, "OUTPUT_DIR", tmp_path / "outputs")
    app_module.tasks.clear()

    with TestClient(app_module.app) as client:
        with open(src, "rb") as fh:
            up = client.post(
                "/upload",
                headers={"X-API-Key": "secret-key"},
                files={"files": ("Author - Classic Mode Book.epub", fh, "application/epub+zip")},
            )
        assert up.status_code == 200
        task_id = up.json()["files"][0]["task_id"]

        bad = client.get(f"/process/{task_id}?preset=bogus", headers={"X-API-Key": "secret-key"})
        assert bad.status_code == 400

        ok = client.get(f"/process/{task_id}?preset=classic", headers={"X-API-Key": "secret-key"})
        assert ok.status_code == 200
        events = [json.loads(line[6:]) for line in ok.text.splitlines() if line.startswith("data: ")]
        assert events[-1]["status"] == "done"
        assert events[-1]["report"]["chapters_split"] == 0
        assert events[-1]["report"]["files_trimmed"] == 0


def test_giant_headingless_fallback_splitter_and_table_overflow_css(tmp_path):
    """
    Tests 3-tier fallback splitter for >=30k-word headingless XHTML files + table/word-wrap CSS:
      - Tier 1: NCX #fragment anchors inside one giant headingless XHTML file
      - Tier 2: CSS page-break-before: always inside a giant headingless XHTML file
      - Tier 3: Continuous prose balanced chunking inside a giant headingless XHTML file
      - Table fixed width attrs / nowrap styles scrubbed + responsive CSS & break-word rules injected
    """
    src = tmp_path / "giant_tiers.epub"
    dst = tmp_path / "giant_tiers_out.epub"

    para_1k = "<p>" + " ".join(["narrative"] * 1000) + "</p>"
    table_html = (
        '<table width="800" style="width: 800px; min-width: 600px;">'
        '<tr><td width="400" style="width: 400px; white-space: nowrap;">Column One</td><td>Column Two</td></tr>'
        '</table>'
    )

    # Tier 1: 31k words, 0 headings, NCX #fragment anchors + table
    tier1_body = (
        f'<div id="anc_intro">{"".join([para_1k] * 10)}{table_html}</div>'
        f'<div id="anc_one">{"".join([para_1k] * 10)}</div>'
        f'<div id="anc_two">{"".join([para_1k] * 11)}</div>'
    )
    # Tier 2: 31k words, 0 headings, CSS page-break-before: always (no NCX fragments pointing here)
    tier2_body = (
        f'{"".join([para_1k] * 10)}'
        f'<p style="page-break-before: always;">Middle break begins here {"word " * 995}</p>'
        f'{"".join([para_1k] * 9)}'
        f'<p style="page-break-before: always;">Final break begins here {"word " * 995}</p>'
        f'{"".join([para_1k] * 10)}'
    )
    # Tier 3: 35k words, pure continuous paragraphs
    tier3_body = "".join([para_1k] * 35)

    opf = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/opf" version="2.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="uid">urn:uuid:giant</dc:identifier>
    <dc:title>Giant Headingless Anthology</dc:title><dc:creator>Author</dc:creator><dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
    <item id="f1" href="part0000.xhtml" media-type="application/xhtml+xml"/>
    <item id="f2" href="part0001.xhtml" media-type="application/xhtml+xml"/>
    <item id="f3" href="part0002.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine toc="ncx">
    <itemref idref="f1"/><itemref idref="f2"/><itemref idref="f3"/>
  </spine>
</package>"""
    ncx = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">'
        '<head><meta name="dtb:uid" content="urn:uuid:giant"/></head>'
        '<docTitle><text>Giant Headingless Anthology</text></docTitle>'
        '<navMap>'
        '<navPoint id="p1" playOrder="1"><navLabel><text>Introductory Essay</text></navLabel><content src="part0000.xhtml#anc_intro"/></navPoint>'
        '<navPoint id="p2" playOrder="2"><navLabel><text>First Chronicle</text></navLabel><content src="part0000.xhtml#anc_one"/></navPoint>'
        '<navPoint id="p3" playOrder="3"><navLabel><text>Second Chronicle</text></navLabel><content src="part0000.xhtml#anc_two"/></navPoint>'
        '<navPoint id="p4" playOrder="4"><navLabel><text>Part Two</text></navLabel><content src="part0001.xhtml"/></navPoint>'
        '<navPoint id="p5" playOrder="5"><navLabel><text>Part Three</text></navLabel><content src="part0002.xhtml"/></navPoint>'
        '</navMap></ncx>'
    )
    files = {
        "toc.ncx": ncx,
        "part0000.xhtml": _xhtml(tier1_body, title="part0000"),
        "part0001.xhtml": _xhtml(tier2_body, title="part0001"),
        "part0002.xhtml": _xhtml(tier3_body, title="part0002"),
    }
    _write_test_epub(src, files, opf)

    report = process_epub(str(src), str(dst), ProcessingOptions())
    assert report.success, report.error
    # All 3 giant files split -> 3 + 3 + 2 = 8 story chapters total (+1 title page = 9)
    assert report.chapters_split >= 8
    assert report.word_count_story >= 96000  # zero words lost during splitting!

    with zipfile.ZipFile(dst) as zf:
        names = set(zf.namelist())
        assert "OEBPS/part0000_ch001.xhtml" in names
        assert "OEBPS/part0000_ch003.xhtml" in names
        assert "OEBPS/part0001_ch003.xhtml" in names
        assert "OEBPS/part0002_ch002.xhtml" in names

        # Table fixed widths / nowrap scrubbed + house style CSS includes wrap + responsive table rules
        t1_out = zf.read("OEBPS/part0000_ch001.xhtml").decode("utf-8")
        assert 'width="800"' not in t1_out
        assert 'width="400"' not in t1_out
        assert "white-space: nowrap" not in t1_out.lower()
        assert "800px" not in t1_out

        hs_css = zf.read("OEBPS/x4_house_style.css").decode("utf-8")
        assert "overflow-wrap: break-word" in hs_css
        assert "word-wrap: break-word" in hs_css
        assert "table-layout: auto" in hs_css
        assert "td p, th p" in hs_css

        ncx_root = etree.fromstring(zf.read("OEBPS/toc.ncx"))
        labels = [el.text for el in ncx_root.xpath("//*[local-name()='navLabel']/*[local-name()='text']")]
        assert "Introductory Essay" in labels
        assert "First Chronicle" in labels
        assert "Second Chronicle" in labels
        assert "Section 1" in labels
        assert "part0002" not in labels


def _jpg_bytes(w=400, h=600, color=(40, 80, 120)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="JPEG")
    return buf.getvalue()


def test_cover_page_hash_image_preserved_and_windows_case_safe(tmp_path):
    """
    1) A cover XHTML pointing to a hash-named image (e.g. ../Images/image-HASH.jpg) without OPF
       cover metadata registers it in OPF and keeps the real cover even when ornament pruning runs.
    2) Generating a missing cover preserves exact existing 'Images/' directory casing on disk.
    """
    # Part 1: Hash-named cover on coverpage.xhtml + repeated ornament image in chapter1
    src = tmp_path / "hash_cover.epub"
    dst = tmp_path / "hash_cover_out.epub"
    prose = " ".join(["narrative"] * 600)

    opf = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/opf" version="2.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="uid">urn:uuid:hashcov</dc:identifier>
    <dc:title>Hash Cover Book</dc:title><dc:creator>Author</dc:creator><dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="covpg" href="Text/coverpage.xhtml" media-type="application/xhtml+xml"/>
    <item id="c1" href="Text/chapter1.xhtml" media-type="application/xhtml+xml"/>
    <item id="img_hash" href="Images/image-ANFQKZZB.jpg" media-type="image/jpeg"/>
    <item id="img_orn" href="Images/orn.png" media-type="image/png"/>
  </manifest>
  <spine>
    <itemref idref="covpg"/><itemref idref="c1"/>
  </spine>
</package>"""
    files = {
        "Text/coverpage.xhtml": _xhtml('<div><img src="../Images/image-ANFQKZZB.jpg" alt="Hash Cover Book"/></div>', title="Cover"),
        "Text/chapter1.xhtml": _xhtml(
            f'<h1>Chapter 1</h1><p>{prose}</p><p><img src="../Images/orn.png"/></p>'
            f'<p>{prose}</p><p><img src="../Images/orn.png"/></p>',
            title="Chapter 1",
        ),
        "Images/image-ANFQKZZB.jpg": _jpg_bytes(480, 720, (120, 60, 40)),
        "Images/orn.png": _png_bytes(40, 12, (0, 0, 0)),
    }
    _write_test_epub(src, files, opf)

    report = process_epub(str(src), str(dst), ProcessingOptions())
    assert report.success, report.error
    assert report.cover_generated is False

    with zipfile.ZipFile(dst) as zf:
        names = set(zf.namelist())
        assert "OEBPS/Images/image-ANFQKZZB.jpg" in names
        assert not any("cover_generated" in n for n in names)
        cov_out = zf.read("OEBPS/Text/coverpage.xhtml").decode("utf-8")
        assert "../Images/image-ANFQKZZB.jpg" in cov_out
        opf_out = zf.read("OEBPS/content.opf").decode("utf-8")
        assert 'cover-image' in opf_out
        assert 'name="cover"' in opf_out

    # Part 2: Truly coverless book with capital OEBPS/Images/ folder preserves directory casing
    src2 = tmp_path / "nocover_upper.epub"
    dst2 = tmp_path / "nocover_upper_out.epub"
    opf2 = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/opf" version="2.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="uid">urn:uuid:nocov</dc:identifier>
    <dc:title>No Cover Upper</dc:title><dc:creator>Author</dc:creator><dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="c1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
    <item id="ill" href="Images/scene.jpg" media-type="image/jpeg"/>
  </manifest>
  <spine><itemref idref="c1"/></spine>
</package>"""
    _write_test_epub(
        src2,
        {
            "chapter1.xhtml": _xhtml(f'<h1>Chapter 1</h1><p>{prose}</p><p><img src="Images/scene.jpg"/></p>'),
            "Images/scene.jpg": _jpg_bytes(320, 240, (60, 90, 120)),
        },
        opf2,
    )
    rep2 = process_epub(str(src2), str(dst2), ProcessingOptions())
    assert rep2.success, rep2.error
    assert rep2.cover_generated is True
    with zipfile.ZipFile(dst2) as zf:
        opf2_out = zf.read("OEBPS/content.opf").decode("utf-8")
        assert 'href="Images/cover_generated.jpg"' in opf2_out
        assert 'href="images/cover_generated.jpg"' not in opf2_out


def test_critical_foreword_contributor_bio_epigraph_and_centered_images(tmp_path):
    """
    Tests:
      - Critical Foreword (_frw_), contributor bio (_fm2_), frontispiece (_fm3_), and untitled back bio trimmed in story mode
      - Critical Foreword kept in nonfiction mode
      - Headingless epigraph (_epi_ / id="epi") kept & labeled 'Epigraph' in dual TOC (not 'Chapter 1')
      - Surviving inline <p><img/></p> and <div class="fig"><img/><p>cap</p></div> receive .x4-img-wrap centering
    """
    src = tmp_path / "passing_like.epub"
    dst_story = tmp_path / "passing_story.epub"
    dst_nf = tmp_path / "passing_nf.epub"
    prose = " ".join(["story"] * 500)
    fw_prose = " ".join(["criticism"] * 2800)

    opf = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/opf" version="2.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="uid">urn:uuid:passing</dc:identifier>
    <dc:title>Passing Sample</dc:title><dc:creator>Nella Larsen</dc:creator><dc:language>en</dc:language>
    <meta name="cover" content="covimg"/>
  </metadata>
  <manifest>
    <item id="covimg" href="images/cov.jpg" media-type="image/jpeg"/>
    <item id="pic1" href="images/frontispiece.jpg" media-type="image/jpeg"/>
    <item id="pic2" href="images/scene.jpg" media-type="image/jpeg"/>
    <item id="cvi" href="book_cvi_r1.xhtml" media-type="application/xhtml+xml"/>
    <item id="tocp" href="book_toc_r1.xhtml" media-type="application/xhtml+xml"/>
    <item id="ded" href="book_ded_r1.xhtml" media-type="application/xhtml+xml"/>
    <item id="epi" href="book_epi_r1.xhtml" media-type="application/xhtml+xml"/>
    <item id="frw" href="book_frw_r1.xhtml" media-type="application/xhtml+xml"/>
    <item id="fm2" href="book_fm2_r1.xhtml" media-type="application/xhtml+xml"/>
    <item id="fm3" href="book_fm3_r1.xhtml" media-type="application/xhtml+xml"/>
    <item id="c1" href="book_c01_r1.xhtml" media-type="application/xhtml+xml"/>
    <item id="bio" href="book_tail_r1.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="cvi"/><itemref idref="tocp"/><itemref idref="ded"/>
    <itemref idref="epi"/><itemref idref="frw"/><itemref idref="fm2"/>
    <itemref idref="fm3"/><itemref idref="c1"/><itemref idref="bio"/>
  </spine>
</package>"""
    files = {
        "images/cov.jpg": _jpg_bytes(480, 720, (50, 50, 50)),
        "images/frontispiece.jpg": _jpg_bytes(220, 320, (90, 90, 90)),
        "images/scene.jpg": _jpg_bytes(360, 260, (120, 120, 120)),
        "book_cvi_r1.xhtml": _xhtml('<div><img src="images/cov.jpg" alt="Cover"/></div>', title="Cover"),
        "book_toc_r1.xhtml": _xhtml('<p>Table of Contents</p>', title="Passing Sample"),
        "book_ded_r1.xhtml": _xhtml('<p>FOR Carl Van Vechten</p>', title="Passing Sample"),
        "book_epi_r1.xhtml": _xhtml('<div id="epi"><p>One three centuries removed.<br/>—COUNTEE CULLEN</p></div>', title="Passing"),
        "book_frw_r1.xhtml": _xhtml(f'<p class="fmh">CRITICAL FOREWORD</p><p>Mae G. Henderson</p><p>{fw_prose}</p>', title="Passing Sample"),
        "book_fm2_r1.xhtml": _xhtml('<p>MAE G. HENDERSON is professor of English at the University of North Carolina. Author of numerous critical studies.</p>', title="Passing Sample"),
        "book_fm3_r1.xhtml": _xhtml('<p><img src="images/frontispiece.jpg"/></p>', title="Passing Sample"),
        "book_c01_r1.xhtml": _xhtml(
            f'<h1>PART ONE</h1><p>{prose}</p>'
            f'<p><img src="images/scene.jpg" alt="Scene"/></p>'
            f'<div class="fig"><img src="images/scene.jpg" alt="Cap"/><p>Caption line</p></div>',
            title="Passing Sample",
        ),
        "book_tail_r1.xhtml": _xhtml('<p>Jane Author (1978) was born in Seville. She teaches writing and has published two collections.</p>', title="Passing Sample"),
    }
    _write_test_epub(src, files, opf)

    rep_story = process_epub(str(src), str(dst_story), ProcessingOptions(preset="story"))
    assert rep_story.success, rep_story.error

    with zipfile.ZipFile(dst_story) as zf:
        names = set(zf.namelist())
        assert "OEBPS/book_frw_r1.xhtml" not in names
        assert "OEBPS/book_fm2_r1.xhtml" not in names
        assert "OEBPS/book_fm3_r1.xhtml" not in names
        assert "OEBPS/book_tail_r1.xhtml" not in names
        assert "OEBPS/images/frontispiece.jpg" not in names
        assert "OEBPS/book_epi_r1.xhtml" in names
        assert "OEBPS/book_c01_r1.xhtml" in names

        c1_out = zf.read("OEBPS/book_c01_r1.xhtml").decode("utf-8")
        assert c1_out.count("x4-img-wrap") >= 2
        hs_css = zf.read("OEBPS/x4_house_style.css").decode("utf-8")
        assert ".x4-img-wrap, figure" in hs_css
        assert "display: inline-block !important;" in hs_css

        ncx_root = etree.fromstring(zf.read("OEBPS/toc.ncx"))
        labels = [el.text for el in ncx_root.xpath("//*[local-name()='navLabel']/*[local-name()='text']")]
        assert labels == ["Epigraph", "PART ONE"]

    rep_nf = process_epub(str(src), str(dst_nf), ProcessingOptions(preset="nonfiction"))
    assert rep_nf.success, rep_nf.error
    with zipfile.ZipFile(dst_nf) as zf:
        assert "OEBPS/book_frw_r1.xhtml" in set(zf.namelist())


def test_tier2_page_break_front_matter_split_and_section_renumbering(tmp_path):
    """
    Giant single-file headingless book separated by CSS page-break-before: always
    splits tiny front-matter & back-bio pages away from prose, trims them in Pass 2,
    and renumbers surviving story chunks cleanly from Section 1.
    """
    src = tmp_path / "wolfskin_like.epub"
    dst = tmp_path / "wolfskin_out.epub"
    sec_prose = "<p>" + " ".join(["narrative"] * 10500) + "</p>"

    big_html = _xhtml(
        '<div style="text-align:center"><img src="../Images/cov.jpg"/></div>'
        '<div style="page-break-before: always;"/>'
        '<h3>MY BOOK</h3>'
        '<div style="page-break-before: always;"/>'
        '<p>Copyright © 2026 All rights reserved. ISBN 978-1-234567-89-0. Published in 2026.</p>'
        '<div style="page-break-before: always;"/>'
        '<p>To my sister</p>'
        '<div style="page-break-before: always;"/>'
        '<h3>CONTENTS</h3><p><a href="chapter0000.xhtml">My Book</a></p>'
        f'<div style="page-break-before: always;"/>{sec_prose}'
        f'<div style="page-break-before: always;"/>{sec_prose}'
        f'<div style="page-break-before: always;"/>{sec_prose}'
        '<div style="page-break-before: always;"/>'
        '<p>Lara Author (1978) was born in Seville. She teaches writing and has published several novels.</p>'
        '<div style="page-break-before: always;"/>',
        title="My Book",
    )
    opf = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/opf" version="2.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="uid">urn:uuid:wolf</dc:identifier>
    <dc:title>My Book</dc:title><dc:creator>Lara Author</dc:creator><dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="covpg" href="Text/coverpage.xhtml" media-type="application/xhtml+xml"/>
    <item id="body" href="Text/chapter0000.xhtml" media-type="application/xhtml+xml"/>
    <item id="covimg" href="Images/cov.jpg" media-type="image/jpeg"/>
  </manifest>
  <spine><itemref idref="covpg"/><itemref idref="body"/></spine>
</package>"""
    _write_test_epub(
        src,
        {
            "Text/coverpage.xhtml": _xhtml('<div><img src="../Images/cov.jpg"/></div>', title="Cover"),
            "Text/chapter0000.xhtml": big_html,
            "Images/cov.jpg": _jpg_bytes(480, 720, (30, 30, 30)),
        },
        opf,
    )

    report = process_epub(str(src), str(dst), ProcessingOptions(preset="story"))
    assert report.success, report.error
    assert report.word_count_story >= 31000
    assert report.files_trimmed >= 4

    with zipfile.ZipFile(dst) as zf:
        names = set(zf.namelist())
        assert "OEBPS/Images/cov.jpg" in names
        assert not any("cover_generated" in n for n in names)

        ncx_root = etree.fromstring(zf.read("OEBPS/toc.ncx"))
        labels = [el.text for el in ncx_root.xpath("//*[local-name()='navLabel']/*[local-name()='text']")]
        assert labels == ["Section 1", "Section 2", "Section 3"]


def test_toc_polish_author_prefixed_generic_symbols_opening_and_dedup(tmp_path):
    """
    Tests Fix 1:
      - Author-prefixed '<title>Virginia Woolf: Between the Acts</title>' rejected/demoted
      - Generic 'unnamed' and symbol-only '*' headings ignored
      - Headingless opening files before 'Prologue' / 'Part One' become 'Opening' / 'Opening 2' (not 'Chapter 1')
      - Consecutive '(cont.)' fallback labels disambiguated ('Chapter 1 (cont.)', 'Chapter 1 (cont. 2)')
    """
    src = tmp_path / "toc_polish.epub"
    dst = tmp_path / "toc_polish_out.epub"
    prose = " ".join(["narrative"] * 300)

    opf = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/opf" version="2.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="uid">urn:uuid:tocpolish</dc:identifier>
    <dc:title>Between the Acts</dc:title><dc:creator>Virginia Woolf</dc:creator><dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="s1" href="s1.xhtml" media-type="application/xhtml+xml"/>
    <item id="s2" href="s2.xhtml" media-type="application/xhtml+xml"/>
    <item id="s3" href="s3.xhtml" media-type="application/xhtml+xml"/>
    <item id="s4" href="s4.xhtml" media-type="application/xhtml+xml"/>
    <item id="s5" href="s5.xhtml" media-type="application/xhtml+xml"/>
    <item id="s6" href="s6.xhtml" media-type="application/xhtml+xml"/>
    <item id="s7" href="s7.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="s1"/><itemref idref="s2"/><itemref idref="s3"/>
    <itemref idref="s4"/><itemref idref="s5"/><itemref idref="s6"/><itemref idref="s7"/>
  </spine>
</package>"""
    files = {
        "s1.xhtml": _xhtml(f"<p>England, late summer 1939. {prose}</p>", title="Virginia Woolf: Between the Acts"),
        "s2.xhtml": _xhtml(f"<p>Second setting scene. {prose}</p>", title="unnamed"),
        "s3.xhtml": _xhtml(f"<h1>Part One</h1><p>{prose}</p>", title="Part One"),
        "s4.xhtml": _xhtml(f"<h1>*</h1><p>{prose}</p>", title="Virginia Woolf: Between the Acts"),
        "s5.xhtml": _xhtml(f"<h1>Chapter 1</h1><p>{prose}</p>", title="Chapter 1"),
        "s6.xhtml": _xhtml(f"<p>Continuation chunk one. {prose}</p>", title="unnamed"),
        "s7.xhtml": _xhtml(f"<p>Continuation chunk two. {prose}</p>", title="Between the Acts"),
    }
    _write_test_epub(src, files, opf)

    report = process_epub(str(src), str(dst), ProcessingOptions())
    assert report.success, report.error

    with zipfile.ZipFile(dst) as zf:
        ncx_root = etree.fromstring(zf.read("OEBPS/toc.ncx"))
        labels = [el.text for el in ncx_root.xpath("//*[local-name()='navLabel']/*[local-name()='text']")]
        assert "Virginia Woolf: Between the Acts" not in labels
        assert "unnamed" not in labels
        assert "*" not in labels
        assert labels == [
            "Opening",
            "Opening 2",
            "Part One",
            "Part One (cont.)",
            "Chapter 1",
            "Chapter 1 (cont.)",
            "Chapter 1 (cont. 2)",
        ]


def test_matter_leaks_and_mashed_publisher_title_pages(tmp_path):
    """
    Tests Fixes 2 & 3:
      - Praise inside Calibre 'part0003.xhtml' trimmed (not blocked by 'part' regex)
      - Mashed title page ('BUNNYMona Awad www.headofzeus.com') trimmed while 'C:\\MALAGASH>book.exe' stub kept
      - Back <p>Acknowledgments</p>, <p>Also by ...</p>, and founder bio trimmed while story chapters stay kept
    """
    src = tmp_path / "matter_leaks.epub"
    dst = tmp_path / "matter_leaks_out.epub"
    prose = " ".join(["story"] * 350)

    opf = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/opf" version="2.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="uid">urn:uuid:leaks</dc:identifier>
    <dc:title>Death of the Heart, The</dc:title><dc:creator>Elizabeth Bowen</dc:creator><dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="tp1" href="s001.xhtml" media-type="application/xhtml+xml"/>
    <item id="p3" href="part0003.xhtml" media-type="application/xhtml+xml"/>
    <item id="exe" href="com.xhtml" media-type="application/xhtml+xml"/>
    <item id="c1" href="ch01.xhtml" media-type="application/xhtml+xml"/>
    <item id="c2" href="ch02.xhtml" media-type="application/xhtml+xml"/>
    <item id="c3" href="ch03.xhtml" media-type="application/xhtml+xml"/>
    <item id="ack" href="split_028.xhtml" media-type="application/xhtml+xml"/>
    <item id="also" href="bm1.xhtml" media-type="application/xhtml+xml"/>
    <item id="bio" href="split_030.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="tp1"/><itemref idref="p3"/><itemref idref="exe"/>
    <itemref idref="c1"/><itemref idref="c2"/><itemref idref="c3"/>
    <itemref idref="ack"/><itemref idref="also"/><itemref idref="bio"/>
  </spine>
</package>"""
    files = {
        "s001.xhtml": _xhtml("<p>The Death of the Heartby Elizabeth BowenOriginally published in 1940.</p>"),
        "part0003.xhtml": _xhtml(f"<p>Praise for The Death of the Heart ‘Brilliant’ The Guardian. {'review ' * 180}</p>"),
        "com.xhtml": _xhtml(r"<p>C:\DEATH\book.exe</p>"),
        "ch01.xhtml": _xhtml(f"<h1>Chapter 1</h1><p>{prose}</p>"),
        "ch02.xhtml": _xhtml(f"<h2>Notes on an Excavation</h2><p>{prose}</p>"),
        "ch03.xhtml": _xhtml(f"<h2>Credits &amp; Extra Credit</h2><p>{prose}</p>"),
        "split_028.xhtml": _xhtml(f"<p><b>Acknowledgments</b></p><p>I thank everyone who helped. {'thanks ' * 150}</p>"),
        "bm1.xhtml": _xhtml(f"<p>ALSO BY ELIZABETH BOWEN To the North. {'novel ' * 300}</p>"),
        "split_030.xhtml": _xhtml("<p>Jane Doe is the cofounder of Writers Guild and lives in New York.</p>"),
    }
    _write_test_epub(src, files, opf)

    report = process_epub(str(src), str(dst), ProcessingOptions(preset="story"))
    assert report.success, report.error

    with zipfile.ZipFile(dst) as zf:
        names = set(zf.namelist())
        assert "OEBPS/s001.xhtml" not in names
        assert "OEBPS/part0003.xhtml" not in names
        assert "OEBPS/split_028.xhtml" not in names
        assert "OEBPS/bm1.xhtml" not in names
        assert "OEBPS/split_030.xhtml" not in names
        assert "OEBPS/com.xhtml" in names
        assert "OEBPS/ch01.xhtml" in names
        assert "OEBPS/ch02.xhtml" in names
        assert "OEBPS/ch03.xhtml" in names


def test_dead_links_broken_images_zip_metadata_and_image_preclamp(monkeypatch, tmp_path):
    """
    Tests Fixes 4 & 5 + Render Free hardening:
      - Broken <img> missing in source archive stripped along with empty wrapper
      - Dead #_edn1 footnote link + empty <sup> deleted; dead #Afterword + scheme-less domain unwrapped keeping text
      - extract_epub_metadata reads directly from ZIP without temp extraction
      - process_image pre-clamps oversized JPEGs cleanly within bounds
    """
    import epub_processor
    from epub_processor import extract_epub_metadata
    from image_processor import process_image, ImageOptions

    src = tmp_path / "links_imgs.epub"
    dst = tmp_path / "links_imgs_out.epub"
    prose = " ".join(["story"] * 300)

    opf = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/opf" version="2.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="uid">urn:uuid:linksimgs</dc:identifier>
    <dc:title>Links &amp; Images</dc:title><dc:creator>Tester</dc:creator><dc:language>en</dc:language>
    <meta name="cover" content="covimg"/>
  </metadata>
  <manifest>
    <item id="covimg" href="images/cov.jpg" media-type="image/jpeg"/>
    <item id="c1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine><itemref idref="c1"/></spine>
</package>"""
    files = {
        "images/cov.jpg": _jpg_bytes(480, 720, (80, 80, 80)),
        "ch1.xhtml": _xhtml(
            f'<h1>Chapter 1</h1>'
            f'<p>{prose} Fact<sup><a href="missing.xhtml#_edn1">1</a></sup> and see <a href="ch1.xhtml#Afterword">Afterword</a> '
            f'or <a href="support.wordpress.com">support site</a>.</p>'
            f'<p><img src="images/00004.jpg" alt="Missing"/></p>',
            title="Chapter 1",
        ),
    }
    _write_test_epub(src, files, opf)

    # Verify extract_epub_metadata never touches tempfile.mkdtemp
    def _forbid_mkdtemp(*args, **kwargs):
        raise AssertionError("extract_epub_metadata must not extract to temp directory")

    monkeypatch.setattr(epub_processor.tempfile, "mkdtemp", _forbid_mkdtemp)
    meta = extract_epub_metadata(str(src))
    assert meta["title"] == "Links & Images"
    assert meta["author"] == "Tester"
    assert meta["cover_data"] is not None
    monkeypatch.undo()

    report = process_epub(str(src), str(dst), ProcessingOptions())
    assert report.success, report.error

    with zipfile.ZipFile(dst) as zf:
        c1_out = zf.read("OEBPS/ch1.xhtml").decode("utf-8")
        assert "00004.jpg" not in c1_out
        assert "<img" not in c1_out
        assert "_edn1" not in c1_out
        assert "<sup" not in c1_out
        assert "#Afterword" not in c1_out
        assert "Afterword" in c1_out
        assert "support.wordpress.com" not in c1_out
        assert "support site" in c1_out

    # Verify oversized JPEG pre-clamping in process_image
    big_jpg = _jpg_bytes(2400, 3200, (110, 120, 130))
    res = process_image(big_jpg, "big.jpg", ImageOptions.for_device("x4"))
    assert len(res) == 1
    assert res[0].was_converted is True
    assert "clamped 2400x3200->" in res[0].details
    with Image.open(io.BytesIO(res[0].output_bytes)) as out_im:
        assert max(out_im.size) <= 1024


def test_midbook_ad_no_cascade_prose_newsletter_and_midbook_intro_preface(tmp_path):
    """
    Tests mid-book story preservation fixes:
      - Mid-book ad_chapter20.xhtml past 65% is dropped individually without cascading over Ch 15..16
      - Short mid-book prose chapters mentioning 'newsletter' / 'sign up for six workshops' stay kept
      - Mid-book 'Preface' and 'Introduction to The Dietzens' after story_started stay kept,
        while front 'Critical Foreword' / 'Introduction' before story_started trim in story mode and stay in nonfiction
    """
    src = tmp_path / "midbook_safety.epub"
    dst_story = tmp_path / "midbook_story.epub"
    dst_nf = tmp_path / "midbook_nf.epub"
    ch_prose = " ".join(["story"] * 700)

    opf = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/opf" version="2.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="uid">urn:uuid:midbooksafe</dc:identifier>
    <dc:title>Midbook Safety Novel</dc:title><dc:creator>Author</dc:creator><dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="fw" href="frw.xhtml" media-type="application/xhtml+xml"/>
    <item id="c1" href="ch01.xhtml" media-type="application/xhtml+xml"/>
    <item id="c2" href="ch02.xhtml" media-type="application/xhtml+xml"/>
    <item id="nw" href="cH7.xhtml" media-type="application/xhtml+xml"/>
    <item id="ws" href="ch093.xhtml" media-type="application/xhtml+xml"/>
    <item id="pr" href="ch05_preface.xhtml" media-type="application/xhtml+xml"/>
    <item id="it" href="ch06_intro.xhtml" media-type="application/xhtml+xml"/>
    <item id="ad" href="ad_chapter20.xhtml" media-type="application/xhtml+xml"/>
    <item id="c15" href="chapter_15.xhtml" media-type="application/xhtml+xml"/>
    <item id="c16" href="chapter_16.xhtml" media-type="application/xhtml+xml"/>
    <item id="ack" href="back_ack.xhtml" media-type="application/xhtml+xml"/>
    <item id="nts" href="c238_notes.xhtml" media-type="application/xhtml+xml"/>
    <item id="nt1" href="c238_ch1.xhtml" media-type="application/xhtml+xml"/>
    <item id="bio" href="back_bio.xhtml" media-type="application/xhtml+xml"/>
    <item id="tail" href="back_tail.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="fw"/><itemref idref="c1"/><itemref idref="c2"/>
    <itemref idref="nw"/><itemref idref="ws"/><itemref idref="pr"/><itemref idref="it"/>
    <itemref idref="ad"/><itemref idref="c15"/><itemref idref="c16"/>
    <itemref idref="ack"/><itemref idref="nts"/><itemref idref="nt1"/>
    <itemref idref="bio"/><itemref idref="tail"/>
  </spine>
</package>"""
    newsletter_story = (
        "<p>Each Magic Lantern Society newsletter came with an insert on matte paper titled Sales and Wants. "
        + " ".join(["lantern"] * 240)
        + "</p>"
    )
    workshops_story = (
        "<p>Your creativity explodes; you sign up for six workshops in your last semester of school. "
        + " ".join(["writing"] * 240)
        + "</p>"
    )
    files = {
        "frw.xhtml": _xhtml(f"<h1>Critical Foreword</h1><p>{'critique ' * 800}</p>"),
        "ch01.xhtml": _xhtml(f"<h1>Chapter 1</h1><p>{ch_prose}</p>"),
        "ch02.xhtml": _xhtml(f"<h1>Chapter 2</h1><p>{ch_prose}</p>"),
        "cH7.xhtml": _xhtml(newsletter_story),
        "ch093.xhtml": _xhtml(workshops_story),
        "ch05_preface.xhtml": _xhtml(f"<h1>Preface</h1><p>{ch_prose}</p>"),
        "ch06_intro.xhtml": _xhtml(f"<h1>Introduction to The Dietzens</h1><p>{ch_prose}</p>"),
        "ad_chapter20.xhtml": _xhtml("<p>读累了记得休息一会哦~ https://www.chenjin5.com</p>"),
        "chapter_15.xhtml": _xhtml(f"<h1>Three Months Later</h1><p>{ch_prose}</p>"),
        "chapter_16.xhtml": _xhtml(f"<h1>Four Months Later</h1><p>{ch_prose}</p>"),
        "back_ack.xhtml": _xhtml("<p>Acknowledgments</p><p>Thanks to my editor for patient feedback.</p>"),
        "c238_notes.xhtml": _xhtml("<p>Notes</p><p>Source notes begin below.</p>"),
        "c238_ch1.xhtml": _xhtml(f"<h1>Chapter 1: Endnotes</h1><p>{'citation ' * 600}</p>"),
        "back_bio.xhtml": _xhtml("<p>Author Bio</p><p>Author (1980) was born in Dublin and has published three novels.</p>"),
        "back_tail.xhtml": _xhtml("<p>Follow us for more books.</p>"),
    }
    _write_test_epub(src, files, opf)

    rep_story = process_epub(str(src), str(dst_story), ProcessingOptions(preset="story"))
    assert rep_story.success, rep_story.error
    with zipfile.ZipFile(dst_story) as zf:
        names = set(zf.namelist())
        assert "OEBPS/frw.xhtml" not in names
        assert "OEBPS/ad_chapter20.xhtml" not in names
        assert "OEBPS/back_ack.xhtml" not in names
        assert "OEBPS/c238_notes.xhtml" not in names
        assert "OEBPS/c238_ch1.xhtml" not in names
        assert "OEBPS/back_bio.xhtml" not in names
        assert "OEBPS/back_tail.xhtml" not in names
        for kept in (
            "OEBPS/cH7.xhtml",
            "OEBPS/ch093.xhtml",
            "OEBPS/ch05_preface.xhtml",
            "OEBPS/ch06_intro.xhtml",
            "OEBPS/chapter_15.xhtml",
            "OEBPS/chapter_16.xhtml",
        ):
            assert kept in names, f"Expected {kept} to survive"

    rep_nf = process_epub(str(src), str(dst_nf), ProcessingOptions(preset="nonfiction"))
    assert rep_nf.success, rep_nf.error
    with zipfile.ZipFile(dst_nf) as zf:
        names_nf = set(zf.namelist())
        assert "OEBPS/frw.xhtml" in names_nf
        assert "OEBPS/c238_notes.xhtml" in names_nf
        assert "OEBPS/c238_ch1.xhtml" in names_nf
        assert "OEBPS/ad_chapter20.xhtml" not in names_nf
        assert "OEBPS/chapter_15.xhtml" in names_nf


def test_toc_first_header_priority_numbered_epigraph_guard_and_nav_reuse(tmp_path):
    """
    Tests TOC/nav fixes:
      - First .x4-chapter-header .x4-ch-label beats a subsequent subsection .x4-ch-num ('1')
      - Numbered 006_epi_sup_1.xhtml (>300w) is labeled 'Chapter 1', not 'Epigraph',
        while short epi.xhtml (<=300w) is still labeled 'Epigraph'
      - Existing id='nav' (text/nav.xhtml without properties='nav') is updated in-place
        with properties='nav' and no duplicate root nav.xhtml is created
    """
    from epub_structure import fix_toc

    work = tmp_path / "toc_nav_fix"
    text_dir = work / "text"
    text_dir.mkdir(parents=True)

    long_prose = " ".join(["chapter"] * 450)
    (text_dir / "nav.xhtml").write_text(
        '<?xml version="1.0" encoding="utf-8"?>'
        '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Nav</title></head>'
        '<body><nav id="toc"><ol><li><a href="ch1.xhtml">Stale</a></li></ol></nav></body></html>',
        encoding="utf-8",
    )
    (text_dir / "epi.xhtml").write_text(
        _xhtml('<div id="epi"><p>Short epigraph quote.<br/>—POET</p></div>', title="Novel"),
        encoding="utf-8",
    )
    (text_dir / "006_epi_sup_1.xhtml").write_text(
        _xhtml(f"<p>{long_prose}</p>", title="Novel"),
        encoding="utf-8",
    )
    (text_dir / "c_sub.xhtml").write_text(
        _xhtml(
            '<div class="x4-chapter-header"><div class="x4-ch-label">Chosen, New York, 1978</div></div>'
            '<div class="sub_chap"><div class="x4-chapter-header"><div class="x4-ch-num">1</div></div>'
            f'<p>{long_prose}</p></div>',
            title="Novel",
        ),
        encoding="utf-8",
    )

    opf_path = work / "content.opf"
    opf_path.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/opf" version="3.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="uid">urn:uuid:navfix</dc:identifier>
    <dc:title>Novel</dc:title><dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="nav" href="text/nav.xhtml" media-type="application/xhtml+xml"/>
    <item id="epi" href="text/epi.xhtml" media-type="application/xhtml+xml"/>
    <item id="c1" href="text/006_epi_sup_1.xhtml" media-type="application/xhtml+xml"/>
    <item id="c2" href="text/c_sub.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="epi"/><itemref idref="c1"/><itemref idref="c2"/>
  </spine>
</package>""",
        encoding="utf-8",
    )

    fixed, _ = fix_toc(str(work), str(opf_path), force_rebuild=True)
    assert fixed is True
    assert not (work / "nav.xhtml").exists()

    opf_root = etree.fromstring(opf_path.read_bytes())
    nav_items = [
        el for el in opf_root.findall(".//{*}manifest/{*}item")
        if "nav" in (el.get("href") or "").lower()
    ]
    assert len(nav_items) == 1
    assert nav_items[0].get("href") == "text/nav.xhtml"
    assert "nav" in (nav_items[0].get("properties") or "").split()

    ncx_root = etree.fromstring((work / "toc.ncx").read_bytes())
    ncx_labels = [el.text for el in ncx_root.xpath("//*[local-name()='navLabel']/*[local-name()='text']")]
    assert ncx_labels == ["Epigraph", "Chapter 1", "Chosen, New York, 1978"]

    nav_root = etree.fromstring((text_dir / "nav.xhtml").read_bytes())
    nav_labels = ["".join(a.itertext()).strip() for a in nav_root.findall(".//{*}nav//{*}a")]
    assert nav_labels == ncx_labels


def test_straggler_pirate_outline_diacritic_title_and_front_copyright_blob(tmp_path):
    """
    Tests minor straggler cleanups:
      - Pirate manifesto ('Fuck "Copyright" / Read Free or Die') removed
      - Word/Calibre 'Document Outline Local Disk C:\\...' stub removed
      - Diacritic title page ('The Woman in the Dunes by Kōbō Abe' vs OPF 'Kobo Abe') removed
      - Front combined copyright + publisher book list (<1200w) removed
    """
    src = tmp_path / "stragglers.epub"
    dst = tmp_path / "stragglers_out.epub"
    prose = " ".join(["story"] * 600)

    opf = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/opf" version="2.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="uid">urn:uuid:stragglers</dc:identifier>
    <dc:title>The Woman in the Dunes</dc:title><dc:creator>Kobo Abe</dc:creator><dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="pirate" href="Pre0002.xhtml" media-type="application/xhtml+xml"/>
    <item id="tp" href="4.xhtml" media-type="application/xhtml+xml"/>
    <item id="bio" href="5.xhtml" media-type="application/xhtml+xml"/>
    <item id="cop" href="part0000_ch001.html" media-type="application/xhtml+xml"/>
    <item id="c1" href="chapter_1.xhtml" media-type="application/xhtml+xml"/>
    <item id="docout" href="index_split_001.html" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="pirate"/><itemref idref="tp"/><itemref idref="bio"/>
    <itemref idref="cop"/><itemref idref="c1"/><itemref idref="docout"/>
  </spine>
</package>"""
    files = {
        "Pre0002.xhtml": _xhtml('<h1>Fuck "Copyright"</h1><h1>Read Free or Die</h1>'),
        "4.xhtml": _xhtml("<p>The Woman in the Dunes by Kōbō Abe Translated from the Japanese by E. Dale Saunders</p>"),
        "5.xhtml": _xhtml("<p>Kōbō Abe, pseudonym of Kimifusa Abe, was a Japanese writer and playwright.</p>"),
        "part0000_ch001.html": _xhtml(
            f"<p>Defying Press © 2014 All rights reserved. Also by the author: {'Book Title ' * 250}</p>"
        ),
        "chapter_1.xhtml": _xhtml(f"<h1>Prologue</h1><p>{prose}</p>"),
        "index_split_001.html": _xhtml(r"<p>Document Outline</p><p>Local Disk C:\Downloads\book.TXT</p>"),
    }
    _write_test_epub(src, files, opf)

    report = process_epub(str(src), str(dst), ProcessingOptions(preset="story"))
    assert report.success, report.error

    with zipfile.ZipFile(dst) as zf:
        names = set(zf.namelist())
        assert "OEBPS/Pre0002.xhtml" not in names
        assert "OEBPS/4.xhtml" not in names
        assert "OEBPS/5.xhtml" not in names
        assert "OEBPS/part0000_ch001.html" not in names
        assert "OEBPS/index_split_001.html" not in names
        assert "OEBPS/chapter_1.xhtml" in names


def test_opf_image_manifest_updates_with_posix_slashes(tmp_path):
    """Verify .png/.jpeg image conversions in subdirs update OPF manifest with POSIX slashes."""
    from urllib.parse import unquote
    import posixpath

    src = tmp_path / "img_manifest.epub"
    dst = tmp_path / "img_manifest_out.epub"
    prose = " ".join(["story"] * 600)

    opf = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/opf" version="2.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="uid">urn:uuid:opfimg</dc:identifier>
    <dc:title>PNG Cover Book</dc:title><dc:creator>Author</dc:creator><dc:language>en</dc:language>
    <meta name="cover" content="covimg"/>
  </metadata>
  <manifest>
    <item id="covpg" href="Text/Cover.xhtml" media-type="application/xhtml+xml"/>
    <item id="c1" href="Text/ch1.xhtml" media-type="application/xhtml+xml"/>
    <item id="covimg" href="Images/Cover.png" media-type="image/png"/>
    <item id="figimg" href="Images/fig.jpeg" media-type="image/jpeg"/>
  </manifest>
  <spine><itemref idref="covpg"/><itemref idref="c1"/></spine>
</package>"""
    files = {
        "Text/Cover.xhtml": _xhtml('<div><img src="../Images/Cover.png" alt="Cover"/></div>', title="Cover"),
        "Text/ch1.xhtml": _xhtml(f'<h1>Chapter 1</h1><p>{prose}</p><p><img src="../Images/fig.jpeg" alt="Fig"/></p>'),
        "Images/Cover.png": _png_bytes(480, 720, (50, 70, 90)),
        "Images/fig.jpeg": _jpg_bytes(320, 240, (80, 90, 100)),
    }
    _write_test_epub(src, files, opf)

    report = process_epub(str(src), str(dst), ProcessingOptions())
    assert report.success, report.error
    assert report.cover_generated is False

    with zipfile.ZipFile(dst) as zf:
        names = set(zf.namelist())
        assert "OEBPS/Images/Cover.jpg" in names
        assert "OEBPS/Images/fig.jpg" in names
        assert "OEBPS/Images/Cover.png" not in names
        assert "OEBPS/Images/fig.jpeg" not in names

        cov_out = zf.read("OEBPS/Text/Cover.xhtml").decode("utf-8")
        ch1_out = zf.read("OEBPS/Text/ch1.xhtml").decode("utf-8")
        assert "../Images/Cover.jpg" in cov_out
        assert "../Images/fig.jpg" in ch1_out

        opf_root = etree.fromstring(zf.read("OEBPS/content.opf"))
        manifest_hrefs = {el.get("id"): el.get("href") for el in opf_root.findall(".//{*}manifest/{*}item")}
        assert manifest_hrefs["covimg"] == "Images/Cover.jpg"
        assert manifest_hrefs["figimg"] == "Images/fig.jpg"
        for href in manifest_hrefs.values():
            assert "\\" not in href
            assert posixpath.normpath(posixpath.join("OEBPS", unquote(href))) in names


def test_package_hygiene_parent_nav_dedup_manifest_and_unmanifested_clutter(tmp_path):
    """
    Tests all 3 Codex package-hygiene fixes:
      - OEBPS/content.opf referencing ../nav.xhtml produces clean OEBPS/Text/ch1.xhtml links (no host path / C:)
      - Duplicate manifest hrefs and duplicate IDs are deduplicated/disambiguated (preserving cover-image property)
      - Missing page_styles.css removed from manifest and stripped from XHTML <link>
      - Unmanifested clutter (iTunesMetadata.plist, rights.xml, orphan.jpg, stray.ttf) omitted from output ZIP
    """
    from urllib.parse import unquote
    import posixpath

    src = tmp_path / "hygiene.epub"
    dst = tmp_path / "hygiene_out.epub"
    prose = " ".join(["story"] * 600)

    opf = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/opf" version="3.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="uid">urn:uuid:hygiene</dc:identifier>
    <dc:title>Hygiene Novel</dc:title><dc:creator>Author</dc:creator><dc:language>en</dc:language>
    <meta name="cover" content="cov_dup"/>
  </metadata>
  <manifest>
    <item id="nav" href="../nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
    <item id="cov1" href="Images/cover.jpg" media-type="image/jpeg"/>
    <item id="cov_dup" href="Images/cover.jpg" media-type="image/jpeg" properties="cover-image"/>
    <item id="dup_asset" href="Images/fig1.jpg" media-type="image/jpeg"/>
    <item id="dup_asset" href="Images/fig2.jpg" media-type="image/jpeg"/>
    <item id="ch1" href="Text/ch1.xhtml" media-type="application/xhtml+xml"/>
    <item id="ch2" href="Text/ch2.xhtml" media-type="application/xhtml+xml"/>
    <item id="miss_css" href="Styles/page_styles.css" media-type="text/css"/>
  </manifest>
  <spine>
    <itemref idref="ch1"/><itemref idref="ch2"/>
  </spine>
</package>"""
    ch1_html = _xhtml(
        f'<h1>Chapter 1</h1><p>{prose}</p>'
        '<p><img src="../Images/fig1.jpg" alt="1"/><img src="../Images/fig2.jpg" alt="2"/></p>'
    ).replace(
        "</head>",
        '<link rel="stylesheet" type="text/css" href="../Styles/page_styles.css"/></head>',
    )
    ch2_html = _xhtml(f'<h1>Chapter 2</h1><p>{prose}</p>')
    nav_html = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">'
        '<head><title>Nav</title></head><body><nav epub:type="toc"><ol>'
        '<li><a href="OEBPS/Text/ch1.xhtml">Ch 1</a></li>'
        '</ol></nav></body></html>'
    )
    container_xml = (
        '<?xml version="1.0"?>'
        '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
        '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>'
        '</container>'
    )

    with zipfile.ZipFile(src, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", container_xml)
        zf.writestr("nav.xhtml", nav_html)
        zf.writestr("iTunesMetadata.plist", "<plist/>")
        zf.writestr("rights.xml", "<rights/>")
        zf.writestr("OEBPS/content.opf", opf)
        zf.writestr("OEBPS/Images/cover.jpg", _jpg_bytes(480, 720, (50, 50, 50)))
        zf.writestr("OEBPS/Images/fig1.jpg", _jpg_bytes(220, 180, (70, 70, 70)))
        zf.writestr("OEBPS/Images/fig2.jpg", _jpg_bytes(220, 180, (90, 90, 90)))
        zf.writestr("OEBPS/Images/orphan.jpg", _jpg_bytes(200, 200, (120, 120, 120)))
        zf.writestr("OEBPS/fonts/stray.ttf", b"\x00\x01\x00\x00dummy")
        zf.writestr("OEBPS/Text/ch1.xhtml", ch1_html)
        zf.writestr("OEBPS/Text/ch2.xhtml", ch2_html)

    report = process_epub(str(src), str(dst), ProcessingOptions())
    assert report.success, report.error

    with zipfile.ZipFile(dst) as zf:
        names = set(zf.namelist())
        for junk in ("iTunesMetadata.plist", "rights.xml", "OEBPS/Images/orphan.jpg", "OEBPS/fonts/stray.ttf"):
            assert junk not in names, f"Unexpected unmanifested clutter survived: {junk}"
        assert "META-INF/container.xml" in names
        assert "nav.xhtml" in names

        nav_out = etree.fromstring(zf.read("nav.xhtml"))
        nav_hrefs = [a.get("href", "") for a in nav_out.findall(".//{*}nav//{*}a")]
        assert nav_hrefs == ["OEBPS/Text/ch1.xhtml", "OEBPS/Text/ch2.xhtml"]
        for nh in nav_hrefs:
            assert ":" not in nh and "\\" not in nh and nh in names

        opf_root = etree.fromstring(zf.read("OEBPS/content.opf"))
        man_items = list(opf_root.findall(".//{*}manifest/{*}item"))
        ids = [el.get("id") for el in man_items]
        hrefs = [posixpath.normpath(unquote(el.get("href", ""))) for el in man_items]
        assert len(ids) == len(set(ids)), f"Duplicate manifest IDs remain: {ids}"
        assert len(hrefs) == len(set(hrefs)), f"Duplicate manifest hrefs remain: {hrefs}"
        assert "Styles/page_styles.css" not in hrefs
        cov_items = [el for el in man_items if "cover-image" in (el.get("properties") or "").split()]
        assert len(cov_items) == 1
        assert cov_items[0].get("href") == "Images/cover.jpg"

        ch1_out = zf.read("OEBPS/Text/ch1.xhtml").decode("utf-8")
        assert "page_styles.css" not in ch1_out
