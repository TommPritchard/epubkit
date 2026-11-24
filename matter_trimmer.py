"""
Front & Back Matter Trimmer for Xteink X4 Story-First Edition Engine.
Supports:
  - "story" preset (Pure Story default): trims non-story front/back clutter while keeping
    Prologue, Epilogue, Epigraph, and story chapters.
  - "nonfiction" preset (Keep Notes & Intro): keeps Introductions, Forewords, Prefaces,
    Notes/Endnotes, Bibliographies, Appendices, Maps/Glossaries, Indexes, Afterwords.
  - Dead footnote DELETE vs. dead navigation/heading link UNWRAP.
  - Unreferenced image pruning (before image processing runs).
  - Source EPUB health checks (corrupted placeholder chapters, title mismatch, missing files).
"""
from __future__ import annotations

import os
import posixpath
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

from lxml import etree

from html_cleaner import xml_parser, recovery_parser
from epub_structure import (
    NS_OPF, NS_XHTML, NS_EPUB, NS_SVG, NS_XLINK,
    _is_element, _find_element,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BLOCK_TAGS = frozenset({
    "p", "div", "section", "article", "header", "footer", "aside", "nav",
    "h1", "h2", "h3", "h4", "h5", "h6", "li", "ul", "ol", "blockquote",
    "tr", "td", "th", "figcaption", "figure", "hr", "br", "pre", "dd", "dt",
})

WORD_RE = re.compile(r"\b[\w'’-]+\b", re.UNICODE)


def local_name(tag) -> str:
    return tag.split("}")[-1].lower() if isinstance(tag, str) else ""


def norm_match_text(s: str) -> str:
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u00a0", " ").replace("&nbsp;", " ")
    return " ".join(s.split()).strip()


def _strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))


def extract_clean_text(el) -> str:
    """Extract readable text without splitting inline spans (e.g. dropcaps)."""
    if el is None:
        return ""
    parts: list[str] = []

    def walk(node):
        ln = local_name(node.tag)
        if ln in ("script", "style", "head"):
            return
        is_block = ln in BLOCK_TAGS
        if is_block:
            parts.append(" ")
        if node.text:
            parts.append(node.text)
        for child in node:
            walk(child)
            if child.tail:
                parts.append(child.tail)
        if is_block:
            parts.append(" ")

    walk(el)
    return " ".join("".join(parts).split())


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def parse_xhtml_safe(data: bytes) -> tuple[etree._Element, bool]:
    """Parse XHTML/HTML using hardened xml_parser, falling back to recovery_parser."""
    try:
        return etree.fromstring(data, xml_parser()), False
    except Exception:
        return etree.fromstring(data, recovery_parser(data)), True


def get_body(root: etree._Element) -> etree._Element:
    for el in root.iter():
        if local_name(el.tag) == "body":
            return el
    return root


COVER_STEM_TOKENS = ("cover", "cov", "cvi", "coverpage", "cubierta", "titlepage")


def is_cover_stem(href: str) -> bool:
    """Shared cover-filename check across matter_trimmer, house_style, epub_structure, and epub_processor."""
    stem = Path(href).stem.lower()
    return any(tok in stem for tok in COVER_STEM_TOKENS)


def _collect_doc_img_refs(root: etree._Element, rel_dir: str) -> set[str]:
    """Extract normalized image refs from an already-parsed XHTML DOM."""
    refs: set[str] = set()
    for el in root.iter():
        ln = local_name(el.tag)
        if ln == "img" and el.get("src"):
            refs.add(_normalize_rel(rel_dir, el.get("src")))
        elif ln == "image":
            href = el.get(f"{{{NS_XLINK}}}href") or el.get("href") or ""
            if href:
                refs.add(_normalize_rel(rel_dir, href))
        style = el.get("style")
        if style and "url(" in style:
            for m in re.findall(r"url\(([^)]+)\)", style):
                u = m.strip("'\" ")
                if u:
                    refs.add(_normalize_rel(rel_dir, u))
    return refs


def _normalize_rel(base_dir: str, href: str) -> str:
    clean = unquote(href.split("#", 1)[0])
    if not clean:
        return ""
    joined = posixpath.normpath(posixpath.join(base_dir, clean)) if base_dir else posixpath.normpath(clean)
    return joined.lstrip("/")


# ---------------------------------------------------------------------------
# Battle-tested v13 Classification Tables (101 real library EPUBs)
# ---------------------------------------------------------------------------

_INTRO_FOREWORD_PAT = (
    r"^((?:critical|historical|biographical|general|new|selected|translator'?s?|editor'?s?|author'?s?)\s+)*"
    r"(introduction|foreword|preface|vorwort|einleitung|geleitwort)\b"
)
_FRONT_CONTRIBUTOR_BIO_RE = re.compile(
    r"\b(is\s+(?:a\s+|an\s+|the\s+)?(?:(?:emeritus|distinguished|associate|assistant|visiting)\s+)?professor"
    r"|professor\s+of\b|author\s+of\s+(?:numerous|several|many)\b)\b",
    re.I,
)
_FRONT_BIO_ROLE_RE = re.compile(
    r"\b(?:was|is)\s+an?\s+(?:[\w-]+\s+){0,2}(?:writer|novelist|poet|playwright|journalist|essayist|historian)\b",
    re.I,
)
_BACK_AUTHOR_BIO_RE = re.compile(
    r"(?:\(\d{4}\)\s+was\s+born\s+in|\bhas\s+published\b|\bteaches\s+(?:creative\s+)?writing\b"
    r"|\bgraduate\s+of\s+(?:the\s+)?university\b|\bwork\s+has\s+appeared\s+in\b"
    r"|\bis\s+the\s+author\s+of\b|\btranslated\s+into\s+\d+\s+languages\b"
    r"|\bis\s+(?:the\s+|a\s+)?(?:co-?founder|founder)\s+of\b"
    r"|\bis\s+(?:an?\s+)?(?:professor|award[- ]winning|bestselling)\s+"
    r"(?:author|novelist|poet|journalist|historian|critic|translator|editor)\b)",
    re.I,
)
_GENERIC_PART_STEM_RE = re.compile(r"^part\d+$", re.I)

TRASH_TITLE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("cover", re.compile(r"^(cover|book\s*cover|front\s*cover|cover\s*page|jacket|umschlag)$", re.I)),
    ("half_title", re.compile(r"^(half[\s-]*title|half[\s-]*title\s*page|bastard\s*title|schmutztitel)$", re.I)),
    ("title_page", re.compile(r"^(title|title\s*page|full\s*title|titelseite|titel|frontispiece)$", re.I)),
    ("copyright", re.compile(r"^(copyright|copyright\s*page|copyright\s*notice|colophon|imprint|impressum|legal|credits|permissions|publication\s*data|publishing\s*history|terms\s*of\s*use)$", re.I)),
    ("praise", re.compile(r"^(praise|acclaim|advance\s*praise|critical\s*acclaim|reviews|blurbs|what\s+people\s+are\s+saying|press\s+for|praise\s+for\b|acclaim\s+for\b|stimmen\s+zum\s+buch)", re.I)),
    ("also_by", re.compile(r"^(also\s+by|other\s+books\s+by|other\s+titles|books\s+by|by\s+the\s+same\s+author|more\s+from|selected\s+works|read\s+more|booklist|weitere\s+bücher|von\s+demselben\s+autor)", re.I)),
    ("html_toc", re.compile(r"^(contents|table\s+of\s+contents|toc|inhalt|inhaltsverzeichnis|brief\s+contents)$", re.I)),
    ("dedication", re.compile(r"^(dedication|widmung)$", re.I)),
    ("content_advisory", re.compile(r"^(content\s+(advisory|warning|note)|trigger\s+warning|reader'?s?\s+advisory)$", re.I)),
    ("maps_cast_glossary", re.compile(r"^(map|maps|map\s*:|list\s+of\s+maps|list\s+of\s+illustrations|illustrations|figures|tables|characters|cast\s+of\s+characters|dramatis\s+personae|principal\s+characters|glossary|lexicon|pronunciation\s+guide|timeline|chronology|family\s+tree|genealogy|karten|personenverzeichnis|glossar|zeittafel|abbildungen)\b", re.I)),
    ("acknowledgements", re.compile(r"^(#?gratitude|acknowledg(?:e)?ments?|thanks|author'?s?\s+acknowledg(?:e)?ments?|danksagung|dank)$", re.I)),
    ("about_author", re.compile(r"^(about\s+the\s+book|about\s+the\s+author|about\s+the\s+authors|a\s+note\s+about\s+the\s+author|a\s+note\s+on\s+the\s+author|a\s+biography\s+of\b|about\s+the\s+translator|about\s+the\s+publisher|about\s+[\w\s&'-]+(press|books|publishing|house|editions|penguin|vintage|picador|harper|farrar|knopf|grove|bloomsbury|faber|fitzcarraldo)|author\s+bio|biography|über\s+den\s+autor|über\s+die\s+autorin|über\s+das\s+buch|zu\s+diesem\s+buch|verlag)", re.I)),
    ("notes_references", re.compile(r"^(notes|endnotes|footnotes|source\s+notes|chapter\s+notes|notes\s+and\s+sources|references|sources|selected\s+sources|select\s+sources|image\s+sources|works\s+cited|further\s+reading|suggestions\s+for\s+further\s+reading|selected\s+bibliography|select\s+bibliography|bibliography|suggested\s+reading|reading\s+list|anmerkungen|fußnoten|quellen|quellenverzeichnis|literatur|literaturverzeichnis|bildnachweis|picture\s+credits|illustration\s+credits|photo\s+credits|permissions)$", re.I)),
    ("index", re.compile(r"^(index|name\s+index|subject\s+index|register|personenregister|sachregister|stichwortverzeichnis)$", re.I)),
    ("afterword_authors_note", re.compile(r"^(afterword|author'?s?\s+note|a\s+note\s+from\s+the\s+author|a\s+note\s+on\s+sources|note\s+on\s+the\s+text|note\s+on\s+sources|note\s+on\s+translation|historical\s+note|terminal\s+note|translator'?s?\s+(note|afterword)|editor'?s?\s+note|on\s+translating\b|postscript|nachwort|anmerkung\s+des\s+autors|anmerkung\s+der\s+autorin|nachbemerkung)\b", re.I)),
    ("excerpt_promo", re.compile(r"^(excerpt|preview|sneak\s+peek|teaser|bonus|sample\s+chapter|read\s+on|reading\s+group\s+guide|book\s+club|discussion\s+questions|questions\s+for\s+discussion|conversation\s+with|interview\s+with|q\s*&\s*a|newsletter|sign\s+up|become\s+a\s+subscriber|subscribe|other\s+ways\s+to\s+get\s+involved|discover\s+more|discover\s+your\s+next|what'?s\s+next\s+on\s+your\s+reading\s+list|follow\s+penguin|leseprobe|vorschau)", re.I)),
    ("intro_foreword_preface", re.compile(_INTRO_FOREWORD_PAT, re.I)),
    ("appendix", re.compile(r"^(appendix|appendices|anhang)\b", re.I)),
]

KEEP_STORY_TITLE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("prologue", re.compile(r"^(prologue|prolog)\b", re.I)),
    ("epilogue", re.compile(r"^(epilogue|epilog)\b", re.I)),
    ("epigraph", re.compile(r"^(epigraph|motto)\b", re.I)),
    ("chapter", re.compile(r"^(chapter|kapitel|book|buch|part|teil|act|akt|section|canto|letter|interlude|coda)\b", re.I)),
]

FILENAME_HINTS: list[tuple[str, str, re.Pattern]] = [
    ("COVER", "filename:cover", re.compile(r"(^|[/_-])(cover|cov|cvi|coverpage|cubierta)([._-]|$)", re.I)),
    ("TITLE_PAGE", "filename:titlepage", re.compile(r"(^|[/_-])(title\d*|titlepage|title_page|tp|tit|titel)([._-]|$)", re.I)),
    ("TRASH", "filename:halftitle", re.compile(r"(^|[/_-])(half|halftitle|half_title|htp|htit|schmutz)([._-]|$)", re.I)),
    ("TRASH", "filename:copyright", re.compile(r"(^|[/_-])(copyright|copyrightnotice|cop|copy|copy\d+|cpr|colophon|imprint|impressum|cip|legal|crt\d*)([._-]|$)", re.I)),
    ("TRASH", "filename:praise", re.compile(r"(^|[/_-])(praise|blurb|acclaim|reviews|advpraise|qts|jacket)([._-]|$)", re.I)),
    ("TRASH", "filename:also_by", re.compile(r"(^|[/_-])(alsoby|also_by|also|otherbooks|oth|bytheauthor|adc\d*|adcard|backad|orim-ad|ebb|portalist|love_\d+|brandpage|endpage|booklist|landingpage|audio_?link)([._-]|$)", re.I)),
    ("TRASH", "filename:midbook_ad", re.compile(r"(^|[/_-])(ad_?chapter\d*)([._-]|$)", re.I)),
    ("TRASH", "filename:toc", re.compile(r"(^|[/_-])(toc|mini_toc|contents|tableofcontents|nav|inhalt)([._-]|$)", re.I)),
    ("TRASH", "filename:dedication", re.compile(r"(^|[/_-])(ded|dedication|dedic|widmung)([._-]|$)", re.I)),
    ("TRASH", "filename:content_advisory", re.compile(r"(^|[/_-])(content_?advisory|content_?warning|trigger_?warning)([._-]|$)", re.I)),
    ("TRASH", "filename:maps_glossary", re.compile(r"(^|[/_-])(map\d*|maps|illus|ill|ins|ins\d+|loi|lot|lof|gloss|glossary|cast|characters|dramatis|timeline|chron|family)([._-]|$)", re.I)),
    ("TRASH", "filename:acknowledgements", re.compile(r"(^|[/_-])(ack|acknow|acknowledg\w*|thanks|dank)([._-]|$)", re.I)),
    ("TRASH", "filename:about_author", re.compile(r"(^|[/_-])(about|bio|biography|ata\d*|atp\d*|author|atauthor|aboutauthor|about_author|abouttheauthor|abouttheeditor|publisher|pub|advert|ads|promo|newsletter|signup|next-reads|nextreads)([._-]|$)", re.I)),
    ("TRASH", "filename:notes_refs", re.compile(r"(^|[/_-])(note|notes|nts|endnote|endnotes|footnote|footnotes|bib|biblio|bibliography|ref|refs|references|sources|selectedsources|source|credits|cre|permissions|anmerk|quellen|lit)([._-]|$)", re.I)),
    ("TRASH", "filename:index", re.compile(r"(^|[/_-])(idx|index|ind|register)(\d*)$", re.I)),
    ("TRASH", "filename:afterword_note", re.compile(r"(^|[/_-])(aft|after|afterword|ant|authornote|authorsnote|authors_note|postscript|nachwort)([._-]|$)", re.I)),
    ("TRASH", "filename:excerpt_guide", re.compile(r"(^|[/_-])(excerpt|preview|prevcover|prech\d+|teaser|sample|rgg|guide|bookclub|discussion|interview|qa|bonus|leseprobe)([._-]|$)", re.I)),
    ("TRASH", "filename:intro_preface", re.compile(r"(^|[/_-])(intro|itr|introduction|foreword|fwd|frw|pref|preface|vorwort|einleitung)([._-]|$)", re.I)),
    ("TRASH", "filename:appendix", re.compile(r"(^|[/_-])(app|appendix\d*|appendices|anhang)([._-]|$)", re.I)),
    ("KEEP_STORY", "filename:prologue", re.compile(r"(^|[/_-])(prol|prl|prologue|prolog)([._-]|$)", re.I)),
    ("KEEP_STORY", "filename:epilogue", re.compile(r"(^|[/_-])(epil|epl|epilogue|epilog)([._-]|$)", re.I)),
    ("KEEP_STORY", "filename:epigraph", re.compile(r"(^|[/_-])(epi|epig|epigraph|motto)([._-]|$)", re.I)),
]

EPUB_TYPE_MAP: dict[str, tuple[str, str]] = {
    "cover": ("COVER", "epub:type=cover"),
    "titlepage": ("TITLE_PAGE", "epub:type=titlepage"),
    "halftitlepage": ("TRASH", "epub:type=halftitlepage"),
    "copyright-page": ("TRASH", "epub:type=copyright-page"),
    "imprint": ("TRASH", "epub:type=imprint"),
    "colophon": ("TRASH", "epub:type=colophon"),
    "dedication": ("TRASH", "epub:type=dedication"),
    "toc": ("TRASH", "epub:type=toc"),
    "landmarks": ("TRASH", "epub:type=landmarks"),
    "loi": ("TRASH", "epub:type=loi"),
    "lot": ("TRASH", "epub:type=lot"),
    "glossary": ("TRASH", "epub:type=glossary"),
    "index": ("TRASH", "epub:type=index"),
    "bibliography": ("TRASH", "epub:type=bibliography"),
    "endnotes": ("TRASH", "epub:type=endnotes"),
    "rearnotes": ("TRASH", "epub:type=rearnotes"),
    "footnotes": ("TRASH", "epub:type=footnotes"),
    "notes": ("TRASH", "epub:type=notes"),
    "acknowledgments": ("TRASH", "epub:type=acknowledgments"),
    "acknowledgements": ("TRASH", "epub:type=acknowledgements"),
    "afterword": ("TRASH", "epub:type=afterword"),
    "contributors": ("TRASH", "epub:type=contributors"),
    "introduction": ("TRASH", "epub:type=introduction"),
    "foreword": ("TRASH", "epub:type=foreword"),
    "preface": ("TRASH", "epub:type=preface"),
    "appendix": ("TRASH", "epub:type=appendix"),
    "prologue": ("KEEP_STORY", "epub:type=prologue"),
    "epilogue": ("KEEP_STORY", "epub:type=epilogue"),
    "epigraph": ("KEEP_STORY", "epub:type=epigraph"),
}

GUIDE_TYPE_MAP: dict[str, tuple[str, str]] = {
    "cover": ("COVER", "guide=cover"),
    "title-page": ("TITLE_PAGE", "guide=title-page"),
    "toc": ("TRASH", "guide=toc"),
    "copyright-page": ("TRASH", "guide=copyright-page"),
    "dedication": ("TRASH", "guide=dedication"),
    "acknowledgements": ("TRASH", "guide=acknowledgements"),
    "bibliography": ("TRASH", "guide=bibliography"),
    "colophon": ("TRASH", "guide=colophon"),
    "glossary": ("TRASH", "guide=glossary"),
    "index": ("TRASH", "guide=index"),
    "loi": ("TRASH", "guide=loi"),
    "lot": ("TRASH", "guide=lot"),
    "notes": ("TRASH", "guide=notes"),
    "foreword": ("TRASH", "guide=foreword"),
    "preface": ("TRASH", "guide=preface"),
    "epigraph": ("KEEP_STORY", "guide=epigraph"),
}

HIGH_CONFIDENCE_LARGE_TRASH = {
    "notes_references", "bibliography", "index", "excerpt_promo", "also_by",
    "intro_foreword_preface", "appendix", "afterword_authors_note",
    "filename:notes_refs", "filename:excerpt_guide", "filename:intro_preface", "filename:appendix",
    "epub:type=endnotes", "epub:type=rearnotes", "epub:type=footnotes", "epub:type=notes",
    "epub:type=bibliography", "epub:type=index", "epub:type=introduction", "epub:type=foreword",
    "epub:type=preface", "epub:type=appendix", "epub:type=afterword",
    "guide=notes", "guide=bibliography", "guide=index", "guide=foreword", "guide=preface",
}

CASCADE_TRIGGER_REASONS = {
    "excerpt_promo", "filename:excerpt_guide",
    "about_author", "filename:about_author",
    "appendix", "filename:appendix", "epub:type=appendix",
    "notes_references", "filename:notes_refs",
    "epub:type=endnotes", "epub:type=rearnotes", "epub:type=footnotes", "epub:type=notes",
    "bibliography", "epub:type=bibliography", "guide=bibliography",
    "index", "filename:index", "epub:type=index", "guide=index",
    "also_by", "filename:also_by",
    "acknowledgements", "filename:acknowledgements", "epub:type=acknowledgments",
}

NONFICTION_KEEP_REASONS = {
    "intro_foreword_preface", "filename:intro_preface", "epub:type=introduction",
    "epub:type=foreword", "epub:type=preface", "guide=foreword", "guide=preface",
    "notes_references", "filename:notes_refs", "epub:type=endnotes", "epub:type=rearnotes",
    "epub:type=footnotes", "epub:type=notes", "guide=notes",
    "bibliography", "epub:type=bibliography", "guide=bibliography",
    "appendix", "filename:appendix", "epub:type=appendix",
    "maps_cast_glossary", "filename:maps_glossary", "epub:type=glossary",
    "epub:type=loi", "epub:type=lot", "guide=glossary", "guide=loi", "guide=lot",
    "index", "filename:index", "epub:type=index", "guide=index",
    "afterword_authors_note", "filename:afterword_note", "epub:type=afterword",
}

CALIBRE_SPLIT_RE = re.compile(r"^index_split_\d+$", re.I)
CHAPTER_FILE_RE = re.compile(r"(^|[/_-])(chapter|chap|ch\d|part|book)", re.I)
PRAISE_REVIEW_SIGNAL_RE = re.compile(
    r"\b(praise\s+for|acclaim\s+for|new\s+york\s+times|the\s+guardian|the\s+observer|sunday\s+times|"
    r"times\s+literary|publishers\s+weekly|kirkus|wall\s+street\s+journal|new\s+yorker|"
    r"washington\s+post|telegraph|independent|financial\s+times|bestseller|bestselling|"
    r"books?\s+to\s+(watch|look\s+out)|fiction\s+to\s+look\s+out|national\s+book\s+award|"
    r"pulitzer|booker\s+prize|about\s+the\s+book)\b",
    re.I,
)
PROMO_CONTENT_RE = re.compile(
    r"(读累了记得休息一会|\b(explore\s+book\s+giveaways|sneak\s+peeks|deals,\s+and\s+more|discover\s+more|"
    r"become\s+a\s+subscriber|other\s+ways\s+to\s+get\s+involved|"
    r"looking\s+for\s+your\s+next|what's\s+next\s+on\s+your\s+reading\s+list|"
    r"sign\s+up\s+for\s+(?:our|the|free|more|email|news|updates|offers|alerts)\b|"
    r"(?:subscribe\s+to|join|sign\s+up\s+(?:for|to)|receive)\s+(?:our|the|my|free\s+|weekly\s+|monthly\s+|email\s+){0,2}newsletter\b|"
    r"(?:our|the|weekly|monthly|free|email|author|publisher)\s+newsletter\b(?=.{0,80}(?:subscribe|sign\s+up|inbox|www\.|https?://|@|deals|updates|offers|exclusive))|"
    r"reading\s+group|book\s+club|discussion\s+questions|"
    r"about\s+the\s+author|find\s+a\s+full\s+list\s+of\s+our\s+authors|series\s+from\s+open\s+road|follow\s+penguin)\b)",
    re.I,
)
MIDBOOK_AD_RE = re.compile(
    r"(读累了记得休息一会|chenjin5\.com|"
    r"(?:https?://|www\.)\S+.{0,120}\b(?:newsletter|subscribe|giveaways|sneak\s+peeks|more\s+books|read\s+more|download)\b|"
    r"\b(?:newsletter|subscribe|giveaways|sneak\s+peeks|full\s+list\s+of\s+our\s+authors)\b.{0,120}(?:https?://|www\.))",
    re.I,
)
COPYRIGHT_CONTENT_RE = re.compile(
    r"(isbn|copyright\s*©|©\s*\d{4}|all\s+rights\s+reserved|library\s+of\s+congress|"
    r"first\s+published\s+in|first\s+\w+\s+edition|published\s+by\s+.{1,60}(www\.|http|email|press|books|publishing|edition))",
    re.I,
)
PUBLISHER_BIO_FRONT_RE = re.compile(
    r"^(penguin\s+(modern\s+)?classics|vintage\s+classics|oxford\s+world'?s\s+classics|everyman'?s\s+library)\b",
    re.I,
)
ACK_CONTENT_RE = re.compile(
    r"^(with\s+thanks\s+to|grateful\s+acknowledg|the\s+author\s+(wishes|would\s+like)\s+to\s+thank|thank\s+you\b|i\s+(would\s+like\s+to\s+)?thank\b)",
    re.I,
)
FOOTNOTE_LINK_TEXT_RE = re.compile(r"^[\[\(\{]?\s*(\d{1,4}|[ivxlcdm]{1,5}|[*†‡§¶]+)\s*[\]\)\}]?$", re.I)
CORRUPT_PLACEHOLDER_RE = re.compile(r"sorry[, ]+something\s+went\s+wrong\s+loading", re.I)


@dataclass
class TrimReport:
    raw_words: int = 0
    kept_words: int = 0
    trimmed_words: int = 0
    files_trimmed: int = 0
    cover_files_removed: int = 0
    title_files_removed: int = 0
    dead_footnotes_removed: int = 0
    dead_links_unwrapped: int = 0
    orphan_images_removed: int = 0
    cover_xhtml_rel: Optional[str] = None
    kept_spine_rel: list[str] = field(default_factory=list)
    trashed_details: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _compute_common_tokens(stems: list[str]) -> set[str]:
    if len(stems) < 4:
        return set()
    counts: dict[str, int] = {}
    for st in stems:
        for tok in {t.lower() for t in re.split(r"[/_-]+", st) if t and not t.isdigit()}:
            counts[tok] = counts.get(tok, 0) + 1
    threshold = max(3, int(len(stems) * 0.4))
    return {t for t, c in counts.items() if c >= threshold}


def _strip_common_tokens(name: str, common: set[str]) -> str:
    tokens = re.split(r"([/_-])", name)
    return "".join(t for t in tokens if t in ("/", "_", "-") or t.lower() not in common).strip("/_-")


def _apply_word_safeguard(verdict: str, reason: str, words: int) -> tuple[str, str]:
    if verdict == "TRASH" and words > 2500 and reason not in HIGH_CONFIDENCE_LARGE_TRASH:
        return ("KEEP_STORY", f"safeguard>2500w({reason})")
    return verdict, reason


_INTRO_KEYS = frozenset({"introduction", "foreword", "preface"})


def _classify_item(
    idx: int, total: int, rel_path: str, idref: str, props: list[str],
    epub_types: list[str], landmark: str, guide_type: str,
    toc_labels: list[str], headings: list[str], words: int,
    img_count: int, body_text: str, common_tokens: set[str], story_started: bool,
    author: str = "",
) -> tuple[str, str]:
    if "nav" in props:
        return ("TRASH", "manifest:nav")

    # If a chapter has BOTH a story epub:type (chapter/bodymatter/prologue) and an inline footnotes section,
    # always prefer KEEP_STORY over TRASH!
    matched_et = None
    for et in epub_types:
        if et in EPUB_TYPE_MAP and EPUB_TYPE_MAP[et][0] == "KEEP_STORY":
            matched_et = et
            break
    if matched_et is None:
        for et in epub_types:
            if et in EPUB_TYPE_MAP and not (story_started and et in _INTRO_KEYS):
                matched_et = et
                break
    if matched_et is not None:
        v, r = EPUB_TYPE_MAP[matched_et]
        v, r = _apply_word_safeguard(v, r, words)
    else:
        if landmark in EPUB_TYPE_MAP and not (story_started and landmark in _INTRO_KEYS):
            v, r = _apply_word_safeguard(*EPUB_TYPE_MAP[landmark], words)
        elif guide_type in GUIDE_TYPE_MAP and not (story_started and guide_type in _INTRO_KEYS):
            v, r = _apply_word_safeguard(*GUIDE_TYPE_MAP[guide_type], words)
        else:
            v, r = _classify_by_title_filename_content(
                idx, total, rel_path, idref, toc_labels, headings,
                words, img_count, body_text, common_tokens, story_started,
                author=author,
            )

    norm_b = norm_match_text(body_text)
    stem = Path(rel_path).stem

    # Penguin epub:type="preface" on short epigraph quotes
    if r == "epub:type=preface" and 0 < words < 80:
        if not re.match(r"^(preface|foreword|introduction)\b", norm_b, re.I) and re.search(r"[–—]\s*[A-Z]", norm_b):
            return ("KEEP_STORY", "override:preface_is_epigraph")

    # Sanity guards against shifted/broken Calibre TOCs
    if v in ("COVER", "TITLE_PAGE") and words > 100:
        if words < 900 and COPYRIGHT_CONTENT_RE.search(norm_b):
            return ("TRASH", "content:copyright_override_toc")
        return ("KEEP_STORY", "override:toc_cover_too_long")

    if r == "copyright" and words > 1500 and not COPYRIGHT_CONTENT_RE.search(norm_b[:600]):
        return ("KEEP_STORY", "override:toc_copyright_is_story")

    if v == "KEEP_STORY" and not story_started and words < 15 and idx <= 6 and re.match(r"^(for|to)\s+[A-Z]", norm_b, re.I):
        if "." not in norm_b:
            return ("TRASH", "content:dedication_override_toc")

    # Guard short dialogue chapters against front_praise_blurb false positives (allow Calibre generic partNNNN stems)
    if r == "content:front_praise_blurb":
        is_named_ch = bool(CHAPTER_FILE_RE.search(stem)) and not _GENERIC_PART_STEM_RE.match(stem)
        if story_started or is_named_ch or not PRAISE_REVIEW_SIGNAL_RE.search(norm_b):
            return ("KEEP_STORY", "default:story_body")

    return v, r


def _classify_by_title_filename_content(
    idx: int, total: int, rel_path: str, idref: str,
    toc_labels: list[str], headings: list[str], words: int,
    img_count: int, body_text: str, common_tokens: set[str], story_started: bool,
    author: str = "",
) -> tuple[str, str]:
    candidate_titles = []
    if toc_labels:
        candidate_titles.append(norm_match_text(toc_labels[0]))
    if headings:
        candidate_titles.append(norm_match_text(headings[0]))

    for t in candidate_titles:
        for label, pat in KEEP_STORY_TITLE_PATTERNS:
            if pat.search(t):
                return ("KEEP_STORY", f"title:{label}")

    for t in candidate_titles:
        for label, pat in TRASH_TITLE_PATTERNS:
            if label == "intro_foreword_preface" and story_started:
                continue
            if pat.search(t):
                if label == "cover":
                    return ("COVER", "title:cover")
                if label == "title_page":
                    return ("TITLE_PAGE", "title:title_page")
                return _apply_word_safeguard("TRASH", label, words)

    stem = Path(rel_path).stem
    clean_stem = "" if CALIBRE_SPLIT_RE.match(stem) else _strip_common_tokens(stem, common_tokens)
    clean_id = "" if CALIBRE_SPLIT_RE.match(idref) else _strip_common_tokens(idref, common_tokens)
    for tok in (clean_stem, clean_id):
        if not tok:
            continue
        for fv, fr, fpat in FILENAME_HINTS:
            if fr == "filename:intro_preface" and story_started:
                continue
            if fpat.search(tok):
                if fv == "TITLE_PAGE" and idx == 0 and words == 0 and img_count >= 1:
                    return ("COVER", "content:first_image_only")
                return _apply_word_safeguard(fv, fr, words)

    if words == 0 and img_count == 1 and idx == 0:
        return ("COVER", "content:first_image_only")
    if words == 0 and img_count == 0:
        return ("TRASH", "content:empty_page")

    norm_b = norm_match_text(body_text)
    low = norm_b.lower()

    if not story_started and idx <= 6 and words < 30 and re.search(r'\b(fuck\s+"?copyright"?|read\s+free\s+or\s+die)\b', low):
        return ("TRASH", "content:pirate_manifesto")
    if words < 40 and re.search(r"\bdocument\s+outline\b", low) and re.search(r"(?:local\s+disk|[a-z]:\\)", low):
        return ("TRASH", "content:converter_junk_stub")

    if not story_started and idx <= 12 and re.match(_INTRO_FOREWORD_PAT, norm_b[:80], re.I):
        return _apply_word_safeguard("TRASH", "intro_foreword_preface", words)
    if not story_started and idx <= 12 and words < 450 and re.match(r"^(?:advance\s+)?(?:praise|acclaim)\s+for\b", norm_b[:80], re.I):
        return ("TRASH", "praise")
    if not story_started and idx <= 12 and words < 400:
        if _FRONT_CONTRIBUTOR_BIO_RE.search(norm_b):
            return ("TRASH", "about_author")
    if not story_started and idx <= 12 and words < 250 and _FRONT_BIO_ROLE_RE.search(norm_b):
        author_toks = [w for w in re.findall(r"[a-z]{3,}", _strip_accents(author).lower())]
        norm_ascii = _strip_accents(norm_b).lower()
        if (
            re.search(r"\b(?:pseudonym|was\s+born\s+in|born\s+in|professor|author\s+of)\b", norm_ascii)
            or any(tok in norm_ascii for tok in author_toks)
        ):
            return ("TRASH", "about_author")
    if not story_started and idx <= 8 and words < 2500 and re.match(r"^(dramatis\s+personae|cast\s+of\s+characters|principal\s+characters|list\s+of\s+characters)\b", norm_b, re.I):
        return ("TRASH", "maps_cast_glossary")
    if not story_started and words < 250 and idx <= 12 and re.match(r"^(contents|table\s+of\s+contents)\b", norm_b, re.I):
        return ("TRASH", "content:inline_html_toc")
    if not story_started and words < 500 and idx <= 6 and PUBLISHER_BIO_FRONT_RE.search(norm_b):
        return ("TRASH", "content:publisher_series_bio")
    if not story_started and words < 300 and idx <= 12 and (
        ACK_CONTENT_RE.search(norm_b) or re.match(r"^acknowledg(?:e)?ments?\b", norm_b[:40], re.I)
    ):
        return ("TRASH", "content:front_acknowledgements")
    if not story_started and words < 300 and idx <= 12 and re.match(r"^(?:author'?s?\s+note|translator'?s?\s+note)\b", norm_b[:40], re.I):
        return ("TRASH", "afterword_authors_note")
    is_front_copyright_blob = (not story_started and idx <= 6 and words < 1200 and bool(COPYRIGHT_CONTENT_RE.search(norm_b[:400])))
    if (words < 600 and COPYRIGHT_CONTENT_RE.search(norm_b)) or is_front_copyright_blob:
        if ("isbn" in low or "copyright" in low or "©" in norm_b or "all rights reserved" in low or "verlag" in low):
            return ("TRASH", "content:copyright_notice")
    if words < 250 and re.search(r"\b(also by|other books by|books by|other titles)\b", low):
        return ("TRASH", "content:also_by_list")
    if idx >= total - 12 and words < 800:
        if re.match(
            r"^(?:acknowledg(?:e)?ments?(?:\s+and\s+(?:bibliography|notes|sources|credits))?|notes\s+and\s+acknowledg(?:e)?ments?)\b",
            norm_b[:60],
            re.I,
        ):
            return ("TRASH", "acknowledgements")
        if re.match(r"^(?:also\s+by|other\s+books\s+by|books\s+by)\b", norm_b[:60], re.I):
            return ("TRASH", "also_by")
    if words < 400:
        if story_started and idx < total - 8:
            if MIDBOOK_AD_RE.search(norm_b):
                return ("TRASH", "content:promo_or_bio")
        elif PROMO_CONTENT_RE.search(norm_b):
            return ("TRASH", "content:promo_or_bio")
    if not story_started and words < 40 and idx <= 12:
        if re.match(r"^(for|to)\s+\S|^(this\s+book\s+is\s+for|in\s+memory\s+of|in\s+loving\s+memory|dedicated\s+to)\b", norm_b, re.I):
            return ("TRASH", "content:dedication")
    if words < 300 and idx >= total - 12 and re.search(r"^thank you\b|^i (would like to )?thank\b", low):
        return ("TRASH", "content:acknowledgements")
    if words < 350 and idx >= total - 6 and _BACK_AUTHOR_BIO_RE.search(norm_b):
        if not any(re.match(r"^(chapter|kapitel|part|book|epilogue|epilog|\d+)\b", h.strip(), re.I) for h in headings):
            return ("TRASH", "about_author")
    if words < 10 and idx >= total - 3 and low in ("the end", "fin", "ende"):
        return ("TRASH", "content:the_end_stub")

    if words < 15 and img_count >= 1:
        if idx == 0:
            return ("COVER", "content:front_cover_img")
        if not story_started and idx <= 12:
            return ("TRASH", "content:front_promo_or_logo_img")
        if idx >= total - 6:
            return ("TRASH", "content:back_promo_or_logo_img")

    if not story_started and idx <= 12 and words < 450 and (not CHAPTER_FILE_RE.search(stem) or _GENERIC_PART_STEM_RE.match(stem)):
        if PRAISE_REVIEW_SIGNAL_RE.search(norm_b):
            return ("TRASH", "content:front_praise_blurb")

    if not story_started and idx <= 12 and 0 < words <= 20 and re.search(r"(^|[/_-])fm(\d*|[._-])", clean_stem, re.I):
        if "." not in norm_b:
            return ("TITLE_PAGE", "content:front_fm_title")

    return ("KEEP_STORY", "default:story_body")


def is_footnote_link(a_el: etree._Element, target_reason: str) -> bool:
    for k, v in a_el.attrib.items():
        if (k.endswith("type") or k == "role") and "noteref" in v.lower():
            return True
    anc = a_el.getparent()
    while anc is not None:
        if local_name(anc.tag) in ("h1", "h2", "h3", "h4"):
            return False
        anc = anc.getparent()
    parent = a_el.getparent()
    parent_ln = local_name(parent.tag) if parent is not None else ""
    if parent_ln == "sup" or any(local_name(c.tag) == "sup" for c in a_el):
        return True
    cls = (a_el.get("class") or "").lower()
    parent_cls = (parent.get("class") or "").lower() if parent is not None else ""
    is_notes = any(k in target_reason for k in ("note", "endnote", "footnote", "rearnotes", "ref", "bib"))
    txt = extract_clean_text(a_el).strip()
    if is_notes and (FOOTNOTE_LINK_TEXT_RE.match(txt) or any(k in cls or k in parent_cls for k in ("note", "fn", "ref", "sup"))):
        return True
    return False


def _remove_element_keep_tail(el: etree._Element) -> None:
    parent = el.getparent()
    if parent is None:
        return
    if el.tail:
        prev = el.getprevious()
        if prev is not None:
            prev.tail = (prev.tail or "") + el.tail
        else:
            parent.text = (parent.text or "") + el.tail
    parent.remove(el)


def _unwrap_element(el: etree._Element) -> None:
    parent = el.getparent()
    if parent is None:
        return
    idx = list(parent).index(el)
    if el.text:
        if idx > 0:
            prev = parent[idx - 1]
            prev.tail = (prev.tail or "") + el.text
        else:
            parent.text = (parent.text or "") + el.text
    for child in list(el):
        el.remove(child)
        parent.insert(idx, child)
        idx += 1
    if el.tail:
        if idx > 0:
            prev = parent[idx - 1]
            prev.tail = (prev.tail or "") + el.tail
        else:
            parent.text = (parent.text or "") + el.tail
    parent.remove(el)


def trim_epub_matter(
    epub_dir: str,
    opf_path: str,
    preset: str = "story",
    original_filename: str = "",
) -> TrimReport:
    """
    Classify all spine documents, trim non-story matter according to `preset`
    ('story' vs 'nonfiction'), clean dead footnote markers vs nav links,
    prune orphaned images before image processing, and update OPF manifest/spine/guide.
    """
    report = TrimReport()
    opf_tree = etree.parse(opf_path, xml_parser())
    opf_root = opf_tree.getroot()
    opf_dir = str(Path(opf_path).parent)

    manifest = _find_element(opf_root, "manifest")
    spine = _find_element(opf_root, "spine")
    if manifest is None or spine is None:
        return report

    # Extract title/author for duplicate title-page detection & mismatch warning
    title, author = "", ""
    for el in opf_root.iter():
        ln = local_name(el.tag)
        if ln == "title" and not title and (el.text or "").strip():
            title = el.text.strip()
        elif ln == "creator" and not author and (el.text or "").strip():
            author = el.text.strip()

    if original_filename and Path(original_filename).stem.lower() not in ("in", "out", "book", "optimized", "original", "upload"):
        fn_low = Path(original_filename).stem.lower()
        t_toks = [w for w in re.findall(r"[a-z]{4,}", title.lower()) if w not in ("from", "with", "that", "this", "into", "about", "novel", "stories", "book")]
        if t_toks and not any(t in fn_low for t in t_toks):
            report.warnings.append(f"Filename '{original_filename}' does not match book title '{title}'")

    # Build manifest maps
    id_to_item: dict[str, etree._Element] = {}
    nav_rel: Optional[str] = None
    ncx_rel: Optional[str] = None
    cover_img_rel: Optional[str] = None

    for item in manifest:
        if not _is_element(item):
            continue
        iid = item.get("id", "")
        href = unquote(item.get("href", ""))
        props = (item.get("properties") or "").split()
        mtype = (item.get("media-type") or "").lower()
        id_to_item[iid] = item
        if "nav" in props:
            nav_rel = posixpath.normpath(href.replace("\\", "/"))
        if mtype == "application/x-dtbncx+xml":
            ncx_rel = posixpath.normpath(href.replace("\\", "/"))
        if "cover-image" in props:
            cover_img_rel = posixpath.normpath(href.replace("\\", "/"))

    # Also check EPUB2 <meta name="cover" content="...">
    if not cover_img_rel:
        for meta in opf_root.iter():
            if local_name(meta.tag) == "meta" and meta.get("name") == "cover":
                cid = meta.get("content", "")
                if cid in id_to_item:
                    cover_img_rel = posixpath.normpath(unquote(id_to_item[cid].get("href", "")).replace("\\", "/"))
                    break

    # Guide map
    guide_by_rel: dict[str, str] = {}
    for ref in opf_root.iter():
        if local_name(ref.tag) == "reference":
            gtype = (ref.get("type") or "").lower().strip()
            ghref = _normalize_rel("", ref.get("href") or "")
            if ghref and gtype:
                guide_by_rel[ghref] = gtype

    # TOC labels & landmarks
    toc_labels_by_rel: dict[str, list[str]] = {}
    landmarks_by_rel: dict[str, str] = {}

    if ncx_rel:
        ncx_abs = os.path.join(opf_dir, ncx_rel)
        if os.path.exists(ncx_abs):
            try:
                ncx_root, _ = parse_xhtml_safe(Path(ncx_abs).read_bytes())
                ncx_dir = posixpath.dirname(ncx_rel)
                for np in ncx_root.iter():
                    if local_name(np.tag) == "navpoint":
                        lbl, src = "", ""
                        for ch in np:
                            if local_name(ch.tag) == "navlabel":
                                lbl = extract_clean_text(ch)
                            elif local_name(ch.tag) == "content":
                                src = ch.get("src", "")
                        if src:
                            toc_labels_by_rel.setdefault(_normalize_rel(ncx_dir, src), []).append(lbl)
            except Exception:
                pass

    if nav_rel:
        nav_abs = os.path.join(opf_dir, nav_rel)
        if os.path.exists(nav_abs):
            try:
                nav_root, _ = parse_xhtml_safe(Path(nav_abs).read_bytes())
                nav_dir = posixpath.dirname(nav_rel)
                for nav_el in nav_root.iter():
                    if local_name(nav_el.tag) == "nav":
                        ntype = ""
                        for k, v in nav_el.attrib.items():
                            if k.endswith("type"):
                                ntype = v.lower()
                        for a in nav_el.iter():
                            if local_name(a.tag) == "a" and a.get("href"):
                                p = _normalize_rel(nav_dir, a.get("href"))
                                txt = extract_clean_text(a)
                                if "toc" in ntype:
                                    toc_labels_by_rel.setdefault(p, []).append(txt)
                                elif "landmarks" in ntype:
                                    # Fragment landmarks (e.g. chapter1.xhtml#footnotes) target sub-sections, never whole files!
                                    if "#" not in (a.get("href") or ""):
                                        for ak, av in a.attrib.items():
                                            if ak.endswith("type"):
                                                landmarks_by_rel[p] = av.lower().strip()
            except Exception:
                pass

    # Collect spine items
    spine_entries: list[tuple[etree._Element, str, str, list[str]]] = []
    for itemref in spine:
        if not _is_element(itemref):
            continue
        idref = itemref.get("idref", "")
        item = id_to_item.get(idref)
        if item is None:
            continue
        rel = posixpath.normpath(unquote(item.get("href", "")).replace("\\", "/"))
        props = (item.get("properties") or "").split()
        spine_entries.append((itemref, idref, rel, props))

    common_tokens = _compute_common_tokens([Path(rel).stem for _, _, rel, _ in spine_entries])

    # Pass 1: Stream one XHTML file at a time (Render 512MB RAM cap!)
    classified: list[dict] = []
    story_started = False
    corrupt_count = 0
    missing_count = 0

    for idx, (itemref, idref, rel, props) in enumerate(spine_entries):
        abs_path = os.path.join(opf_dir, rel)
        if not os.path.exists(abs_path):
            missing_count += 1
            continue

        raw_bytes = Path(abs_path).read_bytes()
        root, _ = parse_xhtml_safe(raw_bytes)
        body = get_body(root)
        body_text = extract_clean_text(body)
        words = count_words(body_text)
        report.raw_words += words

        if CORRUPT_PLACEHOLDER_RE.search(body_text):
            corrupt_count += 1

        img_count = 0
        img_refs: set[str] = set()
        link_targets: set[str] = set()
        rel_dir = posixpath.dirname(rel)
        headings = []
        for el in body.iter():
            ln = local_name(el.tag)
            if ln == "img":
                img_count += 1
                src = el.get("src")
                if src:
                    img_refs.add(_normalize_rel(rel_dir, src))
            elif ln == "image":
                img_count += 1
                href_v = el.get(f"{{{NS_XLINK}}}href") or el.get("href") or ""
                if href_v:
                    img_refs.add(_normalize_rel(rel_dir, href_v))
            elif ln == "a":
                ahref = (el.get("href") or "").strip()
                if ahref and not ahref.startswith("#") and not ahref.lower().startswith(("http://", "https://", "mailto:", "www.")):
                    link_targets.add(_normalize_rel(rel_dir, ahref))
            elif ln in ("h1", "h2", "h3"):
                ht = extract_clean_text(el)
                if ht:
                    headings.append(ht[:120])
            style = el.get("style")
            if style and "url(" in style:
                for m in re.findall(r"url\(([^)]+)\)", style):
                    u = m.strip("'\" ")
                    if u:
                        img_refs.add(_normalize_rel(rel_dir, u))

        epub_types = []
        body_el_children = [c for c in body if _is_element(c)]
        top_els = [body] + (body_el_children[:1] if len(body_el_children) == 1 else [])
        for el in top_els:
            if local_name(el.tag) == "aside":
                continue
            for k, v in el.attrib.items():
                if (k.endswith("type") and "epub" in k.lower()) or k == f"{{{NS_EPUB}}}type":
                    epub_types.extend(v.lower().split())

        verdict, reason = _classify_item(
            idx=idx, total=len(spine_entries), rel_path=rel, idref=idref,
            props=props, epub_types=epub_types,
            landmark=landmarks_by_rel.get(rel, ""),
            guide_type=guide_by_rel.get(rel, ""),
            toc_labels=toc_labels_by_rel.get(rel, []),
            headings=headings, words=words, img_count=img_count,
            body_text=body_text, common_tokens=common_tokens,
            story_started=story_started, author=author,
        )

        if verdict == "KEEP_STORY" and words >= 500:
            story_started = True

        classified.append({
            "idx": idx, "itemref": itemref, "idref": idref, "rel": rel,
            "words": words, "preview": norm_match_text(body_text)[:220],
            "verdict": verdict, "reason": reason,
            "img_refs": img_refs, "link_targets": link_targets,
        })

    if corrupt_count:
        report.warnings.append(
            f"{corrupt_count} chapter(s) contain placeholder error text ('sorry something went wrong loading...')"
        )
    if missing_count:
        report.warnings.append(f"{missing_count} spine file(s) missing from EPUB archive")
    if report.raw_words < 200:
        report.warnings.append(
            f"Source EPUB contains almost no text ({report.raw_words} words) — file may be corrupt or image-only"
        )

    # Duplicate front title/half-title detection before first >=200w story file (diacritic-insensitive)
    main_title = re.sub(r",\s*(?:the|a|an)$", "", title.split(":", 1)[0].strip(), flags=re.I)
    ascii_title = _strip_accents(main_title).lower()
    ascii_author = _strip_accents(author).lower()
    opf_t = [w for w in re.findall(r"[a-z]{3,}", ascii_title)]
    opf_a = [w for w in re.findall(r"[a-z]{3,}", ascii_author)]
    compact_t = re.sub(r"[^a-z0-9]+", "", ascii_title)
    compact_a = re.sub(r"[^a-z0-9]+", "", ascii_author)
    pub_marker_re = re.compile(
        r"(originally\s+published|translated\s+(?:by|from)|www\.|https?://|press\b|books\b|farrar|straus|giroux|"
        r"digitalized\s+by|new\s+york\b|london\b|editions\b|@)",
        re.I,
    )
    for entry in classified:
        if entry["verdict"] == "KEEP_STORY":
            if entry["words"] >= 200:
                break
            if 0 < entry["words"] <= 35:
                low = _strip_accents(norm_match_text(entry["preview"])).lower()
                if re.search(r"<[^>]+>|\\|\.exe\b", low):
                    continue
                no_init = re.sub(r"\b[a-z]\.", "", low)
                compact = re.sub(r"[^a-z0-9]+", "", low)
                t_hit = (bool(opf_t) and all(w in low for w in opf_t[:3])) or (
                    len(compact_t) >= 4 and compact_t in compact
                )
                a_hit = (bool(opf_a) and all(w in low for w in opf_a[:2])) or (
                    len(compact_a) >= 4 and compact_a in compact
                )
                has_pub_marker = bool(pub_marker_re.search(low))
                if (t_hit and (a_hit or has_pub_marker)) or (entry["words"] <= 12 and (t_hit or a_hit)):
                    if "." not in no_init or has_pub_marker:
                        entry["verdict"] = "TITLE_PAGE"
                        entry["reason"] = "content:dup_title_front"

    # Back-matter cascade (never cascade over an explicit Epilogue or mid-book chapter)
    _EXCERPT_OR_NOTES_TRIGGERS = {
        "excerpt_promo", "filename:excerpt_guide",
        "notes_references", "filename:notes_refs",
        "epub:type=endnotes", "epub:type=rearnotes", "epub:type=footnotes", "epub:type=notes",
        "guide=notes",
    }
    total_story_words = sum(e["words"] for e in classified if e["verdict"] == "KEEP_STORY")
    if total_story_words >= 1000:
        seen = 0
        seen_epilogue = False
        in_cascade = False
        trigger = ""
        for entry in classified:
            if entry["reason"] in ("title:epilogue", "filename:epilogue", "epub:type=epilogue"):
                seen_epilogue = True
                in_cascade = False
                seen += entry["words"]
                continue
            if not in_cascade:
                if entry["verdict"] == "KEEP_STORY":
                    seen += entry["words"]
                elif (
                    seen >= total_story_words * 0.65
                    and entry["reason"] in CASCADE_TRIGGER_REASONS
                    and entry["reason"] not in ("content:promo_or_bio", "filename:midbook_ad")
                ):
                    in_cascade = True
                    trigger = entry["reason"]
            else:
                if entry["verdict"] == "TRASH" and entry["reason"] in _EXCERPT_OR_NOTES_TRIGGERS:
                    trigger = entry["reason"]
                elif entry["verdict"] == "KEEP_STORY":
                    is_named_ch = (
                        entry["reason"] == "title:chapter"
                        or bool(re.search(r"(?:^|[/_-])(?:c\d+|ch\d+|chapter[_-]?\d+)(?:[._-]|$)", Path(entry["rel"]).stem, re.I))
                    )
                    if (
                        not seen_epilogue
                        and is_named_ch
                        and entry["words"] >= 500
                        and trigger not in _EXCERPT_OR_NOTES_TRIGGERS
                    ):
                        in_cascade = False
                        seen += entry["words"]
                        continue
                    entry["verdict"] = "TRASH"
                    entry["reason"] = f"cascade_after:{trigger}"

    # Nonfiction mode ("Keep Notes & Intro"): restore kept scholarly sections
    if preset == "nonfiction":
        for entry in classified:
            if entry["verdict"] == "TRASH" and (
                entry["reason"] in NONFICTION_KEEP_REASONS
                or entry["reason"].startswith("cascade_after:")
            ):
                # Still trash pure promo/ad/about-author/copyright even in nonfiction mode
                if entry["reason"] not in ("excerpt_promo", "about_author", "also_by", "copyright"):
                    entry["verdict"] = "KEEP_STORY"
                    entry["reason"] = f"nonfiction_keep:{entry['reason']}"

    # Keep at most ONE cover XHTML (first COVER); remove old TITLE_PAGEs (house_style builds clean one)
    seen_cover = False
    for entry in classified:
        if entry["verdict"] == "COVER":
            if not seen_cover:
                seen_cover = True
                report.cover_xhtml_rel = entry["rel"]
            else:
                entry["verdict"] = "TRASH"
                entry["reason"] = "duplicate_cover"
                report.cover_files_removed += 1
        elif entry["verdict"] == "TITLE_PAGE":
            entry["verdict"] = "TRASH"
            entry["reason"] = f"replaced_title_page({entry['reason']})"
            report.title_files_removed += 1

    # Safety: never delete ALL story files (if everything got trashed, restore largest)
    kept_story = [e for e in classified if e["verdict"] == "KEEP_STORY"]
    if not kept_story and classified:
        biggest = max(classified, key=lambda e: e["words"])
        biggest["verdict"] = "KEEP_STORY"
        biggest["reason"] = "safety:no_story_fallback"
        kept_story = [biggest]

    trashed_reason_by_rel = {e["rel"]: e["reason"] for e in classified if e["verdict"] == "TRASH"}
    kept_rels = [e["rel"] for e in classified if e["verdict"] in ("COVER", "KEEP_STORY")]
    report.kept_spine_rel = [e["rel"] for e in classified if e["verdict"] == "KEEP_STORY"]

    # Pass 2: Clean dead footnote markers vs dead nav links ONLY in KEPT XHTML files that actually reference a trashed file
    if trashed_reason_by_rel:
        trashed_set = set(trashed_reason_by_rel.keys())
        kept_needing_link_fix = [
            e["rel"] for e in classified
            if e["verdict"] in ("COVER", "KEEP_STORY") and (e["link_targets"] & trashed_set)
        ]
        for rel in kept_needing_link_fix:
            abs_path = os.path.join(opf_dir, rel)
            if not os.path.exists(abs_path):
                continue
            root, _ = parse_xhtml_safe(Path(abs_path).read_bytes())
            rel_dir = posixpath.dirname(rel)
            modified = False
            for a in list(root.iter()):
                if local_name(a.tag) != "a":
                    continue
                href = (a.get("href") or "").strip()
                if not href or href.startswith("#") or re.match(r"^(https?://|mailto:|www\.)", href, re.I):
                    continue
                target_rel = _normalize_rel(rel_dir, href)
                if target_rel in trashed_reason_by_rel:
                    if is_footnote_link(a, trashed_reason_by_rel[target_rel]):
                        parent = a.getparent()
                        _remove_element_keep_tail(a)
                        if parent is not None and local_name(parent.tag) == "sup":
                            if not (parent.text or "").strip() and len(parent) == 0:
                                _remove_element_keep_tail(parent)
                        report.dead_footnotes_removed += 1
                    else:
                        _unwrap_element(a)
                        report.dead_links_unwrapped += 1
                    modified = True
            if modified:
                Path(abs_path).write_bytes(
                    etree.tostring(root, xml_declaration=True, encoding="utf-8", pretty_print=True)
                )

    # Pass 3: Delete trashed XHTML files and remove from OPF manifest/spine/guide
    kept_rels_set = set(kept_rels)
    kept_ids_set = {e["idref"] for e in classified if e["verdict"] in ("COVER", "KEEP_STORY")}
    trashed_ids = set()
    for entry in classified:
        if entry["verdict"] == "TRASH":
            rel = entry["rel"]
            report.trimmed_words += entry["words"]
            report.files_trimmed += 1
            report.trashed_details.append({
                "file": rel, "words": entry["words"], "reason": entry["reason"]
            })
            trashed_ids.add(entry["idref"])
            # Always remove the trashed spine itemref
            if entry["itemref"].getparent() is not None:
                entry["itemref"].getparent().remove(entry["itemref"])
            # Only remove from manifest & disk if NO kept spine entry still uses this idref/file!
            if entry["idref"] not in kept_ids_set:
                item = id_to_item.get(entry["idref"])
                if item is not None and item.getparent() is not None:
                    item.getparent().remove(item)
            if rel not in kept_rels_set:
                abs_path = os.path.join(opf_dir, rel)
                if os.path.exists(abs_path):
                    os.unlink(abs_path)

    # Clean <guide> references pointing to deleted files
    guide_el = _find_element(opf_root, "guide")
    if guide_el is not None:
        for ref in list(guide_el):
            if _is_element(ref):
                ghref = _normalize_rel("", ref.get("href") or "")
                if ghref in trashed_reason_by_rel:
                    guide_el.remove(ref)

    # Pass 4: Prune orphaned images ONLY referenced by deleted matter (reuse Pass 1 img_refs — no 3rd disk read!)
    kept_img_refs: set[str] = set()
    for e in classified:
        if e["verdict"] in ("COVER", "KEEP_STORY"):
            kept_img_refs.update(e["img_refs"])
    report.orphan_images_removed = _prune_unreferenced_images(
        opf_root, opf_dir, kept_rels, cover_img_rel, precollected_img_refs=kept_img_refs
    )

    opf_tree.write(opf_path, xml_declaration=True, encoding="utf-8", pretty_print=True)
    report.kept_words = sum(e["words"] for e in classified if e["verdict"] == "KEEP_STORY")
    return report


def _prune_unreferenced_images(
    opf_root: etree._Element,
    opf_dir: str,
    kept_xhtml_rels: list[str],
    cover_img_rel: Optional[str],
    precollected_img_refs: Optional[set[str]] = None,
) -> int:
    """Delete images in manifest that are no longer referenced by any kept XHTML or CSS file."""
    referenced_rels: set[str] = set()
    if cover_img_rel:
        referenced_rels.add(cover_img_rel)

    if precollected_img_refs is not None:
        referenced_rels.update(precollected_img_refs)
    else:
        # Collect image refs from kept XHTML files
        for rel in kept_xhtml_rels:
            abs_path = os.path.join(opf_dir, rel)
            if not os.path.exists(abs_path):
                continue
            rel_dir = posixpath.dirname(rel)
            root, _ = parse_xhtml_safe(Path(abs_path).read_bytes())
            referenced_rels.update(_collect_doc_img_refs(root, rel_dir))

    # Collect image refs from CSS files
    manifest = _find_element(opf_root, "manifest")
    if manifest is None:
        return 0

    for item in manifest:
        if not _is_element(item):
            continue
        if (item.get("media-type") or "").lower() == "text/css":
            css_rel = posixpath.normpath(unquote(item.get("href", "")).replace("\\", "/"))
            css_abs = os.path.join(opf_dir, css_rel)
            if os.path.exists(css_abs):
                css_dir = posixpath.dirname(css_rel)
                txt = Path(css_abs).read_text(encoding="utf-8", errors="ignore")
                for m in re.findall(r"url\(([^)]+)\)", txt):
                    u = m.strip("'\" ")
                    if u:
                        referenced_rels.add(_normalize_rel(css_dir, u))

    removed = 0
    for item in list(manifest):
        if not _is_element(item):
            continue
        mtype = (item.get("media-type") or "").lower()
        if not mtype.startswith("image/"):
            continue
        props = (item.get("properties") or "").split()
        if "cover-image" in props:
            continue
        img_rel = posixpath.normpath(unquote(item.get("href", "")).replace("\\", "/"))
        if img_rel not in referenced_rels and Path(img_rel).name not in {Path(r).name for r in referenced_rels}:
            abs_img = os.path.join(opf_dir, img_rel)
            if os.path.exists(abs_img):
                os.unlink(abs_img)
            manifest.remove(item)
            removed += 1

    return removed
