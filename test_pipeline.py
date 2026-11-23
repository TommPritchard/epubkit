"""End-to-end pipeline test: build a real EPUB, run it through process_epub.

Covers the image path (Pillow), the XHTML path (lxml) and the upload path
(python-multipart), so a dependency bump that breaks any of them fails here
rather than in production.
"""
import os
import tempfile
import zipfile

import pytest

from PIL import Image, ImageDraw

from epub_processor import process_epub, ProcessingOptions

OPF = '''<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="id">urn:uuid:test</dc:identifier>
    <dc:title>Der Verkäufer</dc:title>
    <dc:creator>Käthe Müller</dc:creator>
    <dc:language>de</dc:language>
    <dc:date>1965-01-01</dc:date>
  </metadata>
  <manifest>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
    <item id="c1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
    <item id="cover" href="cover.png" media-type="image/png" properties="cover-image"/>
    <item id="img1" href="plate.jpg" media-type="image/jpeg"/>
    <item id="css" href="style.css" media-type="text/css"/>
  </manifest>
  <spine><itemref idref="c1"/></spine>
</package>'''

NAV = ('<?xml version="1.0" encoding="utf-8"?><html xmlns="http://www.w3.org/1999/xhtml" '
       'xmlns:epub="http://www.idpf.org/2007/ops"><body><nav epub:type="toc">'
       '<ol><li><a href="ch1.xhtml">Kapitel 1</a></li></ol></nav></body></html>')

# Deliberately malformed (unclosed <br>) so the recovery parser runs.
CH1 = ('<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Kapitel</title>'
       '<link rel="stylesheet" href="style.css"/></head><body>'
       '<h1 class="kapitelüberschrift">Kapitel 1</h1>'
       '<p>Während ihre Schwester die Lehre als Verkäuferin abgebrochen hatte, '
       'trug sie Kopfhörer.<br></p>'
       '<p>»Wie geht’s Mutter übrigens?«</p>'
       '<img src="plate.jpg" alt="Tafel"/>'
       '</body></html>')

CSS = '.kapitelüberschrift { color: #333; } .unused-rule { color: red; }'


def build_epub(path):
    tmp = tempfile.mkdtemp()
    cover = Image.new('RGB', (900, 1400), (200, 60, 60))
    ImageDraw.Draw(cover).rectangle([50, 50, 850, 400], fill=(20, 20, 120))
    cover_path = os.path.join(tmp, 'cover.png')
    cover.save(cover_path)

    plate = Image.new('RGB', (1600, 1200), (30, 160, 90))
    ImageDraw.Draw(plate).ellipse([100, 100, 1500, 1100], fill=(240, 240, 30))
    plate_path = os.path.join(tmp, 'plate.jpg')
    plate.save(plate_path, quality=95)

    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('mimetype', 'application/epub+zip')
        z.writestr('META-INF/container.xml',
                   '<?xml version="1.0"?><container version="1.0" '
                   'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                   '<rootfiles><rootfile full-path="OEBPS/content.opf" '
                   'media-type="application/oebps-package+xml"/></rootfiles></container>')
        z.writestr('OEBPS/content.opf', OPF)
        z.writestr('OEBPS/nav.xhtml', NAV)
        z.writestr('OEBPS/ch1.xhtml', CH1)
        z.writestr('OEBPS/style.css', CSS)
        z.write(cover_path, 'OEBPS/cover.png')
        z.write(plate_path, 'OEBPS/plate.jpg')

def _process():
    work = tempfile.mkdtemp()
    src = os.path.join(work, 'in.epub')
    dst = os.path.join(work, 'out.epub')
    build_epub(src)
    report = process_epub(src, dst, ProcessingOptions())
    return src, dst, report


def test_pipeline_produces_a_smaller_epub():
    src, dst, report = _process()
    assert report.success, report.error
    assert os.path.exists(dst)
    assert os.path.getsize(dst) < os.path.getsize(src)


def test_pipeline_keeps_images():
    _, dst, _ = _process()
    with zipfile.ZipFile(dst) as z:
        images = [n for n in z.namelist()
                  if n.lower().endswith(('.jpg', '.jpeg', '.png'))]
    assert images


def test_pipeline_keeps_text_intact():
    _, dst, _ = _process()
    with zipfile.ZipFile(dst) as z:
        chapter = next(z.read(n).decode('utf-8', 'replace')
                       for n in z.namelist() if n.endswith('ch1.xhtml'))
    assert 'Verkäuferin' in chapter
    assert 'Ã' not in chapter


def test_upload_endpoint_accepts_multipart(monkeypatch, tmp_path):
    pytest.importorskip('httpx', reason='TestClient needs httpx')
    from fastapi.testclient import TestClient
    import app as app_module

    monkeypatch.setattr(app_module, 'CONVERTER_API_KEY', 'test-secret')
    monkeypatch.setattr(app_module, 'UPLOAD_DIR', tmp_path / 'uploads')
    monkeypatch.setattr(app_module, 'OUTPUT_DIR', tmp_path / 'outputs')
    app_module.tasks.clear()

    src = str(tmp_path / 'in.epub')
    build_epub(src)

    with TestClient(app_module.app) as client:
        with open(src, 'rb') as fh:
            response = client.post(
                '/upload',
                headers={'X-API-Key': 'test-secret'},
                files={'files': ('Der Verkäufer.epub', fh, 'application/epub+zip')},
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload['files'][0]['metadata']['title'] == 'Der Verkäufer'


def test_auth_fail_closed_when_key_unset(monkeypatch):
    pytest.importorskip('httpx', reason='TestClient needs httpx')
    from fastapi.testclient import TestClient
    import app as app_module

    monkeypatch.setattr(app_module, 'CONVERTER_API_KEY', '')

    with TestClient(app_module.app) as client:
        assert client.get('/health').status_code == 200
        assert client.get('/').status_code == 404
        assert client.post('/upload').status_code == 404


def test_auth_rejects_invalid_and_accepts_valid_key(monkeypatch):
    pytest.importorskip('httpx', reason='TestClient needs httpx')
    from fastapi.testclient import TestClient
    import app as app_module

    monkeypatch.setattr(app_module, 'CONVERTER_API_KEY', 'super-secret')

    with TestClient(app_module.app) as client:
        assert client.get('/').status_code == 404
        assert client.get('/', headers={'X-API-Key': 'wrong'}).status_code == 404
        assert client.get('/', headers={'X-API-Key': 'super-secret'}).status_code == 200
        # Temporary MP-04 compatibility: ?api_key= query param
        assert client.get('/?api_key=super-secret').status_code == 200


def test_upload_sanitizes_path_traversal_filename(monkeypatch, tmp_path):
    pytest.importorskip('httpx', reason='TestClient needs httpx')
    from pathlib import Path
    from fastapi.testclient import TestClient
    import app as app_module

    upload_dir = tmp_path / 'uploads'
    monkeypatch.setattr(app_module, 'CONVERTER_API_KEY', 'test-secret')
    monkeypatch.setattr(app_module, 'UPLOAD_DIR', upload_dir)
    monkeypatch.setattr(app_module, 'OUTPUT_DIR', tmp_path / 'outputs')
    app_module.tasks.clear()

    src = str(tmp_path / 'in.epub')
    build_epub(src)

    with TestClient(app_module.app) as client:
        with open(src, 'rb') as fh:
            response = client.post(
                '/upload',
                headers={'X-API-Key': 'test-secret'},
                files={'files': ('../../evil\\..\\secret.epub', fh, 'application/epub+zip')},
            )

    assert response.status_code == 200
    file_info = response.json()['files'][0]
    assert file_info['filename'] == 'secret.epub'
    task_id = file_info['task_id']
    assert task_id is not None
    saved_path = Path(app_module.tasks[task_id]['file_path'])
    assert saved_path.parent == upload_dir / task_id
    assert saved_path.name == 'secret.epub'


def test_upload_enforces_max_file_size_streaming(monkeypatch, tmp_path):
    pytest.importorskip('httpx', reason='TestClient needs httpx')
    from fastapi.testclient import TestClient
    import app as app_module

    upload_dir = tmp_path / 'uploads'
    monkeypatch.setattr(app_module, 'CONVERTER_API_KEY', 'test-secret')
    monkeypatch.setattr(app_module, 'UPLOAD_DIR', upload_dir)
    monkeypatch.setattr(app_module, 'OUTPUT_DIR', tmp_path / 'outputs')
    monkeypatch.setattr(app_module, 'MAX_FILE_SIZE', 1024)
    monkeypatch.setattr(app_module, 'UPLOAD_CHUNK_SIZE', 256)
    app_module.tasks.clear()

    src = str(tmp_path / 'big.epub')
    build_epub(src)

    with TestClient(app_module.app) as client:
        with open(src, 'rb') as fh:
            response = client.post(
                '/upload',
                headers={'X-API-Key': 'test-secret'},
                files={'files': ('big.epub', fh, 'application/epub+zip')},
            )

    assert response.status_code == 200
    file_info = response.json()['files'][0]
    assert file_info['task_id'] is None
    assert 'File too large' in file_info['error']
    assert not any(upload_dir.iterdir())


def test_evict_expired_tasks_and_worker_cap(monkeypatch):
    import app as app_module

    app_module.tasks.clear()
    app_module.tasks['old-task'] = {'status': 'done', 'created_at': 100.0}
    app_module.tasks['fresh-task'] = {'status': 'done', 'created_at': 5000.0}

    removed = app_module.evict_expired_tasks(cutoff=1000.0)
    assert removed == 1
    assert 'old-task' not in app_module.tasks
    assert 'fresh-task' in app_module.tasks
    assert app_module.executor._max_workers == 2


def test_cors_restricts_unknown_origin():
    pytest.importorskip('httpx', reason='TestClient needs httpx')
    from fastapi.testclient import TestClient
    import app as app_module

    with TestClient(app_module.app) as client:
        evil = client.options(
            '/upload',
            headers={'Origin': 'https://evil.example.com', 'Access-Control-Request-Method': 'POST'},
        )
        assert evil.headers.get('access-control-allow-origin') is None

        vercel = client.options(
            '/upload',
            headers={'Origin': 'https://reading-ledger.vercel.app', 'Access-Control-Request-Method': 'POST'},
        )
        assert vercel.headers.get('access-control-allow-origin') == 'https://reading-ledger.vercel.app'

        local = client.options(
            '/upload',
            headers={'Origin': 'http://localhost:5173', 'Access-Control-Request-Method': 'POST'},
        )
        assert local.headers.get('access-control-allow-origin') == 'http://localhost:5173'


def test_xml_parser_blocks_external_entities():
    from lxml import etree
    from html_cleaner import xml_parser

    xxe_payload = (
        b'<?xml version="1.0" encoding="utf-8"?>'
        b'<!DOCTYPE root [ <!ENTITY ext SYSTEM "http://127.0.0.1:9/secret"> ]>'
        b'<root>&ext;</root>'
    )
    tree = etree.fromstring(xxe_payload, xml_parser())
    assert tree.text is None or 'http://' not in tree.text
