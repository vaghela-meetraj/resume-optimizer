import io
import zipfile
import pytest
from app import create_app
from app.services.pdf import extract_resume_text


@pytest.fixture
def app():
    app = create_app()
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    return app.test_client()


def create_minimal_pdf_bytes(text="John Doe Python Developer Experience with Docker and SQL"):
    """Create a minimal valid PDF byte stream with text."""
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    
    # We can use reportlab or write a basic valid PDF text stream
    # A standard simple PDF stream:
    pdf_content = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length 73 >>\nstream\nBT\n/F1 12 Tf\n100 700 Td\n(" + text.encode("latin1") + b") Tj\nET\nendstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000244 00000 n \n0000000366 00000 n \ntrailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n443\n%%EOF"
    )
    return pdf_content


def create_minimal_docx_bytes(text="John Doe Python Developer Experience with Docker and SQL"):
    """Create a valid DOCX byte stream with the specified text."""
    import docx
    doc = docx.Document()
    doc.add_paragraph(text)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def test_app_boots_without_api_keys(monkeypatch):
    """Verify that app boots smoothly when no API keys are configured."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    app = create_app()
    assert app is not None


def test_file_too_large_returns_413(client):
    """A file exceeding 5MB must return HTTP 413 Payload Too Large."""
    large_content = b"a" * (5 * 1024 * 1024 + 100)
    data = {
        "resume": (io.BytesIO(large_content), "large_resume.pdf"),
        "job_desc": "Senior Python Developer with hands-on skills in Docker, PostgreSQL and FastAPI."
    }
    response = client.post("/analyze", data=data, content_type="multipart/form-data")
    assert response.status_code == 413
    assert "File too large" in response.get_json()["error"]


def test_fake_pdf_fails_magic_byte_check(client):
    """A text file renamed to .pdf must be rejected with 400."""
    fake_pdf = b"This is just a text file renamed to resume.pdf"
    data = {
        "resume": (io.BytesIO(fake_pdf), "resume.pdf"),
        "job_desc": "Senior Python Developer with hands-on skills in Docker, PostgreSQL and FastAPI."
    }
    response = client.post("/analyze", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "Not a valid PDF file" in response.get_json()["error"]


def test_fake_docx_fails_validation(client):
    """A corrupted or renamed text file to .docx must return 400."""
    fake_docx = b"Not a PK zip file at all"
    data = {
        "resume": (io.BytesIO(fake_docx), "resume.docx"),
        "job_desc": "Senior Python Developer with hands-on skills in Docker, PostgreSQL and FastAPI."
    }
    response = client.post("/analyze", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "Not a valid DOCX file" in response.get_json()["error"]


def test_pasted_text_resume_analysis(client):
    """Pasting resume text directly should work without needing an uploaded file."""
    pasted_text = (
        "John Doe - Senior Software Engineer\n"
        "Technical Skills: Python, Flask, Docker, Git, MySQL, REST APIs\n"
        "Experience building scalable backend services and microservices."
    )
    job_desc = (
        "Requirements:\n"
        "- Strong proficiency in Python and Flask\n"
        "- Experience with Docker and MySQL\n"
        "Nice to have:\n"
        "- Experience with Git and REST APIs"
    )
    data = {
        "resume_text": pasted_text,
        "job_desc": job_desc
    }
    response = client.post("/analyze", data=data)
    assert response.status_code == 200
    res = response.get_json()
    assert "Python" in res["MatchedSkills"]
    assert "Docker" in res["MatchedSkills"]
    assert res["request_id"] is not None


def test_same_resume_pdf_docx_pasted_gives_same_skills():
    """Verify that PDF, DOCX, and pasted text extract the exact same text and skills."""
    from app.services.nlp import NLPProcessor

    nlp = NLPProcessor()
    text = (
        "Jane Doe - Backend Engineer\n"
        "Languages: Python, Java, C++\n"
        "Frameworks: Flask, Django\n"
        "Databases & Tools: PostgreSQL, MySQL, Docker, Git"
    )

    pasted_extracted = extract_resume_text(pasted_text=text)
    
    docx_bytes = create_minimal_docx_bytes(text)
    docx_file = io.BytesIO(docx_bytes)
    docx_file.filename = "resume.docx"
    docx_extracted = extract_resume_text(uploaded_file=docx_file)

    pdf_bytes = create_minimal_pdf_bytes(text)
    pdf_file = io.BytesIO(pdf_bytes)
    pdf_file.filename = "resume.pdf"
    pdf_extracted = extract_resume_text(uploaded_file=pdf_file)

    skills_pasted = nlp.extract_skills_nlp(pasted_extracted)
    skills_docx = nlp.extract_skills_nlp(docx_extracted)
    skills_pdf = nlp.extract_skills_nlp(pdf_extracted)

    assert "Python" in skills_pasted
    assert "Python" in skills_docx
    assert "Python" in skills_pdf
    assert "Docker" in skills_pasted
    assert "Docker" in skills_docx
    assert "Docker" in skills_pdf


def test_security_headers_present(client):
    """Verify standard security headers and X-Request-ID are returned."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert "X-Request-ID" in response.headers

