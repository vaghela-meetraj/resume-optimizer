import io
import zipfile
import docx
from pypdf import PdfReader


MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
MAX_PAGE_COUNT = 10
MIN_TEXT_LENGTH = 50


def extract_pdf_bytes(file_bytes: bytes) -> str:
    """Extract and validate text from PDF bytes."""
    if len(file_bytes) == 0:
        raise ValueError("Uploaded file is empty.")
    if len(file_bytes) > MAX_FILE_SIZE:
        raise ValueError("File too large (max 5MB).")

    # Magic byte check
    if not file_bytes.startswith(b'%PDF-'):
        raise ValueError("Not a valid PDF file. The file is corrupted or improperly formatted.")

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception as e:
        raise ValueError(f"Invalid or corrupted PDF file: {e}")

    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are not supported. Please remove the password and try again.")

    page_count = len(reader.pages)
    if page_count == 0:
        raise ValueError("PDF file contains no pages.")
    if page_count > MAX_PAGE_COUNT:
        raise ValueError(f"PDF exceeds maximum allowed page count of {MAX_PAGE_COUNT} pages.")

    text_parts = []
    for page in reader.pages:
        try:
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text_parts.append(page_text.strip())
        except Exception:
            continue

    if not text_parts:
        raise ValueError(
            "This PDF appears to be a scanned image or photo without selectable text. "
            "Please save it as a digital text PDF, DOCX, or paste your resume text directly."
        )

    full_text = "\n\n".join(text_parts).strip()
    if len(full_text) < MIN_TEXT_LENGTH:
        raise ValueError("Extracted text is too short. Please upload a detailed resume.")

    return full_text


def extract_docx_bytes(file_bytes: bytes) -> str:
    """Extract and validate text from DOCX bytes including paragraphs and tables."""
    if len(file_bytes) == 0:
        raise ValueError("Uploaded file is empty.")
    if len(file_bytes) > MAX_FILE_SIZE:
        raise ValueError("File too large (max 5MB).")

    # Magic byte and zip structure check
    if not file_bytes.startswith(b'PK'):
        raise ValueError("Not a valid DOCX file. File must be a valid Word document.")

    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            namelist = z.namelist()
            if "word/document.xml" not in namelist:
                raise ValueError("Corrupted DOCX file: missing document stream.")
    except Exception as e:
        raise ValueError(f"Corrupted or invalid DOCX archive: {e}")

    try:
        doc = docx.Document(io.BytesIO(file_bytes))
    except Exception as e:
        raise ValueError(f"Failed to parse DOCX document: {e}")

    text_parts = []
    for p in doc.paragraphs:
        if p.text and p.text.strip():
            text_parts.append(p.text.strip())

    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                text_parts.append(row_text)

    if not text_parts:
        raise ValueError("DOCX document contains no readable text.")

    full_text = "\n\n".join(text_parts).strip()
    if len(full_text) < MIN_TEXT_LENGTH:
        raise ValueError("Extracted text is too short. Please upload a detailed resume.")

    return full_text


def extract_resume_text(uploaded_file=None, pasted_text: str = None) -> str:
    """
    Unified entry point for resume text extraction.
    Accepts an uploaded file (PDF or DOCX) or raw pasted text.
    """
    # 1. Pasted text mode
    if pasted_text and pasted_text.strip():
        cleaned = pasted_text.strip()
        if len(cleaned) < MIN_TEXT_LENGTH:
            raise ValueError(f"Pasted resume text must be at least {MIN_TEXT_LENGTH} characters.")
        return cleaned

    # 2. File upload mode
    if not uploaded_file:
        raise ValueError("Please provide either a resume file (PDF or DOCX) or paste your resume text.")

    filename = getattr(uploaded_file, "filename", "").lower()

    # Reject unsupported extensions with specific helpful guidance
    if filename.endswith(".doc") and not filename.endswith(".docx"):
        raise ValueError("Legacy .doc format is not supported. Please save as .docx or .pdf, or paste your text.")
    if any(filename.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]):
        raise ValueError("Image files cannot be read directly. Please save as PDF or DOCX, or paste the text.")

    try:
        content = uploaded_file.read()
    except Exception as e:
        raise RuntimeError(f"Failed reading uploaded file: {e}")

    if filename.endswith(".docx"):
        return extract_docx_bytes(content)
    elif filename.endswith(".pdf"):
        return extract_pdf_bytes(content)
    else:
        # Fallback inspection by magic bytes if filename is generic
        if content.startswith(b'%PDF-'):
            return extract_pdf_bytes(content)
        elif content.startswith(b'PK'):
            return extract_docx_bytes(content)
        raise ValueError("Unsupported file format. Please upload a .pdf or .docx file, or paste your text.")


# Backward compatibility alias
def extract_pdf_text(uploaded_file) -> str:
    return extract_resume_text(uploaded_file=uploaded_file)
