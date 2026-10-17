import io

import PyPDF2 as pdf


def extract_pdf_text(uploaded_file) -> str:
    MAX_FILE_SIZE = 10 * 1024 * 1024
    try:
        file_content = uploaded_file.read()
    except Exception as e:
        raise RuntimeError(f"Failed to read uploaded file: {e}")
    if len(file_content) == 0:
        raise ValueError("Uploaded file is empty.")
    if len(file_content) > MAX_FILE_SIZE:
        raise ValueError("File too large. Maximum allowed size is 10MB.")
    try:
        reader = pdf.PdfReader(io.BytesIO(file_content))
    except Exception as e:
        raise ValueError(f"Invalid or corrupted PDF file: {e}")
    if len(reader.pages) == 0:
        raise ValueError("PDF file contains no pages.")
    text_parts = []
    for page in reader.pages:
        try:
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text_parts.append(page_text)
        except Exception:
            continue
    if not text_parts:
        raise ValueError("No readable text found. If this is a scanned PDF or image, please convert it to a standard text PDF first.")
    full_text = " ".join(text_parts).strip()
    if len(full_text) < 50:
        raise ValueError("Extracted text is too short. Please check your PDF.")
    return full_text
