from observability import observation

# -----------------------------------
# Document Processing (PyMuPDF / PyPDF)
# -----------------------------------


def _read_with_pymupdf(pdf_path):

    import fitz

    with fitz.open(pdf_path) as doc:
        pages = [page.get_text() for page in doc]

    return pages


def _read_with_pypdf(pdf_path):

    from pypdf import PdfReader

    reader = PdfReader(pdf_path)

    return [page.extract_text() or "" for page in reader.pages]


def read_pdf(pdf_path):
    """Extract text with PyMuPDF, falling back to PyPDF."""

    with observation("extract-pdf", input={"pdf": str(pdf_path)}) as span:

        try:
            pages = _read_with_pymupdf(pdf_path)
            extractor = "pymupdf"
        except Exception as e:
            print(f"PyMuPDF failed ({e}) - falling back to PyPDF")
            pages = _read_with_pypdf(pdf_path)
            extractor = "pypdf"

        text = "\n".join(p for p in pages if p)

        if span:
            span.update(
                output={"chars": len(text)},
                metadata={"extractor": extractor, "pages": len(pages)}
            )

    return text
