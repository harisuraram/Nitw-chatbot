import os
import pdfplumber
from pdf2image import convert_from_path
import pytesseract

# Folder containing your PDFs
pdf_folder = "./pdfs"
# Folder to save extracted text files
output_folder = "./data"

os.makedirs(output_folder, exist_ok=True)

# Loop through all PDFs in the folder
for pdf_file in os.listdir(pdf_folder):
    if not pdf_file.lower().endswith(".pdf"):
        continue

    pdf_path = os.path.join(pdf_folder, pdf_file)
    txt_file_path = os.path.join(output_folder, pdf_file.rsplit(".", 1)[0] + ".txt")

    full_text = ""

    try:
        # Try extracting text using pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Extract text
                page_text = page.extract_text() or ""
                full_text += page_text + "\n"

                # Extract tables
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        full_text += " | ".join([str(cell) if cell else "" for cell in row]) + "\n"
                    full_text += "\n"

        # If no text was found, fallback to OCR
        if not full_text.strip():
            print(f"No text found in {pdf_file} using pdfplumber. Falling back to OCR...")
            images = convert_from_path(pdf_path)
            for img in images:
                ocr_text = pytesseract.image_to_string(img)
                full_text += ocr_text + "\n"

        # Clean up extra spaces/newlines
        full_text = "\n".join([line.strip() for line in full_text.splitlines() if line.strip()])

        # Save to txt file
        with open(txt_file_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        print(f"✅ Extracted text saved for: {pdf_file}")

    except Exception as e:
        print(f"❌ Failed to process {pdf_file}: {e}")