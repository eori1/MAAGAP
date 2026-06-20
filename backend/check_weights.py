import docx

doc = docx.Document(r'c:\Users\ASUS\Desktop\Tisis\docs\UG-CICT-THESIS-MANUSCRIPT_LATEST_UPDATE.docx')

keywords = ['weight', 'coefficient', 'logistic', 'formula', 'delay probability', 'overrun probability', '1.6', '1.4', '0.9', 'typhoon', 'infrastructure']
with open(r'c:\Users\ASUS\Desktop\Tisis\docs\extracted_weights.md', 'w', encoding='utf-8') as f:
    f.write("--- paragraphs with keywords ---\n")
    for p in doc.paragraphs:
        text = p.text.strip().lower()
        if any(k in text for k in keywords):
            f.write(p.text.strip() + "\n")
            f.write("-" * 40 + "\n")
