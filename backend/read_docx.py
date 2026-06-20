import docx

doc = docx.Document(r'c:\Users\ASUS\Desktop\Tisis\docs\UG-CICT-THESIS-MANUSCRIPT_LATEST_UPDATE.docx')
text = '\n'.join([p.text for p in doc.paragraphs if p.text.strip()])
print(text[:2500])
