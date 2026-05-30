import pdfplumber
import nltk
# nltk.download('punkt_tab')
def pdf_text(pdf_path,sentence_per_chunk=3,overlap=1):
    text=""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text=page.extract_text()
            if page_text:
                text+=page_text +"\n"
    sentences=nltk.sent_tokenize(text)

    i=0
    chunks=[]
    while i<len(sentences):
        chunk=" ".join(sentences[i:i+sentence_per_chunk])
        chunks.append(chunk)
        i+=sentence_per_chunk-overlap
    return chunks

    # for i in range(0,len(text),chunk_size):
    #     chunk=text[i:i+chunk_size]
    #     chunks.append(chunk)
    # return chunks




# print(pdf_text("C:\\Users\\ajayg\\Downloads\\Python_developer_resume.pdf"))



