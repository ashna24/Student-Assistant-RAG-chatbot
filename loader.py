from pypdf import PdfReader

class Document:
    def __init__(self,filepath):
        self.filepath = filepath
        self.text = ""

    #for reading the pdf 
    def Reader(self):
        reader = PdfReader(self.filepath)
        for t in reader.pages:
            self.text +=t.extract_text()
        print(self.text)

    #for chunking
    def chunking(self, overlap= 50 , chunk_size= 200):
        start = 0 
        chunk = []
        while start < len(self.text):
            end = start + chunk_size
            chunk.append(self.text[start:end])
            start += chunk_size - overlap
        return chunk

