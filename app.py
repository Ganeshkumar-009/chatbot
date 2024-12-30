import os
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import streamlit as st
def extract_text_from_pdf(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None
def split_text_into_chunks(text, chunk_size=1000, overlap=200):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks
def find_most_relevant_chunk(chunks, query):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(chunks + [query])
    query_vector = tfidf_matrix[-1]
    chunk_vectors = tfidf_matrix[:-1]
    similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
    best_chunk_idx = similarities.argmax()
    return chunks[best_chunk_idx]
def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=500, min_length=200, do_sample=False)
    return summary[0]['summary_text']
def add_custom_css():
    css = f"""
    <style>
        .stApp {{
            background: url("https://www.technetexperts.com/wp-content/uploads/2024/08/AI-And-Robotics-1024x574.jpg") no-repeat center center fixed;
            background-size: cover;
            background-color: rgba(0, 0, 0, 0.7);
            background-blend-mode: overlay;
            color: white;
        }}
        .stContainer {{
            background-color: rgba(0, 0, 0, 0.5);
            padding: 20px;
            border-radius: 10px;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: white;
            text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.7);
        }}
        p {{
            color: white;
        }}
        .stButton>button {{
            background-color: #ffffff;
            color: #000000;
            border-radius: 5px;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
def extract_headings(text):
    lines = text.split("\n")
    headings = []
    for line in lines:
        line = line.strip()
        if len(line) > 0 and (line.isupper() or len(line) < 60 and re.match(r"^[A-Za-z :]+$", line)):
            headings.append(line)
        elif re.match(r"^\d+(\.\d+)\s+.", line):
            headings.append(line)
    return list(dict.fromkeys(headings))
def interact_with_chatbot(chunks):
    st.markdown('<div class="stContainer">', unsafe_allow_html=True)
    st.header("Ask Your Questions")
    query = st.text_input("Enter your question:")
    if query:
        best_chunk = find_most_relevant_chunk(chunks, query)
        summarized_chunk = summarize_text(best_chunk)
        st.subheader("Answer:")
        st.write(summarized_chunk)
    st.markdown('</div>', unsafe_allow_html=True)
def main():
    st.title("AI CHATBOT FOR PDF INTERACTION")
    st.write("Upload a PDF file, and this app will let you ask questions and get summarized answers.")
    add_custom_css()
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file:
        with st.spinner("Extracting text from PDF..."):
            pdf_text = extract_text_from_pdf(uploaded_file)
        if pdf_text:
            st.success("Text extracted successfully.")
            headings = extract_headings(pdf_text)
            if headings:
                st.subheader("Extracted Headings:")
                for heading in headings:
                    st.write(f"- {heading}")            
            chunks = split_text_into_chunks(pdf_text)
            interact_with_chatbot(chunks)
        else:
            st.error("Failed to extract text from the PDF.")
if __name__ == "__main__":
    main()
