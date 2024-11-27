import streamlit as st
from transformers import pipeline
from gtts import gTTS
import os
from PyPDF2 import PdfReader
from docx import Document
import tempfile
import textwrap
from PIL import Image
import io
import logging

logging.basicConfig(level=logging.DEBUG)
logging.debug("Checkpoint: App Started")


# Initialize image captioner pipeline
captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

# Add custom styling with HTML/CSS
st.markdown("""
    <style>
    body {
        background-color: #f7f8fa;
        font-family: Arial, sans-serif;
    }
    .title {
        color: #2c3e50;
        text-align: center;
        font-size: 40px;
        margin-bottom: 20px;
    }
    .subtitle {
        color: #34495e;
        font-size: 24px;
        margin-top: 20px;
        border-bottom: 2px solid #ecf0f1;
        padding-bottom: 10px;
    }
    .button {
        background-color: #3498db;
        border: none;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Function to extract text and image captions from files
def extract_text_and_captions(file):
    text = ""
    image_captions = []
    
    if file.name.endswith(".pdf"):
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
            # Extract images from PDF pages
            if "/XObject" in page.get("/Resources", {}):
                x_objects = page["/Resources"]["/XObject"].get_object()
                for obj in x_objects:
                    if x_objects[obj]["/Subtype"] == "/Image":
                        # Save the image temporarily
                        size = (x_objects[obj]["/Width"], x_objects[obj]["/Height"])
                        data = x_objects[obj].get_data()
                        image = Image.open(io.BytesIO(data))
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img_file:
                            image.save(tmp_img_file.name)
                            # Generate caption
                            caption = captioner(tmp_img_file.name)[0]['generated_text']
                            image_captions.append(caption)
    
    elif file.name.endswith(".docx"):
        doc = Document(file)
        for para in doc.paragraphs:
            text += para.text
        # Process images in the DOCX file
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img_file:
                    tmp_img_file.write(rel.target_part.blob)
                    tmp_img_file.close()
                    # Generate caption for the image
                    caption = captioner(tmp_img_file.name)[0]['generated_text']
                    image_captions.append(caption)
    
    elif file.name.endswith(".txt"):
        text = file.read().decode("utf-8")
    else:
        text = "Unsupported file format."
    
    return text, image_captions

# Function for Text-to-Speech (TTS)
def text_to_speech(text, output_path="output_audio.mp3"):
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(output_path)
    return output_path

# Summarization function with truncation and chunking
def summarize_text(text):
    max_input_length = 1024  # BART model token limit
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    # Tokenize and split into chunks if necessary
    wrapped_text = textwrap.fill(text, width=max_input_length)
    chunks = wrapped_text.split('\n')
    
    summaries = []
    
    for chunk in chunks:
        if len(chunk.split()) > max_input_length:
            chunk = chunk[:max_input_length]  # Truncate if chunk exceeds max length
        
        # Get summary for the chunk
        summary = summarizer(chunk, max_length=150, min_length=40, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    
    return ' '.join(summaries)

# Sentiment analysis function
def analyze_sentiment(text):
    sentiment_pipeline = pipeline("sentiment-analysis")
    sentiment = sentiment_pipeline(text)
    return sentiment[0]['label'], sentiment[0]['score']

# Streamlit UI
st.markdown('<h1 class="title">📄 Document Intelligence Project</h1>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, TXT):", type=["pdf", "docx", "txt"])

if uploaded_file:
    # Extract text and image captions
    with st.spinner("🔍 Extracting text and analyzing images..."):
        document_text, image_captions = extract_text_and_captions(uploaded_file)
    
    # Check if text extraction was successful
    if document_text == "Unsupported file format.":
        st.error("❌ The uploaded file format is unsupported. Please upload a PDF, DOCX, or TXT file.")
    else:
        # Display extracted text
        st.markdown('<h2 class="subtitle">📜 Extracted Text</h2>', unsafe_allow_html=True)
        st.write(document_text)

        # Display image captions
        if image_captions:
            st.markdown('<h2 class="subtitle">🖼️ Image Captions</h2>', unsafe_allow_html=True)
            for idx, caption in enumerate(image_captions, start=1):
                st.write(f"**Image {idx}:** {caption}")

        # Add buttons for functionalities
        if st.button("📄 Summarize Document"):
            with st.spinner("✨ Summarizing..."):
                summary = summarize_text(document_text)
                st.markdown('<h2 class="subtitle">📝 Summary</h2>', unsafe_allow_html=True)
                st.write(summary)
                st.download_button("📥 Download Summary", summary, file_name="summary.txt")

        if st.button("📊 Analyze Sentiment"):
            with st.spinner("✨ Analyzing sentiment..."):
                sentiment_label, sentiment_score = analyze_sentiment(document_text)
                st.markdown('<h2 class="subtitle">💡 Sentiment Analysis</h2>', unsafe_allow_html=True)
                st.write(f"**Label:** {sentiment_label}")
                st.write(f"**Confidence:** {sentiment_score * 100:.2f}%")

        if st.button("🔊 Convert Text to Audio"):
            with st.spinner("✨ Converting to audio..."):
                audio_path = text_to_speech(document_text)
                st.markdown('<h2 class="subtitle">🔈 Text-to-Speech</h2>', unsafe_allow_html=True)
                st.audio(audio_path, format="audio/mp3")
                with open(audio_path, "rb") as file:
                    st.download_button("📥 Download Audio", file, file_name="output_audio.mp3")

# Footer
st.markdown('<h3 style="text-align: center; margin-top: 50px;">Developed by Mahima Bhandari 💻</h3>', unsafe_allow_html=True)
