import streamlit as st
import requests
import fitz  
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import docx2txt 

st.title("YouSum")
st.write("**An AI-powered text summarizer**")

# Load the pre-trained model and tokenizer
model_name = "mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Function to generate a summary
def generate_summary(text):
    tokenized_text = tokenizer.encode(text, return_tensors="pt")
    summary_ids = model.generate(tokenized_text, max_length=150)
    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return output

st.write("\nWelcome to YouSum, create a summary with an AI-Powered Solution")

# Input text area
input_option = st.selectbox("Select input source:", ("Paste Text", "Upload File", "Enter URL"))

if input_option == "Paste Text":
    input_text = st.text_area("Enter your text to summarize:", height=200)
    if st.button("Generate Summary"):
        if input_text:
            summary = generate_summary(input_text)
            st.subheader("Summary:")
            st.write(summary)
        else:
            st.warning("Please enter text to summarize.")

elif input_option == "Upload File":
    uploaded_file = st.file_uploader("Upload a file:", type=["txt", "pdf", "docx"])
    if uploaded_file:
        try:
            if uploaded_file.type == "text/plain":
                file_contents = uploaded_file.read()
                summary = generate_summary(file_contents.decode("utf-8"))
            elif uploaded_file.type == "application/pdf":
                pdf_text = ""
                pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                for page in pdf_document:
                    pdf_text += page.get_text()
                summary = generate_summary(pdf_text)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc_text = docx2txt.process(uploaded_file)  
                summary = generate_summary(doc_text)
            else:
                st.warning("Unsupported file format.")
                summary = None

            if summary is not None:
                st.subheader("Summary:")
                st.write(summary)
        except Exception as e:
            st.error(f"Error processing the uploaded file: {str(e)}"
