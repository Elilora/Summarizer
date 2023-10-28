import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.title("YouSum")
st.write("**An AI-powered text summarizer**")

# Load the pre-trained model and tokenizer
model_name = "minhtoan/t5-finetune-bbc-news"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Function to generate a summary
def generate_summary(text):
    tokenized_text = tokenizer.encode(text, return_tensors="pt")
    summary_ids = model.generate(tokenized_text, max_length=150)
    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return output

st.write("\nWelcome to YouSum, create sumary text with AI Powered Solution")
# Input text area
input_text = st.text_area("\n\nEnter your text to summarize:",height= 200)
if input_text:
    # Generate a summary when the user submits the text
    if st.button("Generate Summary"):
        summary = generate_summary(input_text)
        st.subheader("Summary:")
        st.write(summary)


