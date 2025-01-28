import streamlit as st
import json
from transformers import pipeline
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

background_image_url = "https://www.freepik.com/free-photos-vectors/black-website-background"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url({background_image_url});
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    .css-ffhzg2 {{
        background-color: transparent;
    }}
    .stMarkdown, .stWrite, .stText, .stTitle, .stButton {{
        color: white;
    }}
    .stTextInput input {{
        color: white;
        background-color: black;
    }}
    .stButton button {{
        background-color: black;
        color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Risk Analysis Using Watson AI")
st.write("Upload a document (txt, pdf, csv) for legal and risk analysis using Watson AI.")

uploaded_file = st.file_uploader("Upload your file", type=["txt", "pdf", "csv"])

if uploaded_file is not None:
    st.write(f"File uploaded: {uploaded_file.name}")
    
    file_content = uploaded_file.read().decode("utf-8")
    
    st.write("Document preview (first 1000 characters):")
    st.text(file_content[:1000])

    def risk_detection(chunks):
        model_name = "google/flan-t5-base"
        nlp = pipeline("text2text-generation", model=model_name)

        results = []
        for chunk in chunks:
            prompt = (
                """You are a legal and risk analysis expert. Analyze the following text and provide a detailed report with:
                Analysis: Identify at least two specific risks, hidden obligations, or dependencies in the text. Explain their nature, implications, and any underlying issues requiring attention. Focus on legal, financial, operational, or compliance aspects for the given below context."""
                + chunk
            )
            result = nlp(prompt, max_length=200, do_sample=False)
            results.append({"context": chunk, "analysis": result[0]['generated_text']})

            prompt = (
                """You are a legal and risk analysis expert. Analyze the following text and provide a detailed report with:
                Recommendations: Provide at least two clear and practical recommendations to address the identified risks, fulfill obligations, or manage dependencies. Ensure the recommendations are actionable, relevant, and include specific steps or strategies for resolution for the given below context and if there are no recommendations, state that no recommendations are possible for the given context."""
                + chunk
            )
            result = nlp(prompt, max_length=200, do_sample=False)
            results.append({"recommendations": result[0]['generated_text']})

        return results

    chunks = [file_content[i:i+1000] for i in range(0, len(file_content), 1000)]
    analysis_results = risk_detection(chunks)
    
    st.write("Risk Analysis Results:")
    for result in analysis_results:
        if "context" in result:
            st.write(f"**Context:**\n{result['context']}")
        if "analysis" in result:
            st.write(f"**Analysis:**\n{result['analysis']}")
        if "recommendations" in result:
            st.write(f"**Recommendations:**\n{result['recommendations']}")

    email = "tonystark1696969@gmail.com"
    password = "lsut uhmh ilch sibt"
    recipient_email = "ambarish.singh22@vit.edu"

    def send_email(subject, body):
        message = MIMEMultipart()
        message['From'] = email
        message['To'] = recipient_email
        message['Subject'] = subject
        message.attach(MIMEText(body, 'plain'))

        try:
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.set_debuglevel(1)
                server.starttls()
                server.login(email, password)
                server.send_message(message)
                st.write("Email sent successfully!")
        except Exception as e:
            st.write(f"Failed to send email: {e}")

    email_subject = "Risk Analysis Results"
    email_body = "Here are the results of the risk analysis you requested:\n\n"

    for result in analysis_results:
        if "context" in result:
            email_body += f"**Context:**\n{result['context']}\n"
        if "analysis" in result:
            email_body += f"**Analysis:**\n{result['analysis']}\n"
        if "recommendations" in result:
            email_body += f"**Recommendations:**\n{result['recommendations']}\n"

    send_email(email_subject, email_body)
