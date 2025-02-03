%%writefile app.py
import streamlit as st
import os
import json
import gspread
import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from oauth2client.service_account import ServiceAccountCredentials
from transformers import pipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from ibm_watsonx_ai.foundation_models.extensions.langchain import WatsonxLLM

def load_and_preprocess(file_path):
    """Load and split a text document into 1000-character chunks."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as file:
        document = file.read()

    chunks = [document[i:i+1000] for i in range(0, len(document), 1000)]
    return chunks

def risk_detection(chunks):
    """Analyze legal risks using an NLP model."""
    model_name = "google/flan-t5-base"
    nlp = pipeline("text2text-generation", model=model_name)

    results = []
    for chunk in chunks:
        prompt_analysis = (
            "Analyze the following text for legal risks, hidden obligations, or compliance issues:\n\n"
            + chunk
        )
        analysis_result = nlp(prompt_analysis, max_length=200, do_sample=False)
        analysis_text = analysis_result[0]['generated_text']

        prompt_recommendations = (
            "Provide recommendations to mitigate identified risks in the following text:\n\n"
            + chunk
        )
        recommendations_result = nlp(prompt_recommendations, max_length=200, do_sample=False)
        recommendations_text = recommendations_result[0]['generated_text']

        results.append({
            "context": chunk,
            "analysis": analysis_text,
            "recommendations": recommendations_text
        })
    return results

def export_to_sheets(data, sheet_name, credentials_file):
    """Export analysis results to Google Sheets."""
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_file, scope)
    client = gspread.authorize(creds)

    spreadsheet = client.create(sheet_name)
    sheet = spreadsheet.sheet1

    df = pd.DataFrame(data)
    sheet.update([df.columns.values.tolist()] + df.values.tolist())

    return spreadsheet.url

def process_and_query(file_path, query, credentials_file, sheet_name):
    """Run document analysis, query Watson AI, and save results."""
    chunks = load_and_preprocess(file_path)

    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings()
    docsearch = Chroma.from_documents(texts, embeddings)

    credentials = {
        "url": "https://eu-de.ml.cloud.ibm.com",
        "apikey": "ygkx3537EypWqZ6ziVsZe_2TWa52ha7nSiCdRJAfXMBu"
    }
    project_id = "4ec8c16d-4406-4dd2-92da-41d1718164ff"

    model_id = 'google/flan-ul2'
    parameters = {
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
        GenParams.MIN_NEW_TOKENS: 50,
        GenParams.MAX_NEW_TOKENS: 200,
        GenParams.TEMPERATURE: 0.5
    }
    model = Model(model_id=model_id, params=parameters, credentials=credentials, project_id=project_id)
    flan_ul2_llm = WatsonxLLM(model=model)

    qa = RetrievalQA.from_chain_type(
        llm=flan_ul2_llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=False
    )

    result = qa.invoke(query)
    analysis = risk_detection(chunks)

    output_path = "risk_analysis.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=4)

    sheet_url = export_to_sheets(analysis, sheet_name, credentials_file)
    return result, analysis, sheet_url

def send_email(email, password, recipient_email, subject, body):
    """Send an email with analysis results."""
    message = MIMEMultipart()
    message['From'] = email
    message['To'] = recipient_email
    message['Subject'] = subject
    message.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(email, password)
            server.send_message(message)
            st.success("✅ Email sent successfully!")
    except Exception as e:
        st.error(f"❌ Failed to send email: {e}")

def main():
    st.title("🚀 Legal and Risk Analysis Tool")

    uploaded_file = st.file_uploader("📂 Upload a text file")

    if uploaded_file is not None:
        file_path = os.path.join("/tmp", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        query = st.text_input("🔍 Enter your query")
        credentials_file = "alien-baton-446917-f0-a58499645f4c.json"
        sheet_name = st.text_input("📊 Enter Google Sheet name")
        email = "chsdv2004@gmail.com"
        password = "fmxe kxdz koiy qzra"
        recipient_email = st.text_input("📩 Enter recipient email")

        if st.button("Query Document"):
            if query and file_path:
                result, _, _ = process_and_query(file_path, query, credentials_file, sheet_name)
                st.write("✅ Query Result:", result)
            else:
                st.error("Please provide a valid query and upload a document.")

        if st.button("Analyze and Send Email"):
            if file_path and credentials_file and sheet_name and email and password and recipient_email:
                _, analysis, sheet_url = process_and_query(file_path, query, credentials_file, sheet_name)
                st.write("✅ Google Sheets URL:", sheet_url)
                st.json(analysis)

                subject = "Legal and Risk Analysis Results"
                body = f"Here are the results of the legal and risk analysis:\n\nGoogle Sheets URL: {sheet_url}\n\nAnalysis Results: {json.dumps(analysis, indent=4)}"
                send_email(email, password, recipient_email, subject, body)
            else:
                st.error("Please provide all required inputs.")

if _name_ == "_main_":
    main()
