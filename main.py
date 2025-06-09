import os
import shutil
from PIL import Image
import streamlit as st
from pdf2image import convert_from_path
import torch
import torch.nn as nn
from torchvision import transforms
import plotly.graph_objects as go
from dotenv import load_dotenv

from google import genai
from google.genai import types

# ---- Model Definition ----
# Flatten helper
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# CNN Model using Sequential
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2),  # -> (16, 1104, 702)
            nn.ReLU(),
            nn.MaxPool2d(2),                         # -> (16, 552, 351)

            nn.Conv2d(16, 32, kernel_size=3),           # -> (32, 550, 349)
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),                         # -> (32, 275, 174)

            nn.Conv2d(32, 64, kernel_size=3),           # -> (64, 273, 172)
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),                         # -> (64, 136, 86)

            nn.Conv2d(64, 128, kernel_size=3),          # -> (128, 136, 86)
            nn.ReLU(),
            nn.MaxPool2d(2),                         # -> (128, 68, 43)

            nn.Conv2d(128, 256, kernel_size=3),          # -> (256, 66, 41)
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),                         # -> (256, 33, 20)

            nn.Conv2d(256, 512, kernel_size=3),          # -> (512, 66, 41)
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),                         # -> (512, 33, 20)
            nn.AdaptiveAvgPool2d((7, 7)),

            Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

@st.cache_resource(show_spinner=False)
def load_model(path="model/model_weights.pth"):
    device = "cpu"
    model = CNN().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((2212, 1408)),  # Ensure all images are the same size
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def predict_image(image):
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        prob = output.item()
        pred = 0 if prob > 0.5 else 1
    return pred, prob

def generate_consultation_from_pdf(pdf_bytes):
    load_dotenv()
    api_key = os.getenv("GOOGLE_AI_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_AI_API_KEY in environment variables")

    client = genai.Client(api_key=api_key)

    prompt = """
        You are an AI neurologist assistant analyzing EEG (Electroencephalogram) report images extracted from a multi-page PDF. Each page shows EEG brainwave data.

            1. For each EEG image/page, classify it as either **Normal** or **Abnormal** based on neural patterns.
            2. After analyzing all pages, provide an overall summary:
                - Whether the EEG report is mostly normal or abnormal.
                - Key findings observed in abnormal pages (e.g., unusual neural activity, spikes, slowing, epileptiform discharges).
            3. Generate a detailed medical consultation including:
                - Explanation of what the abnormal or normal findings indicate.
                - Possible neurological conditions related to abnormal patterns.
                - Recommended next steps for the patient, such as clinical tests, follow-ups, or specialist consultations.
                - Lifestyle advice or precautions if applicable.
            4. Write the consultation clearly and compassionately as if speaking to a patient.
            5. Output the entire analysis and consultation in markdown or HTML format for easy reading.

            Treat the EEG data carefully and give professional, accurate advice based on current neurological knowledge.
        """

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            types.Part.from_bytes(
                data=pdf_bytes,
                mime_type="application/pdf",
            ),
            prompt
        ]
    )
    return response.text

st.set_page_config(
    page_title="🧠 EEG Graph Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        border-right: 2px solid #203a43;
        color: #ffffff;
    }
    /* Sidebar widgets */
    .css-1d391kg button, 
    .css-1d391kg select, 
    .css-1d391kg input, 
    .css-1d391kg .stFileUploader>div>div {
        background: #2c5364 !important;
        color: #e0e0e0 !important;
        border: none !important;
    }
    .css-1d391kg label, .css-1d391kg .stMarkdown p, .css-1d391kg .stMarkdown span {
        color: #cfd8dc !important;
    }

    /* Main content */
    .css-1d391kg + div {
        background: #203a43;
        padding: 2rem 3rem;
        color: #e0e0e0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    h1, h2, h3 {
        color: #a7c7e7;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Metric styling: Remove background, set text color */
    .stMetric > div {
        background-color: transparent !important;
        color: #a7c7e7 !important;
        font-weight: 700;
        border: none !important;
        box-shadow: none !important;
    }

    /* Buttons */
    .stButton>button {
        background-color: #2c5364;
        color: #e0e0e0;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 8px 16px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #203a43;
        color: #a7c7e7;
    }

    /* Image captions */
    .stImage > figcaption {
        color: #a7c7e7 !important;
        font-weight: 600;
    }

    /* Consultation output styling */
    .consultation-output {
        background-color: #1e2d3d;
        padding: 20px;
        border-radius: 12px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #cfd8dc;
        font-size: 16px;
        line-height: 1.5;
        max-height: 600px;
        overflow-y: auto;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

tabs = st.tabs(["EEG Analysis", "Consultation"])
consultation_text = ""

with tabs[0]:
    st.sidebar.title("⚙️ Configuration")

    uploaded_pdf = st.sidebar.file_uploader("Upload EEG Report (PDF)", type=["pdf"])

    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        Upload a PDF EEG report to analyze its pages.
        The model classifies EEG pages as Normal or Abnormal.
        """
    )

    st.title("🧠 EEG Graph Analyzer")
    st.markdown("Analyze EEG reports with state-of-the-art CNN model.")

    if uploaded_pdf:
        with st.spinner("🔄 Processing EEG data... Please wait."):
            category_folder = "Patient_Reports"
            if os.path.exists(category_folder):
                shutil.rmtree(category_folder)
            os.makedirs(category_folder, exist_ok=True)

            if os.path.exists('Data'):
                shutil.rmtree('Data')
            os.makedirs('Data', exist_ok=True)
      
            pdf_path = r"Data/uploaded_report.pdf"
            with open(pdf_path, "wb") as f:
                f.write(uploaded_pdf.read())

            images = convert_from_path(pdf_path)
            crop_box = (60, 186, 2272, 1594)
            predictions = []
            confidences = []

            for i, image in enumerate(images):
                cropped_image = image.crop(crop_box)
                image_name = f"{os.path.splitext(uploaded_pdf.name)[0]}_{i+1}.png"
                image_path = os.path.join(category_folder, image_name)
                cropped_image.save(image_path, "PNG")

                pred, prob = predict_image(cropped_image)
                predictions.append((image_path, pred, prob))
                confidences.append(prob)

        st.success("✅ EEG analysis complete!")

        st.header("📷 EEG Page Analysis")
        for i, (img_path, pred, prob) in enumerate(predictions):
            label = "Abnormal" if pred == 1 else "Normal"
            emoji = "⚡"

            col1, col2 = st.columns([3, 1])
            with col1:
                img = Image.open(img_path)
                st.image(img, caption=f"Page {i+1}", width=600)
            with col2:
                st.markdown(f"### {emoji} Analysis for Page {i+1}")
                st.markdown(f"### {label}")
                st.metric(label="Confidence", value=f"{prob:.2%}")

        st.header("📊 Overall Report Analysis")
        total_pages = len(predictions)
        abnormal_count = sum(1 for _, p, _ in predictions if p == 1)
        normal_count = total_pages - abnormal_count

        col1, col2 = st.columns(2)
        col1.metric("Total Pages", total_pages)
        col2.metric("Abnormal Pages", abnormal_count)

        fig = go.Figure(data=[go.Pie(
            labels=["Normal", "Abnormal"],
            values=[normal_count, abnormal_count],
            hole=0.6,
            marker_colors=["#57bc90", "#ff6f61"],
            textinfo="percent+label"
        )])
        fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)

        # Sidebar Warning / Good News message
        if abnormal_count > 0:
            st.sidebar.markdown(
                """
                <div style="
                    background-color: #ffcccc; 
                    padding: 15px; 
                    border-radius: 10px; 
                    font-size: 18px; 
                    font-weight: bold; 
                    color: #a80000;
                    box-shadow: 2px 2px 8px rgba(168, 0, 0, 0.3);
                    ">
                    ⚠️ <strong>Warning:</strong> This EEG indicates unusual neural patterns.<br>
                    It is strongly recommended to seek advice from a certified neurologist for a clinical diagnosis.
                </div>
                """, unsafe_allow_html=True
            )
        else:
            st.sidebar.markdown(
                """
                <div style="
                    background-color: #d4edda; 
                    padding: 15px; 
                    border-radius: 10px; 
                    font-size: 18px; 
                    font-weight: bold; 
                    color: #155724;
                    box-shadow: 2px 2px 8px rgba(21, 87, 36, 0.3);
                    ">
                    🎉 Good news! The EEG graph is within normal limits.
                </div>
                """, unsafe_allow_html=True
            )

        # Confidence over Pages line chart
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            y=confidences,
            mode="lines+markers",
            line=dict(color="#203a43", width=3),
            marker=dict(size=8),
            name="Confidence"
        ))
        fig_line.update_layout(
            title="Confidence Scores Across Pages",
            xaxis_title="Page Number",
            yaxis_title="Confidence Score",
            yaxis_range=[0, 1],
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#a7c7e7')
        )
        st.plotly_chart(fig_line, use_container_width=True)

        # Generate Consultation Automatically
        with st.spinner("🤖 Generating detailed medical consultation..."):
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            try:
                consultation_text = generate_consultation_from_pdf(pdf_bytes)
                st.sidebar.success("✅ Consultation generated successfully.")
            except Exception as e:
                consultation_text = f"⚠️ Error generating consultation: {e}"
                st.sidebar.error(f"⚠️ Error generating consultation.")

    else:
        st.info("Please upload an EEG PDF report to begin analysis.")

with tabs[1]:
    st.header("🩺 AI-Generated Medical Consultation")
    if consultation_text:
        st.markdown(f'<div class="consultation-output">{consultation_text}</div>', unsafe_allow_html=True)
    else:
        st.info("Medical consultation will appear here after analysis.")