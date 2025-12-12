import streamlit as st
from PIL import Image
import numpy as np
import os
from datetime import datetime
import tempfile
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import our custom modules
from model.detector import PPEDetector
from utils.visualization import draw_detections, create_summary_overlay
from utils.pdf_generator import generate_safety_report
from utils.gemini_recommendations import GeminiRecommendations


# Page configuration
st.set_page_config(
    page_title="SafetyVision AI",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .violation-box {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .compliant-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_detector():
    """Load YOLOv8 model (cached for performance)"""
    # Fixed confidence threshold at 0.5
    return PPEDetector(confidence_threshold=0.5)


@st.cache_resource
def load_gemini():
    """Load Gemini recommender (cached for performance)"""
    # Use Gemini API key from Streamlit secrets or environment
    api_key = st.secrets.get('GEMINI_API_KEY', os.getenv('GEMINI_API_KEY'))
    return GeminiRecommendations(api_key=api_key)


def process_image(image_source, source_type="upload"):
    """
    Process image from either upload or webcam
    
    Args:
        image_source: PIL Image or UploadedFile
        source_type: "upload" or "webcam"
    """
    # Convert to PIL Image if needed
    if source_type == "webcam":
        image = image_source
    else:
        image = Image.open(image_source)
    
    # Save to temp file for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        image.save(tmp_file.name)
        temp_image_path = tmp_file.name
    
    # Run detection
    with st.spinner("🔍 Running PPE detection..."):
        detector = load_detector()
        detections = detector.detect(temp_image_path)
        summary = detector.classify_compliance(detections)
        
        # Store in session state
        st.session_state['detections'] = detections
        st.session_state['summary'] = summary
        st.session_state['image_path'] = temp_image_path
    
    # Create visualizations
    with st.spinner("🎨 Creating visualizations..."):
        annotated_image = draw_detections(
            temp_image_path,
            detections,
            output_path=None
        )
        
        final_image = create_summary_overlay(annotated_image, summary)
        
        annotated_path = temp_image_path.replace('.jpg', '_annotated.jpg')
        Image.fromarray(final_image).save(annotated_path)
        st.session_state['annotated_path'] = annotated_path
    
    st.success("✅ Detection complete!")


def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🦺 SafetyVision AI</h1>', unsafe_allow_html=True)
    st.markdown("### Automated PPE Compliance Detection System")
    st.markdown("---")
    
    # Input method selection
    st.subheader("📸 Choose Input Method")
    input_method = st.radio(
        "Select how to provide an image:",
        ["📤 Upload Image", "📷 Use Webcam"],
        horizontal=True
    )
    
    st.markdown("---")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if input_method == "📤 Upload Image":
            st.subheader("📤 Upload Image")
            uploaded_file = st.file_uploader(
                "Choose an image of workers at the site",
                type=['jpg', 'jpeg', 'png', 'webp'],
                help="Upload a clear image showing workers. Best results with good lighting."
            )
            
            if uploaded_file is not None:
                # Display original image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image")
                
                # Analyze button
                if st.button("🚀 Analyze PPE Compliance", type="primary", use_container_width=True, key="analyze_upload"):
                    process_image(uploaded_file, source_type="upload")
        
        else:  # Webcam mode
            st.subheader("📷 Webcam Capture")
            camera_image = st.camera_input("Take a picture of workers at the site")
            
            if camera_image is not None:
                # Display captured image
                image = Image.open(camera_image)
                st.image(image, caption="Captured Image")
                
                # Analyze button
                if st.button("🚀 Analyze PPE Compliance", type="primary", use_container_width=True, key="analyze_webcam"):
                    process_image(image, source_type="webcam")
    
    with col2:
        st.subheader("🔍 Detection Results")
        
        # Display results if available
        if 'summary' in st.session_state:
            summary = st.session_state['summary']
            
            # Status box
            if summary['status'] == 'VIOLATION DETECTED':
                st.markdown(
                    f'<div class="violation-box"><h3>⚠️ {summary["status"]}</h3></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="compliant-box"><h3>✅ {summary["status"]}</h3></div>',
                    unsafe_allow_html=True
                )
            
            # Metrics
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Total People", summary['total_people'])
            col_b.metric("Compliant", summary['compliant'], delta=None)
            col_c.metric("Violations", summary['violations'], delta=None, delta_color="inverse")
            
            # Compliance rate
            st.progress(summary['compliance_rate'] / 100)
            st.caption(f"Compliance Rate: {summary['compliance_rate']:.1f}%")
        else:
            st.info("👆 Upload an image or capture from webcam to begin detection")
    
    # Annotated image and recommendations
    if 'annotated_path' in st.session_state:
        st.markdown("---")
        
        col3, col4 = st.columns([1, 1])
        
        with col3:
            st.subheader("📸 Annotated Image")
            st.image(st.session_state['annotated_path'])
            st.caption("🟢 Green = Compliant | 🔴 Red = Violation | 🟡 Yellow = Warning")
        
        with col4:
            st.subheader("🤖 AI Safety Recommendations")
            
            # Generate recommendations
            with st.spinner("🤖 Generating AI recommendations..."):
                recommender = load_gemini()
                recommendations = recommender.generate_recommendations(
                    st.session_state['summary'],
                    st.session_state['detections']
                )
                st.session_state['recommendations'] = recommendations
            
            # Display recommendations
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"**{i}.** {rec}")
            
            # Show status
            if recommender.enabled:
                st.caption("✨ Powered by Google Gemini AI")
            else:
                st.caption("📋 Using rule-based recommendations")
    
    # PDF Generation
    if 'recommendations' in st.session_state:
        st.markdown("---")
        st.subheader("📄 Generate Safety Report")
        
        col5, col6 = st.columns([2, 1])
        
        with col5:
            st.write("Create a comprehensive PDF report with:")
            st.markdown("""
            - ✅ Compliance status summary
            - ✅ Annotated image with detections
            - ✅ AI-generated recommendations
            - ✅ Timestamp and metadata
            """)
        
        with col6:
            if st.button("📥 Generate PDF Report", type="primary", use_container_width=True):
                with st.spinner("📄 Creating PDF report..."):
                    # Generate PDF
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    pdf_filename = f"safety_report_{timestamp}.pdf"
                    
                    pdf_path = generate_safety_report(
                        image_path=st.session_state['annotated_path'],
                        detections=st.session_state['detections'],
                        summary=st.session_state['summary'],
                        recommendations=st.session_state['recommendations'],
                        output_path=pdf_filename
                    )
                    
                    # Read PDF for download
                    with open(pdf_path, 'rb') as pdf_file:
                        pdf_bytes = pdf_file.read()
                    
                    st.session_state['pdf_bytes'] = pdf_bytes
                    st.session_state['pdf_filename'] = pdf_filename
                
                st.success("✅ PDF report generated!")
        
        # Download button
        if 'pdf_bytes' in st.session_state:
            st.download_button(
                label="📥 Download PDF Report",
                data=st.session_state['pdf_bytes'],
                file_name=st.session_state['pdf_filename'],
                mime='application/pdf',
                use_container_width=True
            )
    
    # Footer with About and Tech Stack
    st.markdown("---")
    
    # About and Tech Stack at bottom
    col_footer1, col_footer2 = st.columns([1, 1])
    
    with col_footer1:
        st.markdown("### About SafetyVision AI")
        st.info(
            "SafetyVision AI uses computer vision to detect PPE violations "
            "and generate comprehensive safety reports with AI-powered recommendations. "
            "Built for Oil & Gas + Utilities industries to enhance workplace safety."
        )
    
    with col_footer2:
        st.markdown("### Tech Stack")
        st.code("YOLOv8 | Google Gemini | Streamlit | ReportLab")
        st.caption("Confidence Threshold: 0.5 (optimized for safety-critical applications)")
    
    # Credits
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>Built with ❤️ using YOLOv8, Google Gemini, and Streamlit | "
        "Monica Nieckula - Oil & Gas + Utilities AI Hackathon</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()