import streamlit as st
from PIL import Image
import numpy as np
import os
from datetime import datetime
import tempfile

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
    initial_sidebar_state="expanded"
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
    return PPEDetector(confidence_threshold=0.5)


@st.cache_resource
def load_gemini():
    """Load Gemini recommender (cached for performance)"""
    api_key = os.getenv('GEMINI_API_KEY')
    return GeminiRecommendations(api_key=api_key)


def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🦺 SafetyVision AI</h1>', unsafe_allow_html=True)
    st.markdown("### Automated PPE Compliance Detection System")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Confidence threshold slider
        confidence_threshold = st.slider(
            "Detection Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.1,
            help="Minimum confidence for detections (lower = more detections, higher = fewer false positives)"
        )
        
        # Gemini API key input (optional)
        st.markdown("---")
        st.subheader("🤖 AI Recommendations")
        gemini_api_key = st.text_input(
            "Gemini API Key (Optional)",
            type="password",
            help="Enter your Google Gemini API key for AI-powered recommendations. Leave blank to use rule-based fallback."
        )
        
        if gemini_api_key:
            os.environ['GEMINI_API_KEY'] = gemini_api_key
        
        st.markdown("---")
        st.markdown("### About")
        st.info(
            "SafetyVision AI uses computer vision to detect PPE violations "
            "and generate comprehensive safety reports with AI-powered recommendations."
        )
        
        st.markdown("### Tech Stack")
        st.code("YOLOv8 | Gemini | Streamlit | ReportLab")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image of workers at the site",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Upload a clear image showing workers. Best results with good lighting."
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", width="stretch")
            
            # Save to temp file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                image.save(tmp_file.name)
                temp_image_path = tmp_file.name
    
    with col2:
        st.subheader("🔍 Detection Results")
        
        if uploaded_file is not None:
            # Run detection button
            if st.button("🚀 Analyze PPE Compliance", type="primary", use_container_width=True):
                
                with st.spinner("🔍 Running PPE detection..."):
                    # Load detector
                    detector = load_detector()
                    detector.confidence_threshold = confidence_threshold
                    
                    # Run detection
                    detections = detector.detect(temp_image_path)
                    summary = detector.classify_compliance(detections)
                    
                    # Store in session state
                    st.session_state['detections'] = detections
                    st.session_state['summary'] = summary
                    st.session_state['image_path'] = temp_image_path
                
                with st.spinner("🎨 Creating visualizations..."):
                    # Create annotated image
                    annotated_image = draw_detections(
                        temp_image_path,
                        detections,
                        output_path=None
                    )
                    
                    # Add summary overlay
                    final_image = create_summary_overlay(annotated_image, summary)
                    
                    # Store annotated image
                    annotated_path = temp_image_path.replace('.jpg', '_annotated.jpg')
                    Image.fromarray(final_image).save(annotated_path)
                    st.session_state['annotated_path'] = annotated_path
                
                st.success("✅ Detection complete!")
        
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
    
    # Annotated image and recommendations
    if 'annotated_path' in st.session_state:
        st.markdown("---")
        
        col3, col4 = st.columns([1, 1])
        
        with col3:
            st.subheader("📸 Annotated Image")
            st.image(st.session_state['annotated_path'], width="stretch")
            st.caption("Red boxes = violations | Green boxes = compliant")
        
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
            
            # If using fallback, show info
            if not recommender.enabled:
                st.info("💡 Using rule-based recommendations. Add Gemini API key in sidebar for AI-powered insights.")
    
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
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>Built with ❤️ using YOLOv8, Google Gemini, and Streamlit | "
        "Monica Nieckula - Oil & Gas + Utilities AI Hackathon</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()