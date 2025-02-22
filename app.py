# app.py
import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import time

# Configure retries for robustness
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[500, 502, 503, 504]
)
http = requests.Session()
http.mount("http://", HTTPAdapter(max_retries=retry_strategy))

def process_cv(file_content: bytes, filename: str) -> dict:
    """Process CV with improved error handling and timeout"""
    try:
        with st.spinner("Processing CV... This may take up to 30 seconds"):
            files = {
                "file": (filename, file_content, "application/pdf")
            }
            response = http.post(
                "http://127.0.0.1:8000/",
                files=files,
                timeout=30
            )
            if response.status_code == 400:
                st.error(f"Server Error: {response.json()['detail']}")
                return None
            return response.json()
    except requests.exceptions.Timeout:
        st.error("⏳ Request timed out. Please try with a smaller PDF file.")
    except requests.exceptions.ConnectionError:
        st.error("📡 Cannot connect to the server. Please ensure the backend is running.")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
    return None

def main():
    st.title("📄 CV Shortlisting AI")
    
    # File size limit (5MB)
    MAX_FILE_SIZE = 5 * 1024 * 1024
    
    uploaded_file = st.file_uploader(
        "Upload CV (PDF format, max 5MB)",
        type="pdf",
        help="Please ensure your CV is in PDF format and under 5MB"
    )
    
    if uploaded_file:
        file_size = len(uploaded_file.getvalue())
        
        if file_size > MAX_FILE_SIZE:
            st.error("File size exceeds 5MB limit. Please upload a smaller file.")
            return
            
        st.info(f"Processing file: {uploaded_file.name} ({file_size/1024:.1f} KB)")
        
        if st.button("Analyze CV"):
            # Pass both file content and filename
            result = process_cv(uploaded_file.getvalue(), uploaded_file.name)
            
            if result and "cv_score" in result:
                score = result["cv_score"]
                st.success(f"Analysis Complete!")
                st.metric("CV Score", f"{score:.1f}/100")
                st.progress(score/100)

if __name__ == "__main__":
    main()