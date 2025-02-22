# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pypdf
import io
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=3)

async def process_in_thread(func, *args):
    """Run CPU-intensive functions in thread pool"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, func, *args)

def extract_text_from_pdf(pdf_file: bytes) -> str:
    """Optimized PDF text extraction"""
    try:
        # Validate PDF format
        if not pdf_file.startswith(b'%PDF'):
            raise ValueError("Invalid PDF format")
            
        pdf_reader = pypdf.PdfReader(io.BytesIO(pdf_file))
        
        if len(pdf_reader.pages) == 0:
            raise ValueError("PDF has no pages")
            
        text = " ".join(
            page.extract_text() 
            for page in pdf_reader.pages 
            if page.extract_text().strip()
        )
        
        if not text.strip():
            raise ValueError("No text could be extracted from PDF")
            
        return text
    except Exception as e:
        logger.error(f"PDF extraction error: {str(e)}")
        raise ValueError(f"PDF processing error: {str(e)}")

@app.post("/")
async def shortlist_cv(file: UploadFile = File(...)):
    """Endpoint with improved error handling and timeout protection"""
    try:
        logger.debug(f"Received file: {file.filename}")
        logger.debug(f"Content type: {file.content_type}")
        
        # Validate file extension
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400, 
                detail="Only PDF files are supported"
            )
        
        # Read file content
        contents = await file.read()
        
        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Empty file received"
            )
        
        # Process PDF in thread pool
        try:
            text = await process_in_thread(extract_text_from_pdf, contents)
            logger.debug(f"Extracted text length: {len(text)}")
        except ValueError as ve:
            raise HTTPException(
                status_code=400,
                detail=str(ve)
            )
        except Exception as e:
            logger.error(f"Unexpected error in PDF processing: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing PDF: {str(e)}"
            )
        
        # Score CV in thread pool
        try:
            from model import cv_scorer
            score = await process_in_thread(cv_scorer.score_cv, text)
            return {"cv_score": score}
        except Exception as e:
            logger.error(f"Error in CV scoring: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error scoring CV: {str(e)}"
            )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}