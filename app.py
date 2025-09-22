from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict
import os
import psutil
import logging

# Import the router and TOGETHER_MODEL from your pdf_processor.py file
# Assuming pdf_processor.py is in the same directory as app.py
from pdf_processor import router as pdf_processor_router, GROQ_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="NexNotes AI Backend API")

# Update CORS for frontend deployment (e.g., Vercel, local development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://*.vercel.app",  # For Vercel deployments
        "http://localhost:3000", # For local React development
        "*"                      # Wildcard for broader access during development/testing.
                                 # For production, it's safer to list specific domains.
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Include the router from pdf_processor.py ---
# This registers all endpoints defined in pdf_processor.py under the root path
app.include_router(pdf_processor_router)

# --- Utility function to get memory usage (retained for health check) ---
def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # in MB

# --- Simplified general API endpoints ---

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    memory_usage = get_memory_usage()
    return {
        "status": "healthy",
        "memory_usage_mb": f"{memory_usage:.2f}",
        "active_ai_system": "Together AI (via pdf_processor)",
        "message": "Backend is running and ready to process requests for summaries, questions, and flashcards."
    }

@app.get("/")
async def root():
    return {"message": "NexNotes AI Backend is running. Visit /docs for API documentation."} 

@app.get("/v1/models")
async def get_available_models():
    """
    Returns information about the primary AI model used in the backend.
    """
    return {
        "models": [
            {
                "id": GROQ_MODEL,
                "name": GROQ_MODEL.split('/')[-1], # Extract model name from TOGETHER_MODEL path
                "description": "Powerful instruction-tuned language model for text generation, summarization, question answering, and flashcard creation."
            }
        ]
    }