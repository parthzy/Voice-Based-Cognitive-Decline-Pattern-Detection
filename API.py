from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile
import os
from speech_processor import process_audio_file

app = FastAPI(
    title="MemoTag Speech Intelligence API",
    description="API for analyzing speech patterns and detecting cognitive impairment risks",
    version="1.0.0"
)

@app.post("/analyze-speech/")
async def analyze_speech(audio_file: UploadFile = File(...)):
    """
    Analyze speech audio file for cognitive impairment risk factors.
    
    Parameters:
    - audio_file: Audio file in WAV format
    
    Returns:
    - Dictionary containing extracted features and risk score
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Process the audio file
        result = process_audio_file(temp_path)
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        return JSONResponse(
            content={
                "status": "success",
                "data": result
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"} 
