from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import shutil
import json
import subprocess
import uuid
from typing import Optional
from traductor import VideoTranslatorPipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)

pipelines = {}
project_status = {}

def update_status(project_id, status: str):
    project_status[project_id] = status

@app.post("/api/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...), target_language: str = Form("es")):
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    project_id = file.filename
    pipe = VideoTranslatorPipeline(file_path, target_language=target_language)
    pipelines[project_id] = pipe
    
    update_status(project_id, "extracting_audio")
    try:
        pipe.free_gpu()
        pipe.extract_audio()
        pipe.separate_audio()
        update_status(project_id, "idle")
    except Exception as e:
        update_status(project_id, f"error: {str(e)}")
        
    return {"project_id": project_id, "status": "Files extracted & separated"}

@app.post("/api/youtube")
async def download_youtube_video(background_tasks: BackgroundTasks, url: str = Form(...), target_language: str = Form("es"), source_language: str = Form(None)):
    import hashlib
    raw_hash = str(hashlib.md5(url.encode()).hexdigest())
    url_hash = raw_hash[0:8]
    project_id = f"yt_{url_hash}.mp4"
    file_path = f"uploads/{project_id}"
    
    cmd = ["yt-dlp", "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]", "--merge-output-format", "mp4", "-o", file_path, url]
    
    if not os.path.exists(file_path):
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Failed to download from YouTube: {e}"})
        
    pipe = VideoTranslatorPipeline(file_path, target_language=target_language, source_language=source_language)
    pipelines[project_id] = pipe
    
    update_status(project_id, "extracting_audio")
    try:
        pipe.free_gpu()
        if not os.path.exists(pipe.audio_track_orig):
            pipe.extract_audio()
        if not os.path.exists(pipe.no_vocals_track):
            pipe.separate_audio()
        update_status(project_id, "idle")
    except Exception as e:
        update_status(project_id, f"error: {str(e)}")
        
    return {"project_id": project_id, "status": "YouTube files extracted & separated"}

def process_transcribe(project_id):
    update_status(project_id, "transcribing")
    try:
        pipe = pipelines[project_id]
        whisper_data = pipe.transcribe_audio()
        speaker_data = pipe.detect_speakers()
        pipeline_data = pipe.assign_speakers(whisper_data, speaker_data)
        pipe.save_json(pipeline_data)
        update_status(project_id, "idle")
    except Exception as e:
        update_status(project_id, f"error: {str(e)}")

@app.post("/api/transcribe/{project_id}")
async def transcribe(project_id: str, background_tasks: BackgroundTasks):
    if project_id not in pipelines:
        return JSONResponse(status_code=404, content={"error": "Project not found"})
    background_tasks.add_task(process_transcribe, project_id)
    return {"message": "Transcription started"}

def process_translate(project_id):
    update_status(project_id, "translating")
    try:
        pipe = pipelines[project_id]
        with open(pipe.json_path, "r", encoding="utf-8") as f:
            pipeline_data = json.load(f)
            
        for item in pipeline_data:
            text = item.get("original_text", "")
            if text:
                item["translated_text"] = pipe.translate_text(text)
                pipe.save_json(pipeline_data)
        update_status(project_id, "idle")
    except Exception as e:
        update_status(project_id, f"error: {str(e)}")

@app.post("/api/translate/{project_id}")
async def translate(project_id: str, background_tasks: BackgroundTasks):
    if project_id not in pipelines:
        return JSONResponse(status_code=404, content={"error": "Project not found"})
    background_tasks.add_task(process_translate, project_id)
    return {"message": "Translation started"}

def process_generate_audio(project_id, item_id=None, voice_override=None):
    update_status(project_id, "generating_audio")
    try:
        pipe = pipelines[project_id]
        with open(pipe.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if voice_override:
            print(f"[INFO] Aplicando voz '{voice_override}' a {'segmento ' + str(item_id) if item_id else 'todos los segmentos'}")
            for item in data:
                if item_id is None or item["id"] == item_id:
                    item["voice"] = voice_override
            pipe.save_json(data)
        
        updated_data = pipe.generate_voice_segments(data, item_id=item_id)
        pipe.save_json(updated_data)
        update_status(project_id, "idle")
    except Exception as e:
        print(f"[ERROR generate_audio] {e}")
        update_status(project_id, f"error: {str(e)}")

@app.post("/api/generate-audio/{project_id}")
async def generate_audio(project_id: str, background_tasks: BackgroundTasks, voice: Optional[str] = Query(None)):
    if project_id not in pipelines:
        return JSONResponse(status_code=404, content={"error": "Project not found"})
    background_tasks.add_task(process_generate_audio, project_id, None, voice)
    return {"message": f"Audio generation started" + (f" with voice {voice}" if voice else "")}

@app.post("/api/generate-audio/{project_id}/{item_id}")
async def generate_audio_single(project_id: str, item_id: int, background_tasks: BackgroundTasks, voice: Optional[str] = Query(None)):
    if project_id not in pipelines:
        return JSONResponse(status_code=404, content={"error": "Project not found"})
    background_tasks.add_task(process_generate_audio, project_id, item_id, voice)
    return {"message": f"Audio generation started for segment {item_id}" + (f" with voice {voice}" if voice else "")}

def process_assemble(project_id):
    update_status(project_id, "assembling")
    try:
        pipe = pipelines[project_id]
        with open(pipe.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pipe.assemble_final_video(data)
        update_status(project_id, "done")
    except Exception as e:
        update_status(project_id, f"error: {str(e)}")

@app.post("/api/assemble/{project_id}")
async def assemble(project_id: str, background_tasks: BackgroundTasks):
    if project_id not in pipelines:
        return JSONResponse(status_code=404, content={"error": "Project not found"})
    background_tasks.add_task(process_assemble, project_id)
    return {"message": "Assembly started"}

@app.get("/api/status/{project_id}")
async def get_status(project_id: str):
    return {"status": project_status.get(project_id, "not_found")}

@app.get("/api/data/{project_id}")
async def get_data(project_id: str):
    if project_id not in pipelines:
        return []
    pipe = pipelines[project_id]
    if os.path.exists(pipe.json_path):
        with open(pipe.json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

@app.put("/api/data/{project_id}")
async def update_data(project_id: str, data: list):
    if project_id not in pipelines:
        return JSONResponse(status_code=404, content={"error": "Project not found"})
    pipe = pipelines[project_id]
    pipe.save_json(data)
    return {"message": "Data saved"}

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
