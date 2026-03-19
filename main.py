import subprocess
import requests
import json
import os
import gc

class VideoTranslatorPipeline:
    def __init__(self, video_path, target_language="es"):
        self.video_path = video_path
        self.target_language = target_language
        self.base_name = os.path.splitext(video_path)[0]
        self.audio_track_orig = f"{self.base_name}_original_track.wav"
        self.json_path = f"{self.base_name}_data.json"
        self.chunks_dir = f"{self.base_name}_chunks"
        self.final_video = f"{self.base_name}_DOBLADO_FLUIDO.mp4"

    def run_pipeline(self):
        print(f"--- 🚀 INICIANDO PRODUCCIÓN DE DOBLAJE PROFESIONAL ---")
        
        # 0. Limpieza de VRAM inicial
        self.unload_ollama()

        # 1. Extracción de audio para Whisper
        print("\n1. [FFmpeg] Extrayendo pista de audio...")
        self.extract_audio()

        # 2. Transcripción con Whisper
        print("\n2. [Whisper] Generando transcripción y tiempos...")
        pipeline_data = self.transcribe_audio()

        # 3. Traducción con Gemma 2 9B
        print(f"\n3. [Gemma 2] Traduciendo al {self.target_language}...")
        total_items = len(pipeline_data)
        for i, item in enumerate(pipeline_data):
            print(f"   -> Traduciendo segmento {item['id']}/{total_items}...")
            is_last = (i == total_items - 1)
            item['translated_text'] = self.translate_text(item['original_text'], unload_after=is_last)
            self.save_json(pipeline_data)

        # 4. Generación de Voz con Kokoro
        print(f"\n4. [Kokoro] Generando locuciones sincronizadas...")
        pipeline_data = self.generate_voice_segments(pipeline_data)
        self.save_json(pipeline_data)

        # 5. Ensamble Final de Pista Única
        print(f"\n5. [FFmpeg] Ensamblando video con mezcla fluida...")
        self.assemble_final_video(pipeline_data)

        return pipeline_data

    def unload_ollama(self):
        """Libera la VRAM de la GPU enviando una señal de keep_alive=0"""
        url = "http://localhost:11434/api/generate"
        payload = {"model": "gemma2:9b", "keep_alive": 0}
        try: requests.post(url, json=payload, timeout=5)
        except: pass

    def extract_audio(self):
        command = [
            "ffmpeg", "-i", self.video_path, 
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", 
            self.audio_track_orig, "-y"
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def transcribe_audio(self):
        from faster_whisper import WhisperModel
        # Usamos float16 para optimizar tus 4GB de VRAM
        model = WhisperModel("turbo", device="cuda", compute_type="float16")
        segments, _ = model.transcribe(self.audio_track_orig, beam_size=5)
        
        data = []
        for i, segment in enumerate(segments):
            data.append({
                "id": i + 1,
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "duration": max(0.1, round(segment.end - segment.start, 2)),
                "original_text": segment.text.strip(),
                "translated_text": None,
                "audio_file": None
            })
        
        del model
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except: pass
        return data

    def translate_text(self, text, unload_after=False):
        url = "http://localhost:11434/api/generate"
        prompt = (f"Eres un traductor de doblaje experto. Traduce al español: '{text}'. "
                  f"IMPORTANTE: Mantén la frase breve y concisa para que quepa en el tiempo. "
                  f"Responde solo con la traducción.")
        
        payload = {
            "model": "gemma2:9b",
            "prompt": prompt, 
            "stream": False,
            "keep_alive": 0 if unload_after else "5m",
            "options": {"temperature": 0.2}
        }
        try:
            response = requests.post(url, json=payload, timeout=60)
            return response.json().get("response", "").strip().replace('"', '')
        except Exception as e:
            return f"[Error: {e}]"

    def generate_voice_segments(self, pipeline_data):
        from kokoro import KPipeline
        import soundfile as sf
        import numpy as np

        # Idioma e para español, a para inglés (americano)
        lang = 'e' if self.target_language.lower().startswith('es') else 'a'
        pipeline = KPipeline(lang_code=lang)
        
        if not os.path.exists(self.chunks_dir):
            os.makedirs(self.chunks_dir)

        for item in pipeline_data:
            texto = item['translated_text']
            if not texto or "[Error]" in texto: continue

            # Seleccionar voz femenina de alta calidad
            voz = 'ef_dora' if lang == 'e' else 'af_bella'
            
            # Generar audio con velocidad ligeramente superior para asegurar sincronía
            generator = pipeline(texto, voice=voz, speed=1.1)
            audios = [audio for _, _, audio in generator]
            
            if audios:
                final_audio = np.concatenate(audios)
                ruta_wav = os.path.join(self.chunks_dir, f"seg_{item['id']:03d}.wav")
                sf.write(ruta_wav, final_audio, 24000)
                item['audio_file'] = ruta_wav
        
        del pipeline
        gc.collect()
        return pipeline_data

    def assemble_final_video(self, pipeline_data):
        valid_segments = [item for item in pipeline_data if item['audio_file'] and os.path.exists(item['audio_file'])]
        
        if not valid_segments:
            print("Error: No hay audios válidos para procesar.")
            return

        inputs = ["-i", self.video_path]
        filter_complex = ""
        
        for i, item in enumerate(valid_segments):
            inputs.extend(["-i", item['audio_file']])
            idx = i + 1
            
            # Medimos duración real del audio generado
            prob_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", item['audio_file']]
            actual_dur = float(subprocess.check_output(prob_cmd).decode().strip())
            
            # Sincronización: delay según el segundo de inicio de Whisper
            # afade: suavizado de 0.1s al final para evitar cortes secos
            filter_complex += (
                f"[{idx}:a]afade=t=out:st={actual_dur-0.1}:d=0.1,"
                f"adelay={int(item['start']*1000)}|{int(item['start']*1000)}[a{idx}];"
            )

        # Unión de todas las pistas con amix
        # dropout_transition=0 permite que si un audio se alarga, se mezcle con el siguiente
        labels = "".join([f"[a{i+1}]" for i in range(len(valid_segments))])
        filter_complex += f"{labels}amix=inputs={len(valid_segments)}:duration=longest:dropout_transition=0:normalize=0[audio_out]"

        cmd = ["ffmpeg"] + inputs + [
            "-filter_complex", filter_complex,
            "-map", "0:v", "-map", "[audio_out]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            self.final_video, "-y"
        ]

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✨ ¡ÉXITO! Video final guardado como: {self.final_video}")

    def save_json(self, data):
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    video_input = "video_prueba.mp4" 
    if os.path.exists(video_input):
        pipeline = VideoTranslatorPipeline(video_input)
        pipeline.run_pipeline()
    else:
        print(f"El archivo {video_input} no existe.")