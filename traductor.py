import subprocess
import requests
import json
import os
import gc
import torch
import re
from num2words import num2words


class VideoTranslatorPipeline:

    # -------------------------------------------------

    def __init__(self, video_path, target_language="es", source_language=None):

        self.video_path = video_path
        self.target_language = target_language
        self.source_language = source_language  # None = Whisper autodetecta
        self.base_name = os.path.splitext(video_path)[0]

        self.audio_track_orig = f"{self.base_name}_original_track.wav"
        self.no_vocals_track = os.path.join("separated", "htdemucs", f"{self.base_name}_original_track", "no_vocals.wav")
        self.json_path = f"{self.base_name}_data.json"
        self.chunks_dir = f"{self.base_name}_chunks"
        self.final_video = f"{self.base_name}_DOBLADO.mp4"

    # -------------------------------------------------

    def free_gpu(self):

        print("Liberando VRAM...")

        try:
            subprocess.run(["pkill", "-u", os.getenv("USER", ""), "ollama"],
                           stderr=subprocess.DEVNULL)
        except Exception:
            pass

        self.clear_vram()

    # -------------------------------------------------

    def clear_vram(self):

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # -------------------------------------------------

    def run_pipeline(self):

        print(" INICIANDO PIPELINE")

        self.free_gpu()

        self.extract_audio()

        self.separate_audio()

        whisper_data = self.transcribe_audio()

        speaker_data = self.detect_speakers()

        pipeline_data = self.assign_speakers(whisper_data, speaker_data)

        self.save_json(pipeline_data)

        for item in pipeline_data:

            text = item["original_text"]

            item["translated_text"] = self.translate_text(text)

            self.save_json(pipeline_data)

        pipeline_data = self.generate_voice_segments(pipeline_data)

        self.assemble_final_video(pipeline_data)

        print(" Proceso finalizado")

    # -------------------------------------------------

    def extract_audio(self):

        print("Extrayendo audio...")

        cmd = [
            "ffmpeg", "-i", self.video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            self.audio_track_orig, "-y"
        ]

        subprocess.run(cmd)

    # -------------------------------------------------

    def separate_audio(self):

        print("Separando voces y música con Demucs...")

        cmd = ["demucs", "--two-stems=vocals", self.audio_track_orig]
        subprocess.run(cmd)

    # -------------------------------------------------

    def transcribe_audio(self):

        print("Whisper transcribiendo...")

        from faster_whisper import WhisperModel

        # Intentar en GPU, caer a CPU si no hay VRAM
        try:
            model = WhisperModel(
                "small",
                device="cuda",
                compute_type="int8_float16",
                cpu_threads=4,
                num_workers=1
            )
            print("Whisper cargado en CUDA")
        except (RuntimeError, Exception) as e:
            print(f"VRAM insuficiente para Whisper ({e}), usando CPU...")
            self.clear_vram()
            model = WhisperModel(
                "small",
                device="cpu",
                compute_type="int8",
                cpu_threads=4,
                num_workers=1
            )
            print("Whisper cargado en CPU")

        segments, _ = model.transcribe(
            self.audio_track_orig,
            beam_size=5,
            vad_filter=True,
            language=self.source_language  # None = autodetectar
        )

        data = []

        for i, seg in enumerate(segments):

            data.append({
                "id": i + 1,
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "duration": round(seg.end - seg.start, 2),
                "speaker": "unknown",
                "original_text": seg.text.strip(),
                "translated_text": None,
                "audio_file": None
            })

        del model
        self.clear_vram()

        return data

    # -------------------------------------------------

    def detect_speakers(self):

        print("Detectando hablantes...")

        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token="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"
        )

        pipeline.to(torch.device("cpu"))

        diarization = pipeline(self.audio_track_orig)
        diarization = diarization.speaker_diarization

        speakers = []

        for segment, track, speaker in diarization.itertracks(yield_label=True):

            print(f"{speaker}: {segment.start:.2f} -> {segment.end:.2f}")

            speakers.append({
                "start": segment.start,
                "end": segment.end,
                "speaker": speaker
            })

        del pipeline
        self.clear_vram()

        return speakers

    # -------------------------------------------------

    def assign_speakers(self, whisper_data, speaker_data):

        print("Asignando hablantes al texto...")

        last_known_speaker = speaker_data[0]["speaker"] if speaker_data else "SPEAKER_00"

        for segment in whisper_data:

            start = segment["start"]
            end = segment["end"]

            speaker = "UNKNOWN"

            for sp in speaker_data:

                if start >= sp["start"] and end <= sp["end"]:
                    speaker = sp["speaker"]
                    break

            if speaker == "UNKNOWN":
                speaker = last_known_speaker
            else:
                last_known_speaker = speaker

            segment["speaker"] = speaker

        return whisper_data

    # -------------------------------------------------

    def translate_text(self, text):

        url = "http://localhost:11434/api/generate"

        prompt = (
            f"Translate the following segment into Spanish. "
            f"IMPORTANT: Do not translate proper names, locations, or brands; keep them as they are in the original or their romanized form. "
            f"Make the translation concise. "
            f"Text: {text}"
        )

        payload = {
            "model": "demonbyron/HY-MT1.5-7B",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1
            }
        }

        r = requests.post(url, json=payload, timeout=60)
        raw = r.json()["response"].strip()

        return self._clean_translation(raw)

    # -------------------------------------------------

    def _clean_translation(self, text):
        """Elimina preámbulos, notas y basura que el modelo pueda añadir."""

        # Eliminar posibles repeticiones del prompt
        prompt_leak_patterns = [
            r"(?i)(?:es\s+|son\s+)?importante[s]?[\s:]*no\s+.*?concisa\.",
            r"(?i)importante:\s*no\s*traduzca.*?concisa\.",
        ]

        # Eliminar líneas que parezcan explicaciones del modelo
        noise_patterns = [
            r"(?i)^(claro[,.]?|aquí tienes[:]?|la traducción[:]?|traducción[:]?|"
            r"here('s| is)[:]?|sure[,.]?|of course[,.]?)[^\n]*\n*",
            r"(?i)\n*\(.*?(nota|note|traducción|translation).*?\)",
            r"(?i)\n*\*.*?\*",                # *texto en asteriscos*
            r"(?i)\n*(note|nota)[:\s].*$",    # líneas que empiezan con Nota:
        ]

        result = text
        for pattern in prompt_leak_patterns + noise_patterns:
            result = re.sub(pattern, "", result, flags=re.MULTILINE | re.DOTALL)

        # Si quedaron múltiples líneas vacías, limpiar
        result = re.sub(r"\n{3,}", "\n\n", result)

        # Si después de limpiar hay una línea de intro seguida del texto real,
        # quedarse solo con la última parte significativa
        lines = [l.strip() for l in result.strip().splitlines() if l.strip()]
        if lines:
            result = " ".join(lines)

        return result.strip()

    # -------------------------------------------------

    def _normalize_numbers(self, text):
        """Reemplaza los números en formato dígito por su escritura, útil para Kokoro"""
        
        def replace_num(match):
            num_str = match.group(0).replace(",", "").replace(".", "")
            try:
                # Convertir a texto en español
                return num2words(int(num_str), lang='es')
            except ValueError:
                return match.group(0)

        # Buscar números enteros solos o con formato de miles (8,000 / 8000 / 8.000)
        return re.sub(r'\b\d{1,3}([,.]?\d{3})*\b', replace_num, text)

    # -------------------------------------------------

    def generate_voice_segments(self, data):

        print("Generando TTS...")

        # Liberar agresivamente antes de cargar Kokoro
        self.free_gpu()

        from kokoro import KPipeline
        import soundfile as sf
        import numpy as np

        # Intentar en GPU; si no hay VRAM suficiente, caer a CPU
        device = 'cuda'
        try:
            pipeline = KPipeline(lang_code='e', device=device)
            print(f"Kokoro cargado en {device.upper()}")
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            print(f"VRAM insuficiente ({e}), cambiando a CPU...")
            self.clear_vram()
            device = 'cpu'
            pipeline = KPipeline(lang_code='e', device=device)
            print("Kokoro cargado en CPU")

        if not os.path.exists(self.chunks_dir):
            os.makedirs(self.chunks_dir)

        for item in data:

            text = item["translated_text"]
            speaker = item["speaker"]

            if not text:
                continue

            # Kokoro lee los números dígito a dígito (ej: 8000 -> 8 0 0 0)
            # Normalizamos convirtiéndolos en texto
            text_for_tts = self._normalize_numbers(text)

            if speaker == "SPEAKER_00":
                voice = "ef_dora"
            else:
                voice = "ef_dora"

            try:
                generator = pipeline(text_for_tts, voice=voice, speed=1.1)
                audios = [audio for _, _, audio in generator]
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                print(f"OOM en segmento {item['id']}, reintentando en CPU...")
                self.clear_vram()
                del pipeline
                pipeline = KPipeline(lang_code='e', device='cpu')
                generator = pipeline(text, voice=voice, speed=1.1)
                audios = [audio for _, _, audio in generator]

            if audios:

                audio = np.concatenate(audios)

                path = os.path.join(
                    self.chunks_dir,
                    f"seg_{item['id']:03d}.wav"
                )

                sf.write(path, audio, 24000)

                item["audio_file"] = path
                item["audio_duration"] = round(len(audio) / 24000, 3)

        del pipeline
        self.clear_vram()

        return data

    # -------------------------------------------------

    def assemble_final_video(self, data):

        print("Ensamblando video...")

        valid = [x for x in data if x["audio_file"]]

        has_bg = getattr(self, "no_vocals_track", None) and os.path.exists(self.no_vocals_track)

        inputs = ["-i", self.video_path]
        if has_bg:
            inputs += ["-i", self.no_vocals_track]
            audio_start_idx = 2
        else:
            audio_start_idx = 1

        filter_complex = ""

        for i, item in enumerate(valid):

            inputs += ["-i", item["audio_file"]]

            idx = i + audio_start_idx
            delay = int(item["start"] * 1000)
            slot = item["duration"]
            audio_dur = item.get("audio_duration", slot)

            parts = f"[{idx}:a]"

            if audio_dur > slot:
                # Acelerar para que quepa en el slot (máx 1.4x)
                speed = min(round(audio_dur / slot, 4), 1.4)
                parts += f"atempo={speed},"
                print(f"  Seg {item['id']}: {audio_dur:.2f}s > slot {slot:.2f}s → atempo={speed:.2f}x")

            # Recortar al slot y aplicar delay
            parts += f"atrim=duration={slot},adelay={delay}|{delay}[a{idx}];"

            filter_complex += parts

        labels = "".join([f"[a{i+audio_start_idx}]" for i in range(len(valid))])

        if has_bg:
            labels = f"[1:a]{labels}"
            mix_count = len(valid) + 1
        else:
            mix_count = len(valid)

        filter_complex += (
            f"{labels}amix=inputs={mix_count}"
            ":duration=longest:normalize=0[aout]"
        )

        cmd = ["ffmpeg"] + inputs + [
            "-filter_complex", filter_complex,
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac",
            self.final_video, "-y"
        ]

        subprocess.run(cmd)

    # -------------------------------------------------

    def save_json(self, data):

        with open(self.json_path, "w", encoding="utf-8") as f:

            json.dump(data, f, indent=4, ensure_ascii=False)


# -----------------------------------------------------

# Códigos de idioma comunes para referencia
IDIOMAS = {
    "1": ("ja", "Japonés"),
    "2": ("en", "Inglés"),
    "3": ("zh", "Chino"),
    "4": ("ko", "Coreano"),
    "5": ("fr", "Francés"),
    "6": ("de", "Alemán"),
    "7": ("pt", "Portugués"),
    "8": ("it", "Italiano"),
    "9": ("ru", "Ruso"),
    "0": (None, "Autodetectar"),
}

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Doblador de video con IA")
    parser.add_argument("video", nargs="?", help="Ruta del video a doblar")
    parser.add_argument("--lang", default="es", help="Idioma destino (default: es)")
    parser.add_argument("--source", default=None, help="Idioma fuente (ej: ja, en, zh). Sin valor = autodetectar")
    args = parser.parse_args()

    video = args.video
    source_lang = args.source

    # Si no se pasó argumento de video, mostrar menú interactivo
    if not video:

        videos = sorted([
            f for f in os.listdir(".")
            if f.lower().endswith((".mp4", ".mkv", ".avi", ".mov", ".webm"))
        ])

        if not videos:
            print(" No se encontraron videos en la carpeta actual.")
            exit(1)

        print("\n Videos disponibles:")
        for i, v in enumerate(videos, 1):
            print(f"  [{i}] {v}")

        while True:
            try:
                opcion = int(input("\n¿Cuál quieres doblar? (número): "))
                if 1 <= opcion <= len(videos):
                    video = videos[opcion - 1]
                    break
                print(f"Elige un número entre 1 y {len(videos)}.")
            except ValueError:
                print("Escribe un número válido.")

    # Si no se pasó --source, preguntar el idioma fuente
    if source_lang is None:

        print("\n Idioma del video original:")
        for k, (code, nombre) in IDIOMAS.items():
            print(f"  [{k}] {nombre}" + (f" ({code})" if code else ""))

        opcion_lang = input("\n¿Cuál es el idioma del video? (número): ").strip()
        code, nombre = IDIOMAS.get(opcion_lang, (None, "Autodetectar"))
        source_lang = code
        print(f"✔ Idioma seleccionado: {nombre}")

    if not os.path.exists(video):
        print(f" Video no encontrado: {video}")
        exit(1)

    print(f"\n▶ Procesando: {video}")
    pipeline = VideoTranslatorPipeline(
        video,
        target_language=args.lang,
        source_language=source_lang
    )
    pipeline.run_pipeline()
