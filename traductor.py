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
        self.source_language = source_language  # None activa autodección en Whisper.
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

        print("Iniciando pipeline...")

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

        print("Proceso finalizado.")

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

        # Intento de carga en GPU, con fallback a CPU si falta VRAM.
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
            language=self.source_language  # None activa autodetectado
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

        import os
        hf_token = os.environ.get("HF_TOKEN", None)
        try:
            # Carga desde caché local si los modelos están descargados
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=hf_token
            )
        except Exception as e:
            print(f"[PYANNOTE] Error cargando modelo: {e}")
            raise

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

        lang_names_en = {"es": "Spanish", "en": "English", "fr": "French", "ja": "Japanese", "zh": "Chinese"}
        lang_names_zh = {"es": "西班牙语", "en": "英语", "fr": "法语", "ja": "日语", "zh": "中文"}
        
        tgt_en = lang_names_en.get(self.target_language, "Spanish")
        tgt_zh = lang_names_zh.get(self.target_language, "西班牙语")
        
        # Verificación de idioma chino
        is_zh = getattr(self, "source_language", None) == "zh" or self.target_language == "zh"
        
        if is_zh:
            prompt = (
                f"将以下文本翻译为{tgt_zh}，只输出译文，不要解释：\n{text}"
            )
        else:
            prompt = (
                f"Translate to {tgt_en}. Output only the translation, nothing else:\n{text}"
            )

        payload = {
            "model": "demonbyron/HY-MT1.5-7B",
            "prompt": prompt,
            "stream": False,
            "options": {
                "top_k": 20,
                "top_p": 0.6,
                "repeat_penalty": 1.05,
                "temperature": 0.7
            }
        }

        r = requests.post(url, json=payload, timeout=60)
        raw = r.json()["response"].strip()

        return self._clean_translation(raw)

    # -------------------------------------------------

    def _clean_translation(self, raw, original_prompt=None):
        """Elimina texto adicional o reglas incluidas por el modelo."""

        text = raw.strip()

        # Patrones de frases introductorias
        intro_patterns = [
            r"(?i)^(claro[,.]?|por supuesto[,.]?|aquí tienes[:]?|here(?:'s| is)[:]?|sure[,.]?|of course[,.]?|la traducción[:]?|traducción[:]?)\s*\n+",
            r"(?i)^(claro[,.]?|por supuesto[,.]?|aquí tienes[:]?|here(?:'s| is)[:]?|sure[,.]?|of course[,.]?|la traducción[:]?|traducción[:]?)\s+",
        ]

        # Patrones de instrucciones reflejadas por el modelo
        rule_patterns = [
            r"(?i)(translate\s+to\s+\w+\.?\s*output\s+only.*?:\s*\n*)",
            r"(?i)(output\s+only\s+the\s+translation.*?:\s*\n*)",
            r"(?i)(将以下.*?不要.*?：\s*\n*)",
            r"(?i)(critical\s+rules?:.*?)(\n|$)",
            r"(?i)^\d+\.\s+DO\s+NOT.*$",
            r"(?i)^\d+\.\s+Keep.*$",
            r"(?i)(注意只需要输出翻译后的结果.*?(\n|$))",
        ]

        # Notas al final del texto
        suffix_patterns = [
            r"(?i)\n+(note|nota)[:\s].*$",
            r"(?i)\n+\(.*?(nota|note|traducción|translation).*?\)",
            r"(?i)\n+\*.*?\*",
        ]

        result = text
        for pattern in intro_patterns + rule_patterns + suffix_patterns:
            result = re.sub(pattern, "", result, flags=re.MULTILINE | re.DOTALL)

        # Extrae la traducción omitiendo el texto original si fue incluido
        lines = [l.strip() for l in result.strip().splitlines() if l.strip()]
        if lines:
            result = " ".join(lines)

        return result.strip()

    # -------------------------------------------------

    def _normalize_numbers(self, text):
        """Convierte números a texto para la síntesis de voz."""
        
        def replace_num(match):
            num_str = match.group(0).replace(",", "").replace(".", "")
            try:
                # Conversión a texto en español
                return num2words(int(num_str), lang='es')
            except ValueError:
                return match.group(0)

        # Búsqueda de números enteros y formatos de miles
        return re.sub(r'\b\d{1,3}([,.]?\d{3})*\b', replace_num, text)

    # -------------------------------------------------

    def generate_voice_segments(self, data, item_id=None):

        print("Generando TTS...")

        # Liberación de memoria previa a la carga de Kokoro
        self.free_gpu()

        from kokoro import KPipeline
        import soundfile as sf
        import numpy as np

        # Fallback a CPU en caso de memoria VRAM insuficiente
        device = 'cuda'
        
        pipelines = {}
        def get_pipeline(lang_code):
            if lang_code not in pipelines:
                try:
                    pipelines[lang_code] = KPipeline(lang_code=lang_code, device=device)
                    print(f"Kokoro ({lang_code}) cargado en {device.upper()}")
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    print(f"VRAM insuficiente ({e}), cambiando a CPU...")
                    self.clear_vram()
                    fallback = 'cpu'
                    pipelines[lang_code] = KPipeline(lang_code=lang_code, device=fallback)
                    print(f"Kokoro ({lang_code}) cargado en CPU")
            return pipelines[lang_code]

        if not os.path.exists(self.chunks_dir):
            os.makedirs(self.chunks_dir)

        for item in data:
            if item_id is not None and item["id"] != item_id:
                continue

            text = item.get("translated_text", "")
            speaker = item.get("speaker", "SPEAKER_00")

            if not text:
                continue

            # Normalización de números para síntesis en Kokoro
            text_for_tts = self._normalize_numbers(text)

            voice = item.get("voice", "ef_dora")
            if not voice:
                voice = "ef_dora"
            
            # Persistencia de la voz seleccionada en el objeto
            item["voice"] = voice

            print(f"[DEBUG-VOICE] Segmento {item['id']} ha solicitado explícitamente la voz: {voice}")

            lang_code = voice[0]
            pipeline = get_pipeline(lang_code)

            try:
                generator = pipeline(text_for_tts, voice=voice, speed=1.1)
                audios = [audio for _, _, audio in generator]
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                print(f"OOM en segmento {item['id']}, reintentando en CPU...")
                self.clear_vram()
                fallback_pipe = KPipeline(lang_code=lang_code, device='cpu')
                pipelines[lang_code] = fallback_pipe
                generator = fallback_pipe(text_for_tts, voice=voice, speed=1.1)
                audios = [audio for _, _, audio in generator]

            if audios:

                audio = np.concatenate(audios)

                path = os.path.join(
                    self.chunks_dir,
                    f"seg_{item['id']:03d}_{voice}.wav"
                )

                sf.write(path, audio, 24000)

                item["audio_file"] = path
                item["audio_duration"] = round(float(len(audio)) / 24000.0, 3)

        del pipeline
        self.clear_vram()

        return data

    # -------------------------------------------------

    def _get_video_duration(self):
        """Obtiene la duración total del video usando ffprobe."""
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", self.video_path],
            capture_output=True, text=True
        )
        return float(result.stdout.strip())

    def _detect_video_encoder(self):
        """Detecta el mejor encoder de video disponible en este sistema."""
        candidates = [
            ("h264_nvenc",  ["-c:v", "h264_nvenc"]),
            ("libx264",     ["-c:v", "libx264", "-preset", "fast", "-crf", "18"]),
            ("libx265",     ["-c:v", "libx265", "-preset", "fast", "-crf", "22"]),
            ("mpeg4",       ["-c:v", "mpeg4", "-q:v", "4"]),
            ("libvpx",      ["-c:v", "libvpx", "-crf", "10", "-b:v", "0"]),
        ]
        for name, flags in candidates:
            result = subprocess.run(
                ["ffmpeg", "-f", "lavfi", "-i", "color=black:s=64x64:d=0.1",
                 "-vframes", "1"] + flags + ["-f", "null", "-"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"[ENCODER] Usando encoder de video: {name}")
                return flags
        # Fallback a copia sin re-encoding
        print("[ENCODER] Ningún encoder encontrado, intentando copy...")
        return ["-c:v", "copy"]

    # -------------------------------------------------

    def assemble_final_video(self, data):
        """
        Ensamblado con sincronía híbrida:
        1. Aceleración de audio (hasta 1.25x) si excede la duración original.
        2. Ajuste de segmento de video si la aceleración de audio no es suficiente.
        Requiere libx264.
        """
        print("Ensamblando video con sincronía híbrida...")

        ATEMPO_THRESHOLD = 1.25

        valid = sorted(
            [x for x in data if x.get("audio_file") and os.path.exists(x["audio_file"])],
            key=lambda x: x["start"]
        )

        if not valid:
            raise RuntimeError("No hay segmentos de audio válidos para ensamblar.")

        # Procesamiento de pista de fondo
        bg_path = os.path.abspath(self.no_vocals_track)
        has_bg = os.path.exists(bg_path)
        if has_bg:
            print(f"[BG] Usando pista de fondo: {bg_path}")
        else:
            print("[BG] No se encontró pista de fondo.")

        # Cálculo de la duración total del video
        try:
            video_total_dur = self._get_video_duration()
        except Exception:
            video_total_dur = valid[-1]["end"] + 2.0

        # ------------------------------------------------------------------
        # Fase 1: Cálculo de nueva línea de tiempo
        # ------------------------------------------------------------------
        # video_segs: list of (v_start, v_end, stretch_factor)
        # audio_segs: list of (audio_file, atempo, new_delay_seconds)

        video_segs = []
        audio_segs = []
        new_cursor = 0.0
        prev_v_end = 0.0

        for item in valid:
            v_start = float(item["start"])
            v_end   = float(item["end"])
            slot    = v_end - v_start
            if slot <= 0:
                continue

            audio_dur = float(item.get("audio_duration", slot))
            ratio = audio_dur / slot

            # Gap previo al segmento
            if v_start > prev_v_end + 0.01:
                gap = v_start - prev_v_end
                video_segs.append((prev_v_end, v_start, 1.0))
                new_cursor += gap

            # Capa 1: Ajuste de velocidad del audio
            if ratio <= ATEMPO_THRESHOLD:
                atempo      = float(max(1.0, round(ratio, 4)))
                v_stretch   = 1.0
                effective_a = audio_dur / atempo
            else:
                # Capa 2: Estiramiento de video
                atempo    = ATEMPO_THRESHOLD
                effective_a = audio_dur / atempo
                v_stretch = effective_a / slot
                print(f"  Seg {item['id']}: ratio={ratio:.2f}x → atempo={atempo}x, "
                      f"video_stretch={v_stretch:.2f}x")

            video_segs.append((v_start, v_end, v_stretch))
            audio_segs.append((item["audio_file"], atempo, new_cursor))

            new_cursor += effective_a
            prev_v_end = v_end

        # Manejo de gap final
        if prev_v_end < video_total_dur - 0.01:
            video_segs.append((prev_v_end, video_total_dur, 1.0))

        # ------------------------------------------------------------------
        # Fase 2: Codificación y ensamblado de video y audio
        # ------------------------------------------------------------------
        import tempfile, shutil

        tmp_dir = tempfile.mkdtemp(prefix="doblado_")
        seg_files = []

        try:
            # Paso A: Codificación de segmentos de video a temporales
            for i, (vs, ve, stretch) in enumerate(video_segs):
                tmp_out = os.path.join(tmp_dir, f"vseg_{i:03d}.ts")
                seg_cmd = [
                    "ffmpeg",
                    "-ss", f"{vs:.4f}", "-to", f"{ve:.4f}",
                    "-i", self.video_path,
                    "-vf", f"setpts=(PTS-STARTPTS)*{stretch:.6f}",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-an",
                    "-avoid_negative_ts", "make_zero",
                    "-f", "mpegts",
                    tmp_out, "-y"
                ]
                print(f"  [V-seg {i}] {vs:.2f}→{ve:.2f}s x{stretch:.2f}")
                res = subprocess.run(seg_cmd, capture_output=True, text=True)
                if res.returncode != 0:
                    print(f"  [V-seg {i}] ERROR: {res.stderr[-500:]}")
                    raise subprocess.CalledProcessError(res.returncode, seg_cmd, res.stderr)
                seg_files.append(tmp_out)

            # Paso B: Unión secuencial de segmentos de video
            concat_list = os.path.join(tmp_dir, "concat.txt")
            with open(concat_list, "w") as f:
                for sf in seg_files:
                    f.write(f"file '{sf}'\n")

            video_joined = os.path.join(tmp_dir, "video_joined.mp4")
            join_cmd = [
                "ffmpeg",
                "-f", "concat", "-safe", "0",
                "-i", concat_list,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-an",
                video_joined, "-y"
            ]
            
            print(f"[CONCAT] Uniendo {len(seg_files)} segmentos de video secuencialmente...")
            res = subprocess.run(join_cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"[CONCAT] ERROR: {res.stderr[-2000:]}")
                raise subprocess.CalledProcessError(res.returncode, join_cmd, res.stderr)

            # Paso C: Mezcla de audio final con video
            audio_inputs = []
            if has_bg:
                audio_inputs += ["-i", bg_path]
            for af, _, _ in audio_segs:
                audio_inputs += ["-i", af]

            a_offset_local = 1
            if has_bg:
                a_offset_local = 2

            fc = ""
            if has_bg:
                fc += "[1:a]volume=0.4[bg];"

            a_out_labels = []
            for i, (af, atempo, delay_s) in enumerate(audio_segs):
                idx = a_offset_local + i
                delay_ms = int(delay_s * 1000)
                chunk = f"[{idx}:a]dynaudnorm=p=0.9:m=100,"
                if atempo > 1.001:
                    chunk += f"atempo={atempo:.4f},"
                chunk += f"adelay={delay_ms}|{delay_ms}[ao{i}];"
                fc += chunk
                a_out_labels.append(f"[ao{i}]")

            if has_bg:
                all_a = "[bg]" + "".join(a_out_labels)
                mix_n = len(audio_segs) + 1
            else:
                all_a = "".join(a_out_labels)
                mix_n = len(audio_segs)

            fc += f"{all_a}amix=inputs={mix_n}:duration=longest:normalize=0[aout]"

            final_cmd = (
                ["ffmpeg", "-i", video_joined]
                + audio_inputs
                + [
                    "-filter_complex", fc,
                    "-map", "0:v",
                    "-map", "[aout]",
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    self.final_video, "-y"
                ]
            )

            print(f"[MIX] Añadiendo las pistas de audio ({len(audio_segs)})...")
            res = subprocess.run(final_cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"[MIX] ERROR: {res.stderr[-3000:]}")
                raise subprocess.CalledProcessError(res.returncode, final_cmd, res.stderr)

            print("Video ensamblado correctamente.")

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


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

    # Mostrar menú interactivo si no se provee argumento de video
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

    # Solicitar el idioma de origen si no se proporciona
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
