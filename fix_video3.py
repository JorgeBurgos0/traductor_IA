import json
import os
from traductor import VideoTranslatorPipeline

# Cargar JSON
with open("video_prueba3_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Instanciar solo para re-generar audios malos
pipeline = VideoTranslatorPipeline("video_prueba3.mp4")
pipeline.json_path = "video_prueba3_data.json"

ids_con_error = [34, 84, 101, 111]

for item in data:
    if item["id"] in ids_con_error:
        # Volver a limpiar usando el nuevo regex en traductor.py
        texto_limpio = pipeline._clean_translation(item["translated_text"])
        item["translated_text"] = texto_limpio
        print(f"ID {item['id']} corregido a: {texto_limpio}")
        
        # Eliminar el archivo de audio viejo para forzar que Kokoro lo regenere
        if item.get("audio_file") and os.path.exists(item["audio_file"]):
            os.remove(item["audio_file"])
            item["audio_file"] = None

# Guardar JSON parcheado
pipeline.save_json(data)

# Re-generar solo los faltantes y reensamblar
data = pipeline.generate_voice_segments(data)
pipeline.save_json(data)
pipeline.assemble_final_video(data)
print("Video reparado con éxito!")
