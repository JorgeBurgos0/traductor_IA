# Guía de Migración y Configuración para Fedora 40/41

Este documento contiene las instrucciones exactas para restaurar este proyecto (`traductor_IA`) en una instalación limpia de Fedora. Está diseñado para que **tú (el humano)** hagas la parte del sistema operativo, y **Antigravity (la IA)** pueda leerlo y ejecutar el resto de los pasos automáticamente.

## 🧑‍💻 Fase 1: Lo que debes hacer tú (El Humano)

Antes de pedirle a Antigravity que configure el entorno, necesitas instalar los drivers de NVIDIA privativos a nivel de sistema (ya que requieren permisos root y reiniciar el equipo). Abre una terminal en tu nuevo Fedora y ejecuta:

1. **Habilitar repositorios RPM Fusion libres y no-libres:**
   ```bash
   sudo dnf install https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm https://mirrors.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm
   ```
2. **Instalar los drivers de video (Serie 570+) y soporte CUDA:**
   ```bash
   sudo dnf install akmod-nvidia xorg-x11-drv-nvidia-cuda
   ```
3. **¡REINICIA TU COMPUTADORA!** *(Nota vital: Espera unos 5 minutos antes de reiniciar para darle tiempo al sistema de compilar el módulo del kernel en segundo plano).*

---

## 🤖 Fase 2: Instrucciones para Antigravity

**PROMPT PARA TI:** Cuando estés en Fedora y abras tu editor en el proyecto, simplemente dile a Antigravity:
> *"Lee el archivo MIGRACION_FEDORA.md y ejecuta la Fase 2 automáticamente para configurar mi entorno."*

### ⚠️ INSTRUCCIONES INTERNAS PARA ANTIGRAVITY ⚠️
Antigravity: Si el usuario te ha pedido ejecutar este archivo de configuración, por favor realiza las siguientes tareas en orden de manera autónoma usando comandos de terminal (`run_command`):

#### 2.1. Instalar Dependencias del Sistema (`dnf`)
El proyecto necesita herramientas de compilación de C++, desarrollo de Python y codecs de video/audio libres de patentes (FFmpeg de RPM Fusion). Ejecuta:
```bash
sudo dnf group install "Development Tools" "Development Libraries" -y
sudo dnf install python3-devel ffmpeg -y
```

#### 2.2. Limpiar y Re-crear el Entorno Virtual (`venv`)
Es probable que el usuario haya arrastrado por error carpetas de entorno de Debian. Bórralas y crea un entorno limpio con el Python de Fedora:
```bash
# Eliminar posibles restos de Debian para evitar incompatibilidades binarias
rm -rf venv env .venv __pycache__

# Crear y activar entorno fresco
python3 -m venv venv
source venv/bin/activate

# Actualizar pip básico
pip install --upgrade pip
```

#### 2.3. Instalar Dependencias de Python (`requirements.txt`)
Verifica si el archivo `requirements.txt` existe. Si existe, instálalo:
```bash
source venv/bin/activate
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Falta el archivo requirements.txt."
fi
```

#### 2.4. Aceleración por GPU (CUDA / PyTorch)
*(Antigravity: Verifica en traductor.py si se requiere PyTorch, Transformers, TensorFlow o Whisper. Si es así, reinstala la versión de PyTorch específica para CUDA 12.4+ para asegurar la compatibilidad con los drivers NVIDIA 570+ que el usuario instaló).*
Ejemplo (ajústalo según lo que lea el proyecto):
```bash
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
