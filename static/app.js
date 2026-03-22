let projectId = null;
let currentData = [];
let pollInterval = null;
let currentAudio = null;

const statusPanel = document.getElementById('current-status');
const videoPreview = document.getElementById('video-preview');
const videoContainer = document.getElementById('video-container');

function switchTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

    event.target.classList.add('active');
    document.getElementById('tab-' + tabId).classList.add('active');
}

async function pollStatus() {
    if (!projectId) return;
    try {
        const res = await fetch(`/api/status/${projectId}`);
        const data = await res.json();

        statusPanel.innerText = data.status;

        if (data.status === 'idle') {
            clearInterval(pollInterval);
            pollInterval = null;
            await loadData();
            unlockSteps();
        } else if (data.status === 'done') {
            clearInterval(pollInterval);
            pollInterval = null;

            const baseName = projectId.substring(0, projectId.lastIndexOf('.')) || projectId;
            const videoUrl = `/uploads/${baseName}_DOBLADO.mp4`;

            statusPanel.innerHTML = `Video ensamblado exitosamente. <a href="${videoUrl}" target="_blank" style="color:#58a6ff;">Ver HD</a>`;

            videoPreview.src = videoUrl;
            videoContainer.classList.remove('empty');

            unlockSteps();
        } else if (data.status.startsWith('error')) {
            clearInterval(pollInterval);
            pollInterval = null;
            alert(data.status);
            unlockSteps();
        } else {
            if (data.status === 'translating') {
                await loadData();
            }
        }
    } catch (e) {
        console.error(e);
    }
}

function startPolling() {
    if (!pollInterval) {
        pollInterval = setInterval(pollStatus, 2000);
    }
}

function unlockSteps() {
    if (!projectId) return;
    document.getElementById('step-transcribe').classList.remove('disabled');
    document.getElementById('step-translate').classList.remove('disabled');
    document.getElementById('step-audio').classList.remove('disabled');
    document.getElementById('step-assemble').classList.remove('disabled');
}

function lockSteps() {
    document.getElementById('step-transcribe').classList.add('disabled');
    document.getElementById('step-translate').classList.add('disabled');
    document.getElementById('step-audio').classList.add('disabled');
    document.getElementById('step-assemble').classList.add('disabled');
}

async function loadData() {
    if (!projectId) return;
    // Prevención de caché para carga de JSON
    const res = await fetch(`/api/data/${projectId}?t=${new Date().getTime()}`);
    currentData = await res.json();

    if (currentData.length > 0) {
        document.getElementById('btn-save-data').classList.remove('hidden');
        renderTable();
    }
}

const voiceOptions = [
    // Español (Latino / España)
    { id: 'ef_dora', name: 'ES Dora (Mujer)' },
    { id: 'em_alex', name: 'ES Alex (Hombre)' },
    { id: 'em_santa', name: 'ES Santa (Hombre)' },

    // Inglés USA
    { id: 'af_heart', name: 'US Heart (Mujer)' },
    { id: 'af_alloy', name: 'US Alloy (Mujer)' },
    { id: 'af_aoede', name: 'US Aoede (Mujer)' },
    { id: 'af_bella', name: 'US Bella (Mujer)' },
    { id: 'af_jessica', name: 'US Jessica (Mujer)' },
    { id: 'af_kore', name: 'US Kore (Mujer)' },
    { id: 'af_nicole', name: 'US Nicole (Mujer)' },
    { id: 'af_nova', name: 'US Nova (Mujer)' },
    { id: 'af_river', name: 'US River (Mujer)' },
    { id: 'af_sarah', name: 'US Sarah (Mujer)' },
    { id: 'af_sky', name: 'US Sky (Mujer)' },
    { id: 'am_adam', name: 'US Adam (Hombre)' },
    { id: 'am_echo', name: 'US Echo (Hombre)' },
    { id: 'am_eric', name: 'US Eric (Hombre)' },
    { id: 'am_fenrir', name: 'US Fenrir (Hombre)' },
    { id: 'am_liam', name: 'US Liam (Hombre)' },
    { id: 'am_michael', name: 'US Michael (Hombre)' },
    { id: 'am_onyx', name: 'US Onyx (Hombre)' },
    { id: 'am_puck', name: 'US Puck (Hombre)' },

    // Inglés Británico
    { id: 'bf_alice', name: 'UK Alice (Mujer)' },
    { id: 'bf_emma', name: 'UK Emma (Mujer)' },
    { id: 'bf_isabella', name: 'UK Isabella (Mujer)' },
    { id: 'bf_lily', name: 'UK Lily (Mujer)' },
    { id: 'bm_daniel', name: 'UK Daniel (Hombre)' },
    { id: 'bm_fable', name: 'UK Fable (Hombre)' },
    { id: 'bm_george', name: 'UK George (Hombre)' },
    { id: 'bm_lewis', name: 'UK Lewis (Hombre)' },

    // Japonés
    { id: 'jf_alpha', name: 'JA Alpha (Mujer)' },
    { id: 'jf_gongitsune', name: 'JA Gongitsune (Mujer)' },
    { id: 'jf_nezumi', name: 'JA Nezumi (Mujer)' },
    { id: 'jf_tebukuro', name: 'JA Tebukuro (Mujer)' },
    { id: 'jm_kumo', name: 'JA Kumo (Hombre)' },

    // Chino Mandarín
    { id: 'zf_xiaobei', name: 'ZH Xiaobei (Mujer)' },
    { id: 'zf_xiaoni', name: 'ZH Xiaoni (Mujer)' },
    { id: 'zf_xiaoxiao', name: 'ZH Xiaoxiao (Mujer)' },
    { id: 'zf_xiaoyi', name: 'ZH Xiaoyi (Mujer)' },
    { id: 'zm_yunjian', name: 'ZH Yunjian (Hombre)' },
    { id: 'zm_yunxi', name: 'ZH Yunxi (Hombre)' },
    { id: 'zm_yunxia', name: 'ZH Yunxia (Hombre)' },
    { id: 'zm_yunyang', name: 'ZH Yunyang (Hombre)' },

    // Francés
    { id: 'ff_siwis', name: 'FR Siwis (Mujer)' },

    // Hindi
    { id: 'hf_alpha', name: 'HI Alpha (Mujer)' },
    { id: 'hf_beta', name: 'HI Beta (Mujer)' },
    { id: 'hm_omega', name: 'HI Omega (Hombre)' },
    { id: 'hm_psi', name: 'HI Psi (Hombre)' },

    // Italiano
    { id: 'if_sara', name: 'IT Sara (Mujer)' },
    { id: 'im_nicola', name: 'IT Nicola (Hombre)' },

    // Portugués (Brasil)
    { id: 'pf_dora', name: 'PT Dora (Mujer)' },
    { id: 'pm_alex', name: 'PT Alex (Hombre)' },
    { id: 'pm_santa', name: 'PT Santa (Hombre)' }
];

// Inicialización de selector global de voces
(function populateGlobalVoiceSelect() {
    const sel = document.getElementById('global-voice-select');
    if (!sel) return;
    voiceOptions.forEach(v => {
        const opt = document.createElement('option');
        opt.value = v.id;
        opt.textContent = v.name;
        sel.appendChild(opt);
    });
})();

function getGlobalVoice() {
    const sel = document.getElementById('global-voice-select');
    return sel ? sel.value : 'ef_dora';
}

function renderTable() {
    const tbody = document.querySelector('#segments-table tbody');
    tbody.innerHTML = '';

    currentData.forEach((item, index) => {
        const tr = document.createElement('tr');

        let voiceSelect = `<select id="voice-sel-${item.id}" class="cell-edit-sm" onchange="updateVoice(${index}, ${item.id}, this.value)">`;
        let curVoice = item.voice || '';
        if (!curVoice) {
            voiceSelect += `<option value="" disabled selected>-- Elige --</option>`;
        }
        voiceOptions.forEach(v => {
            let selected = (v.id === curVoice) ? 'selected' : '';
            voiceSelect += `<option value="${v.id}" ${selected}>${v.name}</option>`;
        });
        voiceSelect += `</select>`;

        let audioControls = '';
        if (item.audio_file) {
            let url = `/${item.audio_file}`.replace('//', '/');
            // Evitar que el navegador reproduzca el WAV viejo (con la voz original de Dora) de la caché
            audioControls = `
                <div class="audio-ctrl">
                    <button class="audio-btn" title="Reproducir" onclick="playAudio('${url}?t=${new Date().getTime()}')">▶</button>
                    <button class="audio-btn outline-btn" title="Pausar" onclick="stopAudio()">⏸</button>
                    <button class="audio-btn outline-btn" title="Regenerar" onclick="regenAudio(${item.id})">↻</button>
                </div>
            `;
        } else {
            audioControls = `<button class="audio-btn outline-btn" onclick="regenAudio(${item.id})">Generar</button>`;
        }

        tr.innerHTML = `
            <td>${item.id}</td>
            <td><input type="text" class="cell-edit-sm" value="${item.speaker}" onchange="updateItem(${index}, 'speaker', this.value)"></td>
            <td style="font-size:0.8rem; color:#8b949e;">${item.start} - ${item.end}</td>
            <td><textarea class="cell-edit-area" onchange="updateItem(${index}, 'original_text', this.value)">${item.original_text || ''}</textarea></td>
            <td><textarea class="cell-edit-area" onchange="updateItem(${index}, 'translated_text', this.value)">${item.translated_text || ''}</textarea></td>
            <td>${voiceSelect}</td>
            <td>${audioControls}</td>
        `;
        tbody.appendChild(tr);
    });
}

function updateItem(index, key, value) {
    currentData[index][key] = value;
}

// Actualización de voz en memoria y DOM
function updateVoice(index, itemId, value) {
    currentData[index]['voice'] = value;
    console.log(`[VOICE] Segment ${itemId} voice set to: ${value}`);
}

function playAudio(url) {
    if (currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
    }
    currentAudio = new Audio(url);
    currentAudio.play();
}

function stopAudio() {
    if (currentAudio) {
        currentAudio.pause();
    }
}

// Regenerates audio for a single segment ID
async function regenAudio(itemId) {
    // Read voice directly from the row's select element
    const rowSel = document.getElementById(`voice-sel-${itemId}`);
    const rowVoice = rowSel ? rowSel.value : null;

    if (!rowVoice) {
        alert(`Seleccione una voz en la fila del segmento ${itemId} antes de regenerar.`);
        return;
    }

    // Actualización local
    const item = currentData.find(d => d.id === itemId);
    if (item) item.voice = rowVoice;

    lockSteps();
    statusPanel.innerText = `Regenerando segmento ${itemId} con voz: ${rowVoice}...`;

    // Persistencia de estado
    await fetch(`/api/data/${projectId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(currentData)
    });

    // Pasar la voz directo en la URL como parámetro
    await fetch(`/api/generate-audio/${projectId}/${itemId}?voice=${encodeURIComponent(rowVoice)}`, { method: 'POST' });
    startPolling();
}

document.getElementById('btn-upload').addEventListener('click', async () => {
    const isYoutube = document.getElementById('tab-youtube').offsetParent !== null;
    const sourceLang = document.getElementById('source-lang').value;
    const targetLang = document.getElementById('target-lang').value;

    statusPanel.innerText = "Iniciando descarga/carga...";
    lockSteps();

    const formData = new FormData();
    formData.append('target_language', targetLang);
    if (sourceLang) formData.append('source_language', sourceLang);

    let endpoint = '/api/upload';

    if (isYoutube) {
        const url = document.getElementById('youtube-url').value;
        if (!url) return alert("Ingresa URL de YouTube");
        formData.append('url', url);
        endpoint = '/api/youtube';
    } else {
        const fileInput = document.getElementById('video-file');
        if (!fileInput.files[0]) {
            alert("Selecciona un video local");
            unlockSteps();
            return;
        }
        formData.append('file', fileInput.files[0]);
    }

    try {
        const res = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });
        const data = await res.json();

        if (data.status && data.status.startsWith("error")) {
            alert(data.status);
            unlockSteps();
            statusPanel.innerText = "Error: " + data.status;
            return;
        }

        projectId = data.project_id;
        statusPanel.innerText = data.status || "Video preparado.";

        videoPreview.src = `/uploads/${projectId}`;
        videoContainer.classList.remove('empty');

        // Start polling allows the frontend to refresh the cache statuses and table immediately
        startPolling();

    } catch (e) {
        alert("Error: " + e.message);
        unlockSteps();
    }
});

document.getElementById('btn-transcribe').addEventListener('click', async () => {
    lockSteps();
    await fetch(`/api/transcribe/${projectId}`, { method: 'POST' });
    startPolling();
});

document.getElementById('btn-translate').addEventListener('click', async () => {
    lockSteps();
    await fetch(`/api/translate/${projectId}`, { method: 'POST' });
    startPolling();
});

document.getElementById('btn-save-data').addEventListener('click', async () => {
    try {
        const res = await fetch(`/api/data/${projectId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(currentData)
        });
        const d = await res.json();
        alert(d.message || "Guardado!");
    } catch (e) {
        alert(e);
    }
});

document.getElementById('btn-audio').addEventListener('click', async () => {
    const globalVoice = getGlobalVoice();
    if (!globalVoice) {
        alert('Por favor elige una voz antes de generar.');
        return;
    }
    lockSteps();
    statusPanel.innerText = `Generando voces con: ${globalVoice}...`;

    // Envío de la voz como parámetro URL
    await fetch(`/api/generate-audio/${projectId}?voice=${encodeURIComponent(globalVoice)}`, { method: 'POST' });
    startPolling();
});

document.getElementById('btn-assemble').addEventListener('click', async () => {
    lockSteps();
    await fetch(`/api/assemble/${projectId}`, { method: 'POST' });
    startPolling();
});
