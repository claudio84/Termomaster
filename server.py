import os
import re
import io
import json
import math
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from groq import Groq
from pydantic import BaseModel
from pypdf import PdfReader

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI(title="TermoMaster AI")

KB_FILE = "knowledge_base.json"

# ==================== GESTIONE ARCHIVIO CONOSCENZA (RAG) ====================

def load_kb():
    if os.path.exists(KB_FILE):
        try:
            with open(KB_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_kb(data):
    with open(KB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def chunk_text(text: str, chunk_size: int = 350, overlap: int = 50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def search_kb(query: str, top_k: int = 5) -> str:
    kb = load_kb()
    if not kb:
        return ""
    
    query_tokens = [w.lower() for w in re.findall(r'\w+', query) if len(w) > 2]
    if not query_tokens:
        return ""

    scored_chunks = []
    for item in kb:
        content = item.get("text", "")
        source = item.get("source", "Manuale")
        page = item.get("page", None)
        item_type = item.get("type", "pdf")
        
        content_lower = content.lower()
        score = 0
        for token in query_tokens:
            count = content_lower.count(token)
            if count > 0:
                # Dà peso doppio alle esperienze di cantiere dirette del tecnico
                multiplier = 2.5 if item_type == "esperienza_campo" else 1.0
                score += (1 + math.log(count)) * (1.5 if token.isdigit() or len(token) > 5 else 1.0) * multiplier
        
        if score > 0:
            tag = "🛠️ ESPERIENZA DI CANTIERE PRECEDENTE" if item_type == "esperienza_campo" else f"MANUALE/LEZIONE: {source}"
            header_src = f"[{tag}" + (f" - Pag. {page}" if page else "") + "]"
            scored_chunks.append((score, f"{header_src}\n{content}"))
            
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    best = [c[1] for c in scored_chunks[:top_k]]
    return "\n\n---\n\n".join(best)

# ==================== PROMPT & MOTORE AI ====================

BASE_SYSTEM_PROMPT = """
Sei TermoMaster AI, assistente tecnico diagnostico dedicato per frigoristi, bruciatoristi e caldaisti.
Rispondi SEMPRE ed ESCLUSIVAMENTE in lingua italiana corretta, fluida e professionale.
Parli come un tecnico senior esperto: conciso, pratico, autorevole, zero convenevoli e orientato alla risoluzione rapida del guasto.

Regole operative fondamentali:
1. Auto-apprendimento e Memorizzazione Interventi di Cantiere:
   - Se l'utente ti comunica come ha risolto un guasto (es. "Ho risolto: era...", "Alla fine il problema era...", "Salva questo caso: ..."), devi:
     1. Confermare la memorizzazione con un breve riepilogo (Modello, Problema riscontrato, Causa reale e Soluzione).
     2. Aggiungere OBBLIGATORIAMENTE alla fine della risposta il blocco di salvataggio nel seguente formato esatto:
        <<<SALVA_INTERVENTO: {"modello": "Nome/Modello", "guasto": "Sintomo o errore", "soluzione": "Causa e soluzione reale applicata"}>>>

2. Gestione "Bassa Pressione" su Pompe di Calore Aria-Acqua:
   - Se l'utente menziona "bassa pressione" senza specificare, distingui SEMPRE chiaramente tra:
     a) Circuito Idraulico (Acqua): Mancanza d'acqua nell'impianto (manometro impianto < 1 bar, allarme pressostato acqua).
     b) Circuito Frigorifero (Gas): Bassa pressione di aspirazione (richiede verifica di SH e SC).

3. Refrigerazione / Bruciatori:
   - Bassa asp + Alto SH = Sottocarica, perdita o valvola strozzata.
   - Alta asp + Basso SH = Sovralimentazione o compressore inefficiente.
   - Bruciatori: analisi combustione (O2, CO2, lambda, ppm CO) e segnale ionizzazione.

4. Utilizzo delle Esperienze di Cantiere e Manuali:
   - Se tra i dati trovi una "ESPERIENZA DI CANTIERE PRECEDENTE", citale con massima priorità: ricorda al tecnico che su quell'impianto/errore è già stata trovata quella specifica soluzione.

5. Formato Risposte Diagnostiche Ordinarie:
   - [DIAGNOSI]: Causa probabile in 1-2 frasi chiare.
   - [CONTROLLI PRIORITARI]:
     1. Test più rapido (non invasivo / elettrico o verifica pressioni di base).
     2. Verifica meccanica / idraulica.
     3. Intervento invasivo (solo se i primi falliscono).
"""

def get_best_model():
    try:
        models = [m.id for m in client.models.list().data]
        excluded = ["whisper", "allam", "orpheus", "guard", "embed", "safeguard", "vision"]
        valid = [m for m in models if not any(x in m.lower() for x in excluded)]
        for m in valid:
            if "llama" in m.lower() and "70b" in m.lower():
                return m
        for m in valid:
            if "70b" in m.lower() or "llama-3" in m.lower():
                return m
        return valid[0] if valid else "llama-3.3-70b-versatile"
    except Exception:
        return "llama-3.3-70b-versatile"

def query_groq(prompt_text: str) -> str:
    model_name = get_best_model()
    
    # Cerca nei manuali e negli interventi di cantiere salvati in precedenza
    context_docs = search_kb(prompt_text)
    
    system_instruction = BASE_SYSTEM_PROMPT
    if context_docs:
        system_instruction += f"\n\nDATABASE TECNICO (MANUALI ED ESPERIENZE DI CANTIERE SALVATE):\n{context_docs}\n\nUsa prioritariamente questi dati reali per guidare la diagnosi."

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt_text}
        ],
        temperature=0.2
    )
    raw_text = completion.choices[0].message.content
    cleaned_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
    
    # Intercetta e salva l'esperienza di cantiere se generata
    match = re.search(r'<<<SALVA_INTERVENTO:\s*({.*?})>>>', cleaned_text, flags=re.DOTALL)
    if match:
        try:
            case_data = json.loads(match.group(1))
            modello = case_data.get("modello", "Impianto generico")
            guasto = case_data.get("guasto", "Guasto non specificato")
            soluzione = case_data.get("soluzione", "Soluzione applicata")
            
            chunk_content = (
                f"INTERVENTO DI CANTIERE RISOLTO\n"
                f"Apparecchio/Modello: {modello}\n"
                f"Guasto/Codice Errore: {guasto}\n"
                f"Causa e Soluzione Effettiva: {soluzione}\n"
                f"Esito: Risolto sul campo."
            )
            
            kb = load_kb()
            kb.append({
                "source": f"Intervento: {modello}",
                "page": None,
                "text": chunk_content,
                "type": "esperienza_campo"
            })
            save_kb(kb)
            print(f"--> [MEMORIZZATO INTERVENTO SUL CAMPO]: {modello} - {guasto}")
        except Exception as e:
            print(f"--> Errore salvataggio intervento: {e}")
            
        # Rimuove il tag tecnico dalla risposta mostrata all'utente
        cleaned_text = re.sub(r'<<<SALVA_INTERVENTO:\s*{.*?}>>>', '', cleaned_text, flags=re.DOTALL).strip()

    return cleaned_text

# ==================== ENDPOINT API ====================

class TextRequest(BaseModel):
    message: str

@app.get("/api/kb-stats")
def kb_stats():
    kb = load_kb()
    manuali = len(set(item.get("source", "") for item in kb if item.get("type") != "esperienza_campo"))
    interventi = len([item for item in kb if item.get("type") == "esperienza_campo"])
    return JSONResponse({
        "total_chunks": len(kb),
        "manuali_count": manuali,
        "interventi_count": interventi
    })

@app.post("/api/chat")
async def chat_endpoint(req: TextRequest):
    if not req.message.strip():
        return JSONResponse({"reply": "Messaggio vuoto."})
    try:
        reply = query_groq(req.message)
        return JSONResponse({"reply": reply})
    except Exception as e:
        return JSONResponse({"reply": f"Errore diagnosi: {str(e)}"}, status_code=500)

@app.post("/api/chat-audio")
async def chat_audio_endpoint(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        transcription = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=(file.filename or "audio.webm", audio_bytes),
            language="it"
        )
        user_text = transcription.text
        if not user_text.strip():
            return JSONResponse({"transcript": "", "reply": "Audio non comprensibile o muto."})
        reply = query_groq(user_text)
        return JSONResponse({"transcript": user_text, "reply": reply})
    except Exception as e:
        return JSONResponse({"reply": f"Errore elaborazione audio: {str(e)}"}, status_code=500)

@app.post("/api/upload-document")
async def upload_document_endpoint(file: UploadFile = File(...)):
    filename = file.filename or "documento"
    content_bytes = await file.read()
    new_chunks = []

    try:
        if filename.lower().endswith(".pdf"):
            pdf_reader = PdfReader(io.BytesIO(content_bytes))
            for page_num, page in enumerate(pdf_reader.pages, start=1):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    chunks = chunk_text(page_text.strip())
                    for c in chunks:
                        new_chunks.append({
                            "source": filename,
                            "page": page_num,
                            "text": c,
                            "type": "pdf"
                        })
        elif any(filename.lower().endswith(ext) for ext in [".mp3", ".m4a", ".wav", ".ogg", ".opus", ".aac", ".webm", ".flac"]):
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=(filename, content_bytes),
                language="it"
            )
            audio_text = transcription.text
            if audio_text and audio_text.strip():
                chunks = chunk_text(audio_text.strip())
                for c in chunks:
                    new_chunks.append({
                        "source": filename,
                        "page": None,
                        "text": c,
                        "type": "audio"
                    })
        else:
            return JSONResponse({"status": "error", "message": "Formato non supportato."}, status_code=400)

        if not new_chunks:
            return JSONResponse({"status": "warning", "message": f"Nessun testo estraibile da {filename}."}, status_code=200)

        kb = load_kb()
        kb.extend(new_chunks)
        save_kb(kb)

        return JSONResponse({
            "status": "ok",
            "message": f"'{filename}' memorizzato con successo ({len(new_chunks)} sezioni indicizzate).",
            "total_chunks": len(kb)
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": f"Errore elaborazione: {str(e)}"}, status_code=500)

# ==================== INTERFACCIA WEB ====================

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return r"""
<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>TermoMaster AI</title>
    <meta name="theme-color" content="#0f172a">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background-color: #0f172a; color: #f8fafc; display: flex; flex-direction: column; height: 100dvh; }
        header { background: #1e293b; padding: 12px 16px; border-bottom: 1px solid #334155; display: flex; align-items: center; justify-content: space-between; }
        .header-title h1 { font-size: 1.1rem; color: #38bdf8; font-weight: 700; letter-spacing: 0.5px; }
        .header-title small { font-size: 0.72rem; color: #94a3b8; display: block; }
        .badge { font-size: 0.75rem; background: #0284c7; padding: 4px 10px; border-radius: 12px; font-weight: 600; }
        #chat-window { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
        .msg { max-width: 90%; padding: 14px 16px; border-radius: 14px; font-size: 0.95rem; line-height: 1.6; word-wrap: break-word; }
        .user { align-self: flex-end; background: #0284c7; color: white; border-bottom-right-radius: 2px; white-space: pre-wrap; }
        .bot { align-self: flex-start; background: #1e293b; border: 1px solid #334155; color: #e2e8f0; border-bottom-left-radius: 2px; }
        .system-notice { align-self: center; background: rgba(56, 189, 248, 0.1); border: 1px solid #0284c7; color: #38bdf8; font-size: 0.8rem; border-radius: 8px; padding: 6px 12px; text-align: center; }
        .bot strong { color: #38bdf8; }
        .diagnostic-header { color: #38bdf8; font-weight: bold; display: block; margin-top: 10px; font-size: 1rem; border-bottom: 1px solid #334155; padding-bottom: 3px; }
        .transcript-tag { font-size: 0.75rem; color: #94a3b8; margin-bottom: 4px; font-style: italic; }
        #input-bar { background: #1e293b; padding: 10px 12px; border-top: 1px solid #334155; display: flex; gap: 8px; align-items: center; }
        #text-input { flex: 1; background: #0f172a; border: 1px solid #334155; color: white; padding: 10px 14px; border-radius: 24px; font-size: 0.95rem; outline: none; }
        #text-input:focus { border-color: #38bdf8; }
        button { border: none; outline: none; cursor: pointer; border-radius: 50%; width: 42px; height: 42px; display: flex; align-items: center; justify-content: center; transition: all 0.2s; font-size: 1.1rem; }
        #send-btn { background: #38bdf8; color: #0f172a; }
        #mic-btn { background: #334155; color: #f8fafc; }
        #attach-btn { background: #334155; color: #f8fafc; }
        #mic-btn.recording { background: #ef4444; animation: pulse 1.2s infinite; }
        @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.1); } 100% { transform: scale(1); } }
        .loading { display: flex; gap: 4px; padding: 12px 16px; }
        .dot { width: 8px; height: 8px; background: #38bdf8; border-radius: 50%; animation: bounce 1.4s infinite ease-in-out both; }
        .dot:nth-child(1) { animation-delay: -0.32s; }
        .dot:nth-child(2) { animation-delay: -0.16s; }
        @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1.0); } }
    </style>
</head>
<body>
    <header>
        <div class="header-title">
            <h1>🔧 TermoMaster AI</h1>
            <small id="kb-info">📚 Memoria: 0 manuali | 🛠️ 0 interventi</small>
        </div>
        <span class="badge">ONLINE</span>
    </header>
    
    <div id="chat-window">
        <div class="msg bot">Pronto per la diagnosi. Puoi caricare manuali (📎), chiedere verifiche o raccontarmi come hai risolto un guasto per farlo memorizzare.</div>
    </div>
    
    <div id="input-bar">
        <input type="file" id="file-input" accept=".pdf,audio/*,.mp3,.m4a,.wav,.ogg,.opus,.aac" style="display:none">
        <button id="attach-btn" title="Carica PDF o Lezione Audio">📎</button>
        <button id="mic-btn" title="Registra vocale">🎤</button>
        <input type="text" id="text-input" placeholder="Descrivi guasto o soluzione trovata..." autocomplete="off">
        <button id="send-btn" title="Invia">➤</button>
    </div>

    <script>
        const chatWindow = document.getElementById('chat-window');
        const textInput = document.getElementById('text-input');
        const sendBtn = document.getElementById('send-btn');
        const micBtn = document.getElementById('mic-btn');
        const attachBtn = document.getElementById('attach-btn');
        const fileInput = document.getElementById('file-input');
        const kbInfo = document.getElementById('kb-info');

        let mediaRecorder = null;
        let audioChunks = [];

        async function updateKbStats() {
            try {
                const res = await fetch('/api/kb-stats');
                const data = await res.json();
                kbInfo.innerText = `📚 Memoria: ${data.manuali_count} manuali | 🛠️ ${data.interventi_count} interventi`;
            } catch (e) {}
        }
        updateKbStats();

        function formatBotMessage(text) {
            let safe = text
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;");
            
            safe = safe.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
            safe = safe.replace(/\[([A-Z\s_]+)\]/g, '<span class="diagnostic-header">[$1]</span>');
            safe = safe.replace(/\n/g, "<br>");
            return safe;
        }

        function appendMessage(text, sender, isVoice = false) {
            const div = document.createElement('div');
            div.className = `msg ${sender}`;
            if (sender === 'user') {
                if (isVoice) {
                    div.innerHTML = `<div class="transcript-tag">🎤 Vocale trascritto:</div>` + text.replace(/\n/g, '<br>');
                } else {
                    div.innerText = text;
                }
            } else if (sender === 'system-notice') {
                div.innerHTML = text;
            } else {
                div.innerHTML = formatBotMessage(text);
            }
            chatWindow.appendChild(div);
            chatWindow.scrollTop = chatWindow.scrollHeight;
            return div;
        }

        function showLoading(msg = "") {
            const div = document.createElement('div');
            div.className = 'msg bot loading';
            div.id = 'loading-dots';
            div.innerHTML = msg ? `<span style="color:#38bdf8;font-size:0.85rem;margin-right:8px">${msg}</span>` : '';
            div.innerHTML += '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';
            chatWindow.appendChild(div);
            chatWindow.scrollTop = chatWindow.scrollHeight;
        }

        function hideLoading() {
            const el = document.getElementById('loading-dots');
            if (el) el.remove();
        }

        async function sendText() {
            const text = textInput.value.trim();
            if (!text) return;
            textInput.value = '';
            appendMessage(text, 'user');
            showLoading();

            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: text })
                });
                const data = await res.json();
                hideLoading();
                appendMessage(data.reply, 'bot');
                updateKbStats();
            } catch (err) {
                hideLoading();
                appendMessage("Errore di connessione con il server.", 'bot');
            }
        }

        sendBtn.addEventListener('click', sendText);
        textInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') sendText(); });

        attachBtn.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', async () => {
            const file = fileInput.files[0];
            if (!file) return;

            showLoading(`Elaborazione e memorizzazione di ${file.name}...`);
            const formData = new FormData();
            formData.append('file', file);

            try {
                const res = await fetch('/api/upload-document', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();
                hideLoading();

                if (data.status === 'ok') {
                    appendMessage(`✅ ${data.message}`, 'system-notice');
                    updateKbStats();
                } else {
                    appendMessage(`⚠️ ${data.message || 'Errore durante il caricamento.'}`, 'system-notice');
                }
            } catch (err) {
                hideLoading();
                appendMessage("Errore di rete durante il caricamento del file.", 'system-notice');
            }
            fileInput.value = '';
        });

        micBtn.addEventListener('click', async () => {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
                micBtn.classList.remove('recording');
                return;
            }

            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    stream.getTracks().forEach(t => t.stop());
                    
                    showLoading();
                    const formData = new FormData();
                    formData.append('file', audioBlob, 'rec.webm');

                    try {
                        const res = await fetch('/api/chat-audio', {
                            method: 'POST',
                            body: formData
                        });
                        const data = await res.json();
                        hideLoading();
                        if (data.transcript) {
                            appendMessage(data.transcript, 'user', true);
                        }
                        appendMessage(data.reply, 'bot');
                        updateKbStats();
                    } catch (err) {
                        hideLoading();
                        appendMessage("Errore durante l'invio del vocale.", 'bot');
                    }
                };

                mediaRecorder.start();
                micBtn.classList.add('recording');
            } catch (err) {
                alert("Permesso microfono non concesso o dispositivo non supportato.");
            }
        });
    </script>
</body>
</html>
    """

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
