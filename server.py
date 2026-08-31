import os
import re
import io
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from groq import Groq
from pydantic import BaseModel

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI(title="TermoMaster AI")

SYSTEM_PROMPT = """
Sei TermoMaster AI, assistente tecnico diagnostico dedicato per frigoristi, bruciatoristi e caldaisti.
Rispondi SEMPRE ed ESCLUSIVAMENTE in lingua italiana corretta, fluida e professionale.
Parli come un tecnico senior esperto: conciso, pratico, autorevole, zero convenevoli e orientato alla risoluzione rapida del guasto.

Regole operative fondamentali:
1. Gestione "Bassa Pressione" su Pompe di Calore Aria-Acqua:
   - Se l'utente menziona "bassa pressione" senza specificare, distingui SEMPRE chiaramente tra:
     a) Circuito Idraulico (Acqua): Mancanza d'acqua nell'impianto (manometro impianto < 1 bar, allarme pressostato acqua). Cause tipiche: rubinetto di carico chiuso, perdite visibili o vaso d'espansione scarico/bucato.
     b) Circuito Frigorifero (Gas): Bassa pressione di aspirazione (richiede verifica di Surriscaldamento SH e Sottoraffreddamento SC).

2. Refrigerazione / Pompe di Calore (Circuito Gas):
   - Bassa aspirazione + Alto SH = Sottocarica di refrigerante, perdita nel circuito o valvola termostatica/elettronica strozzata.
   - Alta aspirazione + Basso SH = Sovralimentazione dell'evaporatore o compressione inefficiente.

3. Bruciatori / Combustione:
   - Analisi fumi: O2, CO2, CO, rendimento e lambda.
   - Verifica segnale fiamma (uA), pressione ugello/rete ed elettrodi d'accensione/ionizzazione.

4. Formato Risposte Diagnostiche:
   - [DIAGNOSI]: Causa probabile e spiegazione tecnica in 1-2 frasi chiare.
   - [CONTROLLI PRIORITARI]:
     1. Test più rapido (non invasivo / elettrico o verifica manometri/pressioni di base).
     2. Verifica meccanica / idraulica.
     3. Intervento invasivo (solo se i controlli precedenti risultano regolari).
"""

def get_best_model():
    """Seleziona il modello Llama 70B per la massima accuratezza linguistica e tecnica"""
    try:
        models = [m.id for m in client.models.list().data]
        excluded = ["whisper", "allam", "orpheus", "guard", "embed", "safeguard", "vision"]
        valid = [m for m in models if not any(x in m.lower() for x in excluded)]
        
        # 1. Cerca prima i modelli Llama 70B (massima qualità in italiano)
        for m in valid:
            if "llama" in m.lower() and "70b" in m.lower():
                return m
        # 2. Cerca altri modelli da 70B o Llama
        for m in valid:
            if "70b" in m.lower():
                return m
        for m in valid:
            if "llama-3" in m.lower():
                return m
        return valid[0] if valid else "llama-3.3-70b-versatile"
    except Exception:
        return "llama-3.3-70b-versatile"

def query_groq(prompt_text: str) -> str:
    model_name = get_best_model()
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text}
        ],
        temperature=0.2  # Bassa temperatura = risposte precise, rigorose e senza allucinazioni
    )
    raw_text = completion.choices[0].message.content
    cleaned_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
    return cleaned_text

class TextRequest(BaseModel):
    message: str

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
        header { background: #1e293b; padding: 14px 18px; border-bottom: 1px solid #334155; display: flex; align-items: center; justify-content: space-between; }
        header h1 { font-size: 1.15rem; color: #38bdf8; font-weight: 700; letter-spacing: 0.5px; }
        header span { font-size: 0.75rem; background: #0284c7; padding: 3px 8px; border-radius: 12px; font-weight: 600; }
        #chat-window { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
        .msg { max-width: 90%; padding: 14px 16px; border-radius: 14px; font-size: 0.95rem; line-height: 1.6; word-wrap: break-word; }
        .user { align-self: flex-end; background: #0284c7; color: white; border-bottom-right-radius: 2px; white-space: pre-wrap; }
        .bot { align-self: flex-start; background: #1e293b; border: 1px solid #334155; color: #e2e8f0; border-bottom-left-radius: 2px; }
        .bot strong { color: #38bdf8; }
        .diagnostic-header { color: #38bdf8; font-weight: bold; display: block; margin-top: 10px; font-size: 1rem; border-bottom: 1px solid #334155; padding-bottom: 3px; }
        .transcript-tag { font-size: 0.75rem; color: #94a3b8; margin-bottom: 4px; font-style: italic; }
        #input-bar { background: #1e293b; padding: 10px 12px; border-top: 1px solid #334155; display: flex; gap: 8px; align-items: center; }
        #text-input { flex: 1; background: #0f172a; border: 1px solid #334155; color: white; padding: 10px 14px; border-radius: 24px; font-size: 0.95rem; outline: none; }
        #text-input:focus { border-color: #38bdf8; }
        button { border: none; outline: none; cursor: pointer; border-radius: 50%; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; transition: all 0.2s; }
        #send-btn { background: #38bdf8; color: #0f172a; }
        #mic-btn { background: #334155; color: #f8fafc; }
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
        <h1>🔧 TermoMaster AI</h1>
        <span>ONLINE</span>
    </header>
    <div id="chat-window">
        <div class="msg bot">Pronto per la diagnosi. Scrivi i dati o premi il microfono per registrare una nota vocale dal cantiere.</div>
    </div>
    <div id="input-bar">
        <button id="mic-btn" title="Registra vocale">🎤</button>
        <input type="text" id="text-input" placeholder="Descrivi il guasto o parametri..." autocomplete="off">
        <button id="send-btn" title="Invia">➤</button>
    </div>

    <script>
        const chatWindow = document.getElementById('chat-window');
        const textInput = document.getElementById('text-input');
        const sendBtn = document.getElementById('send-btn');
        const micBtn = document.getElementById('mic-btn');

        let mediaRecorder = null;
        let audioChunks = [];

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
            } else {
                div.innerHTML = formatBotMessage(text);
            }
            chatWindow.appendChild(div);
            chatWindow.scrollTop = chatWindow.scrollHeight;
            return div;
        }

        function showLoading() {
            const div = document.createElement('div');
            div.className = 'msg bot loading';
            div.id = 'loading-dots';
            div.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';
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
            } catch (err) {
                hideLoading();
                appendMessage("Errore di connessione con il server.", 'bot');
            }
        }

        sendBtn.addEventListener('click', sendText);
        textInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') sendText(); });

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
