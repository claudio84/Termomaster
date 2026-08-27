import os
import html
import requests
from fastapi import FastAPI, Request, Response
from groq import Groq

# Inizializza client Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI(title="TermoMaster AI Gateway")

SYSTEM_PROMPT = """
Sei TermoMaster AI, assistente tecnico diagnostico dedicato per frigoristi, bruciatoristi e caldaisti.
Parli come un tecnico senior esperto: conciso, pratico, zero convenevoli e orientato alla risoluzione del guasto.

Regole operative:
1. Refrigerazione / Pompe di Calore:
   - Analizza sempre Surriscaldamento (SH), Sottoraffreddamento (SC) e Delta T.
   - Bassa asp + Alto SH = Sottocarica, perdita o valvola termostatica/elettronica strozzata.
   - Alta asp + Basso SH = Sovralimentazione evaporatore o compressore inefficiente.
   - Alta condensazione + Alto Delta T idraulico = Scambio insufficiente / scarsa portata acqua.

2. Bruciatori / Combustione:
   - Analizza fumi: O2, CO2, CO, rendimento e lambda.
   - Monitora segnale fiamma (uA), pressione ugello/gas ed elettrodi.

3. Formato Risposte Diagnostiche:
   - [DIAGNOSI]: Causa probabile in 1-2 frasi.
   - [CONTROLLI PRIORITARI]:
     1. Test più rapido (non invasivo / elettrico).
     2. Verifica meccanica / idraulica.
     3. Intervento invasivo (solo se i primi falliscono).
"""

def get_active_chat_model():
    """Seleziona dinamicamente il miglior modello attivo disponibile nell'account Groq"""
    preferred_models = [
        "llama-3.2-3b-preview",
        "llama-3.2-1b-preview",
        "mixtral-8x7b-32768",
        "llama3-70b-8192",
        "llama3-8b-8192"
    ]
    try:
        available = [m.id for m in client.models.list().data if "whisper" not in m.id]
        for model in preferred_models:
            if model in available:
                return model
        return available[0] if available else "llama-3.2-3b-preview"
    except Exception:
        return "llama-3.2-3b-preview"

@app.get("/")
def home():
    return {"status": "TermoMaster AI Online", "active_model": get_active_chat_model()}

@app.post("/whatsapp-webhook")
async def whatsapp_webhook(request: Request):
    form_data = await request.form()
    body = form_data.get("Body", "")
    media_url = form_data.get("MediaUrl0", None)
    media_type = form_data.get("MediaContentType0", "")
    
    print(f"--> [MESSAGGIO RICEVUTO]: '{body}' | Media: {media_url}")
    testo_ricevuto = body

    # Trascrizione note vocali
    if media_url and "audio" in media_type:
        try:
            print("--> Scaricamento ed elaborazione nota vocale...")
            audio_resp = requests.get(media_url)
            temp_filename = "temp_audio.ogg"
            with open(temp_filename, "wb") as f:
                f.write(audio_resp.content)
                
            with open(temp_filename, "rb") as af:
                trascrizione = client.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=(temp_filename, af.read()),
                    language="it"
                )
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            testo_ricevuto = trascrizione.text
            print(f"--> [VOCALE TRASCRITTO]: {testo_ricevuto}")
        except Exception as e:
            print(f"--> [ERRORE AUDIO]: {e}")
            testo_ricevuto = f"Errore trascrizione vocale: {e}"

    if not testo_ricevuto or not testo_ricevuto.strip():
        return Response(content="<Response></Response>", media_type="application/xml")

    # Diagnosi AI con modello rilevato automaticamente
    model_to_use = get_active_chat_model()
    print(f"--> Chiamata a Groq con modello: {model_to_use}...")
    try:
        chat_completion = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": testo_ricevuto}
            ]
        )
        testo_risposta = chat_completion.choices[0].message.content
        print(f"--> [RISPOSTA GENERATA]:\n{testo_risposta}")
    except Exception as e:
        print(f"--> [ERRORE GROQ]: {e}")
        testo_risposta = f"Errore AI: {str(e)}"

    # Risposta TwiML per WhatsApp
    escaped_reply = html.escape(testo_risposta)
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>{escaped_reply}</Message>
</Response>"""
    return Response(content=twiml, media_type="application/xml")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
