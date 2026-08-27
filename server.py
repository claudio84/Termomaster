import os
import json
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

@app.get("/")
def home():
    return {"status": "TermoMaster AI Online"}

@app.post("/whatsapp-webhook")
async def whatsapp_webhook(request: Request):
    form_data = await request.form()
    body = form_data.get("Body", "")
    media_url = form_data.get("MediaUrl0", None)
    media_type = form_data.get("MediaContentType0", "")
    
    print(f"--> [MESSAGGIO RICEVUTO]: '{body}' | Media: {media_url}")
    
    testo_ricevuto = body

    # Gestione note vocali da WhatsApp
    if media_url and "audio" in media_type:
        try:
            print("--> Scaricamento ed elaborazione audio...")
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

    # Chiamata Groq con modello attivo
    try:
        print("--> Chiamata a Groq...")
        chat_completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
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

    # TwiML XML valido
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
