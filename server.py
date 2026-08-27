import os
import json
import requests
from fastapi import FastAPI, Request, Response
from groq import Groq

# Inizializza client Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI(title="TermoMaster AI Gateway")

SYSTEM_PROMPT = """
Sei TermoMaster AI, assistente tecnico per frigoristi, bruciatoristi e caldaisti.
Rispondi in modo conciso, pratico e diretto:
- [DIAGNOSI]: Causa probabile in 1-2 frasi.
- [CONTROLLI PRIORITARI]:
  1. Test rapido/elettrico.
  2. Controllo meccanico/idraulico.
  3. Verifica invasiva.
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

    # Gestione note vocali
    if media_url and "audio" in media_type:
        try:
            print("--> Scaricamento nota vocale...")
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
            testo_ricevuto = f"Errore audio: {e}"

    if not testo_ricevuto or not testo_ricevuto.strip():
        print("--> Messaggio vuoto, skip.")
        return Response(content="<Response></Response>", media_type="text/xml")

    # Elaborazione Groq Llama 3
    try:
        print("--> Chiamata a Groq...")
        chat_completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": testo_ricevuto}
            ]
        )
        testo_risposta = chat_completion.choices[0].message.content
        print(f"--> [RISPOSTA AI GENERATA]:\n{testo_risposta}")
    except Exception as e:
        print(f"--> [ERRORE GROQ]: {e}")
        testo_risposta = f"Errore AI: {str(e)}"

    # TwiML standard compatibile WhatsApp
    twiml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>{testo_risposta}</Message>
</Response>"""

    return Response(content=twiml_response, media_type="text/xml")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
