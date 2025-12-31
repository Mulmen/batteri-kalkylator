# Batterikalkylator (Streamlit)

Den här appen kör batteri+sol+spot-modellen i en webbläsare via Streamlit.

## Kör lokalt
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Kör över nätet (enkelt)
### Alternativ A: Streamlit Community Cloud (gratis)
1. Lägg upp denna mapp som ett GitHub-repo.
2. Gå till Streamlit Community Cloud och välj "New app".
3. Välj repo + branch och sätt:
   - **Main file path:** `app.py`
4. Deploy.

### Alternativ B: Hugging Face Spaces (gratis)
1. Skapa en Space av typen **Streamlit**.
2. Ladda upp `app.py` och `requirements.txt`.
3. Space bygger och publicerar en URL automatiskt.

### Alternativ C: Docker (valfri plattform: Render, Fly.io, Azure, AWS, mm.)
```bash
docker build -t batteri-webapp .
docker run -p 8501:8501 batteri-webapp
```
