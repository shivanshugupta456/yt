# Generative AI Streamlit App

This repo contains a Streamlit application that generates a video-like MP4 from text prompts using Stable Diffusion XL frames.

## Run locally

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Use Google Colab GPU

If you want faster generation, run this repo on Google Colab with GPU enabled:

1. Upload the repo to Google Drive or GitHub.
2. Open a new Colab notebook and enable GPU: `Runtime` > `Change runtime type` > `GPU`.
3. Install dependencies and copy the app files.
4. Run the generation script directly from Colab, or use the Streamlit app with `pyngrok` for browser access.

Example Colab commands:

```python
!pip install -r /path/to/requirements.txt
!streamlit run /path/to/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

Then use `ngrok` or a Colab tunnel to access the UI.

## Deploy on Streamlit Cloud

1. Push this repo to GitHub.
2. In Streamlit Cloud, create a new app from this repository.
3. Set the main file to `streamlit_app.py`.
4. Deploy.
