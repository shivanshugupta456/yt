import importlib.util
from pathlib import Path

module_path = Path(__file__).resolve().parent / "YT MAKERS LAB" / "text_to_video.py"
spec = importlib.util.spec_from_file_location("yt_text_to_video_app", module_path)
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

if __name__ == "__main__":
    app.run_streamlit_ui()
