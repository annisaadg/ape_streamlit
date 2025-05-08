import base64
import streamlit as st

def render_video(video_path):
    try:
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        b64_video = base64.b64encode(video_bytes).decode()
        video_html = f"""
            <div style="display: flex; justify-content: center; margin-top: 20px;">
                <video width="640" height="360" controls style="border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.2);">
                    <source src="data:video/mp4;base64,{b64_video}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </div>
        """
        st.markdown(video_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Tidak dapat memutar video. Error: {e}")
