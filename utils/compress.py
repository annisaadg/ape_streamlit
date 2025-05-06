import streamlit as st

import ffmpeg
import os

def compress(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Ambil nama file saja (tanpa path)
    filename = os.path.basename(input_file)  # hasil: 'img_0063.mov'
    name_without_ext = os.path.splitext(filename)[0]  # hasil: 'img_0063'

    output_path = os.path.join(output_dir, "compressed")
    output_path = os.path.join(output_path, f"compressed_{name_without_ext}.mp4")
    
    stream = ffmpeg.input(input_file)
    stream = ffmpeg.output(stream, output_path, vcodec='libx264', crf=28, preset='slow')
    ffmpeg.run(stream)

    return output_path