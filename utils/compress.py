import ffmpeg
import os

def compress(input_file, output_dir):
    compressed_dir = os.path.join(output_dir, "compressed")
    os.makedirs(compressed_dir, exist_ok=True)

    filename = os.path.basename(input_file)  

    output_path = os.path.join(compressed_dir, filename)

    stream = ffmpeg.input(input_file)
    stream = ffmpeg.output(stream, output_path, vcodec='libx264', crf=28, preset='slow')
    ffmpeg.run(stream)

    return output_path
