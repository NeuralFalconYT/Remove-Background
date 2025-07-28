import cv2
import numpy as np
import os
import shutil
import subprocess
import glob
from tqdm.auto import tqdm
import uuid
import re
from zipfile import ZipFile

gpu = False
os.makedirs("./results",exist_ok=True)

def apply_green_screen(image_path, save_path,foreground_segmenter):
    """
    Replaces the background of the input image with green using a segmentation model.

    Args:
        image_path (str): Path to the input image.
        segmenter (SoftForegroundSegmenter): Initialized segmentation model.
        save_path (str, optional): If provided, saves the result to this path.

    Returns:
        np.ndarray: The green screen composited image.
    """

    # Load image with alpha if available
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Remove transparency if present
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # Convert to RGB for the model
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get segmentation mask
    mask = foreground_segmenter.estimate_foreground_segmentation(image_rgb)

    # Normalize and convert mask to 0-255 uint8
    if mask.max() <= 1.0:
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    if mask.ndim == 2:
        mask_gray = mask
    elif mask.shape[2] == 1:
        mask_gray = mask[:, :, 0]
    else:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    _, binary_mask = cv2.threshold(mask_gray, 128, 255, cv2.THRESH_BINARY)

    # Create green background
    green_bg = np.full_like(image_rgb, (0, 255, 0), dtype=np.uint8)

    # Create 3-channel mask
    mask_3ch = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    # Composite: foreground from image, background as green
    output_rgb = np.where(mask_3ch == 255, image_rgb, green_bg)

    # Convert back to BGR for OpenCV
    output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)

    # Save if path is given
    if save_path:
        cv2.imwrite(save_path, output_bgr)

    return output_bgr


def cam_green_screen(image,foreground_segmenter):

    # Remove transparency if present
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # Convert to RGB for the model
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get segmentation mask
    mask = foreground_segmenter.estimate_foreground_segmentation(image_rgb)

    # Normalize and convert mask to 0-255 uint8
    if mask.max() <= 1.0:
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    if mask.ndim == 2:
        mask_gray = mask
    elif mask.shape[2] == 1:
        mask_gray = mask[:, :, 0]
    else:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    _, binary_mask = cv2.threshold(mask_gray, 128, 255, cv2.THRESH_BINARY)

    # Create green background
    green_bg = np.full_like(image_rgb, (0, 255, 0), dtype=np.uint8)

    # Create 3-channel mask
    mask_3ch = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    # Composite: foreground from image, background as green
    output_rgb = np.where(mask_3ch == 255, image_rgb, green_bg)

    # Convert back to BGR for OpenCV
    output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
    return output_bgr



def create_transparent_foreground(image_path,foreground_segmenter):
    uid = uuid.uuid4().hex[:8].upper()
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    base_name = re.sub(r'[^a-zA-Z\s]', '', base_name)
    base_name = base_name.strip().replace(" ", "_").replace("__","_")
    save_path = f"./results/{base_name}_{uid}.png"
    save_path = os.path.abspath(save_path)

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = foreground_segmenter.estimate_foreground_segmentation(image_rgb)

    if mask.max() <= 1.0:
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    if mask.ndim == 3 and mask.shape[2] == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    _, alpha = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    rgba_image = np.dstack((image_rgb, alpha))
    cv2.imwrite(save_path, cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGRA))

    return image_rgb, rgba_image, save_path




def remove_background_batch_images(img_list, foreground_segmenter):
    # Create unique temp directory
    uid = uuid.uuid4().hex[:8].upper()
    temp_dir = os.path.abspath(f"./results/bg_removed_{uid}")
    os.makedirs(temp_dir, exist_ok=True)

    # Process each image
    for image_path in tqdm(img_list, desc="Removing Backgrounds"):
        _, _, save_path = create_transparent_foreground(image_path, foreground_segmenter)
        shutil.move(save_path, os.path.join(temp_dir, os.path.basename(save_path)))

    # Create zip file
    zip_path = f"{temp_dir}.zip"
    with ZipFile(zip_path, 'w') as zipf:
        for root, _, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=temp_dir)
                zipf.write(file_path, arcname=arcname)
    # shutil.rmtree(temp_dir)
    return os.path.abspath(zip_path)

def get_sorted_paths(directory, extension="png"):
    """
    Returns full paths of all images with the given extension, sorted by filename (without extension).
    """
    extension = extension.lstrip(".").lower()
    pattern = os.path.join(directory, f"*.{extension}")
    files = glob.glob(pattern)
    files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    return files


def extract_all_frames_ffmpeg_gpu(video_path, output_dir="frames", extension="png", use_gpu=True):
    """
    Extracts all frames from a video using ffmpeg, with optional GPU acceleration.
    Returns a sorted list of full paths to the extracted frames.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    extension = extension.lstrip(".")
    output_pattern = os.path.join(output_dir, f"%05d.{extension}")

    command = [
        "ffmpeg", "-i", video_path, output_pattern
    ]
    if use_gpu:
        command.insert(1, "cuda")
        command.insert(1, "-hwaccel")

    print("Running command:", " ".join(command))
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return get_sorted_paths(output_dir, extension)



def green_screen_batch(frames, foreground_segmenter,output_dir="green_screen_frames"):
    """
    Applies green screen background to a batch of frames and saves the results.

    Args:
        frames (List[str]): List of image paths.
        output_dir (str): Directory to save green-screened output.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    green_screen_frames=[]
    for frame in tqdm(frames, desc="Processing green screen frames"):
        save_image_path=os.path.join(output_dir, os.path.basename(frame))
        result = apply_green_screen(
            frame,
            save_image_path,
            foreground_segmenter
        )
        green_screen_frames.append(save_image_path)
    return green_screen_frames


def green_screen_video_maker(original_video, green_screen_frames, batch_size=100):
    """
    Creates video chunks from green screen frames based on original video's properties.

    Args:
        original_video (str): Path to the original video file (to read FPS, size).
        green_screen_frames (List[str]): List of green screen frame paths.
        batch_size (int): Number of frames per chunked video.
    """
    temp_folder = "temp_video"
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)
    os.makedirs(temp_folder, exist_ok=True)

    # Get video info from original video
    cap = cv2.VideoCapture(original_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    total_frames = len(green_screen_frames)
    num_chunks = (total_frames + batch_size - 1) // batch_size  # Ceiling division

    for chunk_idx in tqdm(range(num_chunks), desc="Processing video chunks"):
        chunk_path = os.path.join(temp_folder, f"{chunk_idx+1}.mp4")
        out = cv2.VideoWriter(chunk_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        start_idx = chunk_idx * batch_size
        end_idx = min(start_idx + batch_size, total_frames)

        for frame_path in green_screen_frames[start_idx:end_idx]:
            frame = cv2.imread(frame_path)
            frame = cv2.resize(frame, (width, height))  # Ensure matching resolution
            out.write(frame)

        out.release()



def merge_video_chunks(output_path="final_video.mp4", temp_folder="temp_video", use_gpu=True):
    """
    Merges all video chunks from temp_folder into a final single video.
    """
    os.makedirs("./results", exist_ok=True)
    output_path = f"../results/{output_path}"  # relative to temp_folder
    file_list_path = os.path.join(temp_folder, "chunks.txt")
    chunk_files=sorted(
            [f for f in os.listdir(temp_folder) if f.lower().endswith("mp4")],
            key=lambda x: int(os.path.splitext(x)[0])
        )

    with open(file_list_path, "w") as f:
        for chunk in chunk_files:
            f.write(f"file '{chunk}'\n")  # ✅ No './' prefix

    ffmpeg_cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "chunks.txt"]

    if use_gpu:
        ffmpeg_cmd += ["-c:v", "h264_nvenc", "-preset", "fast"]
    else:
        ffmpeg_cmd += ["-c", "copy"]

    ffmpeg_cmd.append(output_path)

    # ✅ Run from inside temp_folder, so chunks.txt and mp4 files are local
    subprocess.run(ffmpeg_cmd, cwd=temp_folder, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def extract_audio_from_video(video_path, output_audio_path="output_audio.wav", format="wav", sample_rate=16000, channels=1):
    """
    Extracts audio from a video file using ffmpeg.

    Args:
        video_path (str): Path to the input video file.
        output_audio_path (str): Path to save the extracted audio (e.g., .wav or .mp3).
        format (str): 'wav' or 'mp3'
        sample_rate (int): Sampling rate in Hz (e.g., 16000 for ASR models)
        channels (int): Number of audio channels (1=mono, 2=stereo)
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_audio_path) or ".", exist_ok=True)

    # Build ffmpeg command
    if format.lower() == "wav":
        command = [
            "ffmpeg", "-y",               # Overwrite output
            "-i", video_path,            # Input video
            "-vn",                       # Disable video
            "-ac", str(channels),        # Audio channels (1 = mono)
            "-ar", str(sample_rate),     # Audio sample rate
            "-acodec", "pcm_s16le",      # WAV codec
            output_audio_path
        ]
    elif format.lower() == "mp3":
        command = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-ac", str(channels),
            "-ar", str(sample_rate),
            "-acodec", "libmp3lame",     # MP3 codec
            output_audio_path
        ]
    else:
        raise ValueError("Unsupported format. Use 'wav' or 'mp3'.")

    # Run command silently
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def add_audio(video_path, audio_path, output_path, use_gpu=False):
    """
    Replaces the audio of a video with a new audio track.

    Args:
        video_path (str): Path to the video file.
        audio_path (str): Path to the audio file.
        output_path (str): Path where the final video will be saved.
        use_gpu (bool): If True, use GPU-accelerated video encoding.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    command = [
        "ffmpeg", "-y",                     # Overwrite without asking
        "-i", video_path,                  # Input video
        "-i", audio_path,                  # Input audio
        "-map", "0:v:0",                   # Use video from first input
        "-map", "1:a:0",                   # Use audio from second input
        "-shortest"                        # Trim to the shortest stream (audio/video)
    ]

    if use_gpu:
        command += ["-c:v", "h264_nvenc", "-preset", "fast"]
    else:
        command += ["-c:v", "copy"]

    command += ["-c:a", "aac", "-b:a", "192k", output_path]

    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)



def remove_background_from_video(uploaded_video_path,foreground_segmenter):
    # 🔁 Generate a single UUID to use for all related files
    uid = uuid.uuid4().hex[:8].upper()

    # Define all output paths using that UUID
    base_name = os.path.splitext(os.path.basename(uploaded_video_path))[0]
    base_name = re.sub(r'[^a-zA-Z\s]', '', base_name) 
    base_name = base_name.strip().replace(" ", "_")

    temp_video_path = f"./results/{base_name}_chunks_{uid}.mp4"
    audio_path = f"./results/{base_name}_audio_{uid}.wav"
    final_output_path = f"./results/{base_name}_final_{uid}.mp4"

    # Step 1: Extract frames
    frames = extract_all_frames_ffmpeg_gpu(
        video_path=uploaded_video_path,
        output_dir="frames",
        extension="png",
        use_gpu=gpu
    )

    # Step 2: Remove background (green screen)
    green_screen_frames = green_screen_batch(frames,foreground_segmenter)

    # Step 3: Rebuild video from frames
    green_screen_video_maker(uploaded_video_path, green_screen_frames, batch_size=100)

    # Step 4: Merge video chunks
    merge_video_chunks(output_path=os.path.basename(temp_video_path), use_gpu=gpu)

    # Step 5: Extract original audio
    extract_audio_from_video(uploaded_video_path, output_audio_path=audio_path)

    # Step 6: Add audio back
    add_audio(
        video_path=temp_video_path,
        audio_path=audio_path,
        output_path=final_output_path,
        use_gpu=True
    )

    return os.path.abspath(final_output_path)
