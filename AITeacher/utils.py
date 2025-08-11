#!/usr/bin/env python3
"""
PDF Seamless Teacher - Utility Functions Module
================================================================
Core utility functions for PDF processing, AI analysis, audio generation,
and video creation using Gemini API, F5-TTS, and Manim integration.

Dependencies:
- Google GenAI (Gemini API)
- F5-TTS for voice cloning
- Manim for video generation
- PDF2Image for document processing
================================================================
"""

# ╭──────────────────────────────────────────────────────────────╮
# │                        IMPORTS SECTION                       │
# ╰──────────────────────────────────────────────────────────────╯

import os
import uuid
import subprocess
import requests
import base64
import re
import shutil
import time
import socket
import errno
from pathlib import Path
from typing import List, Optional, Tuple
from io import BytesIO

# PDF and Image Processing
from pdf2image import convert_from_bytes
from PIL import Image

# Audio Processing
from pydub import AudioSegment

# Google GenAI imports
import google.genai as genai
from google.genai import types

# Video Processing (with fallback handling)
try:
    from moviepy.editor import VideoFileClip, concatenate_videoclips
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("Warning: MoviePy not available. Video concatenation will be limited.")

# Manimator integration (with fallback handling)
try:
    from manimator import generate_video_from_scenes
    MANIMATOR_AVAILABLE = True
except ImportError:
    MANIMATOR_AVAILABLE = False
    print("Warning: Manimator module not available. Using fallback video generation.")

# ╭──────────────────────────────────────────────────────────────╮
# │                    CONFIGURATION CONSTANTS                   │
# ╰──────────────────────────────────────────────────────────────╯

# Gemini API Configuration
GEMINI_MODEL = "gemma-3-27b-it"
GEMINI_TIMEOUT = 30  # seconds

# F5-TTS Configuration
F5_TTS_MAX_RETRIES = 3
F5_TTS_TIMEOUT = 120  # seconds
F5_TTS_MAX_CHARS = 400

# Manim Configuration
MANIM_TIMEOUT = int(os.getenv("MANIM_TIMEOUT", "120"))
FINAL_VIDEO_DIR = Path("final_video_output")
TEMP_SCENE_DIR_PREFIX = "temp_scene_"

# Audio Processing Configuration
CROSSFADE_DURATION_MS = 50
MIN_AUDIO_DURATION = 0.1  # seconds
MIN_FILE_SIZE = 1024  # bytes

# Network Retry Configuration
RETRY_DELAY_BASE = 1  # Base delay in seconds
RETRY_DELAY_MAX = 60  # Maximum delay in seconds
RETRY_BACKOFF_MULTIPLIER = 2  # Exponential backoff multiplier

# ╭──────────────────────────────────────────────────────────────╮
# │                   NETWORK RETRY UTILITIES                    │
# ╰──────────────────────────────────────────────────────────────╯

def is_connection_error(exception: Exception) -> bool:
    """
    Check if an exception is a connection-related error that should trigger retry.
    
    Args:
        exception: Exception to check
        
    Returns:
        True if it's a connection error, False otherwise
    """
    # Check for specific error codes and messages
    error_indicators = [
        "104",  # Connection reset by peer
        "connection reset by peer",
        "connection refused",
        "connection timeout",
        "connection aborted",
        "network is unreachable",
        "host is unreachable",
        "timeout",
        "timed out",
        "connection error",
        "connection failed",
        "unable to connect",
        "can't connect",
        "cannot connect"
    ]
    
    error_str = str(exception).lower()
    
    # Check for socket errors
    if hasattr(exception, 'errno'):
        connection_errnos = [
            errno.ECONNRESET,    # Connection reset by peer
            errno.ECONNREFUSED,  # Connection refused
            errno.ECONNABORTED,  # Connection aborted
            errno.ETIMEDOUT,     # Connection timed out
            errno.EHOSTUNREACH,  # Host unreachable
            errno.ENETUNREACH,   # Network unreachable
        ]
        if exception.errno in connection_errnos:
            return True
    
    # Check for string indicators
    return any(indicator in error_str for indicator in error_indicators)

def retry_with_backoff(func, *args, **kwargs):
    """
    Execute a function with unlimited retries and exponential backoff for connection errors.
    
    Args:
        func: Function to execute
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Result of the function execution
    """
    attempt = 0
    delay = RETRY_DELAY_BASE
    
    while True:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if is_connection_error(e):
                attempt += 1
                print(f"🔄 Connection error detected (attempt {attempt}): {e}")
                print(f"⏳ Retrying in {delay} seconds...")
                time.sleep(delay)
                
                # Exponential backoff with maximum delay
                delay = min(delay * RETRY_BACKOFF_MULTIPLIER, RETRY_DELAY_MAX)
                continue
            else:
                # Re-raise non-connection errors
                raise

# ╭──────────────────────────────────────────────────────────────╮
# │                   PDF PROCESSING UTILITIES                   │
# ╰──────────────────────────────────────────────────────────────╯

def convert_pdf_to_images(pdf_bytes: bytes) -> List[Image.Image]:
    """
    Converts raw PDF bytes into a list of PIL Image objects.
    
    Args:
        pdf_bytes: Raw PDF file content as bytes
        
    Returns:
        List of PIL Image objects, one per PDF page
        
    Raises:
        Exception: If PDF conversion fails
    """
    try:
        images = convert_from_bytes(pdf_bytes, dpi=200, fmt='JPEG')
        print(f"✅ Successfully converted PDF to {len(images)} images")
        return images
    except Exception as e:
        print(f"❌ PDF conversion failed: {e}")
        raise

def image_to_base64(image: Image.Image) -> str:
    """
    Converts a PIL Image object to a base64 encoded string.
    
    Args:
        image: PIL Image object
        
    Returns:
        Base64 encoded string representation of the image
    """
    try:
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=85, optimize=True)
        encoded = base64.b64encode(buffered.getvalue()).decode()
        return encoded
    except Exception as e:
        print(f"❌ Image encoding failed: {e}")
        raise

# ╭──────────────────────────────────────────────────────────────╮
# │                   GEMINI API UTILITIES                       │
# ╰──────────────────────────────────────────────────────────────╯

def get_gemini_client() -> genai.Client:
    """
    Initialize and return Gemini API client.
    
    Returns:
        Configured Gemini API client
        
    Raises:
        ValueError: If GEMINI_API_KEY is not set
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable not set. "
            "Please set your API key: export GEMINI_API_KEY='your_key_here'"
        )
    return genai.Client(api_key=api_key)

def is_gemini_available() -> bool:
    """
    Check if Gemini API is available and properly configured.
    
    Returns:
        True if API is accessible, False otherwise
    """
    def _test_connection():
        client = get_gemini_client()
        # Test with a minimal request
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents="Test connection"
        )
        print("✅ Gemini API connection verified")
        return True
    
    try:
        return retry_with_backoff(_test_connection)
    except Exception as e:
        print(f"❌ Gemini API not available: {e}")
        return False

def _upload_image_to_gemini(image: Image.Image) -> str:
    """
    Upload image to Gemini API and return file reference.
    
    Args:
        image: PIL Image object to upload
        
    Returns:
        Uploaded file reference for API calls
    """
    def _upload_with_retry():
        # Convert PIL image to bytes
        img_buffer = BytesIO()
        image.save(img_buffer, format='JPEG', quality=90)
        img_buffer.seek(0)
        
        # Create temporary file for upload
        temp_img_path = f"temp_img_{uuid.uuid4()}.jpg"
        
        try:
            with open(temp_img_path, 'wb') as f:
                f.write(img_buffer.getvalue())
            
            # Upload to Gemini API
            client = get_gemini_client()
            uploaded_file = client.files.upload(file=temp_img_path)
            
            return uploaded_file
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
    
    return retry_with_backoff(_upload_with_retry)

# ╭──────────────────────────────────────────────────────────────╮
# │                    AI ANALYSIS PROMPTS                       │
# ╰──────────────────────────────────────────────────────────────╯

def gemma3_teaching_prompt() -> str:
    """
    Returns the system prompt for generating teaching content.
    
    Returns:
        Formatted prompt string for educational content generation
    """
    return (
        "You are an expert teacher with years of experience in education. "
        "Analyze the provided content and create a comprehensive teaching explanation. "
        
        "REQUIREMENTS:\n"
        "- Write as a single, continuous paragraph\n"
        "- Explain everything in detail for beginners\n"
        "- Use clear, simple language\n"
        "- No lists, bullet points, or section headings\n"
        "- No special symbols or formatting\n"
        "- Focus on clarity and educational value\n"
        "- Make the explanation flow naturally\n"
        "- Ensure content is engaging and informative\n\n"
        
        "Create a detailed teaching explanation that helps students understand the content thoroughly."
    )

def gemma3_video_scene_prompt() -> str:
    """
    Returns the system prompt for generating video scene descriptions.
    
    Returns:
        Formatted prompt string for video scene generation
    """
    return (
        "You are a professional video scene designer specializing in educational content. "
        "Create detailed scene descriptions for Manim educational videos. "
        
        "REQUIREMENTS:\n"
        "- Focus on visual elements and animations\n"
        "- Describe mathematical concepts and diagrams\n"
        "- Specify what should appear on screen\n"
        "- Detail how elements should move or transform\n"
        "- Emphasize key educational points visually\n"
        "- No audio or narration instructions\n"
        "- Be specific about visual representations\n"
        "- Consider animation timing and flow\n\n"
        
        "Create a comprehensive visual scene description that will help students learn through animation."
    )

# ╭──────────────────────────────────────────────────────────────╮
# │                  AI CONTENT ANALYSIS FUNCTIONS               │
# ╰──────────────────────────────────────────────────────────────╯

def analyze_page_with_gemma3(image: Image.Image) -> str:
    """
    Uses Gemma 3 via Gemini API to analyze an image and generate a teaching script.
    
    Args:
        image: PIL Image object of the PDF page
        
    Returns:
        Generated teaching script as a string
        
    Raises:
        Exception: If API call fails
    """
    def _analyze_with_retry():
        prompt = gemma3_teaching_prompt()
        
        print("🧠 Analyzing page content for teaching script...")
        
        # Upload image to Gemini API
        uploaded_file = _upload_image_to_gemini(image)
        
        # Generate content with image and prompt
        client = get_gemini_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[uploaded_file, prompt],
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=1000
            )
        )
        
        content = response.text.strip()
        
        # Clean up content
        content = re.sub(r'\([^)]*\)', '', content)  # Remove parenthetical text
        content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
        
        print(f"✅ Generated teaching script ({len(content)} characters)")
        return content
    
    try:
        return retry_with_backoff(_analyze_with_retry)
    except Exception as e:
        print(f"❌ Error analyzing page with Gemini API: {e}")
        raise

def analyze_page_for_video_scene(image: Image.Image) -> str:
    """
    Uses Gemma 3 via Gemini API to analyze an image and generate a video scene description.
    
    Args:
        image: PIL Image object of the PDF page
        
    Returns:
        Generated video scene description as a string
        
    Raises:
        Exception: If API call fails
    """
    def _analyze_scene_with_retry():
        prompt = gemma3_video_scene_prompt()
        
        print("🎬 Analyzing page content for video scene...")
        
        # Upload image to Gemini API
        uploaded_file = _upload_image_to_gemini(image)
        
        # Generate content with image and prompt
        client = get_gemini_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[uploaded_file, prompt],
            config=types.GenerateContentConfig(
                temperature=0.8,
                max_output_tokens=800
            )
        )
        
        content = response.text.strip()
        
        print(f"✅ Generated video scene description ({len(content)} characters)")
        return content
    
    try:
        return retry_with_backoff(_analyze_scene_with_retry)
    except Exception as e:
        print(f"❌ Error generating video scene with Gemini API: {e}")
        raise

# ╭──────────────────────────────────────────────────────────────╮
# │                   F5-TTS VOICE UTILITIES                     │
# ╰──────────────────────────────────────────────────────────────╯

def get_all_voice_samples() -> List[str]:
    """
    Scans the voice samples directory and returns a sorted list of .wav files.
    
    Returns:
        List of voice sample filenames
    """
    voice_samples_dir = Path.cwd() / 'F5-TTS' / 'voice_samples'
    voice_samples_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        voice_files = sorted([f.name for f in voice_samples_dir.glob('*.wav')])
        print(f"📁 Found {len(voice_files)} voice samples")
        return voice_files
    except OSError as e:
        print(f"❌ Error accessing voice samples directory: {e}")
        return []

def call_f5_tts(text: str, name_prefix: str, voice_sample_path: str, max_retries: int = F5_TTS_MAX_RETRIES) -> Tuple[Optional[str], Optional[str]]:
    """
    Calls F5-TTS to generate audio from text using a voice sample.
    
    Args:
        text: Text to convert to speech
        name_prefix: Prefix for output filenames
        voice_sample_path: Path to voice sample file
        max_retries: Maximum number of retry attempts
        
    Returns:
        Tuple of (audio_file_path, text_file_path) or (None, text_file_path) if failed
    """
    if not text or not text.strip():
        print("⚠️ Skipping empty or whitespace-only text chunk")
        return None, None
    
    textfile = None
    outwav = None
    
    def _f5_tts_with_retry():
        nonlocal textfile, outwav
        
        # Generate unique identifiers
        uid = str(uuid.uuid4())
        textfile = Path.cwd() / "F5-TTS" / "texts" / f"tts_input_{uid}.txt"
        outwav = Path.cwd() / "F5-TTS" / "audio" / f"{name_prefix}_{uid}.wav"
        
        # Create directories
        textfile.parent.mkdir(parents=True, exist_ok=True)
        outwav.parent.mkdir(parents=True, exist_ok=True)
        
        # Write text to file
        with open(textfile, "w", encoding="utf-8") as f:
            f.write(text)
        
        # Prepare F5-TTS command
        python_exec = "/home/campus/nikhil/AITeacher/aiteacher/bin/python3.11"
        worker_script = "/home/campus/nikhil/AITeacher/F5-TTS/f5_tts_worker.py"
        
        cmd = [python_exec, worker_script, str(textfile), str(outwav)]
        if voice_sample_path:
            cmd.append(voice_sample_path)
        
        print(f"🎤 Generating audio with unlimited retries...")
        
        # Execute F5-TTS
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True, 
            timeout=F5_TTS_TIMEOUT
        )
        
        print(f"✅ F5-TTS completed successfully")
        if result.stdout:
            print(f"📝 F5-TTS output: {result.stdout[:200]}...")
        
        return str(outwav), str(textfile)
    
    # First try with traditional retry logic for non-connection errors
    for attempt in range(max_retries):
        try:
            return _f5_tts_with_retry()
            
        except subprocess.CalledProcessError as e:
            if is_connection_error(e):
                print(f"🔄 Connection error in F5-TTS, switching to unlimited retry mode...")
                break
            else:
                print(f"❌ F5-TTS subprocess failed (attempt {attempt + 1}/{max_retries})")
                print(f"STDOUT: {e.stdout}")
                print(f"STDERR: {e.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"⏰ F5-TTS timed out (attempt {attempt + 1}/{max_retries})")
            
        except Exception as e:
            if is_connection_error(e):
                print(f"🔄 Connection error in F5-TTS, switching to unlimited retry mode...")
                break
            else:
                print(f"❌ Unexpected F5-TTS error (attempt {attempt + 1}/{max_retries}): {e}")
    
    # If we reach here due to connection errors, use unlimited retry
    try:
        return retry_with_backoff(_f5_tts_with_retry)
    except Exception as e:
        if not is_connection_error(e):
            print(f"❌ Failed to generate audio for: {text[:50]}...")
            return None, str(textfile) if textfile else None
        else:
            # This should not happen as retry_with_backoff handles connection errors
            raise

# ╭──────────────────────────────────────────────────────────────╮
# │                   TEXT PROCESSING UTILITIES                  │
# ╰──────────────────────────────────────────────────────────────╯

def chunk_text(text: str, max_chars: int = F5_TTS_MAX_CHARS) -> List[str]:
    """
    Splits a large text into smaller chunks based on sentences.
    
    Args:
        text: Input text to chunk
        max_chars: Maximum characters per chunk
        
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Check if adding this sentence would exceed the limit
        if len(current_chunk) + len(sentence) + 1 > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += (" " + sentence) if current_chunk else sentence
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Filter out empty chunks
    chunks = [chunk for chunk in chunks if chunk.strip()]
    
    print(f"📝 Split text into {len(chunks)} chunks (max {max_chars} chars each)")
    return chunks

# ╭──────────────────────────────────────────────────────────────╮
# │                   AUDIO PROCESSING UTILITIES                 │
# ╰──────────────────────────────────────────────────────────────╯

def concatenate_audios_with_crossfade(wav_paths: List[str], output_path: str, crossfade_ms: int = CROSSFADE_DURATION_MS) -> Optional[str]:
    """
    Combines multiple WAV files into a single file with crossfade transitions.
    
    Args:
        wav_paths: List of paths to WAV files
        output_path: Path for the output combined file
        crossfade_ms: Crossfade duration in milliseconds
        
    Returns:
        Output file path if successful, None otherwise
    """
    if not wav_paths:
        print("❌ No audio files provided for concatenation")
        return None
    
    print(f"🎵 Concatenating {len(wav_paths)} audio files...")
    
    valid_chunks = []
    
    # Validate and load audio chunks
    for i, path in enumerate(wav_paths):
        if not path or not os.path.exists(path):
            print(f"⚠️ Skipping missing file: {path}")
            continue
        
        if os.path.getsize(path) < MIN_FILE_SIZE:
            print(f"⚠️ Skipping small file: {path}")
            continue
        
        try:
            chunk = AudioSegment.from_wav(path)
            if chunk.duration_seconds > MIN_AUDIO_DURATION:
                valid_chunks.append(chunk)
                print(f"✅ Loaded audio chunk {i + 1}: {chunk.duration_seconds:.2f}s")
            else:
                print(f"⚠️ Skipping short audio chunk: {path}")
        except Exception as e:
            print(f"❌ Failed to load audio file {path}: {e}")
    
    if not valid_chunks:
        print("❌ No valid audio chunks found for concatenation")
        return None
    
    try:
        # Start with the first chunk
        combined = valid_chunks[0]
        
        # Add remaining chunks with crossfade
        for i, chunk in enumerate(valid_chunks[1:], 1):
            print(f"🔗 Adding chunk {i + 1} with {crossfade_ms}ms crossfade...")
            combined = combined.append(chunk, crossfade=crossfade_ms)
        
        # Export the final combined audio
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined.export(output_path, format="wav")
        
        duration = combined.duration_seconds
        print(f"✅ Successfully created combined audio: {duration:.2f}s at {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Failed to concatenate audio files: {e}")
        return None

def cleanup_files(file_paths: List[str]) -> None:
    """
    Deletes a list of temporary files safely.
    
    Args:
        file_paths: List of file paths to delete
    """
    if not file_paths:
        return
    
    cleaned_count = 0
    for path in file_paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
                cleaned_count += 1
            except OSError as e:
                print(f"⚠️ Could not remove file {path}: {e}")
    
    if cleaned_count > 0:
        print(f"🧹 Cleaned up {cleaned_count} temporary files")

# ╭──────────────────────────────────────────────────────────────╮
# │                   MANIM SETUP VERIFICATION                   │
# ╰──────────────────────────────────────────────────────────────╯

def _check_manim_setup() -> bool:
    """
    Verify that required dependencies are available for video generation.
    
    Returns:
        True if setup is valid
        
    Raises:
        RuntimeError: If setup is invalid
        FileNotFoundError: If required files are missing
    """
    print("🔍 Verifying Manim setup...")
    
    # Check if manimator module is available
    if not MANIMATOR_AVAILABLE:
        raise RuntimeError(
            "Manimator module not available. Please ensure manimator.py is in the same directory."
        )
    
    # Check if GEMINI_API_KEY is set
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY environment variable not set. "
            "Please set your API key: export GEMINI_API_KEY='your_key_here'"
        )
    
    # Check if Manim is installed
    def _check_manim_version():
        result = subprocess.run(
            ["manim", "--version"], 
            check=True, 
            capture_output=True, 
            text=True,
            timeout=10
        )
        print(f"✅ Manim version: {result.stdout.strip()}")
        return True
    
    try:
        retry_with_backoff(_check_manim_version)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            "Manim not found. Please install ManimCE: pip install manim"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Manim command timed out")
    
    # Check if MoviePy is available for video concatenation
    if not MOVIEPY_AVAILABLE:
        print("⚠️ MoviePy not available - video concatenation may be limited")
    
    print("✅ Manim setup verification completed")
    return True

# ╭──────────────────────────────────────────────────────────────╮
# │                   MANIM VIDEO GENERATION                     │
# ╰──────────────────────────────────────────────────────────────╯

def generate_manim_video(scene_descriptions: List[str]) -> Optional[str]:
    """
    Generate a complete video from scene descriptions using the manimator module.
    
    Args:
        scene_descriptions: List of scene descriptions for video generation
        
    Returns:
        Path to the generated video file, or None if failed
    """
    if not scene_descriptions:
        print("❌ No scene descriptions provided for video generation")
        return None
    
    print(f"🎬 Starting video generation for {len(scene_descriptions)} scenes...")
    
    def _generate_with_retry():
        if MANIMATOR_AVAILABLE:
            print("🚀 Using advanced manimator for video generation...")
            return generate_video_from_scenes(scene_descriptions)
        else:
            print("⚠️ Using fallback video generation method...")
            return _generate_manim_video_fallback(scene_descriptions)
    
    # Use manimator if available
    if MANIMATOR_AVAILABLE:
        try:
            return retry_with_backoff(_generate_with_retry)
        except Exception as e:
            if not is_connection_error(e):
                print(f"❌ Manimator failed: {e}")
                print("⚠️ Falling back to legacy method...")
                return _generate_manim_video_fallback(scene_descriptions)
            else:
                # Connection error will be handled by retry_with_backoff
                raise
    else:
        print("⚠️ Using fallback video generation method...")
        return _generate_manim_video_fallback(scene_descriptions)

def _generate_manim_video_fallback(scene_descriptions: List[str]) -> Optional[str]:
    """
    Fallback implementation for video generation when manimator is not available.
    
    Args:
        scene_descriptions: List of scene descriptions
        
    Returns:
        Path to generated video or None if failed
    """
    print("🔄 Fallback video generation method")
    
    # This is a placeholder for the original gemini-cli method
    # In a real implementation, you would include the original video generation logic
    
    def _fallback_with_retry():
        # Verify basic setup
        _check_manim_setup()
        
        # Create output directory
        FINAL_VIDEO_DIR.mkdir(exist_ok=True)
        
        # For now, return None to indicate that fallback is not fully implemented
        print("❌ Fallback video generation not fully implemented")
        print("💡 Please ensure manimator.py is available for full video generation")
        
        return None
    
    try:
        return retry_with_backoff(_fallback_with_retry)
    except Exception as e:
        if not is_connection_error(e):
            print(f"❌ Fallback video generation failed: {e}")
            return None
        else:
            # Connection error will be handled by retry_with_backoff
            raise

# ╭──────────────────────────────────────────────────────────────╮
# │                   UTILITY HELPER FUNCTIONS                   │
# ╰──────────────────────────────────────────────────────────────╯

def get_system_info() -> dict:
    """
    Get system information for debugging and setup verification.
    
    Returns:
        Dictionary containing system information
    """
    info = {
        "gemini_api_available": is_gemini_available(),
        "manimator_available": MANIMATOR_AVAILABLE,
        "moviepy_available": MOVIEPY_AVAILABLE,
        "voice_samples_count": len(get_all_voice_samples()),
        "working_directory": str(Path.cwd()),
        "environment_variables": {
            "GEMINI_API_KEY": "Set" if os.getenv("GEMINI_API_KEY") else "Not Set",
            "GENAI_MODEL": os.getenv("GENAI_MODEL", "Not Set"),
            "MANIM_TIMEOUT": os.getenv("MANIM_TIMEOUT", "Not Set")
        }
    }
    
    return info

def print_system_info() -> None:
    """Print system information for debugging purposes."""
    info = get_system_info()
    
    print("\n" + "="*60)
    print("🔧 SYSTEM INFORMATION")
    print("="*60)
    
    print(f"📡 Gemini API Available: {'✅' if info['gemini_api_available'] else '❌'}")
    print(f"🎬 Manimator Available: {'✅' if info['manimator_available'] else '❌'}")
    print(f"🎞️ MoviePy Available: {'✅' if info['moviepy_available'] else '❌'}")
    print(f"🎤 Voice Samples: {info['voice_samples_count']} found")
    print(f"📁 Working Directory: {info['working_directory']}")
    
    print("\n🔐 Environment Variables:")
    for key, value in info['environment_variables'].items():
        print(f"  {key}: {value}")
    
    print("="*60 + "\n")

# ╭──────────────────────────────────────────────────────────────╮
# │                    MODULE INITIALIZATION                     │
# ╰──────────────────────────────────────────────────────────────╯

def initialize_module() -> bool:
    """
    Initialize the utils module and verify all dependencies.
    
    Returns:
        True if initialization successful, False otherwise
    """
    def _initialize_with_retry():
        print("🚀 Initializing PDF Seamless Teacher utilities...")
        
        # Create necessary directories
        directories = [
            Path.cwd() / "F5-TTS" / "voice_samples",
            Path.cwd() / "F5-TTS" / "texts",
            Path.cwd() / "F5-TTS" / "audio",
            FINAL_VIDEO_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print("✅ Directory structure created")
        
        # Verify API connectivity
        if is_gemini_available():
            print("✅ Gemini API connection verified")
        else:
            print("⚠️ Gemini API not available")
            return False
        
        print("✅ Utils module initialized successfully")
        return True
    
    try:
        return retry_with_backoff(_initialize_with_retry)
    except Exception as e:
        if not is_connection_error(e):
            print(f"❌ Module initialization failed: {e}")
            return False
        else:
            # Connection error will be handled by retry_with_backoff
            raise

# ╭──────────────────────────────────────────────────────────────╮
# │                    END OF UTILS MODULE                       │
# ╰──────────────────────────────────────────────────────────────╯

# Auto-initialize when module is imported
if __name__ != "__main__":
    initialize_module()

# Main execution for testing
if __name__ == "__main__":
    print("🧪 Running utils.py in test mode...")
    print_system_info()
    
    if initialize_module():
        print("✅ All systems ready!")
    else:
        print("❌ System check failed!")
