#!/usr/bin/env python3

import os
import uuid
import subprocess
import base64
import re
import time
import socket
import errno
import json
import concurrent.futures
import threading
import multiprocessing
import asyncio
import shutil
import glob
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from io import BytesIO

import PyPDF2

from pydub import AudioSegment

import google.genai as genai
from google.genai import types

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

try:
    from manimator import generate_video_from_scenes, generate_manim_code_with_opencode, render_single_scene
    MANIMATOR_AVAILABLE = True
except ImportError:
    MANIMATOR_AVAILABLE = False

MAX_WORKERS = min(multiprocessing.cpu_count() * 6, 24)
CONCURRENT_REQUESTS = 8
ENABLE_CACHING = True
CACHE_TIMEOUT = 3600
GPU_ACCELERATION = True
CUDA_VISIBLE_DEVICES = "0"

GEMINI_MODEL = "gemma-3-27b-it"
GEMINI_TIMEOUT = 20
GEMINI_MAX_RETRIES = 3

F5_TTS_MAX_RETRIES = int(os.getenv("F5_TTS_MAX_RETRIES", "3"))
F5_TTS_TIMEOUT = int(os.getenv("F5_TTS_TIMEOUT", "300"))
F5_TTS_MAX_CHARS = 400

MANIM_TIMEOUT = int(os.getenv("MANIM_TIMEOUT", "180"))
FINAL_VIDEO_DIR = Path("final_video_output")

CROSSFADE_DURATION_MS = 30
MIN_AUDIO_DURATION = 0.1
MIN_FILE_SIZE = 1024

WHISPER_MODEL_SIZE = "base"
WHISPER_LANGUAGE = None
WHISPER_DEVICE = "cuda" if GPU_ACCELERATION else "cpu"

RETRY_DELAY_BASE = 0.5
RETRY_DELAY_MAX = 30
RETRY_BACKOFF_MULTIPLIER = 1.5

DEFAULT_PAGES_PER_CHUNK = 5

RESPONSE_CACHE = {}
CACHE_LOCK = threading.Lock()

def get_cache_key(content: str) -> str:
    return hashlib.md5(content.encode()).hexdigest()

def get_cached_response(key: str) -> Optional[str]:
    if not ENABLE_CACHING:
        return None
    
    with CACHE_LOCK:
        if key in RESPONSE_CACHE:
            cached_data, timestamp = RESPONSE_CACHE[key]
            if time.time() - timestamp < CACHE_TIMEOUT:
                return cached_data
            else:
                del RESPONSE_CACHE[key]
    
    return None

def cache_response(key: str, response: str) -> None:
    if not ENABLE_CACHING:
        return
    
    with CACHE_LOCK:
        RESPONSE_CACHE[key] = (response, time.time())

def setup_gpu_environment() -> None:
    if GPU_ACCELERATION:
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
        os.environ["NUMBA_CUDA_DRIVER"] = "/usr/lib/x86_64-linux-gnu/libcuda.so.1"
        os.environ["NUMBA_DISABLE_JIT"] = "0"

@lru_cache(maxsize=128)
def get_gemini_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    
    return genai.Client(api_key=api_key)

def is_connection_error(exception: Exception) -> bool:
    error_indicators = [
        "104",
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
    
    if hasattr(exception, 'errno'):
        connection_errnos = [
            errno.ECONNRESET,
            errno.ECONNREFUSED,
            errno.ECONNABORTED,
            errno.ETIMEDOUT,
            errno.EHOSTUNREACH,
            errno.ENETUNREACH,
        ]
        
        if exception.errno in connection_errnos:
            return True
    
    return any(indicator in error_str for indicator in error_indicators)

def retry_with_backoff(func, *args, **kwargs):
    attempt = 0
    delay = RETRY_DELAY_BASE
    
    while True:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if is_connection_error(e):
                attempt += 1
                
                if "104" in str(e) or "Connection reset by peer" in str(e).lower():
                    time.sleep(0.01)
                    continue
                
                if attempt <= GEMINI_MAX_RETRIES:
                    time.sleep(delay)
                    delay = min(delay * RETRY_BACKOFF_MULTIPLIER, RETRY_DELAY_MAX)
                    continue
                else:
                    raise
            else:
                raise

def generate_timestamped_transcript(audio_path: str) -> Tuple[List[Dict], str]:
    if not WHISPER_AVAILABLE:
        raise Exception(
            "Whisper not available. Please install with: pip install openai-whisper"
        )
    
    def _transcribe_with_retry():
        setup_gpu_environment()
        
        model = whisper.load_model(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE)
        
        result = model.transcribe(
            audio_path,
            word_timestamps=True,
            verbose=False,
            language=WHISPER_LANGUAGE,
            fp16=True,
            best_of=1,
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
        )
        
        timestamped_segments = []
        for segment in result["segments"]:
            timestamped_segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip(),
                "duration": segment["end"] - segment["start"]
            })
        
        full_transcript = result["text"]
        
        return timestamped_segments, full_transcript
    
    try:
        return retry_with_backoff(_transcribe_with_retry)
    except Exception as e:
        raise

def save_timestamped_transcript(segments: List[Dict], output_path: str) -> None:
    def format_timestamp(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments, 1):
            start_time = format_timestamp(segment["start"])
            end_time = format_timestamp(segment["end"])
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{segment['text']}\n\n")

def save_timestamped_transcript_json(segments: List[Dict], output_path: str) -> None:
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)

def generate_chunk_based_audio_and_subtitles(teaching_scripts: List[str], voice_sample_path: str) -> List[Dict]:
    """Generate audio and subtitles for all teaching script chunks."""
    print(f"🎵 Starting chunk-based audio and subtitle generation for {len(teaching_scripts)} scripts")
    
    chunk_data = []
    
    valid_scripts = [(i, script) for i, script in enumerate(teaching_scripts) if script.strip()]
    print(f"📝 Found {len(valid_scripts)} valid scripts out of {len(teaching_scripts)} total")
    
    if not valid_scripts:
        print("❌ No valid scripts found for audio generation")
        return chunk_data
    
    setup_gpu_environment()
    
    def process_chunk(script_data):
        i, script = script_data
        chunk_num = i + 1
        
        print(f"🎵 Processing chunk {chunk_num}: generating audio from script (length: {len(script)} chars)")
        
        audio_path, _ = call_f5_tts(script, f"chunk_{chunk_num}", voice_sample_path)
        
        if audio_path and os.path.exists(audio_path):
            print(f"✅ Audio generated for chunk {chunk_num}: {audio_path}")
            
            print(f"🎯 Generating timestamped transcript for chunk {chunk_num}")
            timestamped_segments, full_transcript = generate_timestamped_transcript(audio_path)
            
            srt_path = f"chunk_{chunk_num}_subtitles_{uuid.uuid4()}.srt"
            save_timestamped_transcript(timestamped_segments, srt_path)
            print(f"✅ Subtitles saved for chunk {chunk_num}: {srt_path}")
            
            audio_segment = AudioSegment.from_wav(audio_path)
            total_duration = audio_segment.duration_seconds
            print(f"⏱️ Chunk {chunk_num} duration: {total_duration:.2f} seconds")
            
            chunk_info = {
                "chunk_number": chunk_num,
                "script": script,
                "audio_path": audio_path,
                "srt_path": srt_path,
                "timestamped_segments": timestamped_segments,
                "total_duration": total_duration,
                "full_transcript": full_transcript,
                "segments": timestamped_segments
            }
            
            print(f"✅ Chunk {chunk_num} processing completed successfully")
            return chunk_info
        else:
            print(f"❌ Audio generation failed for chunk {chunk_num}")
            return None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(valid_scripts))) as executor:
        print(f"🚀 Starting parallel processing with {min(MAX_WORKERS, len(valid_scripts))} workers")
        futures = [executor.submit(process_chunk, script_data) for script_data in valid_scripts]
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                chunk_data.append(result)
    
    print(f"✅ Audio and subtitle generation completed. Generated {len(chunk_data)} valid chunks")
    return chunk_data

def chunk_pdf_by_pages(pdf_bytes: bytes, pages_per_chunk: int = DEFAULT_PAGES_PER_CHUNK) -> List[bytes]:
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
        total_pages = len(pdf_reader.pages)
        
        chunks = []
        for start_page in range(0, total_pages, pages_per_chunk):
            end_page = min(start_page + pages_per_chunk, total_pages)
            
            pdf_writer = PyPDF2.PdfWriter()
            
            for page_num in range(start_page, end_page):
                pdf_writer.add_page(pdf_reader.pages[page_num])
            
            chunk_buffer = BytesIO()
            pdf_writer.write(chunk_buffer)
            chunk_bytes = chunk_buffer.getvalue()
            chunks.append(chunk_bytes)
        
        return chunks
        
    except Exception as e:
        raise

def process_pdf_with_chunking(pdf_path: str, pages_per_chunk: int = DEFAULT_PAGES_PER_CHUNK) -> List[str]:
    """Process PDF into chunks and generate teaching scripts for each chunk."""
    print(f"📄 Processing PDF: {pdf_path} with {pages_per_chunk} pages per chunk")
    
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    
    print(f"📚 PDF loaded, size: {len(pdf_bytes)} bytes")
    
    pdf_chunks = chunk_pdf_by_pages(pdf_bytes, pages_per_chunk)
    print(f"📝 PDF split into {len(pdf_chunks)} chunks")
    
    teaching_scripts = []
    
    for i, chunk in enumerate(pdf_chunks, 1):
        print(f"🔄 Processing chunk {i}/{len(pdf_chunks)}")
        try:
            teaching_script = analyze_pdf_chunk_with_gemma3(chunk)
            teaching_scripts.append(teaching_script)
            print(f"✅ Generated teaching script for chunk {i}")
            
        except Exception as e:
            print(f"❌ Failed to generate teaching script for chunk {i}: {e}")
            teaching_scripts.append("")
    
    print(f"✅ PDF processing completed. Generated {len([s for s in teaching_scripts if s.strip()])} valid teaching scripts")
    return teaching_scripts

def generate_video_scenes_from_scripts(teaching_scripts: List[str]) -> List[str]:
    """Generate detailed video scene descriptions from teaching scripts."""
    print(f"🎬 Generating video scenes from {len(teaching_scripts)} teaching scripts")
    
    video_scenes = []
    
    for i, script in enumerate(teaching_scripts, 1):
        if not script.strip():
            print(f"⚠️ Skipping empty script for scene {i}")
            video_scenes.append("")
            continue
        
        print(f"🎨 Generating scene {i}/{len(teaching_scripts)}")
        try:
            scene_description = generate_scene_from_script(script)
            video_scenes.append(scene_description)
            print(f"✅ Generated scene description {i}")
            
        except Exception as e:
            print(f"❌ Failed to generate scene description {i}: {e}")
            video_scenes.append("")
    
    valid_scenes = len([s for s in video_scenes if s.strip()])
    print(f"✅ Video scene generation completed. Generated {valid_scenes} valid scene descriptions")
    return video_scenes

def generate_scene_from_script(script: str) -> str:
    """Generate highly detailed scene descriptions from script text."""
    print(f"🎨 Generating detailed scene description from script (length: {len(script)} chars)")
    
    def _generate_scene_with_retry():
        client = get_gemini_client()
        
        prompt = f"""
You are a professional video scene designer specializing in educational content. Create EXTREMELY DETAILED scene descriptions for Manim educational videos.

CRITICAL REQUIREMENTS - YOU MUST FOLLOW ALL OF THESE:

1. UNIQUE ELEMENT NAMES: Every single visual element must have a completely unique name, even if they are the same type of object.
   - Use names like: circle_intro_1, circle_main_2, triangle_proof_3, text_title_4, etc.
   - Never reuse names, even for similar objects

2. EXPLICIT COORDINATES WITH NO OVERLAPPING:
   - Position: specify (x, y) coordinates like (2.5, 1.0), (-1.5, -0.5)
   - Size: specify radius, width, height like radius=1.2, width=3.0, height=0.8
   - Movement paths: specify start and end coordinates for all animations
   - MANDATORY: Ensure MINIMUM 2.0 units spacing between any two elements at all times
   - MANDATORY: Keep all elements within screen bounds x=[-6.5, 6.5], y=[-3.5, 3.5]
   - MANDATORY: Check for overlaps before placing any element - if overlap detected, adjust position

3. ARROW POSITIONING RULES:
   - NEVER specify direct coordinates for arrows
   - ALWAYS draw arrows from center of source element to center of target element
   - Position arrows BEHIND the target element (not in front)
   - Specify arrows as: "Draw arrow_name_X from center of element_A to center of element_B, positioned behind element_B"
   - Use buffer zones around elements so arrows don't overlap with other objects

4. COMPLETE TIMING: Specify exact timing for every animation:
   - Start time: when each element appears (e.g., "at 0.5 seconds")
   - Duration: how long each animation takes (e.g., "over 2.0 seconds")
   - Sequence: clear order of operations

5. PRECISE STYLING: Specify exact colors, fonts, sizes:
   - Colors: use specific values like "#FF6B6B", "BLUE", "GREEN"
   - Fonts: specify font families and sizes
   - Stroke width, fill opacity, etc.

6. DETAILED MOVEMENTS: Describe every transformation:
   - Translate: exact start and end positions
   - Rotate: specific angles and rotation centers
   - Scale: exact scaling factors
   - Fade in/out: specific opacity values

7. MATHEMATICAL PRECISION: For educational content:
   - Exact equation positions and formatting
   - Precise diagram layouts with measurements
   - Clear step-by-step mathematical transformations

8. ANTI-OVERLAP VERIFICATION:
   - Before placing any element, check if it would overlap with existing elements
   - If overlap detected, move the element to a non-overlapping position
   - Maintain visual hierarchy and logical flow while avoiding overlaps
   - Use grid-like positioning when necessary to ensure proper spacing

9. SCENE GENERATION ROBUSTNESS:
   - ALWAYS generate a complete, valid scene description
   - If initial attempt seems incomplete, expand with more visual elements
   - Include multiple visual components (shapes, text, animations) in every scene
   - Ensure the scene has substance and educational value
   - Never return empty or minimal descriptions

EXAMPLE FORMAT:
"Create a circle named circle_main_1 at position (0, 0) with radius 1.5 and color '#FF6B6B'. At 0.5 seconds, animate it moving to position (2.0, 1.0) over 1.5 seconds. Create text named text_label_2 with content 'Example' at position (-3.0, -1.5) using font size 24 (ensuring 2.0+ unit spacing from circle_main_1). Draw arrow_connection_3 from center of circle_main_1 to center of text_label_2, positioned behind text_label_2..."

Teaching Script to convert:
{script}

Generate an EXTREMELY DETAILED scene description with unique names, exact coordinates, precise timing, NO OVERLAPPING elements, proper arrow positioning, and maximum visual detail. MANDATORY: All elements must have minimum 2.0 units spacing and stay within screen boundaries. Use every available token to provide comprehensive specifications.
"""
        
        contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
        
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.9,  # Higher creativity for detailed descriptions
                max_output_tokens=8192,  # Maximum possible output
                top_p=0.95,  # High diversity
                top_k=40,  # Balanced selection
                response_mime_type="text/plain"
            )
        )
        
        scene_desc = response.text.strip()
        print(f"✅ Generated scene description (length: {len(scene_desc)} chars)")
        return scene_desc
    
    try:
        return retry_with_backoff(_generate_scene_with_retry)
    except Exception as e:
        print(f"❌ Scene generation failed: {e}")
        raise

def generate_video_scenes_from_chunks(pdf_chunks: List[bytes]) -> List[str]:
    """Generate detailed video scene descriptions directly from PDF chunks."""
    print(f"🎬 Generating video scenes from {len(pdf_chunks)} PDF chunks")
    
    video_scenes = []
    
    for i, chunk in enumerate(pdf_chunks, 1):
        print(f"🎨 Generating scene {i}/{len(pdf_chunks)} from PDF chunk")
        try:
            scene_description = analyze_pdf_chunk_for_video_scene(chunk)
            video_scenes.append(scene_description)
            print(f"✅ Generated scene description {i}")
            
        except Exception as e:
            print(f"❌ Failed to generate scene description {i}: {e}")
            video_scenes.append("")
    
    valid_scenes = len([s for s in video_scenes if s.strip()])
    print(f"✅ Video scene generation completed. Generated {valid_scenes} valid scene descriptions")
    return video_scenes

def is_gemini_available() -> bool:
    def _test_connection():
        client = get_gemini_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents="Test connection"
        )
        return True
    
    try:
        return retry_with_backoff(_test_connection)
    except Exception as e:
        return False

def gemma3_teaching_prompt() -> str:
    return (
        "You are an expert teacher with years of experience in education. "
        "Analyze the provided PDF content and create a comprehensive teaching explanation. "
        "REQUIREMENTS:\n"
        "- Write as multiple paragraphs if needed, but keep each paragraph focused and clear\n"
        "- The transcript must NOT contain any symbols, emoji, emotion gestures, sound effects, or non-speech content\n"
        "- Only provide the plain text that should be spoken by TTS (no [smiles], (laughs), <pause>, etc.)\n"
        "- No lists, bullet points, or section headings\n"
        "- No special symbols or formatting\n"
        "- Focus on clarity and educational value\n"
        "- Make the explanation flow naturally\n"
        "- Ensure content is engaging and informative\n\n"
        "Create a detailed teaching explanation that helps students understand the content thoroughly."
    )

def gemma3_video_scene_prompt() -> str:
    return (
        "You are a professional video scene designer specializing in educational content. "
        "Create EXTREMELY DETAILED scene descriptions for Manim educational videos.\n\n"
        "CRITICAL REQUIREMENTS - YOU MUST FOLLOW ALL OF THESE:\n\n"
        "1. UNIQUE ELEMENT NAMES: Every single visual element must have a completely unique name, even if they are the same type of object.\n"
        "   - Use names like: circle_intro_1, circle_main_2, triangle_proof_3, text_title_4, etc.\n"
        "   - Never reuse names, even for similar objects\n\n"
        "2. EXPLICIT COORDINATES WITH NO OVERLAPPING:\n"
        "   - Position: specify (x, y) coordinates like (2.5, 1.0), (-1.5, -0.5)\n"
        "   - Size: specify radius, width, height like radius=1.2, width=3.0, height=0.8\n"
        "   - Movement paths: specify start and end coordinates for all animations\n"
        "   - MANDATORY: Ensure MINIMUM 2.0 units spacing between any two elements at all times\n"
        "   - MANDATORY: Keep all elements within screen bounds x=[-6.5, 6.5], y=[-3.5, 3.5]\n"
        "   - MANDATORY: Check for overlaps before placing any element - if overlap detected, adjust position\n\n"
        "3. ARROW POSITIONING RULES:\n"
        "   - NEVER specify direct coordinates for arrows\n"
        "   - ALWAYS draw arrows from center of source element to center of target element\n"
        "   - Position arrows BEHIND the target element (not in front)\n"
        "   - Specify arrows as: 'Draw arrow_name_X from center of element_A to center of element_B, positioned behind element_B'\n"
        "   - Use buffer zones around elements so arrows don't overlap with other objects\n\n"
        "4. COMPLETE TIMING: Specify exact timing for every animation:\n"
        "   - Start time: when each element appears (e.g., 'at 0.5 seconds')\n"
        "   - Duration: how long each animation takes (e.g., 'over 2.0 seconds')\n"
        "   - Sequence: clear order of operations\n\n"
        "5. PRECISE STYLING: Specify exact colors, fonts, sizes:\n"
        "   - Colors: use specific values like '#FF6B6B', 'BLUE', 'GREEN'\n"
        "   - Fonts: specify font families and sizes\n"
        "   - Stroke width, fill opacity, etc.\n\n"
        "6. DETAILED MOVEMENTS: Describe every transformation:\n"
        "   - Translate: exact start and end positions\n"
        "   - Rotate: specific angles and rotation centers\n"
        "   - Scale: exact scaling factors\n"
        "   - Fade in/out: specific opacity values\n\n"
        "7. MATHEMATICAL PRECISION: For educational content:\n"
        "   - Exact equation positions and formatting\n"
        "   - Precise diagram layouts with measurements\n"
        "   - Clear step-by-step mathematical transformations\n\n"
        "8. ANTI-OVERLAP VERIFICATION:\n"
        "   - Before placing any element, check if it would overlap with existing elements\n"
        "   - If overlap detected, move the element to a non-overlapping position\n"
        "   - Maintain visual hierarchy and logical flow while avoiding overlaps\n"
        "   - Use grid-like positioning when necessary to ensure proper spacing\n\n"
        "9. SCENE GENERATION ROBUSTNESS:\n"
        "   - ALWAYS generate a complete, valid scene description\n"
        "   - If initial attempt seems incomplete, expand with more visual elements\n"
        "   - Include multiple visual components (shapes, text, animations) in every scene\n"
        "   - Ensure the scene has substance and educational value\n"
        "   - Never return empty or minimal descriptions\n\n"
        "EXAMPLE FORMAT:\n"
        "'Create a circle named circle_main_1 at position (0, 0) with radius 1.5 and color '#FF6B6B'. "
        "Create text named text_label_2 with content 'Example' at position (-3.0, -1.5) using font size 24 "
        "(ensuring 2.0+ unit spacing from circle_main_1). Draw arrow_connection_3 from center of circle_main_1 "
        "to center of text_label_2, positioned behind text_label_2...'\n\n"
        "Generate an EXTREMELY DETAILED scene description with unique names, exact coordinates, "
        "precise timing, NO OVERLAPPING elements, proper arrow positioning, and maximum visual detail. "
        "MANDATORY: All elements must have minimum 2.0 units spacing and stay within screen boundaries. "
        "Use every available token to provide comprehensive specifications."
    )

def analyze_pdf_chunk_with_gemma3(pdf_chunk: bytes) -> str:
    """Generate comprehensive teaching scripts from PDF content with maximum detail."""
    print(f"📚 Analyzing PDF chunk for teaching script (size: {len(pdf_chunk)} bytes)")
    
    def _analyze_with_retry():
        prompt = gemma3_teaching_prompt()
        
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(
                        mime_type="application/pdf",
                        data=pdf_chunk,
                    ),
                    types.Part.from_text(text=prompt),
                ],
            )
        ]
        
        client = get_gemini_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.8,  # Balanced creativity for teaching content
                max_output_tokens=8192,  # Maximum possible output
                top_p=0.95,  # High diversity
                top_k=40,  # Balanced selection
                response_mime_type="text/plain"
            )
        )
        
        content = response.text.strip()
        
        # Clean up any unwanted formatting
        content = re.sub(r'\([^)]*\)', '', content)
        content = re.sub(r'\s+', ' ', content)
        
        print(f"✅ Generated teaching script (length: {len(content)} chars)")
        return content
    
    try:
        return retry_with_backoff(_analyze_with_retry)
    except Exception as e:
        print(f"❌ Teaching script generation failed: {e}")
        raise

def analyze_pdf_chunk_for_video_scene(pdf_chunk: bytes) -> str:
    """Generate highly detailed scene descriptions from PDF chunk content."""
    print(f"🎬 Analyzing PDF chunk for detailed video scene (size: {len(pdf_chunk)} bytes)")
    
    def _analyze_scene_with_retry():
        prompt = gemma3_video_scene_prompt()
        
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(
                        mime_type="application/pdf",
                        data=pdf_chunk,
                    ),
                    types.Part.from_text(text=prompt),
                ],
            )
        ]
        
        client = get_gemini_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.9,  # Higher creativity for detailed descriptions
                max_output_tokens=8192,  # Maximum possible output
                top_p=0.95,  # High diversity
                top_k=40,  # Balanced selection
                response_mime_type="text/plain"
            )
        )
        
        content = response.text.strip()
        print(f"✅ Generated detailed scene description (length: {len(content)} chars)")
        return content
    
    try:
        return retry_with_backoff(_analyze_scene_with_retry)
    except Exception as e:
        print(f"❌ Scene analysis failed: {e}")
        raise

def get_all_voice_samples() -> List[str]:
    voice_samples_dir = Path.cwd() / 'F5-TTS' / 'voice_samples'
    voice_samples_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        voice_files = sorted([f.name for f in voice_samples_dir.glob('*.wav')])
        return voice_files
    except OSError as e:
        return []

def call_f5_tts(text: str, name_prefix: str, voice_sample_path: str, max_retries: int = F5_TTS_MAX_RETRIES) -> Tuple[Optional[str], Optional[str]]:
    if not text or not text.strip():
        return None, None
    
    textfile = None
    outwav = None
    
    def _f5_tts_with_retry():
        nonlocal textfile, outwav
        
        uid = str(uuid.uuid4())
        textfile = Path.cwd() / "F5-TTS" / "texts" / f"tts_input_{uid}.txt"
        outwav = Path.cwd() / "F5-TTS" / "audio" / f"{name_prefix}_{uid}.wav"
        
        textfile.parent.mkdir(parents=True, exist_ok=True)
        outwav.parent.mkdir(parents=True, exist_ok=True)
        
        with open(textfile, "w", encoding="utf-8") as f:
            f.write(text)
        
        python_exec = "/home/campus/nikhil/AITeacher/aiteacher/bin/python3.11"
        worker_script = "/home/campus/nikhil/AITeacher/F5-TTS/f5_tts_worker.py"
        cmd = [python_exec, worker_script, str(textfile), str(outwav)]
        
        if voice_sample_path:
            cmd.append(voice_sample_path)
        
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=F5_TTS_TIMEOUT
        )
        
        return str(outwav), str(textfile)
    
    for attempt in range(max_retries):
        try:
            return _f5_tts_with_retry()
        except subprocess.CalledProcessError as e:
            if is_connection_error(e):
                break
        except subprocess.TimeoutExpired:
            pass
        except Exception as e:
            if is_connection_error(e):
                break
    
    try:
        return retry_with_backoff(_f5_tts_with_retry)
    except Exception as e:
        if not is_connection_error(e):
            return None, str(textfile) if textfile else None
        else:
            raise

def chunk_text(text: str, max_chars: int = F5_TTS_MAX_CHARS) -> List[str]:
    if not text or not text.strip():
        return []
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        if len(current_chunk) + len(sentence) + 1 > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += (" " + sentence) if current_chunk else sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    chunks = [chunk for chunk in chunks if chunk.strip()]
    
    return chunks

def concatenate_audios_with_crossfade(wav_paths: List[str], output_path: str, crossfade_ms: int = CROSSFADE_DURATION_MS) -> Optional[str]:
    if not wav_paths:
        return None
    
    valid_chunks = []
    
    for i, path in enumerate(wav_paths):
        if not path or not os.path.exists(path):
            continue
        
        if os.path.getsize(path) < MIN_FILE_SIZE:
            continue
        
        try:
            chunk = AudioSegment.from_wav(path)
            if chunk.duration_seconds > MIN_AUDIO_DURATION:
                valid_chunks.append(chunk)
        except Exception as e:
            pass
    
    if not valid_chunks:
        return None
    
    try:
        combined = valid_chunks[0]
        
        for i, chunk in enumerate(valid_chunks[1:], 1):
            combined = combined.append(chunk, crossfade=crossfade_ms)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined.export(output_path, format="wav")
        
        return output_path
        
    except Exception as e:
        return None

def combine_video_and_audio(video_path: str, audio_path: str, output_path: str) -> Optional[str]:
    if not MOVIEPY_AVAILABLE:
        return None
    
    try:
        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_path)
        
        final_clip = video_clip.set_audio(audio_clip)
        if final_clip.duration > video_clip.duration:
            final_clip = final_clip.subclip(0, video_clip.duration)
        
        final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', verbose=False, logger=None)
        
        video_clip.close()
        audio_clip.close()
        final_clip.close()
        
        return output_path
    except Exception as e:
        return None

def cleanup_intermediate_files(files_to_keep: List[str]):
    import shutil
    import glob

    files_to_keep_abs = [os.path.abspath(f) for f in files_to_keep if f and os.path.exists(f)]

    patterns_to_delete = [
        "F5-TTS/audio/chunk_*.wav",
        "F5-TTS/texts/tts_input_*.txt",
        "chunk_*_subtitles_*.srt",
        "chunk_*_detailed_scenes_*.txt",
        "scene_segment_*.txt",
        "scene_*.py",
        "final_scene.py",
        "scene.py",
        "temp_pdf_*.pdf",
    ]

    deletion_candidates = glob.glob("F5-TTS/audio/full_*.wav") + \
                          glob.glob("final_video_output/*.mp4") + \
                          glob.glob("final_video_output/combined_workflow_video_*.mp4")

    all_files_to_delete = deletion_candidates
    for pattern in patterns_to_delete:
        all_files_to_delete.extend(glob.glob(pattern))

    for file_path in set(all_files_to_delete):
        if os.path.abspath(file_path) not in files_to_keep_abs:
            try:
                os.remove(file_path)
            except OSError:
                pass

    manim_media_dir = "media"
    if os.path.isdir(manim_media_dir):
        try:
            shutil.rmtree(manim_media_dir)
        except OSError as e:
            pass

def cleanup_files(file_paths: List[str]) -> None:
    if not file_paths:
        return
    
    for path in file_paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError as e:
                pass

def _check_manim_setup() -> bool:
    if not MANIMATOR_AVAILABLE:
        raise RuntimeError(
            "Manimator module not available. Please ensure manimator.py is in the same directory."
        )
    
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY environment variable not set. "
            "Please set your API key: export GEMINI_API_KEY='your_key_here'"
        )
    
    def _check_manim_version():
        result = subprocess.run(
            ["manim", "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10
        )
        return True
    
    try:
        retry_with_backoff(_check_manim_version)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            "Manim not found. Please install ManimCE: pip install manim"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Manim command timed out")
    
    if not MOVIEPY_AVAILABLE:
        pass
    
    return True

def generate_manim_video(scene_descriptions: List[str]) -> Optional[str]:
    """Generate video using Manimator from scene descriptions."""
    print(f"🎬 Starting video generation with {len(scene_descriptions)} scene descriptions")
    
    if not scene_descriptions:
        print("❌ No scene descriptions provided for video generation")
        return None
    
    if not MANIMATOR_AVAILABLE:
        print("❌ Manimator not available, cannot generate video")
        raise RuntimeError("Manimator module not available. Please ensure it's properly installed.")
    
    def _generate_with_retry():
        print("🔄 Attempting to generate video from scenes...")
        result = generate_video_from_scenes(scene_descriptions)
        if result:
            print(f"✅ Video generated successfully: {result}")
        else:
            print("❌ Video generation returned None")
        return result
    
    try:
        return retry_with_backoff(_generate_with_retry)
    except Exception as e:
        print(f"❌ Video generation failed: {e}")
        if not is_connection_error(e):
            return None
        else:
            raise

def generate_detailed_plan_from_subtitles(subtitles: List[dict], gemma_api_keys: List[str]) -> List[dict]:
    if not subtitles:
        return []

    subtitle_groups = []
    group_size = 5
    for i in range(0, len(subtitles), group_size):
        group = subtitles[i:i + group_size]
        if group:
            start_time = group[0].get('start', 0)
            end_time = group[-1].get('end', 0)
            combined_text = " ".join([seg.get('text', '').strip() for seg in group])
            
            subtitle_groups.append({
                'start_time': start_time,
                'end_time': end_time,
                'subtitles': group,
                'combined_text': combined_text
            })

    def generate_plan_for_group(group_data, segment_idx, api_key):
        try:
            client = genai.Client(api_key=api_key)
            
            prompt = f"""Create a brief visual plan for a video segment.

TIME RANGE: {group_data['start_time']:.2f}s - {group_data['end_time']:.2f}s
CONTENT: "{group_data['combined_text']}"

INSTRUCTIONS:
Provide a 2-3 sentence summary describing the key visual elements, diagrams, and animations
that would best represent this content for an educational video. Focus on high-level concepts.

Example: "This segment will introduce the Pythagorean theorem. A right-angled triangle will be drawn,
with its sides labeled 'a', 'b', and 'c'. The equation a^2 + b^2 = c^2 will appear and be highlighted."

Make sure that the total plan follows the exact time range provided.

YOUR SUMMARY:
"""
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[{"role": "user", "parts": [{"text": prompt}]}]
            )
            
            explanation = response.candidates[0].content.parts[0].text.strip()
            return {
                "start_time": group_data['start_time'],
                "end_time": group_data['end_time'],
                "brief_explanation": explanation,
                "segment_text": group_data['combined_text']
            }
        except Exception as e:
            return {
                "start_time": group_data['start_time'],
                "end_time": group_data['end_time'],
                "brief_explanation": f"Visual presentation for content: '{group_data['combined_text'][:100]}...'",
                "segment_text": group_data['combined_text']
            }

    # FIX: Use an ordered approach to collect results from concurrent tasks.
    plan_segments = [None] * len(subtitle_groups)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(gemma_api_keys))) as executor:
        future_to_index = {
            executor.submit(generate_plan_for_group, group, i + 1, gemma_api_keys[i % len(gemma_api_keys)]): i
            for i, group in enumerate(subtitle_groups)
        }
        
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                plan_segments[index] = result
            except Exception as exc:
                print(f'Segment {index} generated an exception: {exc}')
                plan_segments[index] = None

    # Filter out failed tasks and return the ordered list.
    return [p for p in plan_segments if p is not None]


def generate_scene_segment_descriptions(plan_segments: List[dict], gemma_api_keys: List[str]) -> List[str]:
    def generate_description_for_segment(segment_data, segment_idx, api_key):
        try:
            client = genai.Client(api_key=api_key)
            
            prompt = f"""You are an expert Manim scene designer. Generate the most COMPREHENSIVE, HYPER-DETAILED scene description possible for this educational video segment. Every coordinate, dimension, timing, and visual element must be precisely specified.

SEGMENT SPECIFICATIONS:
- Segment Number: {segment_idx}
- EXACT Time Range: {segment_data['start_time']:.3f}s to {segment_data['end_time']:.3f}s
- Total Duration: {segment_data['end_time'] - segment_data['start_time']:.3f}s
- Educational Goal: {segment_data['brief_explanation']}
- Spoken Content: "{segment_data['segment_text']}"

CRITICAL REQUIREMENTS FOR MAXIMUM DETAIL:

1. **UNIQUE NAMING SYSTEM:**
   - Every visual element MUST have a unique identifier
   - Format: [TYPE]_[PURPOSE]_[INDEX] (e.g., CIRCLE_DATAPOINT_001, TEXT_TITLE_MAIN, ARROW_CONNECTION_A2B)
   - NO duplicate names even for identical shapes

2. **PRECISE COORDINATE SYSTEM (Manim: x∈[-7.5,7.5], y∈[-4,4]):**
   - INITIAL_POSITION: [x.xxx, y.xxx] (3 decimal precision)
   - FINAL_POSITION: [x.xxx, y.xxx] (if moving)
   - ANCHOR_POINT: [x.xxx, y.xxx] (reference point for positioning relative elements)
   - RELATIVE_OFFSET: [dx.xxx, dy.xxx] (from anchor if inside containers)

3. **EXACT DIMENSIONS & PROPERTIES:**
   - Rectangles: WIDTH=X.xxx, HEIGHT=Y.xxx, CORNER_RADIUS=Z.xxx
   - Circles: RADIUS=R.xxx, STROKE_WIDTH=W.xxx
   - Text: FONT_SIZE=FS.xx, CHARACTER_WIDTH=CW.xx, LINE_HEIGHT=LH.xx
   - Arrows: LENGTH=L.xxx, TIP_LENGTH=TL.xx, SHAFT_WIDTH=SW.xxx

4. **MICROSECOND-PRECISE TIMING:**
   - APPEAR_AT: [XX.XXX]s (exact moment of first visibility)
   - ANIMATION_START: [XX.XXX]s (when motion/transformation begins)
   - ANIMATION_END: [XX.XXX]s (when motion/transformation completes)
   - DISAPPEAR_AT: [XX.XXX]s (exact moment element vanishes)
   - HIGHLIGHT_DURATION: [X.XXX]s (if element gets emphasized)

5. **DETAILED MOVEMENT SPECIFICATIONS:**
   - PATH_TYPE: [Linear/Curved/Arc/Spiral]
   - TRAJECTORY_POINTS: [(x1,y1), (x2,y2), ..., (xn,yn)]
   - MOVEMENT_SPEED: [units per second]
   - EASING_FUNCTION: [ease_in_out/linear/bounce/elastic]
   - ROTATION_ANGLE: [degrees] (if rotating during movement)

6. **NESTED ELEMENT POSITIONING:**
   - For elements inside containers, specify:
   - CONTAINER_NAME: [parent element identifier]
   - LOCAL_POSITION: [x.xxx, y.xxx] (relative to container center)
   - ALIGNMENT: [center/top_left/bottom_right/etc]
   - PADDING: [top, right, bottom, left] margins

7. **COLOR & STYLE SPECIFICATIONS:**
   - STROKE_COLOR: [#RRGGBB hex code]
   - FILL_COLOR: [#RRGGBB hex code]
   - OPACITY: [0.xxx] (transparency level)
   - Z_INDEX: [integer] (layering order)

8. **COMPREHENSIVE DIAGRAM CATALOG:**
   At the end, provide an exhaustive "DETAILED DIAGRAM SPECIFICATIONS" section listing:
   - Element Type & Unique Name
   - Exact Dimensions (width×height×depth if 3D)
   - Precise Coordinates (x,y,z if applicable)
   - All Properties (colors, fonts, styles)
   - Animation Parameters
   - Interaction Dependencies

EXAMPLE FORMAT:
ELEMENT: RECTANGLE_CONCEPT_MAIN_001
INITIAL_POSITION: [-3.250, 1.500]
DIMENSIONS: WIDTH=2.800, HEIGHT=1.200, CORNER_RADIUS=0.150
STROKE_COLOR: #58C4DD, FILL_COLOR: #FFFFFF, OPACITY: 0.850
APPEAR_AT: 0.500s
ANIMATION_IN: GrowFromCenter (duration=0.800s)
DISAPPEAR_AT: 4.200s
ANIMATION_OUT: ShrinkToCenter (duration=0.600s)

NESTED_ELEMENT: TEXT_LABEL_CONCEPT_001
CONTAINER: RECTANGLE_CONCEPT_MAIN_001
LOCAL_POSITION: [0.000, 0.000] (centered in rectangle)
FONT_SIZE: 24.0, COLOR: #2C3E50
CONTENT: "Core Concept"
APPEAR_AT: 1.300s (0.8s after container)


Generate the most detailed scene description possible using ALL available output tokens. Include every visual element, timing, coordinate, and specification needed for perfect Manim implementation."""
            
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                config=types.GenerateContentConfig(
                    temperature=0.9,
                    max_output_tokens=8192,  # Maximum tokens for detailed output
                    top_p=0.95,
                    top_k=40,
                    response_mime_type="text/plain"
                )
            )
            
            filename = f"scene_segment_{segment_idx}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(response.candidates[0].content.parts[0].text.strip())
            
            return filename
        except Exception as e:
            return None

    # FIX: Use an ordered approach to collect results. This is the critical fix.
    scene_files = [None] * len(plan_segments)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gemma_api_keys)) as executor:
        future_to_index = {
            executor.submit(generate_description_for_segment, segment, i + 1, gemma_api_keys[i % len(gemma_api_keys)]): i
            for i, segment in enumerate(plan_segments)
        }
        
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                filename = future.result()
                scene_files[index] = filename
            except Exception as exc:
                print(f'Scene description for segment {index} generated an exception: {exc}')
                scene_files[index] = None
    
    # Filter out failed tasks and return the final, ordered list of filenames.
    return [f for f in scene_files if f is not None]


def generate_drawings_for_segments(scene_segment_files: List[str]) -> None:
    all_diagrams_text = ""
    for i, scene_file in enumerate(scene_segment_files):
        try:
            with open(scene_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if "LIST OF DIAGRAMS:" in content:
                    diagrams_section = content.split("LIST OF DIAGRAMS:")[1].strip()
                    all_diagrams_text += f"\n--- Requirements from Segment {i+1} ---\n{diagrams_section}\n"
        except Exception as e:
            pass

    if not all_diagrams_text:
        if not os.path.exists("drawings.py"):
            with open("drawings.py", "w", encoding="utf-8") as f:
                f.write("from manim import *\n\n# Reusable drawing functions will be added here.\n")
        return

    prompt = f"""You are a Manim expert. Your task is to write new, reusable drawing functions.

**DIAGRAM REQUIREMENTS FROM VIDEO SEGMENTS:**
{all_diagrams_text}


**INSTRUCTIONS:**
1.  Analyze the **DIAGRAM REQUIREMENTS**.
2.  Create **new, reusable, and parameterized** Python functions for the required diagrams.
3.  Focus on general-purpose functions (e.g., `create_labeled_box`, `create_flow_arrow`) that can be used across multiple scenes.
4.  Each function should return a Manim `VGroup` or `Mobject`.
5.  **OUTPUT ONLY THE PYTHON CODE FOR THE NEW FUNCTIONS.** Do not include existing code or any explanations. Start directly with `def function_name(...):`.

Generate only the Python code for the new functions that should be added to `drawings.py`.
"""
    try:
        cmd = ["opencode", "run", "--mode", "build", "--model", "github-copilot/gpt-4.1", prompt]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=True)
        new_functions_code = result.stdout.strip()

        if new_functions_code:
            with open("drawings.py", "a", encoding="utf-8") as f:
                f.write("\n\n" + new_functions_code)
            
    except Exception as e:
        pass


def generate_and_render_manim_video(scene_segment_file: str, segment_idx: int) -> Optional[str]:
    print(f"🎬 Generating video for segment {segment_idx} from {scene_segment_file}")
    scene_code_file = f"scene_{segment_idx}.py"

    # Base prompt templates for different strategies
    base_prompts = [
        # Strategy 1: Standard approach
        f"""
TASK: Create and render a Manim video for segment {segment_idx}.
Make sure to follow the time frame and content provided correctly, the time should not be more or less, it must be exact.

FILES TO USE:
1. **Scene Description:** `{scene_segment_file}` (Contains the detailed visual plan).
2. **Drawing Library:** `drawings.py` (Contains reusable functions for creating shapes).
3. **Scene Rules:** `Scene_Rules.md` (Contains general animation rules).

STEP-BY-STEP INSTRUCTIONS:
1. **GENERATE CODE:** Read the three files above. Write a complete ManimCE Python script based on the detailed instructions in `{scene_segment_file}`. Save this script as `{scene_code_file}`. The script must import and use functions from `drawings.py`.
2. **EXECUTE CODE:** Run the command: `manim -ql {scene_code_file} YourSceneClassName`. You must determine the `YourSceneClassName` from the code you generated.
3. **VERIFY AND FIX:**
   - If the command runs successfully and creates an MP4 file, your job is done.
   - If the command produces ANY error, you must **FIX THE CODE** in `{scene_code_file}`.
   - The fix must still adhere strictly to the original requirements in `{scene_segment_file}`.
   - After fixing, go back to Step 2 and execute the command again.
4. **REPEAT:** Continue the 'execute -> verify -> fix' loop silently and autonomously until a video is generated successfully.
5. **OUTPUT:** Once successful, you MUST provide ONLY the absolute path to the final rendered MP4 file as your output and nothing else.

CRITICAL: Generate a COMPLETE, self-contained Manim scene that renders successfully from start to finish.
""",
        # Strategy 2: Simplified approach
        f"""
TASK: Create a SIMPLE but complete Manim video for segment {segment_idx}.
Focus on basic shapes and animations that are guaranteed to work.

FILES TO USE:
1. **Scene Description:** `{scene_segment_file}` (Contains the visual plan).
2. **Scene Rules:** `Scene_Rules.md` (Contains animation rules).

SIMPLIFIED INSTRUCTIONS:
1. Create a basic ManimCE script saved as `{scene_code_file}`
2. Use only simple Manim elements: Text, Circle, Rectangle, Arrow, Line
3. Use basic animations: Create, FadeIn, FadeOut, Transform
4. Avoid complex positioning - use simple coordinates
5. Execute: `manim -ql {scene_code_file} YourSceneClassName`
6. Fix any errors and retry until successful
7. Output ONLY the path to the final MP4 file

PRIORITY: Success over complexity. Create a working video first.
""",
        # Strategy 3: Minimal approach
        f"""
TASK: Create the MOST MINIMAL working Manim video for segment {segment_idx}.

ULTRA-SIMPLE APPROACH:
1. Create `{scene_code_file}` with minimal Manim scene
2. Include basic import: from manim import *
3. Create simple Scene class with construct method
4. Add 1-2 basic text elements or shapes
5. Use simple self.play(Create(...)) animations
6. Execute: `manim -ql {scene_code_file} SceneName`
7. Output the MP4 path when successful

GOAL: Get ANY working video output, no matter how simple.
"""
    ]
    
    attempt = 0
    max_timeout = 600  # 10 minutes per attempt
    
    while True:  # UNLIMITED RETRIES - NO GIVING UP
        # Cycle through strategies
        strategy_idx = attempt % len(base_prompts)
        base_prompt = base_prompts[strategy_idx]
        
        attempt += 1
        print(f"🔄 UNLIMITED Attempt {attempt} (Strategy {strategy_idx + 1}) for segment {segment_idx}...")
        
        # Modify prompt based on attempt number
        prompt = base_prompt
        if attempt > 3:
            prompt += f"\n\nPREVIOUS {attempt-1} ATTEMPTS FAILED. You MUST try a completely different approach. Simplify the code structure. Attempt #{attempt}."
        if attempt > 10:
            prompt += f"\n\nCRITICAL: This is attempt #{attempt}. You must generate the most basic possible working Manim scene that compiles and renders successfully."
        
        try:
            print(f"🔧 Running opencode for scene {segment_idx} (attempt {attempt})...")
            cmd = ["opencode", "run", "--mode", "build", "--model", "github-copilot/gpt-4.1", prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=max_timeout)

            if result.returncode == 0:
                output_lines = result.stdout.strip().split('\n')
                video_path = output_lines[-1].strip()
                if os.path.exists(video_path) and video_path.endswith(".mp4"):
                    # Verify the video is complete using manimator's validation
                    from pathlib import Path
                    from manimator import is_complete_video
                    if is_complete_video(Path(video_path)):
                        print(f"✅ Successfully generated complete video (attempt {attempt}): {video_path}")
                        return video_path
                    else:
                        print(f"⚠️ Generated video is incomplete (attempt {attempt}), retrying...")
                        # Continue to next attempt
                else:
                    print(f"❌ No valid video file found (attempt {attempt}): {video_path}")
                    # Continue to next attempt
            else:
                print(f"❌ OpenCode failed with return code {result.returncode} (attempt {attempt})")
                if result.stderr:
                    error_preview = result.stderr[:300] + "..." if len(result.stderr) > 300 else result.stderr
                    print(f"stderr preview: {error_preview}")
                # Continue to next attempt
                
        except subprocess.TimeoutExpired:
            print(f"⏰ OpenCode timed out (attempt {attempt})")
            # Continue to next attempt
        except FileNotFoundError:
            print(f"❌ OpenCode not found - cannot generate video")
            return None  # This is a permanent failure
        except Exception as e:
            error_msg = str(e)[:200] + "..." if len(str(e)) > 200 else str(e)
            print(f"❌ Error in opencode (attempt {attempt}): {error_msg}")
            # Continue to next attempt
        
        # Progressive delay - longer delays for more attempts
        if attempt <= 5:
            delay = 3
        elif attempt <= 10:
            delay = 5
        else:
            delay = 10
            
        print(f"⏱️ Waiting {delay}s before next attempt...")
        time.sleep(delay)


def new_video_generation_workflow(subtitles: List[dict], gemma_api_keys: List[str]) -> List[str]:
    print(f"🎬 Starting NEW video generation workflow with {len(subtitles)} subtitle segments")
    
    # Generate plan segments with unlimited retries
    print("📋 Generating detailed plan from subtitles...")
    plan_segments = None
    plan_attempt = 0
    while not plan_segments:
        plan_attempt += 1
        print(f"🔄 UNLIMITED Plan generation attempt {plan_attempt}...")
        try:
            plan_segments = generate_detailed_plan_from_subtitles(subtitles, gemma_api_keys)
            if plan_segments and len(plan_segments) > 0:
                print(f"✅ Generated {len(plan_segments)} plan segments on attempt {plan_attempt}")
                break
            else:
                print(f"❌ Plan generation attempt {plan_attempt} produced no segments, retrying...")
                plan_segments = None
        except Exception as e:
            print(f"❌ Plan generation attempt {plan_attempt} failed: {e}, retrying...")
            plan_segments = None
        
        time.sleep(3)  # Brief delay between retries

    # Generate scene segment descriptions with unlimited retries
    print("📝 Generating scene segment descriptions...")
    scene_segment_files = None
    scene_attempt = 0
    while not scene_segment_files:
        scene_attempt += 1
        print(f"🔄 UNLIMITED Scene description generation attempt {scene_attempt}...")
        try:
            scene_segment_files = generate_scene_segment_descriptions(plan_segments, gemma_api_keys)
            if scene_segment_files and len(scene_segment_files) > 0:
                print(f"✅ Generated {len(scene_segment_files)} scene segment files on attempt {scene_attempt}")
                break
            else:
                print(f"❌ Scene description generation attempt {scene_attempt} produced no files, retrying...")
                scene_segment_files = None
        except Exception as e:
            print(f"❌ Scene description generation attempt {scene_attempt} failed: {e}, retrying...")
            scene_segment_files = None
        
        time.sleep(3)  # Brief delay between retries
    
    print("🎨 Generating drawings for segments...")
    generate_drawings_for_segments(scene_segment_files)
    
    print(f"🎞️ Rendering {len(scene_segment_files)} video segments with unlimited retries...")
    max_render_workers = 4
    rendered_videos = [None] * len(scene_segment_files)
    
    # First attempt with parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_render_workers) as executor:
        future_to_idx = {executor.submit(generate_and_render_manim_video, scene_file, i + 1): i 
                         for i, scene_file in enumerate(scene_segment_files)}
        
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                video_path = future.result()
                if video_path:
                    rendered_videos[idx] = video_path
                    print(f"✅ Video {idx+1} rendered successfully")
                else:
                    print(f"❌ Video {idx+1} failed to render")
            except Exception as e:
                print(f"❌ Video {idx+1} rendering error: {e}")

    # Unlimited retry loop for failed segments - NO FALLBACKS
    retry_cycle = 0
    
    while True:
        # Identify missing segments
        missing_segments = []
        for i, video_path in enumerate(rendered_videos):
            if not video_path or not os.path.exists(video_path):
                missing_segments.append((scene_segment_files[i], i + 1))
        
        if not missing_segments:
            break  # All videos generated successfully
            
        retry_cycle += 1
        print(f"🔄 UNLIMITED Retry cycle {retry_cycle}: {len(missing_segments)} failed segments...")
        print(f"🎯 Missing segments: {[idx for _, idx in missing_segments]}")
        
        # Retry missing segments one by one for better success rate
        for scene_file, idx in missing_segments:
            print(f"🔧 Retrying segment {idx} until success...")
            video_path = generate_and_render_manim_video(scene_file, idx)
            if video_path and os.path.exists(video_path):
                rendered_videos[idx-1] = video_path
                print(f"✅ Segment {idx} succeeded in retry cycle {retry_cycle}!")
            else:
                print(f"❌ Segment {idx} still failed, will retry in next cycle...")
        
        # Brief pause between retry cycles
        print(f"⏱️ Pausing 10 seconds before next retry cycle...")
        time.sleep(10)
    
    # Final success report
    successful_videos = [v for v in rendered_videos if v and os.path.exists(v)]
    print(f"🎯 PERFECT SUCCESS: All {len(successful_videos)} videos generated successfully after {retry_cycle} retry cycles!")

    # Collect all successful videos
    final_video_paths = []
    for i, video_path in enumerate(rendered_videos):
        if video_path and os.path.exists(video_path):
            final_video_paths.append(video_path)

    final_video_paths.sort()
    print(f"🎯 Final result: {len(final_video_paths)} videos generated successfully")
    return final_video_paths

def get_system_info() -> dict:
    info = {
        "gemini_api_available": is_gemini_available(),
        "manimator_available": MANIMATOR_AVAILABLE,
        "moviepy_available": MOVIEPY_AVAILABLE,
        "whisper_available": WHISPER_AVAILABLE,
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
    info = get_system_info()
    
    print(f"Gemini API Available: {'✅' if info['gemini_api_available'] else '❌'}")
    print(f"Manimator Available: {'✅' if info['manimator_available'] else '❌'}")
    print(f"MoviePy Available: {'✅' if info['moviepy_available'] else '❌'}")
    print(f"Whisper Available: {'✅' if info['whisper_available'] else '❌'}")
    print(f"Voice Samples: {info['voice_samples_count']} found")
    print(f"Working Directory: {info['working_directory']}")
    for key, value in info['environment_variables'].items():
        print(f"  {key}: {value}")

def initialize_module() -> bool:
    def _initialize_with_retry():
        setup_gpu_environment()
        
        directories = [
            Path.cwd() / "F5-TTS" / "voice_samples",
            Path.cwd() / "F5-TTS" / "texts",
            Path.cwd() / "F5-TTS" / "audio",
            FINAL_VIDEO_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        try:
            import torch
            if torch.cuda.is_available():
                pass
            else:
                pass
        except ImportError:
            pass
        
        if is_gemini_available():
            pass
        else:
            return False
        
        if WHISPER_AVAILABLE:
            pass
        else:
            pass
        
        return True
    
    try:
        return retry_with_backoff(_initialize_with_retry)
    except Exception as e:
        if not is_connection_error(e):
            return False
        else:
            raise

if __name__ != "__main__":
    success = initialize_module()

if __name__ == "__main__":
    print_system_info()
    
    if initialize_module():
        pass
    else:
        pass

def cleanup_generated_files_except_final(preserve_final_video: Optional[str] = None, preserve_audio: Optional[str] = None) -> None:
    preserve_patterns = [
        "scene.py",
        "drawings.py", 
        "Scene_Rules.md",
        "requirements.txt",
        "app.py",
        "utils.py",
        "manimator.py"
    ]
    
    if preserve_final_video:
        preserve_patterns.append(os.path.basename(preserve_final_video))
        preserve_patterns.append(preserve_final_video)
    
    if preserve_audio:
        preserve_patterns.append(os.path.basename(preserve_audio))
        preserve_patterns.append(preserve_audio)
    
    cleanup_patterns = [
        "scene_[0-9]*.py",
        "temp_scene_*.py",
        "media/videos/**/scene_[0-9]*.mp4",
        "media/videos/**/Scene[0-9]*.mp4",
        "media/partial_movie_files/",
        "media/Tex/",
        "media/text/", 
        ".manim_cache/",
        "__pycache__/",
        "*.pyc",
        "chunk_*_detailed_scenes_*.txt",
        "scene_segment_*.txt", 
        "plan_segments_*.json",
        "temp_*.srt",
        "*.log",
        "temp_scene_*/",
        "outputs/temp/",
    ]
    
    for pattern in cleanup_patterns:
        for file_path in Path(".").rglob(pattern):
            should_preserve = False
            for preserve_pattern in preserve_patterns:
                if (preserve_pattern in str(file_path) or 
                    file_path.name == preserve_pattern or
                    str(file_path) == preserve_pattern):
                    should_preserve = True
                    break
            
            if should_preserve:
                continue
            
            try:
                if file_path.is_dir():
                    shutil.rmtree(file_path)
                else:
                    file_path.unlink()
            except Exception as e:
                pass

def auto_cleanup_after_video_generation(final_video_path: Optional[str] = None, audio_path: Optional[str] = None) -> None:
    cleanup_generated_files_except_final(
        preserve_final_video=final_video_path,
        preserve_audio=audio_path
    )