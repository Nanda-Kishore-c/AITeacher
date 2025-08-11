#!/usr/bin/env python3
"""
Manimator - AI-Powered Manim Video Generator
================================================================
Advanced Manim video generation system integrated with PDF Seamless Teacher.
Generates educational animations using Gemini API and automatic error correction.

Features:
- AI-powered Manim code generation with Gemma 3
- Automatic error detection and fixing with OpenCode
- Multi-scene video rendering and concatenation
- Robust retry mechanisms and fallback handling
- Interactive chat mode for standalone usage
================================================================
"""

# ╭──────────────────────────────────────────────────────────────╮
# │                        IMPORTS SECTION                       │
# ╰──────────────────────────────────────────────────────────────╯

import json
import os
import re
import shlex
import subprocess
import tempfile
import time
import uuid
import shutil
import socket
import errno
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

# Google GenAI imports
import google.genai as genai
from google.genai import types

# Video processing (with fallback handling)
try:
    from moviepy.editor import VideoFileClip, concatenate_videoclips
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("⚠️ Warning: MoviePy not available. Video concatenation will be disabled.")

# ╭──────────────────────────────────────────────────────────────╮
# │                    CONFIGURATION CONSTANTS                   │
# ╰──────────────────────────────────────────────────────────────╯

# Gemini API Configuration
MODEL_NAME = os.getenv("GENAI_MODEL", "gemma-3-27b-it")
GEMINI_TIMEOUT = 60  # seconds
GEMINI_MAX_RETRIES = 3

# Manim Configuration
MAX_ATTEMPTS = 5
MANIM_TIMEOUT = int(os.getenv("MANIM_TIMEOUT", "120"))  # seconds
MANIM_QUALITY = "low_quality"  # Options: low_quality, medium_quality, high_quality
MANIM_FPS = 24

# OpenCode Configuration
OPENCODE_TIMEOUT = 300  # seconds
OPENCODE_MODEL = "github-copilot/gpt-4.1"

# Directory Configuration
FINAL_VIDEO_DIR = Path("final_video_output")
TEMP_SCENE_DIR_PREFIX = "temp_scene_"
MEDIA_DIR_NAME = "media"

# Video Processing Configuration
VIDEO_CODEC = "libx264"
AUDIO_CODEC = "aac"
VIDEO_BITRATE = "1000k"

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

def _clean_code(code: str) -> str:
    """
    Strip accidental markdown code fences and normalize formatting.
    
    Args:
        code: Raw code string from AI generation
        
    Returns:
        Cleaned code string
    """
    # Remove markdown code fences
    code = re.sub(r"^```(?:python)?\s*", "", code.strip(), flags=re.MULTILINE)
    code = re.sub(r"\s*```$", "", code.strip(), flags=re.MULTILINE)
    
    # Normalize whitespace
    code = re.sub(r'\n\s*\n\s*\n', '\n\n', code)  # Remove excessive blank lines
    code = code.strip()
    
    return code

def _validate_manim_code(code: str) -> Tuple[bool, str, str]:
    """
    Validate generated Manim code for basic requirements.
    
    Args:
        code: Generated Manim code
        
    Returns:
        Tuple of (is_valid, class_name, error_message)
    """
    if not code or not code.strip():
        return False, "", "Empty code generated"
    
    # Check for required imports
    required_imports = ["from manim import", "import manim"]
    has_import = any(imp in code for imp in required_imports)
    if not has_import:
        return False, "", "Missing Manim import statement"
    
    # Extract class name
    class_match = re.search(r"class\s+(\w+)\s*\(.*Scene.*\)", code)
    if not class_match:
        return False, "", "No valid Scene class found"
    
    class_name = class_match.group(1)
    
    # Check for construct method
    if "def construct(self)" not in code:
        return False, class_name, "Missing construct method"
    
    return True, class_name, ""

# ╭──────────────────────────────────────────────────────────────╮
# │                   MANIM CODE GENERATION                      │
# ╰──────────────────────────────────────────────────────────────╯

def generate_manim_code(client: genai.Client, scene_description: str) -> Tuple[str, str]:
    """
    Generate Manim code from scene description using Gemini API.
    
    Args:
        client: Gemini API client
        scene_description: Description of the scene to create
        
    Returns:
        Tuple of (generated_code, class_name)
        
    Raises:
        Exception: If code generation fails after retries
    """
    def _generate_code_with_retry():
        instruction = (
            "You are an expert ManimCE animator and Python developer.\n"
            "Create a complete, working Manim scene that visualizes the educational content described below.\n\n"
            
            "REQUIREMENTS:\n"
            "- Import manim with: from manim import *\n"
            "- Create a class that inherits from Scene\n"
            "- Implement the construct(self) method\n"
            "- Use clear, educational animations\n"
            "- Include text, mathematical formulas, and visual elements\n"
            "- Use appropriate colors and positioning\n"
            "- Add smooth animations and transitions\n"
            "- Ensure code is syntactically correct\n"
            "- Return ONLY Python code, no markdown or explanations\n\n"
            
            f"SCENE DESCRIPTION:\n{scene_description}\n\n"
            
            "Generate complete, executable ManimCE code:"
        )
        
        contents = [types.Content(role="user", parts=[types.Part(text=instruction)])]
        cfg = types.GenerateContentConfig(
            response_mime_type="text/plain",
            temperature=0.7,
            max_output_tokens=1500
        )

        print("🧠 Generating Manim code from scene description...")
        print(f"📝 Scene: {scene_description[:100]}...")

        attempt = 0
        current_instruction = instruction

        while attempt < GEMINI_MAX_RETRIES:
            attempt += 1
            print(f"🔄 Generation attempt {attempt}/{GEMINI_MAX_RETRIES}")
            
            generated = ""
            for chunk in client.models.generate_content_stream(
                model=MODEL_NAME, contents=contents, config=cfg
            ):
                if chunk.text:
                    generated += chunk.text
            
            # Clean and validate the generated code
            cleaned_code = _clean_code(generated)
            is_valid, class_name, error_msg = _validate_manim_code(cleaned_code)
            
            if is_valid:
                print(f"✅ Code generation successful! Class: {class_name}")
                return cleaned_code, class_name
            else:
                print(f"⚠️ Generated code validation failed: {error_msg}")
                if attempt < GEMINI_MAX_RETRIES:
                    print("🔄 Retrying with refined prompt...")
                    # Add validation feedback to the prompt for next attempt
                    current_instruction += f"\n\nPREVIOUS ATTEMPT FAILED: {error_msg}\nPlease fix this issue."
                    contents = [types.Content(role="user", parts=[types.Part(text=current_instruction)])]
        
        raise Exception(f"Failed to generate valid Manim code after {GEMINI_MAX_RETRIES} attempts")
    
    return retry_with_backoff(_generate_code_with_retry)

# ╭──────────────────────────────────────────────────────────────╮
# │                     MANIM EXECUTION                          │
# ╰──────────────────────────────────────────────────────────────╯

def run_manim_render(scene_path: Path, class_name: str, output_dir: Path) -> subprocess.CompletedProcess:
    """
    Execute Manim rendering command.
    
    Args:
        scene_path: Path to the Python scene file
        class_name: Name of the Scene class to render
        output_dir: Directory for output files
        
    Returns:
        Completed subprocess result
    """
    def _run_manim_with_retry():
        # Use just the filename since we're in the correct directory
        scene_filename = scene_path.name
        
        # Construct Manim command with modern flag format
        cmd = [
            "manim",
            "-pql",  # preview, quality low
            scene_filename,
            class_name
        ]
        
        print(f"🚀 Executing Manim: {' '.join(shlex.quote(c) for c in cmd)}")
        print(f"📁 Working directory: {output_dir}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=MANIM_TIMEOUT,
            cwd=output_dir
        )
        
        if result.returncode == 0:
            print("✅ Manim rendering completed successfully")
        else:
            print(f"❌ Manim rendering failed with return code {result.returncode}")
        
        return result
    
    try:
        return retry_with_backoff(_run_manim_with_retry)
    except subprocess.TimeoutExpired:
        print(f"⏰ Manim rendering timed out after {MANIM_TIMEOUT} seconds")
        raise
    except Exception as e:
        if not is_connection_error(e):
            raise
        else:
            raise
def find_rendered_video(output_dir: Path, class_name: str) -> Optional[Path]:
    """
    Find the rendered video file in the Manim output directory.
    
    Args:
        output_dir: Directory where Manim output is stored
        class_name: Name of the rendered scene class
        
    Returns:
        Path to the video file if found, None otherwise
    """
    # Common Manim output paths
    possible_paths = [
        output_dir / "media" / "videos" / "scene" / f"{MANIM_QUALITY}" / f"{class_name}.mp4",
        output_dir / "media" / "videos" / f"{class_name}.mp4",
        output_dir / f"{class_name}.mp4"
    ]
    
    # Search for video files
    for path in possible_paths:
        if path.exists():
            print(f"✅ Found rendered video: {path}")
            return path
    
    # Fallback: search for any mp4 file in media directory
    media_dir = output_dir / "media"
    if media_dir.exists():
        for video_file in media_dir.rglob("*.mp4"):
            if class_name in video_file.name or video_file.stem == class_name:
                print(f"✅ Found video by search: {video_file}")
                return video_file
        
        # Last resort: return first mp4 found
        for video_file in media_dir.rglob("*.mp4"):
            print(f"⚠️ Using fallback video: {video_file}")
            return video_file
    
    print(f"❌ No video file found for class {class_name}")
    return None

# ╭──────────────────────────────────────────────────────────────╮
# │                   OPENCODE ERROR FIXING                      │
# ╰──────────────────────────────────────────────────────────────╯

def opencode_fix_manim_code(py_file: Path, error_output: str = "", scene_idx: int = 1) -> Optional[str]:
    """
    Use OpenCode to automatically fix Manim code errors.
    
    Args:
        py_file: Path to the Python file with errors
        error_output: Error message from Manim rendering
        scene_idx: Scene index for error file naming
        
    Returns:
        Fixed code if successful, None otherwise
    """
    def _opencode_fix_with_retry():
        print(f"🤖 Running OpenCode to fix {py_file.name}...")
        
        # Create error file in the same directory as the Python file
        error_file = py_file.parent / f"error_{scene_idx}.txt"
        
        # Write error output to file
        try:
            error_file.write_text(error_output, encoding='utf-8')
            print(f"📄 Error output saved to: {error_file}")
        except Exception as e:
            print(f"⚠️ Failed to write error file: {e}")
            return None
        
        # Create single-line prompt with file references
        fix_prompt = f"Fix the ManimCE Python file at {py_file.resolve()} so it runs without errors. Use the error output from {error_file.resolve()} to identify and correct issues. Return only the corrected Python code without explanations or markdown formatting. Ensure all syntax is valid ManimCE code and maintain the educational content and animations."

        cmd = [
            "opencode",
            "run",
            "--model", OPENCODE_MODEL,
            fix_prompt
        ]

        print(f"🤖 OpenCode command: {' '.join(shlex.quote(c) for c in cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=OPENCODE_TIMEOUT,
            check=False
        )

        if result.returncode != 0:
            print(f"❌ OpenCode failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return None

        print("✅ OpenCode completed successfully")

        # Extract and clean the fixed code
        fixed_code = _clean_code(result.stdout.strip())
        
        if not fixed_code:
            print("⚠️ OpenCode returned empty response")
            return None
        
        # Validate the fixed code
        is_valid, class_name, error_msg = _validate_manim_code(fixed_code)
        if not is_valid:
            print(f"⚠️ OpenCode fix validation failed: {error_msg}")
            return None
        
        # Compare with original code
        original_code = py_file.read_text()
        if fixed_code != original_code:
            print("✅ OpenCode provided code improvements")
            py_file.write_text(fixed_code)
            return fixed_code
        else:
            print("⚠️ OpenCode returned identical code")
            return None

    try:
        return retry_with_backoff(_opencode_fix_with_retry)
    except subprocess.TimeoutExpired:
        print(f"🚨 OpenCode timed out after {OPENCODE_TIMEOUT} seconds")
        return None
    except Exception as e:
        if not is_connection_error(e):
            print(f"🚨 OpenCode error: {e}")
            return None
        else:
            # This should not happen as retry_with_backoff handles connection errors
            raise

# ╭──────────────────────────────────────────────────────────────╮
# │                   SINGLE SCENE RENDERING                     │
# ╰──────────────────────────────────────────────────────────────╯

def render_single_scene(code: str, class_name: str, scene_idx: int) -> Optional[Path]:
    """
    Render a single Manim scene with automatic error fixing.
    
    Args:
        code: Manim Python code
        class_name: Name of the Scene class
        scene_idx: Scene index for directory naming
        
    Returns:
        Path to rendered video file if successful, None otherwise
    """
    # Create temporary directory for this scene
    temp_dir = Path(f"{TEMP_SCENE_DIR_PREFIX}{scene_idx}")
    temp_dir.mkdir(exist_ok=True)
    
    print(f"🎬 Rendering Scene {scene_idx} (Class: {class_name})")
    
    attempt = 0
    current_code = code
    
    while attempt < MAX_ATTEMPTS:
        attempt += 1
        print(f"🔄 Scene {scene_idx} - Rendering attempt {attempt}/{MAX_ATTEMPTS}")

        # Save current code to file
        scene_file = temp_dir / f"scene_{scene_idx}.py"
        scene_file.write_text(current_code)
        print(f"📄 Saved code to: {scene_file}")
        
        try:
            # Attempt to render with Manim
            result = run_manim_render(scene_file, class_name, temp_dir)

            if result.returncode == 0:
                # Success! Find the rendered video
                video_path = find_rendered_video(temp_dir, class_name)
                if video_path:
                    print(f"✅ Scene {scene_idx} rendered successfully: {video_path}")
                    return video_path
                else:
                    print(f"⚠️ Scene {scene_idx} rendered but video file not found")
                    return None

            # Rendering failed - analyze error and attempt fix
            print(f"❌ Scene {scene_idx} rendering failed")
            print(f"📋 Error output (first 500 chars): {result.stderr[:500]}")

            # Try to fix the code with OpenCode
            if attempt < MAX_ATTEMPTS:
                print(f"🔧 Attempting to fix Scene {scene_idx} with OpenCode...")
                # Pass the scene_idx to the OpenCode function
                fixed_code = opencode_fix_manim_code(scene_file, result.stderr, scene_idx)
                
                if fixed_code:
                    current_code = fixed_code
                    print(f"✅ Scene {scene_idx} code fixed, retrying...")
                    continue
                else:
                    print(f"❌ Scene {scene_idx} - OpenCode could not fix the errors")
                    break
            else:
                print(f"❌ Scene {scene_idx} - Maximum attempts reached")
                break
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Scene {scene_idx} rendering timed out")
            break
        except Exception as e:
            print(f"❌ Scene {scene_idx} unexpected error: {e}")
            break

    print(f"❌ Scene {scene_idx} failed after {attempt} attempts")
    return None

# ╭──────────────────────────────────────────────────────────────╮
# │                   VIDEO CONCATENATION                        │
# ╰──────────────────────────────────────────────────────────────╯

def concatenate_scene_videos(video_paths: List[Path], output_path: Path) -> bool:
    """
    Concatenate multiple scene videos into a single final video.
    
    Args:
        video_paths: List of paths to individual scene videos
        output_path: Path for the final concatenated video
        
    Returns:
        True if successful, False otherwise
    """
    def _concatenate_with_retry():
        if not MOVIEPY_AVAILABLE:
            print("❌ MoviePy not available for video concatenation")
            return False
        
        if not video_paths:
            print("❌ No video files provided for concatenation")
            return False
        
        print(f"🎬 Concatenating {len(video_paths)} scene videos...")
        
        # Load video clips
        clips = []
        for i, path in enumerate(video_paths):
            print(f"📹 Loading clip {i + 1}: {path}")
            clip = VideoFileClip(str(path))
            clips.append(clip)
        
        # Concatenate clips
        print("🔗 Concatenating video clips...")
        final_clip = concatenate_videoclips(clips, method="compose")
        
        # Write final video
        print(f"💾 Writing final video to: {output_path}")
        final_clip.write_videofile(
            str(output_path),
            codec=VIDEO_CODEC,
            audio_codec=AUDIO_CODEC,
            fps=MANIM_FPS,
            bitrate=VIDEO_BITRATE,
            verbose=False,
            logger=None
        )
        
        # Clean up clips
        for clip in clips:
            clip.close()
        final_clip.close()
        
        # Verify output file
        if output_path.exists() and output_path.stat().st_size > 0:
            duration = sum(clip.duration for clip in clips)
            print(f"✅ Final video created successfully!")
            print(f"📊 Duration: {duration:.2f}s, Size: {output_path.stat().st_size / (1024*1024):.2f}MB")
            return True
        else:
            print("❌ Final video file was not created properly")
            return False
    
    try:
        return retry_with_backoff(_concatenate_with_retry)
    except Exception as e:
        if not is_connection_error(e):
            print(f"❌ Video concatenation failed: {e}")
            return False
        else:
            # This should not happen as retry_with_backoff handles connection errors
            raise

# ╭──────────────────────────────────────────────────────────────╮
# │                   CLEANUP UTILITIES                          │
# ╰──────────────────────────────────────────────────────────────╯

def cleanup_temp_directories(num_scenes: int) -> None:
    """
    Clean up temporary scene directories after processing.
    
    Args:
        num_scenes: Number of scenes to clean up
    """
    print("🧹 Cleaning up temporary directories...")
    
    cleaned_count = 0
    for i in range(1, num_scenes + 1):
        temp_dir = Path(f"{TEMP_SCENE_DIR_PREFIX}{i}")
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                cleaned_count += 1
                print(f"🗑️ Removed: {temp_dir}")
            except Exception as e:
                print(f"⚠️ Failed to remove {temp_dir}: {e}")
    
    if cleaned_count > 0:
        print(f"✅ Cleaned up {cleaned_count} temporary directories")
    else:
        print("ℹ️ No temporary directories to clean up")

# ╭──────────────────────────────────────────────────────────────╮
# │                   MAIN VIDEO GENERATION                      │
# ╰──────────────────────────────────────────────────────────────╯

def generate_video_from_scenes(scene_descriptions: List[str]) -> Optional[str]:
    """
    Main function to generate a complete video from scene descriptions.
    This is the primary interface for integration with the PDF Seamless Teacher.
    
    Args:
        scene_descriptions: List of scene descriptions for video generation
        
    Returns:
        Path to the final video file if successful, None otherwise
    """
    def _generate_video_with_retry():
        if not scene_descriptions:
            print("❌ No scene descriptions provided")
            return None
        
        print("🎬 Starting AI-powered Manim video generation...")
        print(f"📋 Processing {len(scene_descriptions)} scenes")
        
        # Initialize Gemini client
        client = get_gemini_client()
        print("✅ Gemini API client initialized")
        
        # Create output directory
        FINAL_VIDEO_DIR.mkdir(exist_ok=True)
        
        # Process each scene
        rendered_videos: List[Path] = []
        
        for i, scene_desc in enumerate(scene_descriptions):
            scene_num = i + 1
            print(f"\n{'='*60}")
            print(f"🎭 PROCESSING SCENE {scene_num}/{len(scene_descriptions)}")
            print(f"{'='*60}")
            print(f"📝 Description: {scene_desc[:150]}...")
            
            try:
                # Generate Manim code
                print(f"🧠 Generating code for Scene {scene_num}...")
                code, class_name = generate_manim_code(client, scene_desc)
                print(f"✅ Code generated for class: {class_name}")
                
                # Render the scene
                video_path = render_single_scene(code, class_name, scene_num)
                
                if video_path:
                    rendered_videos.append(video_path)
                    print(f"✅ Scene {scene_num} completed successfully")
                else:
                    print(f"❌ Scene {scene_num} failed to render")
                    
            except Exception as e:
                print(f"❌ Scene {scene_num} processing failed: {e}")
                continue
        
        # Check if we have any successful renders
        if not rendered_videos:
            print("❌ No scenes were rendered successfully")
            cleanup_temp_directories(len(scene_descriptions))
            return None
        
        print(f"\n🎬 Successfully rendered {len(rendered_videos)}/{len(scene_descriptions)} scenes")
        
        # Concatenate videos if we have multiple scenes
        final_video_path = FINAL_VIDEO_DIR / f"educational_video_{int(time.time())}.mp4"
        
        if len(rendered_videos) == 1:
            # Single video - just copy it
            shutil.copy2(rendered_videos[0], final_video_path)
            print(f"✅ Single scene video saved to: {final_video_path}")
        else:
            # Multiple videos - concatenate them
            if not concatenate_scene_videos(rendered_videos, final_video_path):
                print("❌ Video concatenation failed")
                cleanup_temp_directories(len(scene_descriptions))
                return None
        
        # Cleanup temporary files
        cleanup_temp_directories(len(scene_descriptions))
        
        # Final verification
        if final_video_path.exists() and final_video_path.stat().st_size > 0:
            print(f"\n🎉 VIDEO GENERATION COMPLETED SUCCESSFULLY!")
            print(f"📹 Final video: {final_video_path}")
            print(f"📊 File size: {final_video_path.stat().st_size / (1024*1024):.2f}MB")
            return str(final_video_path)
        else:
            print("❌ Final video verification failed")
            return None
    
    try:
        return retry_with_backoff(_generate_video_with_retry)
    except ValueError as e:
        print(f"❌ Gemini API setup failed: {e}")
        return None
    except Exception as e:
        if not is_connection_error(e):
            print(f"❌ Video generation failed: {e}")
            return None
        else:
            # This should not happen as retry_with_backoff handles connection errors
            raise

# ╭──────────────────────────────────────────────────────────────╮
# │                   INTERACTIVE CHAT MODE                      │
# ╰──────────────────────────────────────────────────────────────╯

def run_interactive_chat_session(client: genai.Client) -> None:
    """
    Interactive chat session for standalone Manim code generation and testing.
    
    Args:
        client: Initialized Gemini API client
    """
    print("=" * 70)
    print("🎨 MANIMATOR - AI-POWERED MANIM CODE GENERATOR")
    print("=" * 70)
    print(f"⏰ Session started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🤖 Using model: {MODEL_NAME}")
    print("\n📋 INSTRUCTIONS:")
    print("• Describe the animation you want to create")
    print("• Type 'END' on a new line to generate and render")
    print("• Type 'status' for session statistics")
    print("• Type 'info' for system information")
    print("• Type 'exit' to quit")
    print("-" * 70)

    # Session statistics
    stats = {
        'processed': 0,
        'successful': 0,
        'failed': 0,
        'total_time': 0.0,
        'start_time': time.time()
    }
    
    buffer: List[str] = []

    while True:
        try:
            line = input(">>> ")
            command = line.strip().lower()
            
            if command == "exit":
                session_duration = time.time() - stats['start_time']
                print(f"\n👋 SESSION SUMMARY")
                print(f"📊 Prompts processed: {stats['processed']}")
                print(f"✅ Successful renders: {stats['successful']}")
                print(f"❌ Failed attempts: {stats['failed']}")
                print(f"⏱️ Total processing time: {stats['total_time']:.2f}s")
                print(f"🕐 Session duration: {session_duration:.2f}s")
                print("Goodbye! 👋")
                return
                
            elif command == "status":
                success_rate = (stats['successful'] / max(stats['processed'], 1)) * 100
                print(f"\n📊 SESSION STATUS")
                print(f"✅ Success rate: {success_rate:.1f}% ({stats['successful']}/{stats['processed']})")
                print(f"⏱️ Average processing time: {stats['total_time']/max(stats['processed'], 1):.2f}s")
                continue
                
            elif command == "info":
                print(f"\n🔧 SYSTEM INFORMATION")
                print(f"🤖 Model: {MODEL_NAME}")
                print(f"🎬 Manim timeout: {MANIM_TIMEOUT}s")
                print(f"🔧 OpenCode model: {OPENCODE_MODEL}")
                print(f"📁 Output directory: {FINAL_VIDEO_DIR}")
                print(f"🎞️ MoviePy available: {'✅' if MOVIEPY_AVAILABLE else '❌'}")
                continue
                
            elif command == "end":
                prompt = "\n".join(buffer).strip()
                buffer.clear()
                
                if not prompt:
                    print("⚠️ Empty prompt - please describe the animation you want to create")
                    continue

                stats['processed'] += 1
                start_time = time.time()

                def _process_prompt():
                    print(f"\n🧠 Generating Manim code...")
                    print(f"📝 Prompt: {prompt[:100]}...")
                    
                    # Generate code
                    code, class_name = generate_manim_code(client, prompt)
                    
                    # Render scene
                    video_path = render_single_scene(code, class_name, 1)
                    
                    return video_path

                try:
                    video_path = retry_with_backoff(_process_prompt)
                    
                    processing_time = time.time() - start_time
                    stats['total_time'] += processing_time
                    
                    if video_path:
                        stats['successful'] += 1
                        print(f"\n🎉 SUCCESS!")
                        print(f"📹 Video saved to: {video_path}")
                        print(f"⏱️ Processing time: {processing_time:.2f}s")
                    else:
                        stats['failed'] += 1
                        print(f"\n❌ FAILED after multiple attempts")
                        print(f"⏱️ Processing time: {processing_time:.2f}s")
                        
                except Exception as e:
                    stats['failed'] += 1
                    processing_time = time.time() - start_time
                    stats['total_time'] += processing_time
                    if not is_connection_error(e):
                        print(f"\n❌ ERROR: {e}")
                        print(f"⏱️ Processing time: {processing_time:.2f}s")
                    else:
                        # This should not happen as retry_with_backoff handles connection errors
                        raise
                
                print("\n🔄 Ready for next prompt")
                
            else:
                buffer.append(line)
                
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted - type 'exit' to quit or continue typing")
        except EOFError:
            print("\n👋 Session ended")
            break

# ╭──────────────────────────────────────────────────────────────╮
# │                   MAIN EXECUTION ENTRY                       │
# ╰──────────────────────────────────────────────────────────────╯

def main() -> None:
    """
    Main function for standalone execution of the manimator.
    """
    def _main_with_retry():
        print("🚀 Initializing Manimator...")
        
        # Check API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ GEMINI_API_KEY environment variable not set")
            print("💡 Please set your API key: export GEMINI_API_KEY='your_key_here'")
            return
        
        # Initialize client
        client = genai.Client(api_key=api_key)
        print("✅ Gemini API client initialized successfully")
        
        # Check dependencies
        print("🔍 Checking dependencies...")
        
        # Check Manim
        result = subprocess.run(["manim", "--version"], capture_output=True, text=True, timeout=10)
        print(f"✅ Manim: {result.stdout.strip()}")
        
        # Check MoviePy
        if MOVIEPY_AVAILABLE:
            print("✅ MoviePy: Available")
        else:
            print("⚠️ MoviePy: Not available - video concatenation limited")
        
        # Check OpenCode
        try:
            subprocess.run(["opencode", "--version"], capture_output=True, timeout=5)
            print("✅ OpenCode: Available")
        except Exception:
            print("⚠️ OpenCode: Not available - automatic error fixing disabled")
        
        print("🎬 Starting interactive chat session...")
        run_interactive_chat_session(client)
    
    try:
        retry_with_backoff(_main_with_retry)
    except Exception as e:
        if not is_connection_error(e):
            if "not found" in str(e).lower():
                print("❌ Manim not found - please install: pip install manim")
            else:
                print(f"❌ Failed to initialize Manimator: {e}")
        else:
            # This should not happen as retry_with_backoff handles connection errors
            raise

# ╭──────────────────────────────────────────────────────────────╮
# │                    MODULE INITIALIZATION                     │
# ╰──────────────────────────────────────────────────────────────╯

# Create necessary directories when module is imported
if __name__ != "__main__":
    FINAL_VIDEO_DIR.mkdir(exist_ok=True)
    print("📁 Manimator directories initialized")

# Main execution
if __name__ == "__main__":
    main()

# ╭──────────────────────────────────────────────────────────────╮
# │                    END OF MANIMATOR MODULE                   │
# ╰──────────────────────────────────────────────────────────────╯
