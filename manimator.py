#!/usr/bin/env python3

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
import concurrent.futures
import glob
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

try:
    from moviepy.editor import VideoFileClip, concatenate_videoclips
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

MAX_ATTEMPTS = 5
MANIM_TIMEOUT = int(os.getenv("MANIM_TIMEOUT", "600"))
MANIM_QUALITY = "low_quality"
MANIM_FPS = 24

OPENCODE_TIMEOUT = 600
OPENCODE_MODEL = "github-copilot/gpt-4.1"
OPENCODE_MAX_RETRIES = 8

FINAL_VIDEO_DIR = Path("final_video_output")
TEMP_SCENE_DIR_PREFIX = "temp_scene_"
MEDIA_DIR_NAME = "media"

VIDEO_CODEC = "libx264"
AUDIO_CODEC = "aac"
VIDEO_BITRATE = "1000k"

RETRY_DELAY_BASE = 1
RETRY_DELAY_MAX = 60
RETRY_BACKOFF_MULTIPLIER = 2

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
                
                time.sleep(delay)
                
                delay = min(delay * RETRY_BACKOFF_MULTIPLIER, RETRY_DELAY_MAX)
                continue
            else:
                raise

def _clean_code(code: str) -> str:
    code = re.sub(r"^```[a-zA-Z0-9]*\n?", "", code, flags=re.MULTILINE)
    code = re.sub(r"\s*```$", "", code.strip(), flags=re.MULTILINE)
    code = re.sub(r'\n\s*\n\s*\n', '\n\n', code)
    code = code.strip()
    return code

def _validate_manim_code(code: str) -> Tuple[bool, str, str]:
    if not code or not code.strip():
        return False, "", "Empty code generated"
    
    required_imports = ["from manim import", "import manim"]
    has_import = any(imp in code for imp in required_imports)
    if not has_import:
        return False, "", "Missing Manim import statement"
    
    class_match = re.search(r"class\s+(\w+)\s*\(.*Scene.*\)", code)
    if not class_match:
        return False, "", "No valid Scene class found"
    
    class_name = class_match.group(1)
    
    if "def construct(self)" not in code:
        return False, class_name, "Missing construct method"
    
    return True, class_name, ""

def generate_manim_code_with_opencode(scene_description: str, subtitle_content: str = "", pdf_content: str = "", total_duration: int = 0) -> Tuple[str, str]:
    import subprocess
    import shlex
    import re
    try:
        generation_prompt = f"""You are a ManimCE expert creating dynamic educational animations with professional diagrams.

Read and follow the rules in Scene_Rules.md file.

SCENE DESCRIPTION:
{scene_description}

CRITICAL REQUIREMENTS:
- FULL SCREEN UTILIZATION: Use entire space x=[-7.5,7.5], y=[-4,4] effectively
- ENHANCED ZONES: Top[2.0,4.0], Upper-Main[0.5,2.0], Lower-Main[-1.0,0.5], Bottom[-4.0,-1.0]
- ABSOLUTE ZERO OVERLAPPING: Minimum 2.0 units spacing between ALL elements at ALL times
- BOUNDARY COMPLIANCE: Keep all elements within x=[-6.5,6.5], y=[-3.5,3.5] to prevent overflow
- DETAILED MOVEMENT: Every animation must specify exact movement paths, directions, speeds
- PROFESSIONAL DIAGRAMS: Use proper Manim shapes (Circle, Rectangle, Arrow, etc.) with precise positioning
- **TRANSLUCENT SHAPES**: Use fill_opacity=0.0 (default) - NO white or colored backgrounds on shapes
- **TEXT RESTRICTIONS**: NO descriptive text paragraphs - ONLY minimal labels for elements
- **EXACT TIMING SYNC**: Every self.play() and self.wait() must have explicit run_time to match target duration EXACTLY
- DYNAMIC ANIMATIONS: Include slide-in/out, spiral, curved paths, transformations
- Remove old elements before adding new ones: self.play(FadeOut(...))
- Elements appear/disappear only during designated time spans
- Minimum 400 lines of code with rich visual content

COMPLETE VIDEO GENERATION REQUIREMENTS:
- GENERATE COMPLETE, FULL MANIM VIDEOS - NO PARTIAL RENDERS
- ENSURE ALL ANIMATIONS FINISH COMPLETELY - DO NOT STOP MIDWAY
- CREATE SELF-CONTAINED SCENES THAT RENDER FROM START TO FINISH
- AVOID ANY INCOMPLETE OR PARTIAL VIDEO SEGMENTS
- GUARANTEE FINAL OUTPUT IS A COMPLETE MP4 VIDEO FILE
- NO PARTIAL MOVIE FILES OR INCOMPLETE RENDERS ALLOWED
- CONTINUE UNTIL ENTIRE ANIMATION SEQUENCE IS COMPLETE

ARROW POSITIONING RULES:
- NEVER use hardcoded coordinates for Arrow() objects
- ALWAYS create arrows using Arrow(start=element1.get_center(), end=element2.get_center())
- Position arrows BEHIND target elements using z_index or add_to_back()
- Example: arrow = Arrow(circle1.get_center(), circle2.get_center()).set_z_index(-1)
- Ensure arrows don't overlap with other elements by using proper spacing

ANTI-OVERLAP SYSTEM:
- Before positioning any element, verify it doesn't overlap with existing elements
- Use grid-based positioning when needed to ensure proper spacing
- Calculate element boundaries and maintain 2.0+ unit buffer zones
- If elements would overlap, adjust positions to maintain visual hierarchy
- Test all animations to ensure moving elements don't create overlaps

MOVEMENT PATTERNS TO INCLUDE:
- Entrance: Slide from edges, spiral in, zoom from center
- Transformations: Morphing shapes, evolving equations
- Explanations: Highlight boxes, moving pointers, tracking elements
- Exit: Directional slides, fade with movement, shrink to points

DIAGRAM REQUIREMENTS:
- Use Circle(), Rectangle(), Polygon(), Arrow() with proper dimensions
- Mathematical diagrams with MathTex(), NumberPlane()
- Educational flowcharts and concept maps
- **TRANSLUCENT BACKGROUNDS**: All shapes use stroke_color only, fill_opacity=0.0 (default)
- **NO WHITE BACKGROUNDS**: Never use fill_color or solid fills on shapes
- Consistent color coding: BLUE="#58C4DD", GOLD="#F0E68C", WHITE="#FFFFFF"
- Proper stroke widths and translucent appearance

ROBUSTNESS REQUIREMENTS:
- ALWAYS generate a complete, functional scene with substantial content
- If scene description seems minimal, expand with relevant educational elements
- Include multiple visual components in every scene
- Ensure every scene has educational value and visual appeal
- Never generate empty or minimal code
- COMPLETE THE ENTIRE ANIMATION SEQUENCE WITHOUT INTERRUPTION
- RENDER FULL VIDEO FROM START TO FINISH - NO PARTIAL OUTPUTS

FINAL VIDEO COMPLETION MANDATE:
- DO NOT STOP GENERATION UNTIL A COMPLETE VIDEO IS PRODUCED
- ENSURE ALL MANIM ANIMATIONS RUN TO COMPLETION
- AVOID ANY PARTIAL RENDERS OR INCOMPLETE SEGMENTS
- GENERATE SELF-CONTAINED, COMPLETE MANIM SCENES ONLY
- FINAL OUTPUT MUST BE A FULLY RENDERED, COMPLETE MP4 VIDEO

Generate a complete, executable ManimCE scene class with dynamic movements, NO OVERLAPPING elements, proper arrow positioning, professional diagrams, and GUARANTEED COMPLETE VIDEO RENDERING. Include imports, class definition, and complete construct method. Generate ONLY the code - no explanations. ENSURE THE GENERATED CODE PRODUCES A COMPLETE, FULL VIDEO WITHOUT ANY PARTIAL RENDERS."""
        def _generate_code_with_retry():
            attempt = 0
            current_prompt = generation_prompt
            while attempt < OPENCODE_MAX_RETRIES:
                attempt += 1
                cmd = [
                    "opencode",
                    "run",
                    "--model", OPENCODE_MODEL,
                    current_prompt
                ]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=OPENCODE_TIMEOUT,
                    check=False
                )
                if result.returncode != 0:
                    if attempt < OPENCODE_MAX_RETRIES:
                        current_prompt += f"\n\nPREVIOUS ATTEMPT FAILED: {result.stderr}\nPlease fix this issue and generate valid ManimCE code."
                        continue
                    else:
                        raise Exception(f"OpenCode failed after {OPENCODE_MAX_RETRIES} attempts")
                cleaned_code = _clean_code(result.stdout.strip())
                if not cleaned_code:
                    if attempt < OPENCODE_MAX_RETRIES:
                        current_prompt += "\n\nPREVIOUS ATTEMPT RETURNED EMPTY CODE. Please generate complete ManimCE code with at least 400 lines that matches the scene description exactly."
                        continue
                    else:
                        raise Exception("OpenCode returned empty code after all attempts")
                class_match = re.search(r"class\s+(\w+)\s*\(.*Scene.*\)", cleaned_code)
                if class_match:
                    class_name = class_match.group(1)
                else:
                    class_name = "GeneratedScene"
                return cleaned_code, class_name
            raise Exception(f"Failed to generate valid Manim code after {OPENCODE_MAX_RETRIES} attempts")
        return retry_with_backoff(_generate_code_with_retry)
    except subprocess.TimeoutExpired:
        raise
    except Exception as e:
        if not is_connection_error(e):
            raise
        else:
            raise

def run_manim_render(scene_path: Path, class_name: str, output_dir: Path) -> subprocess.CompletedProcess:
    def _run_manim_with_retry():
        scene_filename = scene_path.name
        
        cmd = [
            "manim",
            "-pql",
            scene_filename,
            class_name
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=MANIM_TIMEOUT,
            cwd=output_dir
        )
        
        return result
    
    try:
        return retry_with_backoff(_run_manim_with_retry)
    except subprocess.TimeoutExpired:
        raise
    except Exception as e:
        if not is_connection_error(e):
            raise
        else:
            raise

def find_rendered_video(output_dir: Path, class_name: str) -> Optional[Path]:
    scene_name = output_dir.stem 
    possible_paths = [
        output_dir.parent / "media" / "videos" / scene_name / f"{MANIM_QUALITY.replace('_quality', '')}p15" / f"{class_name}.mp4",
        output_dir.parent / "media" / "videos" / scene_name / "480p15" / f"{class_name}.mp4",
        output_dir.parent / "media" / "videos" / f"{class_name}.mp4",
        output_dir.parent / f"{class_name}.mp4"
    ]
    
    # Check the standard paths first, ensuring they're complete videos
    for path in possible_paths:
        if path.exists() and is_complete_video(path):
            print(f"✅ Found complete rendered video: {path}")
            return path
    
    # Search media directory, excluding partial files
    media_dir = output_dir.parent / "media"
    if media_dir.exists():
        for video_file in media_dir.rglob("*.mp4"):
            # Skip partial movie files directory
            if "partial_movie_files" in str(video_file):
                continue
                
            if class_name in str(video_file) and is_complete_video(video_file):
                print(f"✅ Found complete video in media search: {video_file}")
                return video_file
        
        # If we still haven't found anything, look in scene-specific directories
        scene_media_dir = media_dir / "videos" / scene_name
        if scene_media_dir.exists():
            for video_file in scene_media_dir.rglob("*.mp4"):
                # Skip partial movie files
                if "partial_movie_files" in str(video_file):
                    continue
                    
                if is_complete_video(video_file):
                    print(f"✅ Found complete video in scene directory: {video_file}")
                    return video_file
    
    print(f"❌ No complete rendered video found for class: {class_name}")
    return None

def opencode_fix_manim_code(py_file: Path, error_output: str = "", scene_idx: int = 1, scene_description: str = "", scenes_file: str = "") -> Optional[bool]:
    return None

def render_single_scene(code: str, class_name: str, scene_idx: int, scene_description: str = "", scenes_file: str = "") -> Optional[Path]:
    return None

def is_complete_video(video_path: Path) -> bool:
    """Check if a video file is complete and not a partial render."""
    try:
        # Skip any files in partial_movie_files directory
        if "partial_movie_files" in str(video_path):
            return False
        
        # Skip files with suspicious names indicating partial renders
        partial_indicators = [
            "partial",
            "temp",
            "segment_",
            "_tmp",
            "incomplete",
            "rendering"
        ]
        
        file_name_lower = video_path.name.lower()
        if any(indicator in file_name_lower for indicator in partial_indicators):
            return False
        
        # Check if file exists and has reasonable size
        if not video_path.exists() or video_path.stat().st_size < 1024:  # Less than 1KB
            return False
        
        # Try to get video duration using ffmpeg to verify it's complete
        try:
            import subprocess
            result = subprocess.run([
                "ffmpeg", "-i", str(video_path), "-f", "null", "-"
            ], capture_output=True, text=True, timeout=10)
            
            # If ffmpeg can process it without "moov atom not found" error, it's likely complete
            if "moov atom not found" in result.stderr or "Invalid data found" in result.stderr:
                return False
                
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            # If ffmpeg fails, still try to use the file but with caution
            pass
        
        return True
        
    except Exception as e:
        return False

def concatenate_scene_videos(video_paths: List[Path], output_path: Path) -> bool:
    def _concatenate_with_retry():
        if not MOVIEPY_AVAILABLE:
            return False
        
        # Filter out partial and invalid videos
        complete_paths = []
        for p in video_paths:
            if p and is_complete_video(Path(p)):
                complete_paths.append(p)
            else:
                print(f"⚠️ Skipping incomplete/partial video: {p}")
        
        if not complete_paths:
            print("❌ No complete videos found for concatenation")
            return False
        
        print(f"✅ Found {len(complete_paths)} complete videos for concatenation")
        
        try:
            clips = []
            for path in complete_paths:
                try:
                    clip = VideoFileClip(str(path))
                    # Verify the clip has valid duration
                    if clip.duration and clip.duration > 0:
                        clips.append(clip)
                        print(f"✅ Added video: {path} (duration: {clip.duration:.2f}s)")
                    else:
                        print(f"⚠️ Skipping video with invalid duration: {path}")
                        clip.close()
                except Exception as e:
                    print(f"⚠️ Failed to load video {path}: {e}")
                    continue
            
            if not clips:
                print("❌ No valid video clips could be loaded")
                return False
            
            final_clip = concatenate_videoclips(clips, method="compose")
            
            final_clip.write_videofile(
                str(output_path),
                codec=VIDEO_CODEC,
                audio_codec=AUDIO_CODEC,
                fps=MANIM_FPS,
                bitrate=VIDEO_BITRATE,
                verbose=False,
                logger=None
            )

            for clip in clips:
                clip.close()
            final_clip.close()

            if output_path.exists() and output_path.stat().st_size > 0:
                print(f"✅ Final video created successfully: {output_path}")
                return True
            else:
                print("❌ Final video creation failed")
                return False
                
        except Exception as e:
            print(f"❌ Video concatenation failed: {e}")
            return False

    try:
        return retry_with_backoff(_concatenate_with_retry)
    except Exception as e:
        if not is_connection_error(e):
            return False
        else:
            raise

def cleanup_temp_directories(num_scenes: int) -> None:
    for i in range(1, num_scenes + 1):
        temp_dir = Path(f"{TEMP_SCENE_DIR_PREFIX}{i}")
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                pass

def auto_cleanup_generated_files(preserve_final_video: Optional[str] = None, preserve_audio: Optional[str] = None) -> None:
    cleanup_patterns = [
        "scene_*.py",
        "temp_scene_*.py",
        "media/videos/**/scene_*.mp4",
        "media/videos/**/Scene*.mp4",
        "media/partial_movie_files/**/*",
        "media/Tex/**/*",
        "media/text/**/*",
        "__pycache__/**/*",
        "*.pyc",
        "chunk_*_detailed_scenes_*.txt",
        "scene_segment_*.txt",
        "plan_segments_*.json",
        "temp_*.srt",
        "*.log",
        "manim_*.log",
        "temp_scene_*/**/*",
        ".manim_cache/**/*",
    ]
    
    preserved_files = {
        "scene.py",
        "drawings.py",
    }
    
    if preserve_final_video:
        preserved_files.add(os.path.basename(preserve_final_video))
    
    if preserve_audio:
        preserved_files.add(os.path.basename(preserve_audio))
    
    for pattern in cleanup_patterns:
        try:
            import glob
            for file_path in glob.glob(pattern, recursive=True):
                file_name = os.path.basename(file_path)
                
                if file_name in preserved_files:
                    continue
                
                if preserve_final_video and os.path.samefile(file_path, preserve_final_video):
                    continue
                if preserve_audio and os.path.samefile(file_path, preserve_audio):
                    continue
                
                try:
                    if os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                    else:
                        os.remove(file_path)
                except Exception as e:
                    pass
        except Exception as e:
            pass
    
    try:
        empty_dirs = []
        for root, dirs, files in os.walk(".", topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    if not os.listdir(dir_path):
                        empty_dirs.append(dir_path)
                except:
                    pass
        
        for empty_dir in empty_dirs:
            try:
                os.rmdir(empty_dir)
            except:
                pass
    except Exception as e:
        pass

def combine_video_with_audio_and_cleanup(video_path: str, audio_path: str, output_path: str) -> Optional[str]:
    if not MOVIEPY_AVAILABLE:
        return None
    
    try:
        from moviepy.editor import VideoFileClip, AudioFileClip
        
        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_path)
        
        final_clip = video_clip.set_audio(audio_clip)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        final_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True
        )
        
        video_clip.close()
        audio_clip.close()
        final_clip.close()
        
        auto_cleanup_generated_files(
            preserve_final_video=output_path,
            preserve_audio=audio_path
        )
        
        return output_path
        
    except Exception as e:
        return None

def generate_video_from_scenes(scene_descriptions: List[str]) -> Optional[str]:
    if not scene_descriptions:
        return None

    return None

def run_interactive_chat_session() -> None:
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
                return
                
            elif command == "status":
                continue
                
            elif command == "info":
                continue
                
            elif command == "end":
                prompt = "\n".join(buffer).strip()
                buffer.clear()
                
                if not prompt:
                    continue

                stats['processed'] += 1
                start_time = time.time()

                try:
                    segment_file = "scene_segment_1.txt"
                    with open(segment_file, "w", encoding='utf-8') as f:
                        f.write(f"Generate a Manim scene based on the following user prompt:\n\n{prompt}")

                    code, class_name = generate_manim_code_with_opencode(prompt)
                    
                    scene_file = Path("scene.py")
                    scene_file.write_text(code)
                    
                    result = run_manim_render(scene_file, class_name, Path("."))
                    
                    video_path = None
                    if result.returncode == 0:
                        video_path = find_rendered_video(Path("."), class_name)
                    
                    processing_time = time.time() - start_time
                    stats['total_time'] += processing_time
                    
                    if video_path:
                        stats['successful'] += 1
                    else:
                        stats['failed'] += 1
                        
                except Exception as e:
                    stats['failed'] += 1
                    processing_time = time.time() - start_time
                    stats['total_time'] += processing_time
                
            else:
                buffer.append(line)
                
        except KeyboardInterrupt:
            pass
        except EOFError:
            break

def main() -> None:
    def _main_with_retry():
        try:
            result = subprocess.run(["manim", "--version"], capture_output=True, text=True, timeout=10)
        except Exception:
            return
        
        if MOVIEPY_AVAILABLE:
            pass
        else:
            pass
        
        try:
            result = subprocess.run(["opencode", "--version"], capture_output=True, timeout=5)
        except Exception:
            return
        
        run_interactive_chat_session()
    
    try:
        retry_with_backoff(_main_with_retry)
    except Exception as e:
        if not is_connection_error(e):
            if "not found" in str(e).lower():
                pass
            else:
                pass
        else:
            raise

if __name__ != "__main__":
    FINAL_VIDEO_DIR.mkdir(exist_ok=True)

if __name__ == "__main__":
    main()

def run_new_video_generation_workflow(
    subtitles: List[Dict], gemma_api_keys: List[str]
) -> List[str]:
    try:
        from utils import new_video_generation_workflow
        return new_video_generation_workflow(subtitles, gemma_api_keys)
    except ImportError:
        return []
