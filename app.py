#!/usr/bin/env python3

"""
PDF Seamless Teacher - Main Streamlit Application
================================================================
A comprehensive educational tool that converts PDF documents into seamless 
teaching content with AI-generated audio and video.

NEW FEATURES:
- Chunk-based synchronized video generation with precise timestamps
- Individual audio and subtitle files for each PDF chunk
- On-demand scene generation only when video options are selected
- Detailed scene generation using entire subtitle content for manimator integration
- Enhanced download options for chunk-based content

Features:
- PDF analysis with Gemma 3 via Gemini API
- Voice cloning with F5-TTS
- Automated video generation with Manim via manimator
- Interactive Streamlit interface
- Chunk-based audio-video synchronization with exact timing
================================================================
"""

# ╭──────────────────────────────────────────────────────────────╮
# │ IMPORTS SECTION │
# ╰──────────────────────────────────────────────────────────────╯

import streamlit as st
import os
import uuid
import concurrent.futures
import time
import socket
import errno
import glob
from pathlib import Path
import sys

# Video processing imports
try:
    from moviepy.editor import VideoFileClip, concatenate_videoclips
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

# Custom utility imports
from utils import (
    is_gemini_available,
    process_pdf_with_chunking,
    chunk_text,
    call_f5_tts,
    concatenate_audios_with_crossfade,
    get_all_voice_samples,
    cleanup_files,
    generate_manim_video,
    _check_manim_setup,
    is_connection_error,
    retry_with_backoff,
    DEFAULT_PAGES_PER_CHUNK,
    # New imports for on-demand scene generation
    generate_video_scenes_from_scripts,
    # Audio and subtitle generation
    generate_chunk_based_audio_and_subtitles,
    # Video processing utilities
    combine_video_and_audio,
    cleanup_intermediate_files,
    # OLD WORKFLOW FUNCTIONS REMOVED - use new_video_generation_workflow instead
    # render_chunk_based_synchronized_videos,
    # combine_chunk_videos_with_audio,
    WHISPER_AVAILABLE,
    # New workflow
    new_video_generation_workflow
)

# Torch/Streamlit event loop workaround
try:
    import torch
except ImportError:
    torch = None

# ╭──────────────────────────────────────────────────────────────╮
# │ CONFIGURATION CONSTANTS │
# ╰──────────────────────────────────────────────────────────────╯

# Network Retry Configuration for Streamlit-specific operations
STREAMLIT_RETRY_DELAY_BASE = 1  # Base delay in seconds
STREAMLIT_RETRY_DELAY_MAX = 30  # Maximum delay in seconds
STREAMLIT_RETRY_BACKOFF_MULTIPLIER = 2  # Exponential backoff multiplier

# ╭──────────────────────────────────────────────────────────────╮
# │ STREAMLIT RETRY UTILITIES │
# ╰──────────────────────────────────────────────────────────────╯

def streamlit_retry_with_backoff(func, *args, **kwargs):
    """
    Execute a function with unlimited retries and exponential backoff for connection errors.
    Streamlit-specific version with status updates. Error 104 gets immediate unlimited retries.
    
    Args:
        func: Function to execute
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
    
    Returns:
        Result of the function execution
    """
    attempt = 0
    delay = STREAMLIT_RETRY_DELAY_BASE
    
    while True:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if is_connection_error(e):
                attempt += 1
                
                # Special handling for 104 errors - immediate retry
                if "104" in str(e) or "Connection reset by peer" in str(e).lower():
                    error_placeholder = st.empty()
                    with error_placeholder.container():
                        st.warning(
                            f"🔄 **Connection 104 Error (Attempt {attempt})**\n\n"
                            f"Error: {str(e)}\n\n"
                            f"Retrying immediately (unlimited retries for 104 errors)...",
                            icon="🔄"
                        )
                    time.sleep(0.01)  # Minimal delay for 104 errors
                    error_placeholder.empty()
                    continue
                
                # Regular connection errors
                error_placeholder = st.empty()
                with error_placeholder.container():
                    st.warning(
                        f"🔄 **Connection Error Detected (Attempt {attempt})**\n\n"
                        f"Error: {str(e)}\n\n"
                        f"Retrying in {delay} seconds... The system will keep trying until successful.",
                        icon="🔄"
                    )
                
                time.sleep(delay)
                error_placeholder.empty()
                
                # Exponential backoff with maximum delay
                delay = min(delay * STREAMLIT_RETRY_BACKOFF_MULTIPLIER, STREAMLIT_RETRY_DELAY_MAX)
                continue
            else:
                # Re-raise non-connection errors
                raise

def safe_streamlit_operation(operation_name, func, *args, **kwargs):
    """
    Wrapper for Streamlit operations that may encounter connection errors.
    
    Args:
        operation_name: Name of the operation for logging
        func: Function to execute
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
    
    Returns:
        Result of the function execution
    """
    try:
        return streamlit_retry_with_backoff(func, *args, **kwargs)
    except Exception as e:
        if not is_connection_error(e):
            st.error(
                f"❌ **{operation_name} Failed**\n\n"
                f"Error: {str(e)}",
                icon="❌"
            )
            raise
        else:
            # This should not happen as streamlit_retry_with_backoff handles connection errors
            raise

# ╭──────────────────────────────────────────────────────────────╮
# │ SESSION STATE CLEANUP UTILITIES │
# ╰──────────────────────────────────────────────────────────────╯

def force_clear_session_state():
    """Forcefully clear all session state data."""
    keys_to_delete = [
        'seamless_transcript',
        'teaching_scripts', 
        'video_scenes',  # This will only exist after video generation
        'pdf_name',
        'pages_per_chunk',
        'chunk_srt_files',
        'chunk_data_list',
        'detailed_scene_files'
    ]
    
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]
    
    # Force Streamlit to refresh
    st.cache_data.clear()

def cleanup_all_files():
    """Clean up all temporary and persistent files while preserving final outputs."""
    import glob
    
    try:
        # Import enhanced cleanup function from utils
        from utils import auto_cleanup_after_video_generation
        
        # Use enhanced cleanup that preserves final videos and essential files
        auto_cleanup_after_video_generation()
        
        print("✅ Enhanced cleanup completed - preserved final outputs and essential files")
        
    except Exception as e:
        print(f"⚠️ Enhanced cleanup failed, using basic cleanup: {e}")
        
        # Fallback to basic cleanup (but preserve more files)
        try:
            # Only clean temporary audio files (not final ones)
            for pattern in ["F5-TTS/audio/temp_*.wav"]:
                for file in glob.glob(pattern):
                    os.remove(file)
            
            # Only clean temporary text files
            for file in glob.glob("F5-TTS/texts/temp_*.txt"):
                os.remove(file)
            
            # Only clean temporary subtitle files (preserve main subtitles)
            for file in glob.glob("temp_*_subtitles_*.srt"):
                os.remove(file)
            
            # Only clean temporary scene files (preserve scene.py and drawings.py)
            for file in glob.glob("temp_scene_*.txt"):
                os.remove(file)
            
            # DON'T clean final video files anymore
            print("✅ Basic cleanup completed - preserved final outputs")
            
        except Exception as cleanup_error:
            print(f"⚠️ Basic cleanup also failed: {cleanup_error}")

# ╭──────────────────────────────────────────────────────────────╮
# │ PAGE CONFIGURATION │
# ╰──────────────────────────────────────────────────────────────╯

st.set_page_config(
    page_title="PDF Seamless Teacher",
    page_icon="🧑🏫",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ╭──────────────────────────────────────────────────────────────╮
# │ HEADER SECTION │
# ╰──────────────────────────────────────────────────────────────╯

st.title("🧑🏫 PDF Seamless Teacher (Gemma 3 + F5-TTS + Manimator)")
st.markdown(
    """
    **Transform your PDF documents into engaging educational content with perfect chunk-based synchronization!**
    
    Upload a PDF, choose a voice, and generate seamless teaching content with:
    - 🎤 **Individual voice-cloned audio** for each PDF chunk using F5-TTS
    - 🎬 **Synchronized video animations** using Manimator with precise timing
    - 📝 **Timestamped subtitles** for each chunk with accessibility
    - ⏱️ **Perfect timing alignment** between audio and video for each chunk
    - 🎯 **On-demand scene generation** only when video options are selected
    - 🎭 **Detailed scene descriptions** generated for manimator integration
    """
)

# ╭──────────────────────────────────────────────────────────────╮
# │ EMERGENCY RESET SECTION │
# ╰──────────────────────────────────────────────────────────────╯

# Add emergency reset button in sidebar
with st.sidebar:
    st.markdown("### 🔧 System Controls")
    
    if st.button("🔄 **RESET ALL DATA**", type="secondary", use_container_width=True):
        force_clear_session_state()
        st.success("✅ All session data cleared!")
        st.rerun()
    
    if st.button("🗑️ **COMPLETE RESET**", type="primary", use_container_width=True, help="Clear all data and files"):
        force_clear_session_state()
        cleanup_all_files()
        st.success("✅ Complete reset performed!")
        st.rerun()
    
    if st.button("⚡ **RESTART APPLICATION**", type="primary", use_container_width=True):
        # Clear everything
        force_clear_session_state()
        cleanup_all_files()
        
        # Show restart message
        st.error("🔄 Application restarting... Please refresh the page manually.")
        st.stop()

# ╭──────────────────────────────────────────────────────────────╮
# │ SYSTEM CONNECTIVITY CHECK │
# ╰──────────────────────────────────────────────────────────────╯

def check_gemini_connection():
    """Check Gemini API connection with retry logic."""
    return is_gemini_available()

if not safe_streamlit_operation("Gemini API Connection Check", check_gemini_connection):
    st.error(
        "🔌 **Cannot connect to Gemini API**\n\n"
        "Please ensure your `GEMINI_API_KEY` environment variable is set and accessible.",
        icon="🔌"
    )
    st.info(
        "💡 **Setup Instructions:**\n"
        "1. Get your API key from Google AI Studio\n"
        "2. Set environment variable: `export GEMINI_API_KEY='your_key_here'`\n"
        "3. Restart the application"
    )
    st.stop()

# Check Whisper availability for transcription
if not WHISPER_AVAILABLE:
    st.warning(
        "⚠️ **Whisper not available for audio transcription**\n\n"
        "Chunk-based synchronized video generation requires Whisper. Install with: `pip install openai-whisper`",
        icon="⚠️"
    )

# ╭──────────────────────────────────────────────────────────────╮
# │ PDF UPLOAD SECTION │
# ╰──────────────────────────────────────────────────────────────╯

st.markdown("---")
st.markdown("### 📄 Step 1: Upload Your PDF Document")

uploaded_file = st.file_uploader(
    "Choose a PDF file to begin the analysis",
    type="pdf",
    help="Upload any educational PDF document (textbooks, papers, notes, etc.)",
    label_visibility="collapsed"
)

# ╭──────────────────────────────────────────────────────────────╮
# │ PDF PROCESSING SECTION │
# ╰──────────────────────────────────────────────────────────────╯

if uploaded_file is not None:
    # Check if we need to reprocess the PDF
    if ('seamless_transcript' not in st.session_state or 
        st.session_state.get('pdf_name') != uploaded_file.name):
        
        st.markdown("#### 🔄 Processing PDF...")
        
        # Add chunking configuration
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.info(
                f"📚 **PDF Processing Method:** Chunk-based processing (teaching scripts only)\n\n"
                f"The PDF will be split into chunks of {DEFAULT_PAGES_PER_CHUNK} pages each. Each chunk will get its own teaching script. "
                f"Video scenes will be generated later when video generation options are selected.",
                icon="📚"
            )
        
        with col2:
            pages_per_chunk = st.number_input(
                "Pages per chunk:",
                min_value=1,
                max_value=20,
                value=DEFAULT_PAGES_PER_CHUNK,
                help="Number of pages to process together in each chunk"
            )
        
        with st.spinner("Analyzing PDF chunks and generating teaching scripts..."):
            def process_pdf():
                # Save uploaded file temporarily
                temp_pdf_path = f"temp_pdf_{uuid.uuid4()}.pdf"
                try:
                    # Write uploaded file to temporary location
                    with open(temp_pdf_path, "wb") as f:
                        f.write(uploaded_file.read())
                    
                    # Process PDF with chunking - ONLY teaching scripts
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Processing PDF chunks for teaching scripts...")
                    progress_bar.progress(0.1)
                    
                    # Use the updated PDF processing function (returns only teaching scripts)
                    def pdf_processing():
                        return process_pdf_with_chunking(temp_pdf_path, pages_per_chunk)
                    
                    teaching_scripts = safe_streamlit_operation(
                        "PDF Chunk Processing",
                        pdf_processing
                    )
                    
                    progress_bar.progress(0.8)
                    status_text.text("Combining results...")
                    
                    # Combine all teaching scripts into one seamless transcript
                    seamless_transcript = ' '.join([script for script in teaching_scripts if script.strip()])
                    
                    # Store results in session state - NO video scenes yet
                    st.session_state.seamless_transcript = seamless_transcript
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.teaching_scripts = teaching_scripts
                    st.session_state.pages_per_chunk = pages_per_chunk
                    # Explicitly remove video_scenes from session state if it exists
                    if 'video_scenes' in st.session_state:
                        del st.session_state['video_scenes']
                    
                    # Clear progress indicators
                    progress_bar.progress(1.0)
                    status_text.text("Processing complete!")
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                    
                    return True
                    
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_pdf_path):
                        os.remove(temp_pdf_path)
            
            try:
                safe_streamlit_operation("PDF Processing", process_pdf)
            except Exception as e:
                st.error(f"❌ **Failed to process the PDF**\n\nError: {str(e)}", icon="📄")
                st.stop()
    
    # Retrieve processed data
    seamless_transcript = st.session_state.seamless_transcript
    teaching_scripts = st.session_state.get('teaching_scripts', [])
    
    # Success message - updated to reflect no scenes generated yet
    st.success(
        f"✅ **Successfully analyzed '{uploaded_file.name}'**\n\n"
        f"Generated complete teaching transcript from {len(teaching_scripts)} chunks. "
        f"Video scenes will be generated when video options are selected.",
        icon="✅"
    )
    
    # ╭──────────────────────────────────────────────────────────────╮
    # │ CONTENT PREVIEW SECTION │
    # ╰──────────────────────────────────────────────────────────────╯
    
    st.markdown("#### 👀 Content Preview")
    
    # Only show transcript preview initially
    with st.expander("📝 View Teaching Transcript", expanded=False):
        st.markdown("**Generated Teaching Script:**")
        st.write(seamless_transcript)
        st.markdown(f"**Word Count:** {len(seamless_transcript.split())} words")
        st.markdown(f"**Character Count:** {len(seamless_transcript)} characters")
        
        # Show individual chunk scripts if available
        if teaching_scripts and len(teaching_scripts) > 1:
            st.markdown("---")
            st.markdown("**Individual Chunk Scripts:**")
            for i, script in enumerate(teaching_scripts, 1):
                if script.strip():
                    with st.expander(f"Chunk {i} Script", expanded=False):
                        st.write(script)
    
    # Show video scenes only if they have been generated
    if 'video_scenes' in st.session_state and st.session_state.video_scenes:
        with st.expander("🎬 View Video Scene Descriptions", expanded=False):
            st.markdown("**Generated Scene Descriptions:**")
            video_scenes = st.session_state.video_scenes
            for i, scene in enumerate(video_scenes, 1):
                if scene.strip():
                    st.markdown(f"**Scene {i}:**")
                    st.write(scene)
                    if i < len(video_scenes):
                        st.markdown("---")
    else:
        # Note about video scenes
        st.info(
            "🎬 **Video Scene Descriptions** will be generated automatically when you select "
            "video generation options below.",
            icon="💡"
        )
    
    # ╭──────────────────────────────────────────────────────────────╮
    # │ OUTPUT TYPE SELECTION │
    # ╰──────────────────────────────────────────────────────────────╯
    
    st.markdown("---")
    st.markdown("### 🎯 Step 2: Choose Output Type")
    
    output_type = st.radio(
        "What would you like to generate?",
        options=["Audio Only", "Video Only", "Chunk-Based Synchronized Audio & Video with Detailed Scenes"],
        index=2,  # Default to chunk-based synchronized option
        help="Choose whether to generate just audio, just video, or chunk-based synchronized content with detailed scene descriptions",
        horizontal=True
    )
    
    # Show chunk-based synchronization info for the new option
    if output_type == "Chunk-Based Synchronized Audio & Video with Detailed Scenes":
        st.info(
            "🎬 **Chunk-Based Synchronized Video Generation with Manimator**\n\n"
            "This mode generates individual audio files and subtitle files for each PDF chunk, then creates "
            "detailed scene descriptions using the entire subtitle content. These detailed scenes are then "
            "processed by manimator.py to generate and execute Manim code with perfect alignment between "
            "audio and video content for each chunk.",
            icon="🎬"
        )
    
    # ╭──────────────────────────────────────────────────────────────╮
    # │ VOICE SELECTION SECTION │
    # ╰──────────────────────────────────────────────────────────────╯
    
    voice_sample_path = None
    if output_type in ["Audio Only", "Chunk-Based Synchronized Audio & Video with Detailed Scenes"]:
        st.markdown("---")
        st.markdown("### 🎤 Step 3: Choose Voice for Audio Generation")
        
        def get_voices():
            return get_all_voice_samples()
        
        available_voices = safe_streamlit_operation("Voice Samples Loading", get_voices)
        
        if not available_voices:
            st.error(
                "🔊 **No voice samples found**\n\n"
                "Please add at least one `.wav` file to the `F5-TTS/voice_samples/` folder "
                "to enable voice cloning functionality.",
                icon="🔊"
            )
            st.info(
                "💡 **Voice Sample Requirements:**\n"
                "- Format: WAV files only\n"
                "- Duration: 10-30 seconds recommended\n"
                "- Quality: Clear, noise-free audio\n"
                "- Content: Natural speech (any language)"
            )
            if output_type == "Audio Only":
                st.stop()
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_voice_file = st.selectbox(
                    "Select a voice for cloning:",
                    options=available_voices,
                    help="These are the .wav files from your `F5-TTS/voice_samples/` directory.",
                    index=0
                )
                voice_sample_path = os.path.join(os.getcwd(), 'F5-TTS', 'voice_samples', selected_voice_file)
            
            with col2:
                if selected_voice_file:
                    st.markdown("**Selected Voice:**")
                    st.info(f"🎵 {selected_voice_file}")
                    
                    # Display voice sample if it exists
                    if os.path.exists(voice_sample_path):
                        with open(voice_sample_path, 'rb') as audio_file:
                            st.audio(audio_file.read(), format='audio/wav')
    
    # ╭──────────────────────────────────────────────────────────────╮
    # │ MANIMATOR SETUP SECTION │
    # ╰──────────────────────────────────────────────────────────────╯
    
    manimator_ready = False
    if output_type in ["Video Only", "Chunk-Based Synchronized Audio & Video with Detailed Scenes"]:
        st.markdown("---")
        st.markdown("### 🎬 Step 4: Manimator Setup Verification")
        
        def check_manimator():
            _check_manim_setup()
            return True
        
        try:
            safe_streamlit_operation("Manimator Setup Verification", check_manimator)
            manimator_ready = True
            st.success(
                "✅ **Manimator setup verified successfully**\n\n"
                "All required dependencies are available for video generation with detailed scene descriptions.",
                icon="🎬"
            )
        except FileNotFoundError as e:
            st.error(f"❌ **Missing File:** {str(e)}", icon="📄")
            if output_type == "Video Only":
                st.stop()
        except RuntimeError as e:
            st.error(f"❌ **Setup Error:** {str(e)}", icon="⚙️")
            st.info(
                "💡 **Setup Instructions:**\n"
                "1. Install Manim: `pip install manim`\n"
                "2. Set GEMINI_API_KEY environment variable\n"
                "3. Ensure manimator.py is in the project directory"
            )
            if output_type == "Video Only":
                st.stop()
    
    # ╭──────────────────────────────────────────────────────────────╮
    # │ CONTENT GENERATION SECTION │
    # ╰──────────────────────────────────────────────────────────────╯
    
    st.markdown("---")
    st.markdown("### 🚀 Step 5: Generate Content")
    
    # Dynamic button text based on output type
    generate_button_text = {
        "Audio Only": "🎧 Generate Audio",
        "Video Only": "🎬 Generate Video",
        "Chunk-Based Synchronized Audio & Video with Detailed Scenes": "🎬🎧 Generate Chunk-Based Content with Detailed Scenes"
    }
    
    # Generation button
    if st.button(
        generate_button_text[output_type],
        type="primary",
        use_container_width=True,
        help=f"Click to start {output_type.lower()} generation process"
    ):
        
        # Initialize result variables
        final_wav_path = None
        final_video_path = None
        chunk_srt_files = []
        chunk_data_list = []
        detailed_scene_files = []
        
        # ╭──────────────────────────────────────────────────────────────╮
        # │ AUDIO GENERATION │
        # ╰──────────────────────────────────────────────────────────────╯
        
        if output_type == "Audio Only":
            st.markdown("#### 🎧 Audio Generation Process")
            
            with st.spinner(f"Generating audio with the voice of '{selected_voice_file}'... This may take several minutes."):
                def generate_audio():
                    # Chunk the transcript for processing
                    chunks = chunk_text(seamless_transcript, max_chars=400)
                    st.info(f"📝 Processing {len(chunks)} text chunks for audio generation...")
                    
                    # Initialize results storage
                    results = [None] * len(chunks)
                    
                    # Define TTS task function with retry logic
                    def tts_task(args):
                        chunk, idx, voice_path = args
                        
                        def tts_operation():
                            return call_f5_tts(chunk, f"chunk_{idx}", voice_path)
                        
                        # Use retry logic for TTS operations
                        return idx, *streamlit_retry_with_backoff(tts_operation)
                    
                    # Process chunks in parallel
                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                        tasks = [(chunk, idx, voice_sample_path) for idx, chunk in enumerate(chunks)]
                        futures = [executor.submit(tts_task, task) for task in tasks]
                        
                        # Collect results as they complete
                        for future in concurrent.futures.as_completed(futures):
                            idx, wav_path, text_path = future.result()
                            results[idx] = (wav_path, text_path)
                    
                    # Extract successful audio chunks
                    audio_chunk_paths = [r[0] for r in results if r and r[0]]
                    text_chunk_paths = [r[1] for r in results if r and r[0]]
                    
                    if not audio_chunk_paths:
                        st.error(
                            "❌ **Audio generation failed**\n\n"
                            "No valid audio chunks were created. Please check your F5-TTS setup.",
                            icon="❌"
                        )
                        return None
                    else:
                        # Concatenate audio chunks
                        final_wav_path = f"F5-TTS/audio/full_{uuid.uuid4()}.wav"
                        
                        def concatenate_operation():
                            return concatenate_audios_with_crossfade(audio_chunk_paths, final_wav_path)
                        
                        streamlit_retry_with_backoff(concatenate_operation)
                        
                        # Cleanup temporary files
                        cleanup_files(audio_chunk_paths + text_chunk_paths)
                        
                        st.success(
                            f"✅ **Audio generated successfully!**\n\n"
                            f"Created from {len(audio_chunk_paths)} audio segments.",
                            icon="🎉"
                        )
                        return final_wav_path
                
                try:
                    final_wav_path = safe_streamlit_operation("Audio Generation", generate_audio)
                except Exception as e:
                    st.error(
                        f"❌ **Critical error during audio generation**\n\n"
                        f"Error details: {str(e)}",
                        icon="❌"
                    )
        
        # ╭──────────────────────────────────────────────────────────────╮
        # │ CHUNK-BASED SYNCHRONIZED VIDEO GENERATION WITH DETAILED SCENES │
        # ╰──────────────────────────────────────────────────────────────╯
        
        elif output_type == "Chunk-Based Synchronized Audio & Video with Detailed Scenes" and manimator_ready:
            st.markdown("#### 🎬 NEW Video Generation Workflow with Detailed Scenes")

            if not WHISPER_AVAILABLE:
                st.error(
                    "❌ **Whisper not available**\n\n"
                    "The new video generation workflow requires Whisper for transcription. "
                    "Install with: `pip install openai-whisper`", icon="❌"
                )
            else:
                with st.spinner("Running NEW video generation workflow: subtitles → detailed plan → drawings → Manim code → videos..."):
                    # --- Step 1: Run the original workflow to get silent video and audio chunks ---
                    def run_new_workflow():
                        st.info(f"🎤 Generating audio and subtitles for {len(teaching_scripts)} chunks...")
                        chunk_data_list = generate_chunk_based_audio_and_subtitles(teaching_scripts, voice_sample_path)
                        if not chunk_data_list:
                            st.error("❌ No valid chunks processed for audio generation")
                            return None, None, None
                        
                        all_subtitles = []
                        for chunk_data in chunk_data_list:
                            if chunk_data.get('segments'):
                                all_subtitles.extend(chunk_data['segments'])
                        
                        if not all_subtitles:
                            st.error("❌ No subtitles generated for video workflow")
                            return None, None, None

                        gemma_api_keys = [os.getenv("GEMINI_API_KEY")] * 3
                        
                        st.info("🎬 Running NEW workflow: generating video from subtitles...")
                        try:
                            video_files = new_video_generation_workflow(all_subtitles, gemma_api_keys)
                            if not video_files:
                                st.error("❌ New video generation workflow failed to produce video files.")
                                return None, None, None

                            # Combine video segments if necessary
                            if len(video_files) > 1:
                                st.info("🎞️ Combining generated video segments...")
                                silent_video_path = f"final_video_output/combined_workflow_video_{int(time.time())}.mp4"
                                if MOVIEPY_AVAILABLE:
                                    clips = [VideoFileClip(p) for p in video_files if os.path.exists(p)]
                                    if clips:
                                        final_video = concatenate_videoclips(clips)
                                        final_video.write_videofile(silent_video_path, verbose=False, logger=None)
                                        final_video.close()
                                        for clip in clips: clip.close()
                                else:
                                    silent_video_path = video_files[0]
                            else:
                                silent_video_path = video_files[0] if video_files else None

                            return silent_video_path, chunk_data_list, all_subtitles
                        except Exception as e:
                            st.error(f"❌ New workflow failed: {str(e)}")
                            return None, None, None

                    silent_video_path, chunk_data_list, all_subtitles = safe_streamlit_operation(
                        "NEW Video Generation Workflow", run_new_workflow
                    )

                    # --- Step 2: Combine Audio and Video, then Clean Up ---
                    if silent_video_path and chunk_data_list:
                        st.info("🔊 Combining audio chunks into a single track...")
                        
                        # 2a. Concatenate all audio chunks into one file
                        audio_chunk_paths = [chunk['audio_path'] for chunk in chunk_data_list if 'audio_path' in chunk]
                        final_wav_path = f"F5-TTS/audio/full_final_audio_{int(time.time())}.wav"
                        final_wav_path = concatenate_audios_with_crossfade(audio_chunk_paths, final_wav_path)

                        if final_wav_path:
                            # 2b. Merge the silent video with the final audio
                            st.info("🎞️ Merging final video with combined audio...")
                            video_with_audio_path = f"final_video_output/final_video_with_audio_{int(time.time())}.mp4"
                            
                            final_video_path = combine_video_and_audio(silent_video_path, final_wav_path, video_with_audio_path)

                            if final_video_path:
                                st.success("✅ **Final video with audio has been generated!**", icon="🎉")
                                
                                # 2c. Perform cleanup, keeping only the final assets
                                st.info("🧹 Cleaning up intermediate files...")
                                files_to_keep = [final_video_path, final_wav_path]
                                if os.path.exists("drawings.py"):
                                    files_to_keep.append("drawings.py")
                                
                                cleanup_intermediate_files(files_to_keep)
                                st.success("✅ Cleanup complete.")
                            else:
                                st.error("❌ Failed to merge video with audio.")
                        else:
                            st.error("❌ Failed to create combined audio track.")
                    elif silent_video_path:
                         st.warning("⚠️ Video was generated, but no audio was available to merge.")
                         final_video_path = silent_video_path # Use the silent video as the final output
                    else:
                        st.error("❌ Chunk-based synchronized video generation failed.", icon="❌")
        
        # ╭──────────────────────────────────────────────────────────────╮
        # │ TRADITIONAL VIDEO GENERATION │
        # ╰──────────────────────────────────────────────────────────────╯
        
        elif output_type == "Video Only" and manimator_ready:
            st.markdown("#### 🎬 Video Generation Process")
            
            with st.spinner("Generating video scenes and Manim video... This process can take several minutes."):
                def generate_video():
                    # Generate video scenes from teaching scripts on-demand
                    st.info("🎭 Generating video scenes from teaching scripts...")
                    
                    def scene_generation():
                        return generate_video_scenes_from_scripts(teaching_scripts)
                    
                    video_scenes = safe_streamlit_operation(
                        "Video Scene Generation",
                        scene_generation
                    )
                    
                    # Store scenes in session state for future use and display
                    st.session_state.video_scenes = video_scenes
                    
                    st.info(f"🎭 Processing {len(video_scenes)} video scenes with manimator...")
                    
                    def video_operation():
                        return generate_manim_video(video_scenes)
                    
                    final_video_path = streamlit_retry_with_backoff(video_operation)
                    
                    if final_video_path:
                        st.success(
                            "✅ **Video generated successfully using manimator!**\n\n"
                            f"Generated {len(video_scenes)} scenes and rendered video.",
                            icon="🎬"
                        )
                        return final_video_path
                    else:
                        st.error(
                            "❌ **Video generation failed**\n\n"
                            "Please check the console logs for detailed error information.",
                            icon="❌"
                        )
                        return None
                
                try:
                    final_video_path = safe_streamlit_operation("Video Generation", generate_video)
                except Exception as e:
                    st.error(
                        f"❌ **Critical error during video generation**\n\n"
                        f"Error details: {str(e)}",
                        icon="❌"
                    )
        
        # ╭──────────────────────────────────────────────────────────────╮
        # │ RESULTS SECTION │
        # ╰──────────────────────────────────────────────────────────────╯
        
        st.markdown("---")
        st.markdown("### 📥 Generated Content & Downloads")
        
        # Create columns for results display
        if final_wav_path or final_video_path:
            results_cols = st.columns(2)
        else:
            st.warning("⚠️ No content was generated successfully.")
        
        # Audio Results Column
        if final_wav_path and os.path.exists(final_wav_path):
            with results_cols[0]:
                st.markdown("#### 🎧 Generated Audio")
                
                # Display audio player
                with open(final_wav_path, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format="audio/wav")
                
                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download Audio",
                        data=audio_bytes,
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_audio.wav",
                        mime="audio/wav",
                        use_container_width=True,
                        help="Download the generated audio file"
                    )
                
                with col2:
                    st.download_button(
                        label="📄 Download Transcript",
                        data=seamless_transcript,
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcript.txt",
                        mime="text/plain",
                        use_container_width=True,
                        help="Download the teaching transcript"
                    )
        
        # Video Results Column
        if final_video_path and os.path.exists(final_video_path):
            with results_cols[1]:
                st.markdown("#### 🎬 Generated Video")
                
                # Display video player
                with open(final_video_path, "rb") as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)
                
                # Download button
                st.download_button(
                    label="📥 Download Video",
                    data=video_bytes,
                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_video.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                    help="Download the generated video file"
                )
        
        # ╭──────────────────────────────────────────────────────────────╮
        # │ CHUNK-BASED SUBTITLE AND DETAILED SCENE DOWNLOADS │
        # ╰──────────────────────────────────────────────────────────────╯
        
        # Enhanced subtitle downloads for chunk-based generation
        if hasattr(st.session_state, 'chunk_srt_files') and st.session_state.chunk_srt_files:
            st.markdown("#### 📝 Chunk-Based Subtitle Files")
            
            # Create columns for chunk subtitle downloads
            chunk_cols = st.columns(min(len(st.session_state.chunk_srt_files), 3))
            
            for i, srt_file in enumerate(st.session_state.chunk_srt_files):
                col_idx = i % len(chunk_cols)
                with chunk_cols[col_idx]:
                    if os.path.exists(srt_file):
                        with open(srt_file, "r", encoding="utf-8") as f:
                            srt_content = f.read()
                        
                        chunk_num = i + 1
                        st.download_button(
                            label=f"📥 Chunk {chunk_num} Subtitles",
                            data=srt_content,
                            file_name=f"chunk_{chunk_num}_subtitles.srt",
                            mime="text/plain",
                            use_container_width=True,
                            help=f"Download subtitle file for PDF chunk {chunk_num}"
                        )
            
            # Combined subtitles download
            if len(st.session_state.chunk_srt_files) > 1:
                st.markdown("#### 📋 Combined Resources")
                
                # Combine all SRT files
                combined_srt = ""
                for i, srt_file in enumerate(st.session_state.chunk_srt_files):
                    if os.path.exists(srt_file):
                        with open(srt_file, "r", encoding="utf-8") as f:
                            chunk_srt = f.read()
                        # Add chunk separator
                        combined_srt += f"\n\n=== CHUNK {i+1} SUBTITLES ===\n\n"
                        combined_srt += chunk_srt
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 All Chunks Subtitles",
                        data=combined_srt,
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_all_subtitles.srt",
                        mime="text/plain",
                        use_container_width=True,
                        help="Download combined subtitle files from all chunks"
                    )
                
                # Show subtitle preview
                with col2:
                    with st.expander("👀 Preview Combined Subtitles", expanded=False):
                        st.code(combined_srt[:1000] + "..." if len(combined_srt) > 1000 else combined_srt)
        
        # Detailed scene files downloads
        if hasattr(st.session_state, 'detailed_scene_files') and st.session_state.detailed_scene_files:
            st.markdown("#### 🎬 Detailed Scene Description Files")
            
            # Create columns for detailed scene downloads
            scene_cols = st.columns(min(len(st.session_state.detailed_scene_files), 3))
            
            for i, scene_file in enumerate(st.session_state.detailed_scene_files):
                col_idx = i % len(scene_cols)
                with scene_cols[col_idx]:
                    if os.path.exists(scene_file):
                        with open(scene_file, "r", encoding="utf-8") as f:
                            scene_content = f.read()
                        
                        chunk_num = i + 1
                        st.download_button(
                            label=f"📥 Chunk {chunk_num} Detailed Scenes",
                            data=scene_content,
                            file_name=f"chunk_{chunk_num}_detailed_scenes.txt",
                            mime="text/plain",
                            use_container_width=True,
                            help=f"Download detailed scene description for PDF chunk {chunk_num}"
                        )
        
        # ╭──────────────────────────────────────────────────────────────╮
        # │ ADDITIONAL DOWNLOADS │
        # ╰──────────────────────────────────────────────────────────────╯
        
        if (hasattr(st.session_state, 'video_scenes') and st.session_state.video_scenes) or (hasattr(st.session_state, 'chunk_data_list') and st.session_state.chunk_data_list):
            st.markdown("#### 📋 Additional Resources")
            
            # Scene descriptions download
            if output_type == "Chunk-Based Synchronized Audio & Video with Detailed Scenes" and hasattr(st.session_state, 'chunk_data_list'):
                # Synchronized scenes with detailed information
                chunk_data_list = st.session_state.chunk_data_list
                scenes_text = "\n\n" + "="*50 + "\n\n".join([
                    f"SYNCHRONIZED CHUNK {chunk_data['chunk_number']} (Duration: {chunk_data['total_duration']:.2f}s):\n"
                    f"Audio Path: {chunk_data['audio_path']}\n"
                    f"SRT Path: {chunk_data['srt_path']}\n"
                    f"Script: {chunk_data['script'][:200]}...\n"
                    f"Transcript: {chunk_data['full_transcript'][:200]}...\n"
                    f"Note: Detailed scene descriptions were generated and processed by manimator.py"
                    for chunk_data in chunk_data_list
                ])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download Chunk Information",
                        data=scenes_text,
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_chunk_info.txt",
                        mime="text/plain",
                        use_container_width=True,
                        help="Download detailed chunk information with audio and subtitle paths"
                    )
            elif hasattr(st.session_state, 'video_scenes') and st.session_state.video_scenes:
                # Traditional scenes
                video_scenes = st.session_state.video_scenes
                scenes_text = "\n\n" + "="*50 + "\n\n".join([
                    f"SCENE {i+1}:\n{scene}"
                    for i, scene in enumerate(video_scenes) if scene.strip()
                ])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download Scene Descriptions",
                        data=scenes_text,
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_scenes.txt",
                        mime="text/plain",
                        use_container_width=True,
                        help="Download detailed scene descriptions for reference"
                    )
            
            # Individual teaching scripts download if available
            if teaching_scripts and len(teaching_scripts) > 1:
                with col2:
                    individual_scripts = "\n\n" + "="*50 + "\n\n".join([
                        f"CHUNK {i+1} SCRIPT:\n{script}"
                        for i, script in enumerate(teaching_scripts) if script.strip()
                    ])
                    
                    st.download_button(
                        label="📥 Download Individual Scripts",
                        data=individual_scripts,
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_individual_scripts.txt",
                        mime="text/plain",
                        use_container_width=True,
                        help="Download individual chunk teaching scripts"
                    )

# ╭──────────────────────────────────────────────────────────────╮
# │ FOOTER SECTION │
# ╰──────────────────────────────────────────────────────────────╯

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; padding: 20px;">
        <h4>🧑🏫 PDF Seamless Teacher</h4>
        <p><strong>Powered by Gemma 3, F5-TTS, and Manimator</strong></p>
        <p>Transform your educational content with AI-powered voice cloning and synchronized mathematical animations</p>
        <p><em>Now featuring on-demand scene generation and chunk-based synchronized video generation with detailed scene descriptions and manimator integration</em></p>
    </div>
    """,
    unsafe_allow_html=True
)

# ╭──────────────────────────────────────────────────────────────╮
# │ NEW VIDEO GENERATION WORKFLOW INTEGRATION                   │
# ╰──────────────────────────────────────────────────────────────╯
# Replace old video generation logic with new workflow
from utils import new_video_generation_workflow

def run_video_generation_from_subtitles(subtitles, gemma_api_keys):
    """
    Run the new video generation workflow based on subtitles and Gemma API keys.
    Returns list of final video file paths.
    """
    return new_video_generation_workflow(subtitles, gemma_api_keys)