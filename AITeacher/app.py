#!/usr/bin/env python3
"""
PDF Seamless Teacher - Main Streamlit Application
================================================================
A comprehensive educational tool that converts PDF documents into 
seamless teaching content with AI-generated audio and video.

Features:
- PDF analysis with Gemma 3 via Gemini API
- Voice cloning with F5-TTS
- Automated video generation with Manim
- Interactive Streamlit interface
================================================================
"""

# ╭──────────────────────────────────────────────────────────────╮
# │                        IMPORTS SECTION                       │
# ╰──────────────────────────────────────────────────────────────╯

import streamlit as st
import os
import uuid
import concurrent.futures
import time
import socket
import errno
from pathlib import Path

# Custom utility imports
from utils import (
    is_gemini_available,
    convert_pdf_to_images,
    analyze_page_with_gemma3,
    analyze_page_for_video_scene,
    chunk_text,
    call_f5_tts,
    concatenate_audios_with_crossfade,
    get_all_voice_samples,
    cleanup_files,
    generate_manim_video,
    _check_manim_setup,
    is_connection_error,
    retry_with_backoff
)

# ╭──────────────────────────────────────────────────────────────╮
# │                    CONFIGURATION CONSTANTS                   │
# ╰──────────────────────────────────────────────────────────────╯

# Network Retry Configuration for Streamlit-specific operations
STREAMLIT_RETRY_DELAY_BASE = 1  # Base delay in seconds
STREAMLIT_RETRY_DELAY_MAX = 30  # Maximum delay in seconds
STREAMLIT_RETRY_BACKOFF_MULTIPLIER = 2  # Exponential backoff multiplier

# ╭──────────────────────────────────────────────────────────────╮
# │                   STREAMLIT RETRY UTILITIES                  │
# ╰──────────────────────────────────────────────────────────────╯

def streamlit_retry_with_backoff(func, *args, **kwargs):
    """
    Execute a function with unlimited retries and exponential backoff for connection errors.
    Streamlit-specific version with status updates.
    
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
                
                # Show connection error in Streamlit
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
# │                    PAGE CONFIGURATION                        │
# ╰──────────────────────────────────────────────────────────────╯

st.set_page_config(
    page_title="PDF Seamless Teacher", 
    page_icon="🧑‍🏫", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ╭──────────────────────────────────────────────────────────────╮
# │                      HEADER SECTION                          │
# ╰──────────────────────────────────────────────────────────────╯

st.title("🧑‍🏫 PDF Seamless Teacher (Gemma 3 + F5-TTS + Manim)")
st.markdown(
    """
    **Transform your PDF documents into engaging educational content!**
    
    Upload a PDF, choose a voice, and generate a seamless teaching audio track 
    with optional animated video using advanced voice cloning and Manim animations.
    """
)

# ╭──────────────────────────────────────────────────────────────╮
# │                 SYSTEM CONNECTIVITY CHECK                    │
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

# ╭──────────────────────────────────────────────────────────────╮
# │                    PDF UPLOAD SECTION                        │
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
# │                  PDF PROCESSING SECTION                      │
# ╰──────────────────────────────────────────────────────────────╯

if uploaded_file is not None:
    # Check if we need to reprocess the PDF
    if ('seamless_transcript' not in st.session_state or 
        'video_scenes' not in st.session_state or 
        st.session_state.get('pdf_name') != uploaded_file.name):
        
        st.markdown("#### 🔄 Processing PDF...")
        
        with st.spinner("Analyzing PDF and generating teaching script and video scenes..."):
            def process_pdf():
                # Convert PDF to images
                pdf_bytes = uploaded_file.read()
                page_images = convert_pdf_to_images(pdf_bytes)
                
                # Initialize storage for results
                all_transcripts = []
                all_video_scenes = []
                
                # Process each page
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, img in enumerate(page_images):
                    # Update progress
                    progress = (idx + 1) / len(page_images)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing page {idx + 1} of {len(page_images)}...")
                    
                    # Generate teaching transcript with retry logic
                    def generate_transcript():
                        return analyze_page_with_gemma3(img)
                    
                    transcript = safe_streamlit_operation(
                        f"Teaching Transcript Generation (Page {idx + 1})",
                        generate_transcript
                    )
                    all_transcripts.append(transcript)
                    
                    # Generate video scene description with retry logic
                    def generate_scene():
                        return analyze_page_for_video_scene(img)
                    
                    scene_desc = safe_streamlit_operation(
                        f"Video Scene Generation (Page {idx + 1})",
                        generate_scene
                    )
                    all_video_scenes.append(scene_desc)
                
                # Store results in session state
                st.session_state.seamless_transcript = ' '.join(all_transcripts)
                st.session_state.video_scenes = all_video_scenes
                st.session_state.pdf_name = uploaded_file.name
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                return True
            
            try:
                safe_streamlit_operation("PDF Processing", process_pdf)
            except Exception as e:
                st.error(f"❌ **Failed to process the PDF**\n\nError: {str(e)}", icon="📄")
                st.stop()
    
    # Retrieve processed data
    seamless_transcript = st.session_state.seamless_transcript
    video_scenes = st.session_state.video_scenes
    
    # Success message
    st.success(
        f"✅ **Successfully analyzed '{uploaded_file.name}'**\n\n"
        f"Generated {len(video_scenes)} scene descriptions and complete teaching transcript.",
        icon="✅"
    )
    
    # ╭──────────────────────────────────────────────────────────────╮
    # │                   CONTENT PREVIEW SECTION                    │
    # ╰──────────────────────────────────────────────────────────────╯
    
    st.markdown("#### 👀 Content Preview")
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("📝 View Full Teaching Transcript", expanded=False):
            st.markdown("**Generated Teaching Script:**")
            st.write(seamless_transcript)
            st.markdown(f"**Word Count:** {len(seamless_transcript.split())} words")
    
    with col2:
        with st.expander("🎬 View Video Scene Descriptions", expanded=False):
            st.markdown("**Generated Scene Descriptions:**")
            for i, scene in enumerate(video_scenes, 1):
                st.markdown(f"**Scene {i}:**")
                st.write(scene)
                if i < len(video_scenes):
                    st.markdown("---")

    # ╭──────────────────────────────────────────────────────────────╮
    # │                  OUTPUT TYPE SELECTION                       │
    # ╰──────────────────────────────────────────────────────────────╯
    
    st.markdown("---")
    st.markdown("### 🎯 Step 2: Choose Output Type")
    
    output_type = st.radio(
        "What would you like to generate?",
        options=["Audio Only", "Video Only", "Both Audio and Video"],
        index=0,
        help="Choose whether to generate just audio, just video, or both",
        horizontal=True
    )

    # ╭──────────────────────────────────────────────────────────────╮
    # │                   VOICE SELECTION SECTION                    │
    # ╰──────────────────────────────────────────────────────────────╯

    voice_sample_path = None
    if output_type in ["Audio Only", "Both Audio and Video"]:
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
    # │                   MANIM SETUP SECTION                        │
    # ╰──────────────────────────────────────────────────────────────╯

    manim_ready = False
    if output_type in ["Video Only", "Both Audio and Video"]:
        st.markdown("---")
        st.markdown("### 🎬 Step 4: Manim Video Setup Verification")
        
        def check_manim():
            _check_manim_setup()
            return True
        
        try:
            safe_streamlit_operation("Manim Setup Verification", check_manim)
            manim_ready = True
            st.success(
                "✅ **Manim setup verified successfully**\n\n"
                "All required dependencies are available for video generation.",
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
    # │                  CONTENT GENERATION SECTION                  │
    # ╰──────────────────────────────────────────────────────────────╯

    st.markdown("---")
    st.markdown("### 🚀 Step 5: Generate Content")
    
    # Dynamic button text based on output type
    generate_button_text = {
        "Audio Only": "🎧 Generate Audio",
        "Video Only": "🎬 Generate Video", 
        "Both Audio and Video": "🎬🎧 Generate Audio & Video"
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

        # ╭──────────────────────────────────────────────────────────────╮
        # │                    AUDIO GENERATION                          │
        # ╰──────────────────────────────────────────────────────────────╯

        if output_type in ["Audio Only", "Both Audio and Video"]:
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
                            return call_f5_tts(chunk, f"page_{idx}", voice_path)
                        
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
        # │                    VIDEO GENERATION                          │
        # ╰──────────────────────────────────────────────────────────────╯

        if output_type in ["Video Only", "Both Audio and Video"] and manim_ready:
            st.markdown("#### 🎬 Video Generation Process")
            
            with st.spinner("Generating Manim video scenes... This process can take several minutes per scene."):
                def generate_video():
                    st.info(f"🎭 Processing {len(video_scenes)} video scenes...")
                    
                    def video_operation():
                        return generate_manim_video(video_scenes)
                    
                    final_video_path = streamlit_retry_with_backoff(video_operation)
                    
                    if final_video_path:
                        st.success(
                            "✅ **Video generated successfully!**\n\n"
                            f"All {len(video_scenes)} scenes have been rendered and combined.",
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
        # │                    RESULTS SECTION                           │
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
        # │                 ADDITIONAL DOWNLOADS                         │
        # ╰──────────────────────────────────────────────────────────────╯

        if video_scenes:
            st.markdown("#### 📋 Additional Resources")
            
            # Scene descriptions download
            scenes_text = "\n\n" + "="*50 + "\n\n".join([
                f"SCENE {i+1}:\n{scene}" for i, scene in enumerate(video_scenes)
            ])
            
            st.download_button(
                label="📥 Download Scene Descriptions",
                data=scenes_text,
                file_name=f"{os.path.splitext(uploaded_file.name)[0]}_scenes.txt",
                mime="text/plain",
                use_container_width=True,
                help="Download detailed scene descriptions for reference"
            )

# ╭──────────────────────────────────────────────────────────────╮
# │                      FOOTER SECTION                          │
# ╰──────────────────────────────────────────────────────────────╯

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        <p>🧑‍🏫 <strong>PDF Seamless Teacher</strong> - Powered by Gemma 3, F5-TTS, and Manim</p>
        <p>Transform your educational content with AI-powered voice cloning and mathematical animations</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ╭──────────────────────────────────────────────────────────────╮
# │                    END OF APPLICATION                        │
# ╰──────────────────────────────────────────────────────────────╯
