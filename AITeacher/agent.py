#!/usr/bin/env python3
"""
Manim-OpenCode Agent (Fully AI-Powered)
───────────────────────────────────────
1. Generates ManimCE code with Google GenAI (Gemma or Gemini).
2. Renders using: manim -pql filename.py ClassName
3. If rendering fails, uses `opencode` to fix the code.
4. Repeats until success or max attempts reached.
"""

import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, List

import google.genai as genai
from google.genai import types

# ── User-tweakable constants ──────────────────────────────────────
MODEL_NAME = os.getenv("GENAI_MODEL", "gemma-3-27b-it")
MAX_ATTEMPTS = 5
MANIM_TIMEOUT = int(os.getenv("MANIM_TIMEOUT", "120"))  # seconds
# ─────────────────────────────────────────────────────────────────

# ╭──────────────────────── GEMINI HELPERS ──────────────────────╮
def _clean(code: str) -> str:
    """Strip accidental markdown code fences and leading/trailing spaces."""
    code = re.sub(r"^```(?:python)?\s*", "", code.strip(), flags=re.IGNORECASE | re.MULTILINE)
    code = re.sub(r"\s*```$", "", code.strip(), flags=re.MULTILINE)
    return code.strip()

def generate_code(client: genai.Client, prompt: str) -> str:
    """Ask Gemma for raw Manim code and stream it live — retrying infinitely on failure."""
    instruction = (
        "You are an expert ManimCE animator.\n"
        "Return ONLY valid Python3 code (no markdown, no commentary)\n\n"
        f"Prompt:\n{prompt}"
    )
    contents = [types.Content(role="user", parts=[types.Part(text=instruction)])]
    cfg = types.GenerateContentConfig(response_mime_type="text/plain")

    print("\n" + "=" * 50)
    print("🚀 STARTING CODE GENERATION")
    print("=" * 50)

    delay = 1  # start with 1 second delay
    attempt = 0

    while True:
        attempt += 1
        try:
            generated = ""
            for chunk in client.models.generate_content_stream(
                model=MODEL_NAME, contents=contents, config=cfg
            ):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    generated += chunk.text
            print("\n" + "=" * 50)
            print("✅ GENERATION COMPLETE!")
            return _clean(generated)
        except Exception as e:
            print(f"\n⚠️ Generation failed (attempt {attempt}): {str(e)}")
            if "Connection reset by peer" in str(e):
                print(f"💤 Retrying in {delay} seconds...")
                time.sleep(delay)
                delay = min(delay * 2, 60)  # exponential backoff, max 60s
            else:
                print("❌ Unexpected error. Aborting.")
                return ""
# ╰────────────────────────────────────────────────────────────────╯

# ╭───────────────────────── MANIM RUNNER ───────────────────────╮
def run_manim(scene_path: Path, class_name: str) -> subprocess.CompletedProcess:
    cmd = ["manim", "-pql", str(scene_path), class_name]
    print("\n🚀 Executing:", " ".join(shlex.quote(c) for c in cmd))
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=MANIM_TIMEOUT
    )
# ╰────────────────────────────────────────────────────────────────╯

# ╭───────────────── OPENCODE-BASED AUTO-FIXER ──────────────────╮
def opencode_fix(py_file: Path) -> Optional[str]:
    """
    Run `opencode` with GitHub Copilot GPT-4.1 model to fix the Manim file automatically.
    Returns fixed code or None if no change.
    """
    print(f"\n🤖 Running OpenCode with GitHub Copilot to fix {py_file.name}...")

    # Enhanced prompt that references the file instead of including code
    fix_prompt = (
        f"Fix the ManimCE Python file at {py_file.resolve()} so it runs without errors. "
        "Common issues include: import errors, incorrect method calls, "
        "deprecated Manim syntax, missing Scene inheritance, and animation timing issues. "
        "Return only the corrected Python code with no explanations, comments, or markdown formatting."
    )

    # Use the working OpenCode command structure
    cmd = [
        "opencode",
        "run",
        "--model", "github-copilot/gpt-4.1",
        fix_prompt
    ]

    print("🤖 Running command:", " ".join(shlex.quote(c) for c in cmd))

    try:
        # Run OpenCode with the specified model
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            check=False
        )

        if result.returncode != 0:
            print("❌ OpenCode exited with error")
            print("STDERR:\n", result.stderr)
            return None

        print("✅ OpenCode finished successfully.")

        # Extract the fixed code from OpenCode output
        fixed_code = result.stdout.strip()
        
        # Clean any potential markdown formatting
        fixed_code = _clean(fixed_code)
        
        if fixed_code:
            original_code = py_file.read_text()
            if fixed_code != original_code:
                print("✅ OpenCode provided an updated version.")
                # Write the fixed code back to the original file
                py_file.write_text(fixed_code)
                return fixed_code
            else:
                print("⚠️ OpenCode returned the same code.")
                return None
        else:
            print("⚠️ OpenCode returned empty response.")
            return None

    except subprocess.TimeoutExpired:
        print("🚨 OpenCode timed out after 300 seconds")
        return None
    except Exception as e:
        print(f"🚨 Error running OpenCode: {e}")
        return None

# ╰────────────────────────────────────────────────────────────────╯

# ╭────────────────── RENDER / FIX LOOP ─────────────────────────╮
def run_manim_code_until_success(code: str, class_name: str) -> str:
    attempt = 0
    while attempt < MAX_ATTEMPTS:
        attempt += 1
        print(f"\n🔄 ATTEMPT {attempt}/{MAX_ATTEMPTS}")

        # Save scene.py in the current working directory
        scene_file = Path("scene.py")
        scene_file.write_text(code)
        print(f"📄 Saved Manim code to: {scene_file.resolve()}")

        result = run_manim(scene_file, class_name)

        if result.returncode == 0:
            print("\n✅ SUCCESS! Video rendered successfully.")
            return "✅ Success!"

        print("\n❌ MANIM FAILED – analyzing stderr …")
        print("-" * 40)
        print(result.stderr[:800])
        print("-" * 40)

        print("🔁 Launching OpenCode to fix the file...")
        fixed_code = opencode_fix(scene_file)
        if fixed_code:
            code = fixed_code
        else:
            break  # No useful fix was applied

    return "❌ Failed after multiple attempts."
# ╰────────────────────────────────────────────────────────────────╯

# ╭─────────────────── INTERACTIVE CHAT LOOP ────────────────────╮
def run_chat_session(client: genai.Client):
    print("=" * 60)
    print("🎨 MANIM CODE AGENT WITH DETAILED PROGRESS")
    print("=" * 60)
    print(f"⏰ Session started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n📋 INSTRUCTIONS:")
    print("• Paste your Manim prompt below")
    print("• Type 'END' on a new line to execute")
    print("• Type 'status' for session statistics")
    print("• Type 'exit' to quit")
    print("-" * 60)

    stats = dict(processed=0, success=0, failed=0, wall=0.0)
    buffer: List[str] = []

    while True:
        try:
            line = input()
            match line.strip().lower():
                case "exit":
                    print("\n👋 EXITING – SUMMARY")
                    print(f" Prompts processed : {stats['processed']}")
                    print(f" Successful renders: {stats['success']}")
                    print(f" Failed attempts : {stats['failed']}")
                    print(f" Total time : {stats['wall']:.2f}s")
                    return
                case "status":
                    ok = stats['success']
                    total = max(stats['processed'], 1)
                    print(f"\n📊 STATUS – {ok}/{total} succeeded")
                    continue
                case "end":
                    prompt = "\n".join(buffer).strip()
                    buffer.clear()
                    if not prompt:
                        print("⚠️ Empty prompt – please enter something.")
                        continue

                    stats['processed'] += 1
                    t0 = time.time()

                    print(f"\n🧠 Generating Manim code from prompt: {prompt}")
                    code = generate_code(client, prompt)
                    if not code:
                        print("❌ Code generation failed.")
                        stats['failed'] += 1
                        continue

                    # Extract class name
                    class_match = re.search(r"class\s+(\w+)", code)
                    if not class_match:
                        print("❌ Could not find a valid Scene class in generated code.")
                        stats['failed'] += 1
                        continue
                    class_name = class_match.group(1)

                    outcome = run_manim_code_until_success(code, class_name)
                    if outcome.startswith("✅"):
                        stats['success'] += 1
                    else:
                        stats['failed'] += 1
                    stats['wall'] += time.time() - t0
                    print("\n🎯 FINAL RESULT\n" + "=" * 40)
                    print(outcome)
                    print("=" * 40)
                    print("\n🔄 Ready for next prompt")
                case _:
                    buffer.append(line)
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted – type 'exit' to quit or keep typing.")
# ╰────────────────────────────────────────────────────────────────╯

# ╭──────────────────────────── MAIN ────────────────────────────╮
def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set – export it first.")
        return
    client = genai.Client(api_key=api_key)
    print("✅ Google GenAI client initialised.")
    run_chat_session(client)
# ╰────────────────────────────────────────────────────────────────╯

if __name__ == "__main__":
    main()