# Scene Generation Rules for ManimCE Educational Videos

## CRITICAL FILE HANDLING RULES

### DRAWINGS.PY HANDLING
1. **NEVER DELETE OR REWRITE** existing drawings.py if it already exists
2. **ONLY ADD** missing diagram functions that are needed
3. **PRESERVE ALL** existing function definitions exactly as they are
4. **ANALYZE** scene_segment_1.txt to identify what new diagrams are needed
5. **APPEND NEW FUNCTIONS** to the end of the existing file

### SCENE.PY HANDLING  
1. **COMPLETELY REWRITE** scene.py file from scratch every time
2. **DO NOT PRESERVE** any existing scene.py content
3. **START FRESH** with a brand new file
4. **OVERWRITE COMPLETELY** any existing scene.py

---

## DRAWINGS.PY GENERATION RULES

### Core Task
Analyze the scene description and generate a complete drawings.py file with all required diagram functions.

### Critical Requirements
1. **EXTRACT ALL DIAGRAMS**: Identify every shape, diagram, graph, equation, or visual element mentioned
2. **CREATE FUNCTIONS**: Generate a separate function for each diagram type
3. **PERFECT POSITIONING**: Ensure all diagrams fit within Manim bounds x=[-7.5,7.5], y=[-4,4]
4. **PARAMETERIZED**: Make functions flexible with position, scale, color, and rotation parameters
5. **EDUCATIONAL FOCUS**: Optimize diagrams for clear educational presentation
6. **NO OVERLAPS**: Ensure proper spacing and sizing in all diagram functions

### Function Structure Template
```python
def create_[diagram_name](position=ORIGIN, scale=1.0, color=BLUE, rotation=0):
    """Create [specific diagram description]"""
    diagram_group = VGroup()
    
    # Add all components to the group
    # Example: circle = Circle(radius=1*scale, color=color).move_to(position)
    # diagram_group.add(circle)
    
    return diagram_group
```

### Required Imports
```python
from manim import *
import numpy as np
```

### Diagram Creation Guidelines
- Analyze the scene description and create functions for ALL visual elements mentioned
- Mathematical equations, geometric shapes, graphs, flowcharts, etc.
- Each function should be self-contained and reusable
- Include proper docstrings explaining the diagram's purpose

### Output Requirements
- Generate ONLY the complete drawings.py file code. No explanations.
- The file must contain all diagram functions ready for import in scene.py.

---

## SCENE.PY GENERATION RULES

### Core Task
Generate a complete scene.py file using the custom diagram functions from drawings.py.

### Critical Requirements
1. **MANDATORY IMPORT**: Always start with: `from drawings import *`
2. **USE ALL RELEVANT FUNCTIONS**: Use every appropriate diagram function from drawings.py
3. **EXACT TIMING**: Match the exact time frames specified in the scene description
4. **PERFECT POSITIONING**: All elements within x=[-7.5,7.5], y=[-4,4] bounds
5. **NO OVERLAPPING**: Ensure no visual elements overlap at any time
6. **SMOOTH ANIMATIONS**: Professional transitions and movements
7. **MINIMUM 300 LINES**: Generate comprehensive, well-structured code

### Scene Structure Template
```python
from manim import *
from drawings import *  # MANDATORY: Import all diagram functions

class EducationalScene(Scene):
    def construct(self):
        # Use diagram functions with proper timing
        diagram1 = create_some_diagram(position=LEFT*2, scale=1.2)
        diagram2 = create_another_diagram(position=RIGHT*2, scale=0.8)
        
        # Animate with exact timing from scene description
        self.play(FadeIn(diagram1), run_time=2.0)
        self.wait(1.0)
        self.play(Transform(diagram1, diagram2), run_time=3.0)
        # ... continue with precise timing
```

### Animation Guidelines
- Call diagram functions with appropriate position, scale, color parameters
- Use self.play() for animations with exact run_time values
- Use self.wait() for pauses matching the scene description timing
- Include smooth transitions between different diagram states
- Remove old elements before adding new ones when needed

### Output Requirements
- Generate ONLY the complete scene.py code. No explanations or markdown.
- The code must use drawings.py functions and produce perfect educational video.

---

## ENHANCED SCENE GENERATION RULES

### Core Task
Generate a complete Manim scene.py file based on the detailed scene description in scenedesc.txt.

### Critical Requirements
1. **FOLLOW RULES**: Read and strictly follow all rules in RULES.md for animation design
2. **EXACT TIMING**: The animation must match the exact time frames specified in scenedesc.txt
3. **SCREEN BOUNDS**: All elements must stay within x=[-7.5,7.5], y=[-4,4] 
4. **NO OVERLAPPING**: Absolutely no visual elements should overlap at any time
5. **SMOOTH TRANSITIONS**: Professional transitions between scenes and elements
6. **MINIMUM 300 LINES**: Generate at least 300 lines of well-structured code
7. **EDUCATIONAL FOCUS**: Clear, engaging visuals that enhance learning

### Mandatory Drawings.py Usage (When Available)
- ALWAYS start scene.py with: `from drawings import *`
- READ drawings.py to see all available diagram functions
- USE EVERY relevant diagram function from drawings.py in your scene
- Call diagram functions like: `diagram = create_circle_diagram(position=UP, scale=1.5)`
- Add diagrams to scene with: `self.add(diagram)` or `self.play(FadeIn(diagram))`
- Position diagrams using function parameters, not manual positioning
- Scale diagrams appropriately for the scene timing and content

### Execution Steps
1. Analyze scenedesc.txt for exact timing and visual requirements
2. Read drawings.py and identify which diagram functions to use for each scene element (if available)
3. Generate scene.py with complete ManimCE code that imports and uses drawings.py functions
4. Test the code with: `manim -ql scene.py <ClassName>`
5. Fix any errors and ensure perfect execution
6. Continue until a flawless video is generated

### Scene Structure With Drawings
```python
from manim import *
from drawings import *  # MANDATORY: Import all diagram functions

class YourSceneName(Scene):
    def construct(self):
        # Use diagram functions from drawings.py
        diagram1 = create_some_diagram(position=LEFT, scale=1.0)
        diagram2 = create_another_diagram(position=RIGHT, scale=0.8)
        
        # Add to scene with proper timing
        self.play(FadeIn(diagram1), run_time=2.0)
        self.play(Transform(diagram1, diagram2), run_time=3.0)
        # ... continue with exact timing from scenedesc.txt
```

### Scene Structure Without Drawings
```python
from manim import *

class YourSceneName(Scene):
    def construct(self):
        # Create custom visual elements
        # ... your code here
```

### Output Requirements
- Generate ONLY the complete scene.py code. No explanations or markdown.
- The code must be immediately executable and produce a perfect educational video.

---

## MANIM DESIGN RULES (From RULES.md)

### Code Structure
- Single Python code block starting with `from manim import *`
- One scene class: `class GeneratedScene(Scene):`
- All code in `def construct(self):`
- Minimum 400 lines of code
- Set black background: `self.camera.background_color = "#000000"`

### Full Screen Utilization & No Overlapping
- Screen coordinates: x = [-7.5, 7.5], y = [-4, 4]
- Use ENTIRE screen space effectively - no wasted areas
- Enhanced zone system for maximum content:
  * TOP ZONE: y = [2.0, 4.0] - Major titles, key concepts
  * UPPER-MAIN: y = [0.5, 2.0] - Primary diagrams, equations
  * LOWER-MAIN: y = [-1.0, 0.5] - Secondary content, explanations  
  * BOTTOM ZONE: y = [-4.0, -1.0] - Supporting visuals, examples
- Horizontal divisions: LEFT[-7.5,-2.5], CENTER[-2.5,2.5], RIGHT[2.5,7.5]
- **MANDATORY**: Before placing new elements, remove old ones with `self.play(FadeOut(...))`
- Minimum spacing: 1.2 units between elements

### Professional Diagram Creation
- Use proper Manim objects: `Circle()`, `Rectangle()`, `Polygon()`, `Arrow()`
- Mathematical diagrams: `MathTex()`, `NumberPlane()`, `ParametricFunction()`
- Educational illustrations: Flowcharts, concept maps, process diagrams

### Exact Timing Synchronization
- Total animation duration MUST exactly match the target duration specified in scene description
- Every `self.play()` must have explicit `run_time` parameter
- Every `self.wait()` must have explicit duration
- Sum of all run_times and waits = target duration exactly
- Elements appear/disappear only during their designated time spans

### Visual Quality & Movement
- Colors: `BLUE = "#58C4DD"`, `GOLD = "#F0E68C"`, `WHITE = "#FFFFFF"`
- Clean, minimalist design with dynamic movements
- Rich visual content every second with purposeful motion
- Use variety: animated text, moving shapes, flowing equations, directional arrows

---

## SCENE.PY AUTOMATIC EXECUTION AND ERROR FIXING

### Execution Workflow
After generating scene.py, you MUST follow this exact workflow:

1. **GENERATE** complete scene.py code
2. **EXECUTE** immediately: `manim -ql scene.py ClassName`
3. **ANALYZE** any errors that occur
4. **FIX** the code based on error analysis
5. **REPEAT** steps 2-4 until successful execution
6. **CONTINUE** until a perfect video is generated

### Error Analysis and Fixing
- **Import Errors**: Fix missing imports, check drawings.py functions
- **Syntax Errors**: Fix Python syntax issues
- **Manim Errors**: Fix positioning, timing, or animation issues
- **Runtime Errors**: Fix logic errors, variable issues, or object conflicts

### Success Criteria
- The command `manim -ql scene.py ClassName` must execute without errors
- A video file must be successfully generated
- Only then consider the task complete

---