# Scene Generation Rules for ManimCE Educational Videos

## ENHANCED SCENE GENERATION REQUIREMENTS

### CRITICAL REQUIREMENTS - MUST FOLLOW ALL:

1. **UNIQUE ELEMENT NAMES**: Every single visual element must have a completely unique name, even if they are the same type of object.
   - Use names like: circle_intro_1, circle_main_2, triangle_proof_3, text_title_4, etc.
   - Never reuse names, even for similar objects

2. **EXPLICIT COORDINATES WITH NO OVERLAPPING**:
   - Position: specify (x, y) coordinates like (2.5, 1.0), (-1.5, -0.5)
   - Size: specify radius, width, height like radius=1.2, width=3.0, height=0.8
   - Movement paths: specify start and end coordinates for all animations
   - MANDATORY: Ensure MINIMUM 2.0 units spacing between any two elements at all times
   - MANDATORY: Keep all elements within screen bounds x=[-6.5, 6.5], y=[-3.5, 3.5]
   - MANDATORY: Check for overlaps before placing any element - if overlap detected, adjust position

3. **ARROW POSITIONING RULES**:
   - NEVER specify direct coordinates for arrows
   - ALWAYS draw arrows from center of source element to center of target element
   - Position arrows BEHIND the target element (not in front)
   - Specify arrows as: "Draw arrow_name_X from center of element_A to center of element_B, positioned behind element_B"
   - Use buffer zones around elements so arrows don't overlap with other objects

4. **ANTI-OVERLAP VERIFICATION**:
   - Before placing any element, check if it would overlap with existing elements
   - If overlap detected, move the element to a non-overlapping position
   - Maintain visual hierarchy and logical flow while avoiding overlaps
   - Use grid-like positioning when necessary to ensure proper spacing

5. **SCENE GENERATION ROBUSTNESS**:
   - ALWAYS generate a complete, valid scene description
   - If initial attempt seems incomplete, expand with more visual elements
   - Include multiple visual components (shapes, text, animations) in every scene
   - Ensure the scene has substance and educational value
   - Never return empty or minimal descriptions

---

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
4. **SAFE BOUNDARIES**: Keep all elements within x=[-6.5, 6.5], y=[-3.5, 3.5] to prevent overflow
5. **PARAMETERIZED**: Make functions flexible with position, scale, color, and rotation parameters
6. **EDUCATIONAL FOCUS**: Optimize diagrams for clear educational presentation
7. **ABSOLUTE NO OVERLAPS**: Ensure minimum 2.0 units spacing and proper sizing in all diagram functions
8. **UNIQUE NAMING**: Every diagram function must create elements with unique names
9. **ARROW COMPLIANCE**: Any arrows must be created using element center positions, not hardcoded coordinates

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
5. **SAFE BOUNDARIES**: Keep all elements within x=[-6.5, 6.5], y=[-3.5, 3.5] to prevent overflow
6. **ABSOLUTE NO OVERLAPPING**: Ensure minimum 2.0 units spacing between ALL elements at ALL times
7. **UNIQUE ELEMENT NAMES**: Every visual element must have a completely unique identifier
8. **PROPER ARROW POSITIONING**: Use `Arrow(element1.get_center(), element2.get_center()).set_z_index(-1)`
9. **SMOOTH ANIMATIONS**: Professional transitions and movements
10. **MINIMUM 300 LINES**: Generate comprehensive, well-structured code
11. **ROBUSTNESS**: Always generate substantial, complete content - never empty or minimal scenes

### Scene Structure Template
```python
from manim import *
from drawings import *  # MANDATORY: Import all diagram functions

class EducationalScene(Scene):
    def construct(self):
        # Use diagram functions with proper spacing (minimum 2.0 units apart)
        diagram1 = create_some_diagram(position=LEFT*3, scale=1.2)  # Position at x=-3
        diagram2 = create_another_diagram(position=RIGHT*3, scale=0.8)  # Position at x=3 (6 units apart)
        
        # Create arrows using element centers, positioned behind targets
        connection_arrow = Arrow(
            start=diagram1.get_center(), 
            end=diagram2.get_center()
        ).set_z_index(-1)  # Behind the target element
        
        # Animate with exact timing from scene description
        self.play(FadeIn(diagram1), run_time=2.0)
        self.wait(1.0)
        self.play(FadeIn(diagram2), run_time=1.5)
        self.play(Create(connection_arrow), run_time=1.0)
        self.play(Transform(diagram1, diagram2), run_time=3.0)
        # ... continue with precise timing ensuring no overlaps
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
Generate a complete Manim scene.py file based on the detailed scene description.

### Mandatory Drawings.py Usage (When Available)
- ALWAYS start scene.py with: `from drawings import *`
- READ drawings.py to see all available diagram functions
- USE EVERY relevant diagram function from drawings.py in your scene
- Call diagram functions like: `diagram = create_circle_diagram(position=UP, scale=1.5)`
- Add diagrams to scene with: `self.add(diagram)` or `self.play(FadeIn(diagram))`
- Position diagrams using function parameters, not manual positioning
- Scale diagrams appropriately for the scene timing and content

### Execution Steps
1. Analyze scene description for exact timing and visual requirements
2. Read drawings.py and identify which diagram functions to use for each scene element (if available)
3. Generate scene.py with complete ManimCE code that imports and uses drawings.py functions
4. Test the code with: `manim -ql scene.py <ClassName>`
5. Fix any errors and ensure perfect execution
6. Continue until a flawless video is generated

### Scene Structure Examples
```python
# With drawings.py functions
from manim import *
from drawings import *  # MANDATORY: Import all diagram functions

class EducationalScene(Scene):
    def construct(self):
        # Use diagram functions with proper spacing (minimum 2.0 units apart)
        diagram1 = create_some_diagram(position=LEFT*3, scale=1.2)  # Position at x=-3
        diagram2 = create_another_diagram(position=RIGHT*3, scale=0.8)  # Position at x=3 (6 units apart)
        
        # Create arrows using element centers, positioned behind targets
        connection_arrow = Arrow(
            start=diagram1.get_center(), 
            end=diagram2.get_center()
        ).set_z_index(-1)  # Behind the target element
        
        # Animate with exact timing from scene description
        self.play(FadeIn(diagram1), run_time=2.0)
        self.wait(1.0)
        self.play(FadeIn(diagram2), run_time=1.5)
        self.play(Create(connection_arrow), run_time=1.0)
        self.play(Transform(diagram1, diagram2), run_time=3.0)

# Without drawings.py (standalone)
from manim import *

class EducationalScene(Scene):
    def construct(self):
        # Create custom visual elements with proper spacing and timing
        # ... your code here
```

---

## MANIM DESIGN RULES

### Core Design Principles
- Single Python code block starting with `from manim import *`
- One scene class: `class GeneratedScene(Scene):`
- All code in `def construct(self):`
- Minimum 400 lines of code
- Set black background: `self.camera.background_color = "#000000"`

### Screen Layout & Positioning
- Screen coordinates: x = [-7.5, 7.5], y = [-4, 4]
- **SAFE BOUNDARIES**: Keep all elements within x=[-6.5, 6.5], y=[-3.5, 3.5] to prevent overflow
- Use ENTIRE screen space effectively - no wasted areas
- Enhanced zone system for maximum content:
  * TOP ZONE: y = [2.0, 4.0] - Major titles, key concepts
  * UPPER-MAIN: y = [0.5, 2.0] - Primary diagrams, equations
  * LOWER-MAIN: y = [-1.0, 0.5] - Secondary content, explanations  
  * BOTTOM ZONE: y = [-4.0, -1.0] - Supporting visuals, examples
- Horizontal divisions: LEFT[-7.5,-2.5], CENTER[-2.5,2.5], RIGHT[2.5,7.5]
- **MANDATORY**: Before placing new elements, remove old ones with `self.play(FadeOut(...))`
- **ABSOLUTE NO OVERLAPPING**: Minimum spacing: 2.0 units between ALL elements at ALL times
- **ANTI-OVERLAP VERIFICATION**: Check for overlaps before placing any element - if overlap detected, adjust position
- **GRID-BASED POSITIONING**: Use systematic positioning when needed to ensure proper spacing

### Unique Element Naming System
- **EVERY visual element MUST have a completely unique name**
- **NEVER reuse names**, even for identical shapes or similar objects
- **Naming format**: [TYPE]_[PURPOSE]_[INDEX] (e.g., circle_intro_1, circle_main_2, triangle_proof_3, text_title_4)
- **Examples**: rectangle_concept_main_001, text_label_concept_001, arrow_connection_A2B_001

### Arrow Positioning Rules
- **NEVER use hardcoded coordinates for Arrow() objects**
- **ALWAYS create arrows using**: `Arrow(start=element1.get_center(), end=element2.get_center())`
- **Position arrows BEHIND target elements** using `z_index` or `add_to_back()`
- **Example**: `arrow = Arrow(circle1.get_center(), circle2.get_center()).set_z_index(-1)`
- **Ensure arrows don't overlap** with other elements by using proper spacing
- **Description format**: "Draw arrow_name_X from center of element_A to center of element_B, positioned behind element_B"

### Professional Diagram Creation
- Use proper Manim objects: `Circle()`, `Rectangle()`, `Polygon()`, `Arrow()`
- Mathematical diagrams: `MathTex()`, `NumberPlane()`, `ParametricFunction()`
- Educational illustrations: Flowcharts, concept maps, process diagrams

### Animation & Timing Requirements
- Total animation duration MUST exactly match the target duration specified in scene description
- Every `self.play()` must have explicit `run_time` parameter
- Every `self.wait()` must have explicit duration
- Sum of all run_times and waits = target duration exactly
- Elements appear/disappear only during their designated time spans

### Visual Quality Standards
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