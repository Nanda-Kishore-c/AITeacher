from manim import *
import numpy as np

# Color constants for Manim
CYAN = "#00FFFF"
BLUE_E = "#1C758A"
GREEN_B = "#236B21"
ORANGE = "#FF9900"
YELLOW = "#FFFF00"
STEEL_BLUE = "#4682B4"
DARK_GRAY = "#222222"
RED = "#FF0000"
GREEN = "#00FF00"
BLUE = "#0000FF"
WHITE = "#FFFFFF"
BLACK = "#000000"


def create_rectangle(
    width,
    height,
    color=WHITE,
    position=(0, 0),
    stroke_color=WHITE,
    stroke_width=2,
    opacity=1.0,
    corner_radius=0,
    name=None,
):
    rect = RoundedRectangle(
        width=width,
        height=height,
        color=color,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        fill_opacity=opacity,
        corner_radius=corner_radius,
    )
    if len(position) == 2:
        pos3d = np.array([position[0], position[1], 0])
    else:
        pos3d = np.array(position)
    rect.move_to(pos3d)
    if name:
        rect.name = name
    return rect


def create_labeled_rectangle(
    text,
    width,
    height,
    color=WHITE,
    position=(0, 0),
    font_size=24,
    text_color=BLACK,
    stroke_color=WHITE,
    stroke_width=2,
    opacity=1.0,
    corner_radius=0,
    name=None,
):
    rect = create_rectangle(
        width=width,
        height=height,
        color=color,
        position=position,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        opacity=opacity,
        corner_radius=corner_radius,
        name=name,
    )
    label = Text(
        text,
        font_size=font_size,
        color=text_color,
    ).move_to(position)
    group = VGroup(rect, label)
    if name:
        group.name = name
    return group


def create_circle(
    radius,
    stroke_color=WHITE,
    fill_color=None,
    fill_opacity=0,
    position=(0, 0),
    stroke_width=2,
    name=None,
):
    circ = Circle(
        radius=radius,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        fill_color=fill_color if fill_color else stroke_color,
        fill_opacity=fill_opacity,
    )
    if len(position) == 2:
        pos3d = np.array([position[0], position[1], 0])
    else:
        pos3d = np.array(position)
    circ.move_to(pos3d)
    if name:
        circ.name = name
    return circ


def create_arrow(
    start,
    end,
    color=WHITE,
    stroke_width=3,
    buff=0.1,
    curve=False,
    name=None,
):
    # Ensure start and end are 3D
    if len(start) == 2:
        start3d = np.array([start[0], start[1], 0])
    else:
        start3d = np.array(start)
    if len(end) == 2:
        end3d = np.array([end[0], end[1], 0])
    else:
        end3d = np.array(end)
    if curve:
        # Draw a curved arrow (arc between start and end)
        arc = ArcBetweenPoints(
            start=start3d,
            end=end3d,
            angle=PI / 2,
        )
        arrow = Arrow(
            arc.point_from_proportion(0),
            arc.point_from_proportion(1),
            color=color,
            stroke_width=stroke_width,
            buff=0,
        )

    else:
        arrow = Arrow(
            start=start3d,
            end=end3d,
            color=color,
            stroke_width=stroke_width,
            buff=buff,
        )
    if name:
        arrow.name = name
    return arrow


def create_image(
    file,
    width=None,
    height=None,
    position=(0, 0),
    name=None,
):
    img = ImageMobject(file)
    if width and height:
        img.stretch_to_fit_width(width)
        img.stretch_to_fit_height(height)
    elif width:
        img.stretch_to_fit_width(width)
    elif height:
        img.stretch_to_fit_height(height)
    if len(position) == 2:
        pos3d = np.array([position[0], position[1], 0])
    else:
        pos3d = np.array(position)
    img.move_to(pos3d)
    if name:
        img.name = name
    return img


def create_labeled_box(
    text,
    width,
    height,
    box_color=WHITE,
    position=(0, 0),
    font_size=24,
    text_color=BLACK,
    stroke_color=WHITE,
    stroke_width=2,
    opacity=1.0,
    corner_radius=0,
    name=None,
):
    return create_labeled_rectangle(
        text=text,
        width=width,
        height=height,
        color=box_color,
        position=position,
        font_size=font_size,
        text_color=text_color,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        opacity=opacity,
        corner_radius=corner_radius,
        name=name,
    )


def create_grid(
    width,
    height,
    x_step,
    y_step,
    grid_color=GRAY,
    grid_stroke_width=1,
    position=(0, 0),
    name=None,
):
    # Create background rectangle
    bg = create_rectangle(
        width=width,
        height=height,
        color=grid_color,
        position=position,
        stroke_color=grid_color,
        stroke_width=0,
        opacity=1.0,
    )
    # Horizontal lines
    y_min = -height / 2
    y_max = height / 2
    x_min = -width / 2
    x_max = width / 2
    h_lines = VGroup()
    y = y_min
    while y <= y_max + 1e-6:
        line = Line(
            start=[x_min, y],
            end=[x_max, y],
            color=grid_color,
            stroke_width=grid_stroke_width,
        )
        h_lines.add(line)
        y += y_step
    # Vertical lines
    v_lines = VGroup()
    x = x_min
    while x <= x_max + 1e-6:
        line = Line(
            start=[x, y_min],
            end=[x, y_max],
            color=grid_color,
            stroke_width=grid_stroke_width,
        )
        v_lines.add(line)
        x += x_step
    group = VGroup(bg, h_lines, v_lines)
    if name:
        group.name = name
    return group


def create_numbered_box(
    number,
    width,
    height,
    box_color=WHITE,
    position=(0, 0),
    font_size=24,
    text_color=BLACK,
    stroke_color=BLACK,
    stroke_width=2,
    opacity=1.0,
    corner_radius=0,
    name=None,
):
    box = create_rectangle(
        width=width,
        height=height,
        color=box_color,
        position=position,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        opacity=opacity,
        corner_radius=corner_radius,
    )
    label = Text(
        str(number),
        font_size=font_size,
        color=text_color,
    ).move_to(position)
    group = VGroup(box, label)
    if name:
        group.name = name
    return group


def create_text(
    text,
    font_size=24,
    color=WHITE,
    position=(0, 0),
    name=None,
):
    txt = Text(
        text,
        font_size=font_size,
        color=color,
    )
    if len(position) == 2:
        pos3d = np.array([position[0], position[1], 0])
    else:
        pos3d = np.array(position)
    txt.move_to(pos3d)
    if name:
        txt.name = name
    return txt


def create_data_block(
    width,
    height,
    color=BLUE,
    position=(0, 0),
    stroke_color=WHITE,
    stroke_width=2,
    opacity=1.0,
    name=None,
):
    block = create_rectangle(
        width=width,
        height=height,
        color=color,
        position=position,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        opacity=opacity,
    )
    if name:
        block.name = name
    return block


def create_grid_lines(
    width,
    height,
    x_step,
    y_step,
    color=GRAY,
    stroke_width=1,
    name=None,
):
    y_min = -height / 2
    y_max = height / 2
    x_min = -width / 2
    x_max = width / 2
    h_lines = VGroup()
    y = y_min
    while y <= y_max + 1e-6:
        line = Line(
            start=[x_min, y],
            end=[x_max, y],
            color=color,
            stroke_width=stroke_width,
        )
        h_lines.add(line)
        y += y_step
    v_lines = VGroup()
    x = x_min
    while x <= x_max + 1e-6:
        line = Line(
            start=[x, y_min],
            end=[x, y_max],
            color=color,
            stroke_width=stroke_width,
        )
        v_lines.add(line)
        x += x_step
    group = VGroup(h_lines, v_lines)
    if name:
        group.name = name
    return group


def create_labeled_rounded_rectangle(
    text,
    width,
    height,
    box_color=WHITE,
    position=(0, 0),
    font_size=24,
    text_color=BLACK,
    stroke_color=WHITE,
    stroke_width=2,
    opacity=1.0,
    corner_radius=0.2,
    name=None,
):
    rect = RoundedRectangle(
        width=width,
        height=height,
        color=box_color,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        fill_opacity=opacity,
        corner_radius=corner_radius,
    )
    if len(position) == 2:
        pos3d = np.array([position[0], position[1], 0])
    else:
        pos3d = np.array(position)
    rect.move_to(pos3d)
    label = Text(
        text,
        font_size=font_size,
        color=text_color,
    ).move_to(position)
    group = VGroup(rect, label)
    if name:
        group.name = name
    return group


def create_labeled_box(
    width=2.0,
    height=1.0,
    color=WHITE,
    label_text="",
    label_font_size=24,
    label_color=WHITE,
    label_offset=0.0,
    position=(0, 0),
    opacity=1.0,
    box_stroke_width=2,
    box_corner_radius=0,
    label_on_top=True,
    name=None,
):
    """
    Creates a rectangle with an optional label above or inside it.
    Returns a VGroup containing the rectangle and the label.
    """
    rect = RoundedRectangle(
        width=width,
        height=height,
        color=color,
        stroke_width=box_stroke_width,
        fill_opacity=opacity,
        corner_radius=box_corner_radius,
    ).move_to(position)
    if name:
        rect.name = name
    if label_text:
        label = Text(label_text, font_size=label_font_size, color=label_color)
        if label_on_top:
            label.next_to(rect, UP, buff=label_offset)
        else:
            label.move_to(rect.get_center())
        if name:
            label.name = f"{name}_label"
        return VGroup(rect, label)
    else:
        return VGroup(rect)


def create_image_box(
    image_file,
    width=1.0,
    height=1.0,
    position=(0, 0),
    name=None,
):
    """
    Creates an ImageMobject with specified size and position.
    """
    img = ImageMobject(image_file)
    img.width = width
    img.height = height
    img.move_to(position)
    if name:
        img.name = name
    return img


def create_labeled_circle(
    radius=1.0,
    stroke_color=WHITE,
    stroke_width=2,
    fill_color=None,
    fill_opacity=0.0,
    label_text="",
    label_font_size=24,
    label_color=WHITE,
    position=(0, 0),
    label_offset=0.0,
    name=None,
):
    """
    Creates a circle with an optional label above or inside it.
    Returns a VGroup containing the circle and the label.
    """
    circle = Circle(
        radius=radius, color=stroke_color, stroke_width=stroke_width
    ).move_to(position)
    if fill_color is not None:
        circle.set_fill(fill_color, opacity=fill_opacity)
    if name:
        circle.name = name
    if label_text:
        label = Text(label_text, font_size=label_font_size, color=label_color)
        label.next_to(circle, UP, buff=label_offset)
        if name:
            label.name = f"{name}_label"
        return VGroup(circle, label)
    else:
        return VGroup(circle)


def create_flow_arrow(
    start,
    end,
    color=WHITE,
    stroke_width=4,
    buff=0.1,
    dashed=False,
    bidirectional=False,
    name=None,
):
    """
    Creates an Arrow or DoubleArrow from start to end.
    """
    if bidirectional:
        arrow = DoubleArrow(
            start, end, color=color, stroke_width=stroke_width, buff=buff
        )
    else:
        arrow = Arrow(start, end, color=color, stroke_width=stroke_width, buff=buff)
    if dashed:
        arrow = DashedVMobject(arrow)
    if name:
        arrow.name = name
    return arrow


def create_labeled_text(
    text,
    font_size=24,
    color=WHITE,
    position=(0, 0),
    name=None,
):
    """
    Creates a Text mobject at a given position.
    """
    txt = Text(text, font_size=font_size, color=color).move_to(position)
    if name:
        txt.name = name
    return txt


def create_grid(
    grid_size=(4, 4),
    box_size=(0.6, 0.4),
    color=WHITE,
    position=(0, 0),
    box_opacity=1.0,
    box_stroke_width=1,
    box_fill_color=None,
    name=None,
):
    """
    Creates a grid of rectangles (e.g., for memory visualization).
    Returns a VGroup of all rectangles.
    """
    rows, cols = grid_size
    box_w, box_h = box_size
    grid = VGroup()
    for i in range(rows):
        for j in range(cols):
            x = position[0] + (j - (cols - 1) / 2) * box_w
            y = position[1] + ((rows - 1) / 2 - i) * box_h
            rect = Rectangle(
                width=box_w,
                height=box_h,
                color=color,
                stroke_width=box_stroke_width,
                fill_opacity=box_opacity,
            ).move_to((x, y, 0))
            if box_fill_color is not None:
                rect.set_fill(box_fill_color, opacity=box_opacity)
            grid.add(rect)
    if name:
        grid.name = name
    return grid


def create_register_box(
    value_text,
    position=(0, 0),
    width=1.0,
    height=1.0,
    box_color=CYAN,
    text_font_size=16,
    text_color=WHITE,
    name=None,
):
    """
    Creates a register box with a value label above it.
    """
    box = Rectangle(width=width, height=height, color=box_color).move_to(position)
    label = Text(value_text, font_size=text_font_size, color=text_color).move_to(
        [position[0], position[1] + height / 2 + 0.2, 0]
    )
    if name:
        box.name = name
        label.name = f"{name}_label"
    return VGroup(box, label)


def create_data_particle(
    radius=0.15,
    color=YELLOW,
    position=(0, 0),
    name=None,
):
    """
    Creates a small circle representing a data particle.
    """
    circ = Circle(radius=radius, color=color, fill_opacity=1.0).move_to(position)
    if name:
        circ.name = name
    return circ


def create_access_point(
    radius=0.1,
    color=GREEN,
    position=(0, 0),
    name=None,
):
    """
    Creates a small circle representing a memory access point.
    """
    circ = Circle(radius=radius, color=color, fill_opacity=1.0).move_to(position)
    if name:
        circ.name = name
    return circ


def create_access_highlight(
    width=0.6,
    height=0.4,
    color=ORANGE,
    position=(0, 0),
    opacity=0.5,
    name=None,
):
    """
    Creates a semi-transparent rectangle to highlight a memory cell.
    """
    rect = Rectangle(
        width=width, height=height, color=color, fill_opacity=opacity
    ).move_to(position)
    if name:
        rect.name = name
    return rect


def create_address_label(
    address_text,
    font_size=16,
    color=WHITE,
    position=(0, 0),
    name=None,
):
    """
    Creates a text label for a memory address.
    """
    txt = Text(address_text, font_size=font_size, color=color).move_to(position)
    if name:
        txt.name = name
    return txt


def create_labeled_arrow(
    start,
    end,
    label_text="",
    label_font_size=20,
    label_color=WHITE,
    color=WHITE,
    stroke_width=4,
    buff=0.1,
    dashed=False,
    bidirectional=False,
    label_position="above",
    name=None,
):
    """
    Creates an arrow with an optional label above or below.
    """
    arrow = create_flow_arrow(
        start=start,
        end=end,
        color=color,
        stroke_width=stroke_width,
        buff=buff,
        dashed=dashed,
        bidirectional=bidirectional,
        name=name,
    )
    if label_text:
        label = Text(label_text, font_size=label_font_size, color=label_color)
        if label_position == "above":
            label.next_to(arrow, UP, buff=0.1)
        else:
            label.next_to(arrow, DOWN, buff=0.1)
        return VGroup(arrow, label)
    else:
        return arrow


from manim import *


def create_diagram_rectangle_base_1(
    position=ORIGIN,
    scale=1.0,
    color="#58C4DD",
    width=4.0,
    height=2.5,
    label=None,
    label_font_size=24,
    **kwargs,
):
    """
    Creates a reusable rectangle for Hardware/Software blocks.
    Args:
        position: (x, y) coordinates for placement
        scale: Size multiplier
        color: Stroke/border color
        width: Rectangle width (default 4.0)
        height: Rectangle height (default 2.5)
        label: Optional text label
        label_font_size: Font size for label
        **kwargs: Additional customization
    Returns:
        VGroup: Rectangle (with optional label)
    """
    rect = Rectangle(
        width=width * scale, height=height * scale, stroke_color=color, fill_opacity=0.0
    )
    rect.move_to(position)
    group = VGroup(rect)
    if label:
        text = Text(label, font_size=label_font_size * scale, color="#FFFFFF")
        text.move_to(rect.get_center())
        group.add(text)
    return group


def create_diagram_circle_component_1(
    position=ORIGIN,
    scale=1.0,
    color="#58C4DD",
    radius=0.5,
    label=None,
    label_font_size=24,
    **kwargs,
):
    """
    Creates a reusable circle for Processor or similar components.
    Args:
        position: (x, y) coordinates for placement
        scale: Size multiplier
        color: Stroke/border color
        radius: Circle radius (default 0.5)
        label: Optional text label
        label_font_size: Font size for label
        **kwargs: Additional customization
    Returns:
        VGroup: Circle (with optional label)
    """
    circ = Circle(radius=radius * scale, stroke_color=color, fill_opacity=0.0)
    circ.move_to(position)
    group = VGroup(circ)
    if label:
        text = Text(label, font_size=label_font_size * scale, color="#FFFFFF")
        text.move_to(circ.get_center())
        group.add(text)
    return group


def create_diagram_rectangle_component_1(
    position=ORIGIN,
    scale=1.0,
    color="#58C4DD",
    width=1.0,
    height=0.75,
    label=None,
    label_font_size=18,
    **kwargs,
):
    """
    Creates a reusable rectangle for Memory/Monitor components.
    Args:
        position: (x, y) coordinates for placement
        scale: Size multiplier
        color: Stroke/border color
        width: Rectangle width (default 1.0)
        height: Rectangle height (default 0.75)
        label: Optional text label
        label_font_size: Font size for label
        **kwargs: Additional customization
    Returns:
        VGroup: Rectangle (with optional label)
    """
    rect = Rectangle(
        width=width * scale, height=height * scale, stroke_color=color, fill_opacity=0.0
    )
    rect.move_to(position)
    group = VGroup(rect)
    if label:
        text = Text(label, font_size=label_font_size * scale, color="#FFFFFF")
        text.move_to(rect.get_center())
        group.add(text)
    return group


def create_diagram_arrow_base_1(
    start_point, end_point, width=0.1, color="#58C4DD", scale=1.0, **kwargs
):
    """
    Creates a reusable arrow between two points.
    Args:
        start_point: Arrow start (x, y)
        end_point: Arrow end (x, y)
        width: Arrow shaft width
        color: Arrow color
        scale: Size multiplier
        **kwargs: Additional customization
    Returns:
        Arrow: Manim Arrow object
    """
    return Arrow(
        start=start_point,
        end=end_point,
        buff=0,
        stroke_width=width * scale * 24,
        color=color,
        max_tip_length_to_length_ratio=0.25,
    )


def create_diagram_text_label_base_1(
    position=ORIGIN, text="Label", font_size=24, color="#FFFFFF", scale=1.0, **kwargs
):
    """
    Creates a reusable text label.
    Args:
        position: (x, y) coordinates for placement
        text: Text content
        font_size: Font size
        color: Text color
        scale: Size multiplier
        **kwargs: Additional customization
    Returns:
        Text: Manim Text object
    """
    return Text(text, font_size=font_size * scale, color=color).move_to(position)


def create_diagram_ipo_cycle(
    center=ORIGIN,
    scale=1.0,
    color="#58C4DD",
    arrow_color="#F0E68C",
    label_color="#FFFFFF",
    **kwargs,
):
    """
    Creates a circle with four arrows for the IPO cycle.
    Args:
        center: (x, y) coordinates for placement
        scale: Size multiplier
        color: Circle color
        arrow_color: Arrow color
        label_color: Label color
        **kwargs: Additional customization
    Returns:
        VGroup: IPO cycle diagram
    """
    radius = 1.5 * scale
    circ = Circle(radius=radius, stroke_color=color, fill_opacity=0.0)
    circ.move_to(center)
    directions = [UP, RIGHT, DOWN, LEFT]
    labels = ["Input", "Processing", "Output", "Loop"]
    arrows = VGroup()
    texts = VGroup()
    for i, dir in enumerate(directions):
        start = center + dir * (radius * 0.7)
        end = center + dir * (radius * 1.1)
        arr = Arrow(
            start,
            end,
            buff=0,
            stroke_width=8 * scale,
            color=arrow_color,
            max_tip_length_to_length_ratio=0.25,
        )
        arrows.add(arr)
        txt = Text(labels[i], font_size=24 * scale, color=label_color)
        txt.next_to(arr, dir, buff=0.2 * scale)
        texts.add(txt)
    return VGroup(circ, arrows, texts)


def create_diagram_input_device(position=ORIGIN, scale=1.0, color="#F0E68C", **kwargs):
    """
    Creates a generic input device polygon (keyboard/mouse/scanner).
    Args:
        position: (x, y) coordinates for placement
        scale: Size multiplier
        color: Stroke/border color
        **kwargs: Additional customization
    Returns:
        Polygon: Manim Polygon object
    """
    points = [
        [-1.0, -0.5, 0],
        [1.0, -0.5, 0],
        [1.2, 0.0, 0],
        [0.0, 0.7, 0],
        [-1.2, 0.0, 0],
    ]
    points = [np.array(p) * scale + np.array([*position, 0]) for p in points]
    return Polygon(*points, stroke_color=color, fill_opacity=0.0)


def create_diagram_binary_stream(
    start_point, end_point, color="#F0E68C", width=0.08, scale=1.0, **kwargs
):
    """
    Creates a single line representing a bit of binary code.
    Args:
        start_point: Line start (x, y)
        end_point: Line end (x, y)
        color: Line color
        width: Line width
        scale: Size multiplier
        **kwargs: Additional customization
    Returns:
        Line: Manim Line object
    """
    return Line(
        start=start_point,
        end=end_point,
        stroke_color=color,
        stroke_width=width * scale * 24,
    )


def create_diagram_computer_box(
    position=ORIGIN,
    scale=1.0,
    color="#58C4DD",
    label="Computer",
    label_font_size=24,
    **kwargs,
):
    """
    Creates a rectangle labeled "Computer".
    Args:
        position: (x, y) coordinates for placement
        scale: Size multiplier
        color: Stroke/border color
        label: Text label
        label_font_size: Font size for label
        **kwargs: Additional customization
    Returns:
        VGroup: Rectangle with label
    """
    rect = Rectangle(
        width=3.5 * scale, height=2.0 * scale, stroke_color=color, fill_opacity=0.0
    )
    rect.move_to(position)
    text = Text(label, font_size=label_font_size * scale, color="#FFFFFF")
    text.move_to(rect.get_center())
    return VGroup(rect, text)


def create_cpu_container(
    position=ORIGIN, scale=1.0, color="#3498db", width=10.0, height=6.0, **kwargs
):
    """
    Creates a large CPU container rectangle.
    Args:
        position: (x, y) coordinates for placement
        scale: Size multiplier
        color: Stroke/border color
        width: Rectangle width
        height: Rectangle height
        **kwargs: Additional customization
    Returns:
        Rectangle: Manim Rectangle object
    """
    rect = Rectangle(
        width=width * scale, height=height * scale, stroke_color=color, fill_opacity=0.0
    )
    rect.move_to(position)
    return rect


def create_component_circle(
    position=ORIGIN,
    scale=1.0,
    color="#58C4DD",
    radius=2.0,
    label=None,
    label_font_size=24,
    **kwargs,
):
    """
    Creates a large component circle.
    Args:
        position: (x, y) coordinates for placement
        scale: Size multiplier
        color: Stroke/border color
        radius: Circle radius
        label: Optional text label
        label_font_size: Font size for label
        **kwargs: Additional customization
    Returns:
        VGroup: Circle (with optional label)
    """
    circ = Circle(radius=radius * scale, stroke_color=color, fill_opacity=0.0)
    circ.move_to(position)
    group = VGroup(circ)
    if label:
        text = Text(label, font_size=label_font_size * scale, color="#FFFFFF")
        text.move_to(circ.get_center())
        group.add(text)
    return group


def create_input_arrow(
    start_point, end_point, width=0.5, color="#95a5a6", scale=1.0, **kwargs
):
    """
    Creates a wide input arrow.
    Args:
        start_point: Arrow start (x, y)
        end_point: Arrow end (x, y)
        width: Arrow shaft width
        color: Arrow color
        scale: Size multiplier
        **kwargs: Additional customization
    Returns:
        Arrow: Manim Arrow object
    """
    return Arrow(
        start=start_point,
        end=end_point,
        buff=0,
        stroke_width=width * scale * 24,
        color=color,
        max_tip_length_to_length_ratio=0.25,
    )


def create_alu_equation(
    position=ORIGIN, text="A+B", font_size=30, color="#FFFFFF", scale=1.0, **kwargs
):
    """
    Creates a MathTex equation for ALU.
    Args:
        position: (x, y) coordinates for placement
        text: Math equation string
        font_size: Font size
        color: Text color
        scale: Size multiplier
        **kwargs: Additional customization
    Returns:
        MathTex: Manim MathTex object
    """
    eq = MathTex(text, font_size=font_size * scale, color=color)
    eq.move_to(position)
    return eq


def create_control_unit(
    position=ORIGIN,
    scale=1.0,
    color="#e74c3c",
    radius=1.5,
    label="Control Unit",
    label_font_size=12,
    **kwargs,
):
    """
    Creates a control unit circle with label.
    Args:
        position: (x, y) coordinates for placement
        scale: Size multiplier
        color: Stroke/border color
        radius: Circle radius
        label: Text label
        label_font_size: Font size for label
        **kwargs: Additional customization
    Returns:
        VGroup: Circle with label
    """
    circ = Circle(radius=radius * scale, stroke_color=color, fill_opacity=0.0)
    circ.move_to(position)
    text = Text(label, font_size=label_font_size * scale, color="#FFFFFF")
    text.move_to(circ.get_center())
    return VGroup(circ, text)


def create_alu_unit(
    position=ORIGIN,
    scale=1.0,
    color="#2ecc71",
    width=2.5,
    height=1.5,
    label="ALU",
    label_font_size=12,
    **kwargs,
):
    """
    Creates an ALU rectangle with label.
    Args:
        position: (x, y) coordinates for placement
        scale: Size multiplier
        color: Stroke/border color
        width: Rectangle width
        height: Rectangle height
        label: Text label
        label_font_size: Font size for label
        **kwargs: Additional customization
    Returns:
        VGroup: Rectangle with label
    """
    rect = Rectangle(
        width=width * scale, height=height * scale, stroke_color=color, fill_opacity=0.0
    )
    rect.move_to(position)
    text = Text(label, font_size=label_font_size * scale, color="#FFFFFF")
    text.move_to(rect.get_center())
    return VGroup(rect, text)


def create_register(
    position=ORIGIN,
    scale=1.0,
    color="#f39c12",
    width=0.8,
    height=0.5,
    label="Register",
    label_font_size=10,
    **kwargs,
):
    """
    Creates a register rectangle with label.
    Args:
        position: (x, y) coordinates for placement
        scale: Size multiplier
        color: Stroke/border color
        width: Rectangle width
        height: Rectangle height
        label: Text label
        label_font_size: Font size for label
        **kwargs: Additional customization
    Returns:
        VGroup: Rectangle with label
    """
    rect = Rectangle(
        width=width * scale, height=height * scale, stroke_color=color, fill_opacity=0.0
    )
    rect.move_to(position)
    text = Text(label, font_size=label_font_size * scale, color="#FFFFFF")
    text.move_to(rect.get_center())
    return VGroup(rect, text)


def create_data_arrow(
    start_point, end_point, color="#FFFFFF", width=0.2, scale=1.0, **kwargs
):
    """
    Creates a data arrow.
    Args:
        start_point: Arrow start (x, y)
        end_point: Arrow end (x, y)
        color: Arrow color
        width: Arrow shaft width
        scale: Size multiplier
        **kwargs: Additional customization
    Returns:
        Arrow: Manim Arrow object
    """
    return Arrow(
        start=start_point,
        end=end_point,
        buff=0,
        stroke_width=width * scale * 24,
        color=color,
        max_tip_length_to_length_ratio=0.25,
    )


def create_cpu_block(
    position=ORIGIN, scale=1.0, color="#3498db", width=4.0, height=2.0, **kwargs
):
    """
    Creates a CPU block rectangle.
    Args:
        position: (x, y) coordinates for placement
        scale: Size multiplier
        color: Stroke/border color
        width: Rectangle width
        height: Rectangle height
        **kwargs: Additional customization
    Returns:
        Rectangle: Manim Rectangle object
    """
    rect = Rectangle(
        width=width * scale, height=height * scale, stroke_color=color, fill_opacity=0.0
    )
    rect.move_to(position)
    return rect


def create_register_label(
    position=ORIGIN, text="Register", font_size=18, color="#FFFFFF", scale=1.0, **kwargs
):
    """
    Creates a register label.
    Args:
        position: (x, y) coordinates for placement
        text: Label text
        font_size: Font size
        color: Text color
        scale: Size multiplier
        **kwargs: Additional customization
    Returns:
        Text: Manim Text object
    """
    return Text(text, font_size=font_size * scale, color=color).move_to(position)


def create_data_flow_arrow(
    start_point, end_point, width=0.5, color="#2ecc71", scale=1.0, **kwargs
):
    """
    Creates a data flow arrow.
    Args:
        start_point: Arrow start (x, y)
        end_point: Arrow end (x, y)
        width: Arrow shaft width
        color: Arrow color
        scale: Size multiplier
        **kwargs: Additional customization
    Returns:
        Arrow: Manim Arrow object
    """
    return Arrow(
        start=start_point,
        end=end_point,
        buff=0,
        stroke_width=width * scale * 24,
        color=color,
        max_tip_length_to_length_ratio=0.25,
    )


def create_memory_unit(
    position=ORIGIN,
    scale=1.0,
    color="#9b59b6",
    width=4.0,
    height=2.0,
    label="Memory",
    label_font_size=18,
    **kwargs,
):
    """
    Creates a memory unit rectangle with label.
    Args:
        position: (x, y) coordinates for placement
        scale: Size multiplier
        color: Stroke/border color
        width: Rectangle width
        height: Rectangle height
        label: Text label
        label_font_size: Font size for label
        **kwargs: Additional customization
    Returns:
        VGroup: Rectangle with label
    """
    rect = Rectangle(
        width=width * scale, height=height * scale, stroke_color=color, fill_opacity=0.0
    )
    rect.move_to(position)
    text = Text(label, font_size=label_font_size * scale, color="#FFFFFF")
    text.move_to(rect.get_center())
    return VGroup(rect, text)


def create_memory_unit_label(
    position=ORIGIN, text="Memory", font_size=18, color="#FFFFFF", scale=1.0, **kwargs
):
    """
    Creates a memory unit label.
    Args:
        position: (x, y) coordinates for placement
        text: Label text
        font_size: Font size
        color: Text color
        scale: Size multiplier
        **kwargs: Additional customization
    Returns:
        Text: Manim Text object
    """
    return Text(text, font_size=font_size * scale, color=color).move_to(position)


def create_storage_location(
    position=ORIGIN, scale=1.0, color="#f39c12", width=0.8, height=0.8, **kwargs
):
    """
    Creates a storage location rectangle.
    Args:
        position: (x, y) coordinates for placement
        scale: Size multiplier
        color: Stroke/border color
        width: Rectangle width
        height: Rectangle height
        **kwargs: Additional customization
    Returns:
        Rectangle: Manim Rectangle object
    """
    rect = Rectangle(
        width=width * scale, height=height * scale, stroke_color=color, fill_opacity=0.0
    )
    rect.move_to(position)
    return rect


def create_address_label(
    position=ORIGIN, text="Addr", font_size=12, color="#FFFFFF", scale=1.0, **kwargs
):
    """
    Creates an address label.
    Args:
        position: (x, y) coordinates for placement
        text: Label text
        font_size: Font size
        color: Text color
        scale: Size multiplier
        **kwargs: Additional customization
    Returns:
        Text: Manim Text object
    """
    return Text(text, font_size=font_size * scale, color=color).move_to(position)


def create_address_value(
    position=ORIGIN, text="0x00", font_size=12, color="#FFFFFF", scale=1.0, **kwargs
):
    """
    Creates an address value MathTex.
    Args:
        position: (x, y) coordinates for placement
        text: Math string
        font_size: Font size
        color: Text color
        scale: Size multiplier
        **kwargs: Additional customization
    Returns:
        MathTex: Manim MathTex object
    """
    return MathTex(text, font_size=font_size * scale, color=color).move_to(position)


def create_access_arrow(
    start_point, end_point, width=0.5, color="#e67e22", scale=1.0, **kwargs
):
    """
    Creates an access arrow.
    Args:
        start_point: Arrow start (x, y)
        end_point: Arrow end (x, y)
        width: Arrow shaft width
        color: Arrow color
        scale: Size multiplier
        **kwargs: Additional customization
    Returns:
        Arrow: Manim Arrow object
    """
    return Arrow(
        start=start_point,
        end=end_point,
        buff=0,
        stroke_width=width * scale * 24,
        color=color,
        max_tip_length_to_length_ratio=0.25,
    )


def draw_arrow(start_point, end_point, width=0.2, color="#58C4DD", scale=1.0, **kwargs):
    """
    Draws an arrow between two points with specified width.
    Args:
        start_point: Arrow start (x, y)
        end_point: Arrow end (x, y)
        width: Arrow shaft width
        color: Arrow color
        scale: Size multiplier
        **kwargs: Additional customization
    Returns:
        Arrow: Manim Arrow object
    """
    return Arrow(
        start=start_point,
        end=end_point,
        buff=0,
        stroke_width=width * scale * 24,
        color=color,
        max_tip_length_to_length_ratio=0.25,
    )


def rotating_line(
    center_point=ORIGIN,
    radius=1.0,
    angle_speed=PI / 2,
    color="#F0E68C",
    width=0.12,
    scale=1.0,
    **kwargs,
):
    """
    Creates a line rotating around a center point at a given speed.
    Args:
        center_point: Center (x, y)
        radius: Length of the line
        angle_speed: Angular speed (radians/sec)
        color: Line color
        width: Line width
        scale: Size multiplier
        **kwargs: Additional customization
    Returns:
        Line: Manim Line object (to be animated in scene)
    """
    start = np.array(center_point)
    end = start + np.array([radius * scale, 0, 0])
    return Line(start, end, stroke_color=color, stroke_width=width * scale * 24)


def flowing_text(
    start_point,
    end_point,
    text_content="Flow",
    font_size=24,
    color="#FFFFFF",
    scale=1.0,
    **kwargs,
):
    """
    Animates text flowing along a path.
    Args:
        start_point: Path start (x, y)
        end_point: Path end (x, y)
        text_content: Text to animate
        font_size: Font size
        color: Text color
        scale: Size multiplier
        **kwargs: Additional customization
    Returns:
        VGroup: Text and path (for animation)
    """
    path = Line(start_point, end_point, stroke_opacity=0.0)
    text = Text(text_content, font_size=font_size * scale, color=color)
    text.move_to(start_point)
    return VGroup(path, text)


def create_diagram_cpu(
    position=ORIGIN, scale=1.0, color=GREEN, width=3.0, height=2.0, **kwargs
):
    """
    Creates a CPU rectangle.
    Args:
        position: (x, y) coordinates for placement
        scale: Size multiplier
        color: Stroke/border color
        width: Rectangle width
        height: Rectangle height
        **kwargs: Additional customization
    Returns:
        Rectangle: Manim Rectangle object
    """
    rect = Rectangle(
        width=width * scale, height=height * scale, stroke_color=color, fill_opacity=0.0
    )
    rect.move_to(position)
    return rect


def create_diagram_cycle(position=ORIGIN, scale=1.0, color=BLUE, radius=1.5, **kwargs):
    """
    Creates a cycle polygon (e.g., for process cycles).
    Args:
        position: (x, y) coordinates for placement
        scale: Size multiplier
        color: Stroke/border color
        radius: Polygon radius
        **kwargs: Additional customization
    Returns:
        Polygon: Manim Polygon object
    """
    points = [
        position + np.array([np.cos(a) * radius * scale, np.sin(a) * radius * scale, 0])
        for a in np.linspace(0, 2 * PI, 6, endpoint=False)
    ]
    return Polygon(*points, stroke_color=color, fill_opacity=0.0)


def create_diagram_arrow(
    position=ORIGIN,
    scale=1.0,
    color=YELLOW,
    width=0.3,
    height=0.3,
    direction=RIGHT,
    **kwargs,
):
    """
    Creates a small diagram arrow.
    Args:
        position: (x, y) coordinates for placement
        scale: Size multiplier
        color: Arrow color
        width: Arrow shaft width
        height: Arrow length
        direction: Arrow direction (unit vector)
        **kwargs: Additional customization
    Returns:
        Arrow: Manim Arrow object
    """
    start = np.array(position)
    end = start + np.array(direction) * height * scale
    return Arrow(
        start,
        end,
        buff=0,
        stroke_width=width * scale * 24,
        color=color,
        max_tip_length_to_length_ratio=0.25,
    )


def create_diagram_fetch_label(
    position=ORIGIN, font_size=24, color=WHITE, scale=1.0, **kwargs
):
    """
    Creates a MathTex label "Fetch".
    Args:
        position: (x, y) coordinates for placement
        font_size: Font size
        color: Text color
        scale: Size multiplier
        **kwargs: Additional customization
    Returns:
        MathTex: Manim MathTex object
    """
    return MathTex("Fetch", font_size=font_size * scale, color=color).move_to(position)


def create_diagram_decode_label(
    position=ORIGIN, font_size=24, color=WHITE, scale=1.0, **kwargs
):
    """
    Creates a MathTex label "Decode".
    Args:
        position: (x, y) coordinates for placement
        font_size: Font size
        color: Text color
        scale: Size multiplier
        **kwargs: Additional customization
    Returns:
        MathTex: Manim MathTex object
    """
    return MathTex("Decode", font_size=font_size * scale, color=color).move_to(position)


def create_diagram_execute_label(
    position=ORIGIN, font_size=24, color=WHITE, scale=1.0, **kwargs
):
    """
    Creates a MathTex label "Execute".
    Args:
        position: (x, y) coordinates for placement
        font_size: Font size
        color: Text color
        scale: Size multiplier
        **kwargs: Additional customization
    Returns:
        MathTex: Manim MathTex object
    """
    return MathTex("Execute", font_size=font_size * scale, color=color).move_to(
        position
    )


def create_diagram_cycle_text(
    position=ORIGIN,
    text="Continuous Cycle",
    font_size=36,
    color=WHITE,
    scale=1.0,
    **kwargs,
):
    """
    Creates a large cycle text label.
    Args:
        position: (x, y) coordinates for placement
        text: Label text
        font_size: Font size
        color: Text color
        scale: Size multiplier
        **kwargs: Additional customization
    Returns:
        Text: Manim Text object
    """
    return Text(text, font_size=font_size * scale, color=color).move_to(position)


from manim import *


def create_hardware_rectangle(
    position=ORIGIN,
    scale=1.0,
    color="#58C4DD",
    width=5.0,
    height=3.0,
    label="Hardware",
    **kwargs,
):
    """
    Creates a blue rectangle representing hardware with a label.
    """
    rect = Rectangle(
        width=width * scale, height=height * scale, stroke_color=color, fill_opacity=0.0
    )
    rect.move_to(position)
    text = Text(label, color=color, font_size=36 * scale)
    text.move_to(rect.get_center())
    diagram = VGroup(rect, text)
    return diagram


def create_software_rectangle(
    position=ORIGIN,
    scale=1.0,
    color="#A0A0A0",
    width=5.0,
    height=3.0,
    label="Software",
    **kwargs,
):
    """
    Creates a grey rectangle representing software with a label.
    """
    rect = Rectangle(
        width=width * scale, height=height * scale, stroke_color=color, fill_opacity=0.0
    )
    rect.move_to(position)
    text = Text(label, color=color, font_size=36 * scale)
    text.move_to(rect.get_center())
    diagram = VGroup(rect, text)
    return diagram


def create_processor_icon(
    position=ORIGIN, scale=1.0, color="#A0A0A0", width=1.2, height=1.2, **kwargs
):
    """
    Creates a small grey rectangle with internal lines representing a processor.
    """
    rect = Rectangle(
        width=width * scale, height=height * scale, stroke_color=color, fill_opacity=0.0
    )
    rect.move_to(position)
    lines = VGroup(
        Line(
            rect.get_left() + 0.2 * RIGHT, rect.get_right() - 0.2 * RIGHT, color=color
        ).scale(0.7 * scale),
        Line(
            rect.get_top() + 0.2 * DOWN, rect.get_bottom() - 0.2 * DOWN, color=color
        ).scale(0.7 * scale),
    )
    lines.move_to(rect.get_center())
    diagram = VGroup(rect, lines)
    return diagram


def create_memory_icon(
    position=ORIGIN,
    scale=1.0,
    color="#A0A0A0",
    width=1.0,
    height=1.2,
    stacks=3,
    **kwargs,
):
    """
    Creates stacked grey rectangles representing memory.
    """
    stack_group = VGroup()
    for i in range(stacks):
        y_offset = (i - (stacks - 1) / 2) * (height * scale * 0.7)
        rect = Rectangle(
            width=width * scale,
            height=height * scale * 0.5,
            stroke_color=color,
            fill_opacity=0.0,
        )
        rect.move_to(position + y_offset * UP)
        stack_group.add(rect)
    return stack_group


def create_monitor_icon(
    position=ORIGIN, scale=1.0, color="#A0A0A0", width=2.0, height=1.2, **kwargs
):
    """
    Creates a larger grey rectangle representing a monitor.
    """
    rect = Rectangle(
        width=width * scale, height=height * scale, stroke_color=color, fill_opacity=0.0
    )
    stand = Rectangle(
        width=0.3 * scale, height=0.15 * scale, stroke_color=color, fill_opacity=0.0
    )
    stand.next_to(rect, DOWN, buff=0.05 * scale)
    base = Rectangle(
        width=0.7 * scale, height=0.08 * scale, stroke_color=color, fill_opacity=0.0
    )
    base.next_to(stand, DOWN, buff=0.01 * scale)
    group = VGroup(rect, stand, base)
    group.move_to(position)
    return group


def create_arrow_icon(
    start=LEFT, end=RIGHT, scale=1.0, color="#000000", buff=0.0, **kwargs
):
    """
    Creates a black arrow representing the flow of instructions.
    """
    arrow = Arrow(
        start,
        end,
        buff=buff,
        stroke_color=color,
        max_tip_length_to_length_ratio=0.2,
        stroke_width=6 * scale,
    )
    return arrow


def create_circle_main_concept(
    position=ORIGIN, scale=1.0, color="#58C4DD", radius=1.2, label="Concept", **kwargs
):
    """
    Creates a main concept circle with a label.
    """
    circle = Circle(radius=radius * scale, stroke_color=color, fill_opacity=0.0)
    circle.move_to(position)
    text = Text(label, color=color, font_size=36 * scale)
    text.move_to(circle.get_center())
    return VGroup(circle, text)


def create_text_label(
    text, position=ORIGIN, scale=1.0, color="#58C4DD", font_size=32, **kwargs
):
    """
    Creates a text label at a given position.
    """
    label = Text(text, color=color, font_size=font_size * scale)
    label.move_to(position)
    return label


def create_rectangle_input_device(
    position=ORIGIN,
    scale=1.0,
    color="#58C4DD",
    width=2.0,
    height=1.0,
    label="Input",
    **kwargs,
):
    """
    Creates a rectangle for an input device with a label.
    """
    rect = Rectangle(
        width=width * scale, height=height * scale, stroke_color=color, fill_opacity=0.0
    )
    rect.move_to(position)
    text = Text(label, color=color, font_size=28 * scale)
    text.move_to(rect.get_center())
    return VGroup(rect, text)


def create_rectangle_cpu(
    position=ORIGIN,
    scale=1.0,
    color="#58C4DD",
    width=2.5,
    height=1.5,
    label="CPU",
    **kwargs,
):
    """
    Creates a rectangle representing a CPU with a label.
    """
    rect = Rectangle(
        width=width * scale, height=height * scale, stroke_color=color, fill_opacity=0.0
    )
    rect.move_to(position)
    text = Text(label, color=color, font_size=32 * scale)
    text.move_to(rect.get_center())
    return VGroup(rect, text)


def create_mathtex_binary_stream(
    position=ORIGIN, scale=1.0, color="#58C4DD", binary_str="101010", **kwargs
):
    """
    Creates a MathTex object for a binary stream.
    """
    tex = MathTex(binary_str, color=color, font_size=36 * scale)
    tex.move_to(position)
    return tex


def create_diagram_cpu_core(position=ORIGIN, scale=1.0, color="#3498db", **kwargs):
    """
    Creates a CPU core diagram: ALU (circle), Control Unit (rectangle), and labels.
    """
    alu = Circle(radius=0.8 * scale, stroke_color="#2ecc71", fill_opacity=0.0)
    alu.move_to(position + 1.0 * LEFT * scale)
    alu_label = Text("ALU", color="#2ecc71", font_size=28 * scale)
    alu_label.next_to(alu, DOWN, buff=0.1 * scale)
    cu = Rectangle(
        width=1.2 * scale, height=0.8 * scale, stroke_color=color, fill_opacity=0.0
    )
    cu.move_to(position + 1.0 * RIGHT * scale)
    cu_label = Text("Control Unit", color=color, font_size=24 * scale)
    cu_label.next_to(cu, DOWN, buff=0.1 * scale)
    group = VGroup(alu, alu_label, cu, cu_label)
    group.move_to(position)
    return group


def create_diagram_external_devices(
    position=ORIGIN, scale=1.0, color="#e74c3c", **kwargs
):
    """
    Creates a diagram for external devices: Memory and I/O rectangles with labels.
    """
    mem = Rectangle(
        width=1.2 * scale, height=1.6 * scale, stroke_color=color, fill_opacity=0.0
    )
    mem.move_to(position + 1.2 * LEFT * scale)
    mem_label = Text("Memory", color=color, font_size=24 * scale)
    mem_label.next_to(mem, DOWN, buff=0.1 * scale)
    io = Rectangle(
        width=1.2 * scale, height=1.6 * scale, stroke_color="#9b59b6", fill_opacity=0.0
    )
    io.move_to(position + 1.2 * RIGHT * scale)
    io_label = Text("I/O", color="#9b59b6", font_size=24 * scale)
    io_label.next_to(io, DOWN, buff=0.1 * scale)
    group = VGroup(mem, mem_label, io, io_label)
    group.move_to(position)
    return group


def create_diagram_data_flow(position=ORIGIN, scale=1.0, color="#58C4DD", **kwargs):
    """
    Creates a data flow diagram with four arrows between CPU, Memory, and I/O.
    """
    cpu = Rectangle(
        width=1.2 * scale, height=1.2 * scale, stroke_color=color, fill_opacity=0.0
    )
    cpu.move_to(position)
    mem = Rectangle(
        width=1.2 * scale, height=1.2 * scale, stroke_color="#e74c3c", fill_opacity=0.0
    )
    mem.move_to(position + 2.5 * LEFT * scale)
    io = Rectangle(
        width=1.2 * scale, height=1.2 * scale, stroke_color="#9b59b6", fill_opacity=0.0
    )
    io.move_to(position + 2.5 * RIGHT * scale)
    arrow_cpu_mem = Arrow(
        cpu.get_left(), mem.get_right(), stroke_color=color, buff=0.1 * scale
    )
    arrow_mem_cpu = Arrow(
        mem.get_right(), cpu.get_left(), stroke_color=color, buff=0.1 * scale
    )
    arrow_cpu_io = Arrow(
        cpu.get_right(), io.get_left(), stroke_color=color, buff=0.1 * scale
    )
    arrow_io_cpu = Arrow(
        io.get_left(), cpu.get_right(), stroke_color=color, buff=0.1 * scale
    )
    group = VGroup(
        cpu, mem, io, arrow_cpu_mem, arrow_mem_cpu, arrow_cpu_io, arrow_io_cpu
    )
    return group


def create_diagram_cpu(
    position=ORIGIN,
    scale=1.0,
    color="#58C4DD",
    width=2.5,
    height=1.5,
    circle_colors=["#58C4DD", "#F0E68C", "#FFFFFF"],
    **kwargs,
):
    """
    Creates a CPU rectangle with three colored circles inside.
    """
    rect = Rectangle(
        width=width * scale, height=height * scale, stroke_color=color, fill_opacity=0.0
    )
    rect.move_to(position)
    circles = VGroup()
    for i, c in enumerate(circle_colors):
        circ = Circle(radius=0.3 * scale, stroke_color=c, fill_opacity=0.0)
        circ.move_to(rect.get_center() + (i - 1) * 0.7 * RIGHT * scale)
        circles.add(circ)
    group = VGroup(rect, circles)
    return group


def create_diagram_ram(
    position=ORIGIN,
    scale=1.0,
    color="#e74c3c",
    width=2.5,
    height=1.5,
    rows=3,
    cols=5,
    **kwargs,
):
    """
    Creates a RAM rectangle with a grid of smaller rectangles.
    """
    rect = Rectangle(
        width=width * scale, height=height * scale, stroke_color=color, fill_opacity=0.0
    )
    rect.move_to(position)
    cell_w = (width * scale - 0.2 * scale) / cols
    cell_h = (height * scale - 0.2 * scale) / rows
    cells = VGroup()
    for r in range(rows):
        for c in range(cols):
            cell = Rectangle(
                width=cell_w,
                height=cell_h,
                stroke_color=color,
                fill_opacity=0.0,
                stroke_width=1.5,
            )
            x = (c - (cols - 1) / 2) * cell_w
            y = (r - (rows - 1) / 2) * cell_h
            cell.move_to(rect.get_center() + x * RIGHT + y * UP)
            cells.add(cell)
    group = VGroup(rect, cells)
    return group


def create_diagram_arrow(
    start=LEFT, end=RIGHT, scale=1.0, color="#FFFFFF", buff=0.0, pulsate=False, **kwargs
):
    """
    Creates a (pulsating) arrow for data flow visualization.
    """
    arrow = Arrow(
        start,
        end,
        buff=buff,
        stroke_color=color,
        max_tip_length_to_length_ratio=0.2,
        stroke_width=6 * scale,
    )
    if pulsate:
        arrow.add_updater(
            lambda m, dt: m.set_stroke(
                width=6 * scale + 2 * scale * np.sin(dt * 2 * np.pi)
            )
        )
    return arrow


def create_diagram_address_label(position=ORIGIN, scale=1.0, color="#58C4DD", **kwargs):
    """
    Creates a MathTex object displaying 'Address'.
    """
    tex = MathTex(r"\text{Address}", color=color, font_size=36 * scale)
    tex.move_to(position)
    return tex


def create_diagram_cpu_main(
    position=ORIGIN,
    scale=1.0,
    color="#58C4DD",
    width=2.5,
    height=1.5,
    label="CPU",
    **kwargs,
):
    """
    Creates a main CPU rectangle with label.
    """
    rect = Rectangle(
        width=width * scale, height=height * scale, stroke_color=color, fill_opacity=0.0
    )
    rect.move_to(position)
    text = Text(label, color=color, font_size=32 * scale)
    text.move_to(rect.get_center())
    return VGroup(rect, text)


def create_circle_memory_main(
    position=ORIGIN, scale=1.0, color="#e74c3c", radius=0.9, label="Memory", **kwargs
):
    """
    Creates a main memory circle with label.
    """
    circ = Circle(radius=radius * scale, stroke_color=color, fill_opacity=0.0)
    circ.move_to(position)
    text = Text(label, color=color, font_size=28 * scale)
    text.move_to(circ.get_center())
    return VGroup(circ, text)


def create_circle_input_main(
    position=ORIGIN, scale=1.0, color="#58C4DD", radius=0.9, label="Input", **kwargs
):
    """
    Creates a main input circle with label.
    """
    circ = Circle(radius=radius * scale, stroke_color=color, fill_opacity=0.0)
    circ.move_to(position)
    text = Text(label, color=color, font_size=28 * scale)
    text.move_to(circ.get_center())
    return VGroup(circ, text)


def create_rectangle_output_unit(
    position=ORIGIN,
    scale=1.0,
    color="#F0E68C",
    width=2.0,
    height=1.0,
    label="Output",
    **kwargs,
):
    """
    Creates an output unit rectangle with label.
    """
    rect = Rectangle(
        width=width * scale, height=height * scale, stroke_color=color, fill_opacity=0.0
    )
    rect.move_to(position)
    text = Text(label, color=color, font_size=28 * scale)
    text.move_to(rect.get_center())
    return VGroup(rect, text)


def create_text_fetch_instruction(
    position=ORIGIN, scale=1.0, color="#58C4DD", **kwargs
):
    """
    Creates a text label for 'Fetch Instruction'.
    """
    text = Text("Fetch Instruction", color=color, font_size=28 * scale)
    text.move_to(position)
    return text


def create_text_interpret_instruction(
    position=ORIGIN, scale=1.0, color="#58C4DD", **kwargs
):
    """
    Creates a text label for 'Interpret Instruction'.
    """
    text = Text("Interpret Instruction", color=color, font_size=28 * scale)
    text.move_to(position)
    return text


def create_text_retrieve_data(position=ORIGIN, scale=1.0, color="#58C4DD", **kwargs):
    """
    Creates a text label for 'Retrieve Data'.
    """
    text = Text("Retrieve Data", color=color, font_size=28 * scale)
    text.move_to(position)
    return text


def create_diagram_cpu_core_rect(
    position=ORIGIN, scale=1.0, color="#3498db", width=8.0, height=4.0, **kwargs
):
    """
    Creates a CPU core rectangle.
    """
    rect = Rectangle(
        width=width * scale, height=height * scale, stroke_color=color, fill_opacity=0.0
    )
    rect.move_to(position)
    return rect


def create_diagram_memory_rect(
    position=ORIGIN, scale=1.0, color="#e74c3c", width=2.0, height=4.0, **kwargs
):
    """
    Creates a memory rectangle.
    """
    rect = Rectangle(
        width=width * scale, height=height * scale, stroke_color=color, fill_opacity=0.0
    )
    rect.move_to(position)
    return rect


def create_diagram_alu_circle(
    position=ORIGIN, scale=1.0, color="#2ecc71", radius=1.0, **kwargs
):
    """
    Creates an ALU circle.
    """
    circ = Circle(radius=radius * scale, stroke_color=color, fill_opacity=0.0)
    circ.move_to(position)
    return circ


def create_diagram_register_rect(
    position=ORIGIN, scale=1.0, color="#f39c12", width=0.8, height=0.6, **kwargs
):
    """
    Creates a register rectangle.
    """
    rect = Rectangle(
        width=width * scale, height=height * scale, stroke_color=color, fill_opacity=0.0
    )
    rect.move_to(position)
    return rect


def create_diagram_output_device_rect(
    position=ORIGIN, scale=1.0, color="#9b59b6", width=2.0, height=4.0, **kwargs
):
    """
    Creates an output device rectangle.
    """
    rect = Rectangle(
        width=width * scale, height=height * scale, stroke_color=color, fill_opacity=0.0
    )
    rect.move_to(position)
    return rect


def create_diagram_arrow_white(
    start=LEFT, end=RIGHT, scale=1.0, color="#FFFFFF", buff=0.0, **kwargs
):
    """
    Creates a white arrow.
    """
    arrow = Arrow(
        start,
        end,
        buff=buff,
        stroke_color=color,
        max_tip_length_to_length_ratio=0.2,
        stroke_width=6 * scale,
    )
    return arrow


def create_diagram_control_unit_pulse(
    position=ORIGIN, scale=1.0, color="#3498db", width=1.5, height=0.6, **kwargs
):
    """
    Creates a control unit pulse rectangle.
    """
    rect = Rectangle(
        width=width * scale, height=height * scale, stroke_color=color, fill_opacity=0.0
    )
    rect.move_to(position)
    return rect


def create_cpu_container(
    position=ORIGIN, scale=1.0, color="#58C4DD", width=3.0, height=2.0, **kwargs
):
    """
    Creates a reusable rectangle for the CPU.
    """
    rect = Rectangle(
        width=width * scale, height=height * scale, stroke_color=color, fill_opacity=0.0
    )
    rect.move_to(position)
    return rect


def create_alu_core(position=ORIGIN, scale=1.0, color="#2ecc71", radius=0.8, **kwargs):
    """
    Creates a reusable circle for the ALU.
    """
    circ = Circle(radius=radius * scale, stroke_color=color, fill_opacity=0.0)
    circ.move_to(position)
    return circ


def create_data_arrow(
    start=LEFT, end=RIGHT, scale=1.0, color="#58C4DD", buff=0.0, **kwargs
):
    """
    Creates a reusable arrow for data flow.
    """
    arrow = Arrow(
        start,
        end,
        buff=buff,
        stroke_color=color,
        max_tip_length_to_length_ratio=0.2,
        stroke_width=6 * scale,
    )
    return arrow


def create_math_symbol(
    symbol="+", position=ORIGIN, scale=1.0, color="#58C4DD", **kwargs
):
    """
    Creates a reusable MathTex object for a math symbol.
    """
    tex = MathTex(symbol, color=color, font_size=36 * scale)
    tex.move_to(position)
    return tex


from manim import *

# =========================
#   PRODUCTION-READY REUSABLE DIAGRAM FUNCTIONS
# =========================


def create_rectangle_processor_1(
    position=ORIGIN,
    scale=1.0,
    color="#58C4DD",
    label="Processor",
    label_color="#FFFFFF",
    **kwargs,
):
    """
    Creates a rectangle representing a processor with a label.
    """
    rect = Rectangle(
        width=2.2 * scale, height=1.2 * scale, stroke_color=color, fill_opacity=0.0
    )
    text = Text(label, font_size=28 * scale, color=label_color).move_to(
        rect.get_center()
    )
    group = VGroup(rect, text).move_to(position)
    return group


def create_rectangle_memory_1(
    position=ORIGIN,
    scale=1.0,
    color="#F0E68C",
    label="Memory",
    label_color="#000000",
    **kwargs,
):
    """
    Creates a rectangle representing memory with a label.
    """
    rect = Rectangle(
        width=2.2 * scale, height=1.2 * scale, stroke_color=color, fill_opacity=0.0
    )
    text = Text(label, font_size=28 * scale, color=label_color).move_to(
        rect.get_center()
    )
    group = VGroup(rect, text).move_to(position)
    return group


def create_rectangle_monitor_1(
    position=ORIGIN,
    scale=1.0,
    color="#58C4DD",
    label="Monitor",
    label_color="#FFFFFF",
    **kwargs,
):
    """
    Creates a rectangle representing a monitor with a label.
    """
    rect = Rectangle(
        width=2.2 * scale, height=1.2 * scale, stroke_color=color, fill_opacity=0.0
    )
    text = Text(label, font_size=28 * scale, color=label_color).move_to(
        rect.get_center()
    )
    group = VGroup(rect, text).move_to(position)
    return group


def create_line_processor_connection_1(start, end, color="#58C4DD", width=6, **kwargs):
    """
    Creates a line connecting processor to another component.
    """
    line = Line(start, end, stroke_color=color, stroke_width=width)
    return line


def create_line_memory_connection_1(start, end, color="#F0E68C", width=6, **kwargs):
    """
    Creates a line connecting memory to another component.
    """
    line = Line(start, end, stroke_color=color, stroke_width=width)
    return line


def create_line_monitor_connection_1(start, end, color="#58C4DD", width=6, **kwargs):
    """
    Creates a line connecting monitor to another component.
    """
    line = Line(start, end, stroke_color=color, stroke_width=width)
    return line


def create_arrow_instruction_flow_1(
    start, end, color="#58C4DD", buff=0.1, width=8, **kwargs
):
    """
    Creates an arrow representing instruction flow.
    """
    arrow = Arrow(
        start,
        end,
        buff=buff,
        stroke_color=color,
        stroke_width=width,
        max_tip_length_to_length_ratio=0.18,
    )
    return arrow


def create_mathtex_instructions_1(
    position=ORIGIN, scale=1.0, color="#58C4DD", tex="\\text{Instructions}", **kwargs
):
    """
    Creates a MathTex label for instructions.
    """
    label = MathTex(tex, font_size=32 * scale, color=color).move_to(position)
    return label


def create_triangle_cycle(position=ORIGIN, scale=1.0, color="#4CAF50", **kwargs):
    """
    Creates an equilateral triangle (cycle) with specified side length and color.
    """
    side = 2.0 * scale
    triangle = Polygon(
        [0, 0, 0],
        [side, 0, 0],
        [side / 2, side * np.sqrt(3) / 2, 0],
        stroke_color=color,
        fill_opacity=0.0,
    ).move_to(position)
    return triangle


def create_rectangle_base(
    position=ORIGIN, scale=1.0, color="#2196F3", width=2.0, height=1.0, **kwargs
):
    """
    Creates a base rectangle with customizable color, width, and height.
    """
    rect = Rectangle(
        width=width * scale, height=height * scale, stroke_color=color, fill_opacity=0.0
    )
    return rect.move_to(position)


def create_text_label(
    position=ORIGIN, scale=1.0, color="#FFFFFF", text="Label", font_size=24, **kwargs
):
    """
    Creates a text label with specified font size and color.
    """
    label = Text(text, font_size=font_size * scale, color=color).move_to(position)
    return label


def create_arrow_base(start, end, color="#777777", width=0.5, **kwargs):
    """
    Creates a base arrow with specified width and color.
    """
    arrow = Arrow(
        start,
        end,
        stroke_color=color,
        stroke_width=width * 12,
        max_tip_length_to_length_ratio=0.18,
    )
    return arrow


def create_cpu_circle(
    position=ORIGIN, scale=1.0, color="#E91E63", radius=1.5, **kwargs
):
    """
    Creates a circle representing a CPU.
    """
    circle = Circle(radius=radius * scale, stroke_color=color, fill_opacity=0.0)
    return circle.move_to(position)


def create_cpu_internal_line(start, end, color="#FFFFFF", width=0.2, **kwargs):
    """
    Creates an internal line inside the CPU circle.
    """
    line = Line(start, end, stroke_color=color, stroke_width=width * 12)
    return line


def create_binary_digit(
    position=ORIGIN, scale=1.0, color="#000000", digit="0", font_size=18, **kwargs
):
    """
    Creates a MathTex object for a binary digit.
    """
    tex = MathTex(digit, font_size=font_size * scale, color=color).move_to(position)
    return tex


def create_diagram_cpu_rectangle(
    position=ORIGIN,
    scale=1.0,
    color="#58C4DD",
    width=2.2,
    height=1.2,
    label="CPU",
    label_color="#FFFFFF",
    **kwargs,
):
    """
    Creates a rectangle for CPU representation with label.
    """
    rect = Rectangle(
        width=width * scale, height=height * scale, stroke_color=color, fill_opacity=0.0
    )
    text = Text(label, font_size=28 * scale, color=label_color).move_to(
        rect.get_center()
    )
    group = VGroup(rect, text).move_to(position)
    return group


def create_diagram_alu_circle(
    position=ORIGIN,
    scale=1.0,
    color="#F0E68C",
    radius=0.8,
    label="ALU",
    label_color="#000000",
    **kwargs,
):
    """
    Creates a circle for ALU representation with label.
    """
    circle = Circle(radius=radius * scale, stroke_color=color, fill_opacity=0.0)
    text = Text(label, font_size=24 * scale, color=label_color).move_to(
        circle.get_center()
    )
    group = VGroup(circle, text).move_to(position)
    return group


def create_diagram_data_arrow(start, end, color="#58C4DD", width=0.5, **kwargs):
    """
    Creates an arrow for data flow.
    """
    arrow = Arrow(
        start,
        end,
        stroke_color=color,
        stroke_width=width * 12,
        max_tip_length_to_length_ratio=0.18,
    )
    return arrow


def create_cpu_core(
    position=ORIGIN,
    scale=1.0,
    color="#4CAF50",
    width=6.0,
    height=3.0,
    label="CPU Core",
    label_color="#FFFFFF",
    **kwargs,
):
    """
    Creates a rectangle for CPU core with label.
    """
    rect = Rectangle(
        width=width * scale, height=height * scale, stroke_color=color, fill_opacity=0.0
    )
    text = Text(label, font_size=36 * scale, color=label_color).move_to(
        rect.get_center()
    )
    group = VGroup(rect, text).move_to(position)
    return group


def create_control_unit(
    position=ORIGIN,
    scale=1.0,
    color="#FF9800",
    radius=0.7,
    label="CU",
    label_color="#FFFFFF",
    **kwargs,
):
    """
    Creates a circle for Control Unit with label.
    """
    circle = Circle(radius=radius * scale, stroke_color=color, fill_opacity=0.0)
    text = Text(label, font_size=24 * scale, color=label_color).move_to(
        circle.get_center()
    )
    group = VGroup(circle, text).move_to(position)
    return group


def create_alu(
    position=ORIGIN,
    scale=1.0,
    color="#2196F3",
    width=1.5,
    height=1.0,
    label="ALU",
    label_color="#FFFFFF",
    **kwargs,
):
    """
    Creates a rectangle for ALU with label.
    """
    rect = Rectangle(
        width=width * scale, height=height * scale, stroke_color=color, fill_opacity=0.0
    )
    text = Text(label, font_size=24 * scale, color=label_color).move_to(
        rect.get_center()
    )
    group = VGroup(rect, text).move_to(position)
    return group


def create_memory(
    position=ORIGIN,
    scale=1.0,
    color="#E91E63",
    width=1.5,
    height=1.0,
    label="Memory",
    label_color="#FFFFFF",
    **kwargs,
):
    """
    Creates a rectangle for Memory with label.
    """
    rect = Rectangle(
        width=width * scale, height=height * scale, stroke_color=color, fill_opacity=0.0
    )
    text = Text(label, font_size=24 * scale, color=label_color).move_to(
        rect.get_center()
    )
    group = VGroup(rect, text).move_to(position)
    return group


def create_io(
    position=ORIGIN,
    scale=1.0,
    color="#9C27B0",
    width=1.5,
    height=1.0,
    label="I/O",
    label_color="#FFFFFF",
    **kwargs,
):
    """
    Creates a rectangle for I/O with label.
    """
    rect = Rectangle(
        width=width * scale, height=height * scale, stroke_color=color, fill_opacity=0.0
    )
    text = Text(label, font_size=24 * scale, color=label_color).move_to(
        rect.get_center()
    )
    group = VGroup(rect, text).move_to(position)
    return group


def create_register(
    position=ORIGIN,
    scale=1.0,
    color="#00BCD4",
    width=1.0,
    height=0.5,
    label="Register",
    label_color="#FFFFFF",
    **kwargs,
):
    """
    Creates a rectangle for Register with label.
    """
    rect = Rectangle(
        width=width * scale, height=height * scale, stroke_color=color, fill_opacity=0.0
    )
    text = Text(label, font_size=18 * scale, color=label_color).move_to(
        rect.get_center()
    )
    group = VGroup(rect, text).move_to(position)
    return group


def create_memory_unit(
    position=ORIGIN,
    scale=1.0,
    color="#FFEB3B",
    width=3.0,
    height=2.0,
    label="Memory Unit",
    label_color="#000000",
    **kwargs,
):
    """
    Creates a rectangle for Memory Unit with label.
    """
    rect = Rectangle(
        width=width * scale, height=height * scale, stroke_color=color, fill_opacity=0.0
    )
    text = Text(label, font_size=28 * scale, color=label_color).move_to(
        rect.get_center()
    )
    group = VGroup(rect, text).move_to(position)
    return group


def create_data_flow_arrow(start, end, color="#777777", width=0.1, **kwargs):
    """
    Creates an arrow for data flow with specified width and color.
    """
    arrow = Arrow(
        start,
        end,
        stroke_color=color,
        stroke_width=width * 24,
        max_tip_length_to_length_ratio=0.18,
    )
    return arrow


def create_pulse(position=ORIGIN, scale=1.0, color="#FFC107", radius=0.9, **kwargs):
    """
    Creates a circle representing a pulse.
    """
    circle = Circle(radius=radius * scale, stroke_color=color, fill_opacity=0.0)
    return circle.move_to(position)


def create_ram_grid(
    position=ORIGIN,
    scale=1.0,
    color="#58C4DD",
    rows=4,
    cols=4,
    cell_width=0.7,
    cell_height=0.5,
    label_color="#000000",
    **kwargs,
):
    """
    Creates a RAM grid with labeled cells.
    """
    grid = VGroup()
    for i in range(rows):
        for j in range(cols):
            cell = Rectangle(
                width=cell_width * scale,
                height=cell_height * scale,
                stroke_color=color,
                fill_opacity=0.0,
            )
            cell.move_to(
                position
                + np.array(
                    [
                        (j - (cols - 1) / 2) * cell_width * scale * 1.1,
                        (-(i - (rows - 1) / 2)) * cell_height * scale * 1.1,
                        0,
                    ]
                )
            )
            label = MathTex(
                f"{i * cols + j:02X}", font_size=18 * scale, color=label_color
            ).move_to(cell.get_center())
            grid.add(VGroup(cell, label))
    return grid


def create_cpu_unit(
    position=ORIGIN,
    scale=1.0,
    color="#E91E63",
    width=2.0,
    height=1.2,
    label="CPU",
    label_color="#FFFFFF",
    **kwargs,
):
    """
    Creates a rectangle for CPU unit with label.
    """
    rect = Rectangle(
        width=width * scale, height=height * scale, stroke_color=color, fill_opacity=0.0
    )
    text = Text(label, font_size=28 * scale, color=label_color).move_to(
        rect.get_center()
    )
    group = VGroup(rect, text).move_to(position)
    return group


def create_output_unit(
    position=ORIGIN,
    scale=1.0,
    color="#58C4DD",
    width=2.0,
    height=1.2,
    label="Output",
    label_color="#FFFFFF",
    **kwargs,
):
    """
    Creates a rectangle for Output unit (Monitor/Printer) with label.
    """
    rect = Rectangle(
        width=width * scale, height=height * scale, stroke_color=color, fill_opacity=0.0
    )
    text = Text(label, font_size=28 * scale, color=label_color).move_to(
        rect.get_center()
    )
    group = VGroup(rect, text).move_to(position)
    return group


def create_data_circle(
    position=ORIGIN,
    scale=1.0,
    color="#F0E68C",
    radius=0.3,
    label="",
    label_color="#000000",
    **kwargs,
):
    """
    Creates a small data circle with optional label.
    """
    circle = Circle(radius=radius * scale, stroke_color=color, fill_opacity=0.0)
    group = VGroup(circle)
    if label:
        text = Text(label, font_size=16 * scale, color=label_color).move_to(
            circle.get_center()
        )
        group.add(text)
    group.move_to(position)
    return group


def create_address_label(
    position=ORIGIN, scale=1.0, color="#000000", text="Addr", font_size=18, **kwargs
):
    """
    Creates a MathTex address label.
    """
    label = MathTex(text, font_size=font_size * scale, color=color).move_to(position)
    return label


def create_diagram_cpu(
    position=ORIGIN,
    scale=1.0,
    color=GREEN,
    width=3.0,
    height=2.0,
    label="CPU",
    label_color=WHITE,
    **kwargs,
):
    """
    Creates a rectangle for CPU diagram with label.
    """
    rect = Rectangle(
        width=width * scale, height=height * scale, stroke_color=color, fill_opacity=0.0
    )
    text = Text(label, font_size=32 * scale, color=label_color).move_to(
        rect.get_center()
    )
    group = VGroup(rect, text).move_to(position)
    return group


def create_diagram_memory(
    position=ORIGIN,
    scale=1.0,
    color=BLUE,
    radius=1.0,
    label="Memory",
    label_color=WHITE,
    **kwargs,
):
    """
    Creates a circle for Memory diagram with label.
    """
    circle = Circle(radius=radius * scale, stroke_color=color, fill_opacity=0.0)
    text = Text(label, font_size=24 * scale, color=label_color).move_to(
        circle.get_center()
    )
    group = VGroup(circle, text).move_to(position)
    return group


def create_diagram_input(
    position=ORIGIN,
    scale=1.0,
    color=RED,
    radius=1.0,
    label="Input",
    label_color=WHITE,
    **kwargs,
):
    """
    Creates a circle for Input diagram with label.
    """
    circle = Circle(radius=radius * scale, stroke_color=color, fill_opacity=0.0)
    text = Text(label, font_size=24 * scale, color=label_color).move_to(
        circle.get_center()
    )
    group = VGroup(circle, text).move_to(position)
    return group


def create_diagram_output(
    position=ORIGIN,
    scale=1.0,
    color=ORANGE,
    radius=1.0,
    label="Output",
    label_color=WHITE,
    **kwargs,
):
    """
    Creates a circle for Output diagram with label.
    """
    circle = Circle(radius=radius * scale, stroke_color=color, fill_opacity=0.0)
    text = Text(label, font_size=24 * scale, color=label_color).move_to(
        circle.get_center()
    )
    group = VGroup(circle, text).move_to(position)
    return group


def create_diagram_instructions(
    position=ORIGIN,
    scale=1.0,
    color=PURPLE,
    radius=1.0,
    label="Instructions",
    label_color=WHITE,
    **kwargs,
):
    """
    Creates a circle for Instructions diagram with label.
    """
    circle = Circle(radius=radius * scale, stroke_color=color, fill_opacity=0.0)
    text = Text(label, font_size=20 * scale, color=label_color).move_to(
        circle.get_center()
    )
    group = VGroup(circle, text).move_to(position)
    return group


def create_diagram_arrow(start, end, color=GRAY, width=0.5, **kwargs):
    """
    Creates an arrow for diagram flow.
    """
    arrow = Arrow(
        start,
        end,
        stroke_color=color,
        stroke_width=width * 12,
        max_tip_length_to_length_ratio=0.18,
    )
    return arrow


def create_diagram_fetch_label(
    position=ORIGIN, scale=1.0, color=WHITE, text="Fetch", font_size=32, **kwargs
):
    """
    Creates a text label for 'Fetch'.
    """
    label = Text(text, font_size=font_size * scale, color=color).move_to(position)
    return label


def create_diagram_decode_label(
    position=ORIGIN, scale=1.0, color=WHITE, text="Decode", font_size=32, **kwargs
):
    """
    Creates a text label for 'Decode'.
    """
    label = Text(text, font_size=font_size * scale, color=color).move_to(position)
    return label


def create_diagram_execute_label(
    position=ORIGIN, scale=1.0, color=WHITE, text="Execute", font_size=32, **kwargs
):
    """
    Creates a text label for 'Execute'.
    """
    label = Text(text, font_size=font_size * scale, color=color).move_to(position)
    return label


def create_diagram_store_label(
    position=ORIGIN, scale=1.0, color=WHITE, text="Store/Output", font_size=32, **kwargs
):
    """
    Creates a text label for 'Store/Output'.
    """
    label = Text(text, font_size=font_size * scale, color=color).move_to(position)
    return label


# =========================
#   REUSABLE DIAGRAM GROUPS
# =========================


def create_diagram_hardware_components(position=ORIGIN, scale=1.0, **kwargs):
    """
    Creates a reusable diagram of hardware components: processor, memory, monitor, and their connections.
    """
    # Layout: processor (center left), memory (center), monitor (center right)
    proc_pos = position + LEFT * 3.5 * scale
    mem_pos = position
    mon_pos = position + RIGHT * 3.5 * scale
    proc = create_rectangle_processor_1(proc_pos, scale=scale)
    mem = create_rectangle_memory_1(mem_pos, scale=scale)
    mon = create_rectangle_monitor_1(mon_pos, scale=scale)
    line_proc = create_line_processor_connection_1(
        proc_pos + RIGHT * 1.1 * scale, mem_pos + LEFT * 1.1 * scale
    )
    line_mem = create_line_memory_connection_1(
        mem_pos + RIGHT * 1.1 * scale, mon_pos + LEFT * 1.1 * scale
    )
    line_mon = create_line_monitor_connection_1(
        mon_pos + DOWN * 0.6 * scale, mon_pos + DOWN * 1.8 * scale
    )
    group = VGroup(proc, mem, mon, line_proc, line_mem, line_mon).move_to(position)
    return group


def create_diagram_instruction_flow(position=ORIGIN, scale=1.0, **kwargs):
    """
    Creates a reusable diagram for instruction flow: arrow and label.
    """
    start = position + LEFT * 2.0 * scale
    end = position + RIGHT * 2.0 * scale
    arrow = create_arrow_instruction_flow_1(start, end, color="#58C4DD")
    label = create_mathtex_instructions_1(position=position, scale=scale)
    group = VGroup(arrow, label)
    return group


def create_ram_grid_with_labels(position=ORIGIN, scale=1.0, **kwargs):
    """
    Creates a RAM grid with address labels.
    """
    grid = create_ram_grid(position=position, scale=scale)
    # Address labels above each column
    labels = VGroup()
    for j in range(4):
        addr = create_address_label(
            position=position
            + np.array([(j - 1.5) * 0.7 * scale * 1.1, 1.2 * scale, 0]),
            scale=scale,
            text=f"Addr {j}",
            color="#000000",
        )
        labels.add(addr)
    return VGroup(grid, labels)


def create_data_circles_row(
    position=ORIGIN, scale=1.0, count=3, spacing=0.8, color="#F0E68C", **kwargs
):
    """
    Creates a row of data circles.
    """
    group = VGroup()
    for i in range(count):
        circ = create_data_circle(
            position=position + RIGHT * (i - (count - 1) / 2) * spacing * scale,
            scale=scale,
            color=color,
            label=str(i + 1),
        )
        group.add(circ)
    return group


def create_output_unit_group(position=ORIGIN, scale=1.0, **kwargs):
    """
    Creates a group for output unit: monitor and printer.
    """
    monitor = create_output_unit(
        position=position + UP * 0.7 * scale, scale=scale, label="Monitor"
    )
    printer = create_output_unit(
        position=position + DOWN * 0.7 * scale, scale=scale, label="Printer"
    )
    return VGroup(monitor, printer)


# =========================
#   END OF FUNCTION SUITE
# =========================

from manim import *


def create_rectangle_box(
    width=2.0,
    height=1.0,
    color=WHITE,
    position=(0, 0),
    stroke_width=2,
    fill_opacity=1.0,
    corner_radius=0,
    name=None,
    text=None,
    font_size=24,
    text_color=WHITE,
    text_position=None,
    grid_size=None,
    grid_line_color=GRAY,
    grid_line_width=1,
):
    """
    Creates a rectangle (or rounded rectangle) with optional label and grid lines.
    """
    if corner_radius > 0:
        box = RoundedRectangle(
            width=width,
            height=height,
            corner_radius=corner_radius,
            color=color,
            stroke_width=stroke_width,
            fill_opacity=fill_opacity,
        )
    else:
        box = Rectangle(
            width=width,
            height=height,
            color=color,
            stroke_width=stroke_width,
            fill_opacity=fill_opacity,
        )
    box.move_to(position)
    if name:
        box.name = name

    vg = VGroup(box)

    # Add grid lines if requested
    if (
        grid_size is not None
        and isinstance(grid_size, (list, tuple))
        and len(grid_size) == 2
    ):
        rows, cols = grid_size
        x0, y0 = box.get_left()[0], box.get_bottom()[1]
        dx = width / cols
        dy = height / rows
        # Vertical lines
        for i in range(1, cols):
            x = x0 + i * dx
            line = Line(
                [x, y0, 0],
                [x, y0 + height, 0],
                color=grid_line_color,
                stroke_width=grid_line_width,
            )
            vg.add(line)
        # Horizontal lines
        for j in range(1, rows):
            y = y0 + j * dy
            line = Line(
                [x0, y, 0],
                [x0 + width, y, 0],
                color=grid_line_color,
                stroke_width=grid_line_width,
            )
            vg.add(line)

    # Add label if requested
    if text is not None:
        label = Text(str(text), font_size=font_size, color=text_color)
        if text_position is not None:
            label.move_to(box.get_center() + np.array([*text_position, 0]))
        else:
            label.move_to(box.get_center())
        vg.add(label)

    return vg


def create_labeled_rounded_box(
    width=2.0,
    height=1.0,
    color=WHITE,
    position=(0, 0),
    label="",
    font_size=24,
    label_color=WHITE,
    corner_radius=0.25,
    stroke_width=2,
    fill_opacity=1.0,
    name=None,
):
    """
    Creates a rounded rectangle with a centered label.
    """
    box = RoundedRectangle(
        width=width,
        height=height,
        corner_radius=corner_radius,
        color=color,
        stroke_width=stroke_width,
        fill_opacity=fill_opacity,
    ).move_to(position)
    if name:
        box.name = name
    label_mob = Text(str(label), font_size=font_size, color=label_color).move_to(
        box.get_center()
    )
    return VGroup(box, label_mob)


def create_circle_marker(
    radius=0.2,
    color=YELLOW,
    position=(0, 0),
    stroke_width=2,
    fill_opacity=1.0,
    name=None,
):
    """
    Creates a circle marker at a given position.
    """
    circle = Circle(
        radius=radius,
        color=color,
        stroke_width=stroke_width,
        fill_opacity=fill_opacity,
    ).move_to(position)
    if name:
        circle.name = name
    return circle


def create_labeled_text(
    text,
    font_size=24,
    color=WHITE,
    position=(0, 0),
    name=None,
):
    """
    Creates a text label at a given position.
    """
    label = Text(str(text), font_size=font_size, color=color).move_to(position)
    if name:
        label.name = name
    return label


def create_multi_text_labels(
    texts,
    font_size=16,
    color=WHITE,
    positions=None,
    direction="horizontal",
    start_position=(0, 0),
    spacing=1.0,
    name_prefix=None,
):
    """
    Creates a group of text labels in a row or column.
    """
    labels = VGroup()
    for i, t in enumerate(texts):
        if positions is not None:
            pos = positions[i]
        else:
            if direction == "horizontal":
                pos = (start_position[0] + i * spacing, start_position[1], 0)
            else:
                pos = (start_position[0], start_position[1] - i * spacing, 0)
        label = Text(str(t), font_size=font_size, color=color).move_to(pos)
        if name_prefix:
            label.name = f"{name_prefix}_{i}"
        labels.add(label)
    return labels


def create_flow_arrow(
    start,
    end,
    color=WHITE,
    stroke_width=2,
    buff=0.1,
    tip_length=0.3,
    double_arrow=False,
    name=None,
):
    """
    Creates an arrow from start to end.
    """
    arrow_cls = DoubleArrow if double_arrow else Arrow
    arrow = arrow_cls(
        start=start,
        end=end,
        color=color,
        stroke_width=stroke_width,
        buff=buff,
        tip_length=tip_length,
    )
    if name:
        arrow.name = name
    return arrow


def create_polyline(
    points,
    color=WHITE,
    stroke_width=2,
    name=None,
):
    """
    Creates a polyline (Line or series of connected lines) through given points.
    """
    if len(points) == 2:
        line = Line(points[0], points[1], color=color, stroke_width=stroke_width)
    else:
        line = VMobject(color=color, stroke_width=stroke_width)
        line.set_points_as_corners([*points])
    if name:
        line.name = name
    return line


def create_memory_grid(
    width=6.0,
    height=4.0,
    color=BLUE,
    grid_size=(4, 6),
    position=(0, 0),
    stroke_width=2,
    fill_opacity=1.0,
    grid_line_color=GRAY,
    grid_line_width=1,
    name=None,
):
    """
    Creates a memory grid rectangle with internal grid lines.
    """
    rows, cols = grid_size
    box = Rectangle(
        width=width,
        height=height,
        color=color,
        stroke_width=stroke_width,
        fill_opacity=fill_opacity,
    ).move_to(position)
    if name:
        box.name = name
    vg = VGroup(box)
    x0, y0 = box.get_left()[0], box.get_bottom()[1]
    dx = width / cols
    dy = height / rows
    # Vertical lines
    for i in range(1, cols):
        x = x0 + i * dx
        line = Line(
            [x, y0, 0],
            [x, y0 + height, 0],
            color=grid_line_color,
            stroke_width=grid_line_width,
        )
        vg.add(line)
    # Horizontal lines
    for j in range(1, rows):
        y = y0 + j * dy
        line = Line(
            [x0, y, 0],
            [x0 + width, y, 0],
            color=grid_line_color,
            stroke_width=grid_line_width,
        )
        vg.add(line)
    return vg


def create_data_block(
    width=0.8,
    height=0.5,
    color=RED,
    position=(0, 0),
    stroke_width=2,
    fill_opacity=1.0,
    name=None,
):
    """
    Creates a small rectangle representing a data block.
    """
    rect = Rectangle(
        width=width,
        height=height,
        color=color,
        stroke_width=stroke_width,
        fill_opacity=fill_opacity,
    ).move_to(position)
    if name:
        rect.name = name
    return rect


from manim import *


def create_background_rectangle(
    width=15,
    height=9,
    color=BLACK,
    position=(0, 0),
    opacity=1.0,
    stroke_width=0,
    corner_radius=0,
):
    """
    Creates a background rectangle with customizable size, color, position, and style.
    Returns a Rectangle Mobject.
    """
    rect = Rectangle(
        width=width,
        height=height,
        color=color,
        fill_color=color,
        fill_opacity=opacity,
        stroke_width=stroke_width,
        corner_radius=corner_radius,
    )
    rect.move_to(position)
    return rect


def create_centered_text(
    text,
    font_size=36,
    color=WHITE,
    position=(0, 0),
    font="",
    # weight=REGULAR,
    # slant=UPRIGHT,
    line_spacing=0.3,
    z_index=0,
):
    """
    Creates a centered Text mobject with customizable font size, color, and position.
    Returns a Text Mobject.
    """
    txt = Text(
        text,
        font_size=font_size,
        color=color,
        font=font,
        weight=weight,
        slant=slant,
        line_spacing=line_spacing,
    )
    txt.move_to(position)
    txt.z_index = z_index
    return txt


def create_colored_circle(
    radius=1.0,
    color=BLUE,
    position=(0, 0),
    fill_opacity=1.0,
    stroke_width=4,
    stroke_color=None,
):
    """
    Creates a colored circle with customizable radius, color, and position.
    Returns a Circle Mobject.
    """
    circ = Circle(
        radius=radius,
        color=stroke_color if stroke_color else color,
        fill_color=color,
        fill_opacity=fill_opacity,
        stroke_width=stroke_width,
    )
    circ.move_to(position)
    return circ


def create_invisible_text_placeholder(text, position=(0, 0)):
    """
    Creates a Text mobject with zero font size and black color, useful as a placeholder.
    Returns a Text Mobject.
    """
    txt = Text(text, font_size=0, color=BLACK)
    txt.move_to(position)
    return txt
