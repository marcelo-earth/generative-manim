"""15 Manim categories with 5 seed prompts each (75 seeds total)."""

CATEGORIES: dict[str, list[str]] = {
    "geometry_2d": [
        "Create an animation showing a triangle inscribed in a circle, then highlight each vertex",
        "Animate a square morphing into a regular hexagon with smooth transitions",
        "Show the construction of a golden spiral using quarter circles in successive squares",
        "Create a Venn diagram with three overlapping circles that fill with color one by one",
        "Animate the proof that the exterior angles of any polygon sum to 360 degrees",
    ],
    "geometry_3d": [
        "Create a rotating 3D cube with colored faces",
        "Show a sphere being sliced by a plane to reveal a cross-section circle",
        "Animate a 3D coordinate system with labeled axes and a point moving along a parametric curve",
        "Create a 3D surface plot of z = sin(x) * cos(y) that rotates slowly",
        "Show a cylinder transforming into a cone by collapsing the top face to a point",
    ],
    "text_and_latex": [
        "Display the quadratic formula with each term appearing one by one, then highlight the discriminant",
        "Create a title card that types out 'Introduction to Calculus' with a typewriter effect",
        "Show Euler's identity e^(i*pi) + 1 = 0 with each symbol colored differently",
        "Animate the expansion of (a+b)^3 step by step using LaTeX",
        "Display a numbered list of Newton's three laws of motion, fading in sequentially",
    ],
    "graphs_and_charts": [
        "Create an animated bar chart showing monthly sales data with bars growing from zero",
        "Plot a sine wave that draws itself from left to right with a tracing dot",
        "Show a pie chart that assembles slice by slice with percentage labels",
        "Animate a line graph of stock prices with a moving average overlay",
        "Create a scatter plot where points appear one by one and then a best-fit line draws through them",
    ],
    "transforms": [
        "Show a circle transforming into a square using ReplacementTransform",
        "Animate text changing from 'Hello' to 'World' with a smooth morph",
        "Create a grid of dots that transforms into a sine wave pattern",
        "Show the letter A being reflected, rotated, and scaled in sequence",
        "Animate a complex shape being built from simple primitives using successive transforms",
    ],
    "camera_movement": [
        "Create a scene with many objects and zoom into a specific region",
        "Show a number line from -10 to 10 and smoothly pan from left to right",
        "Create a 3D scene where the camera orbits around a central object",
        "Animate a fractal-like pattern and zoom into increasingly fine detail",
        "Show a wide scene with multiple groups, framing each group one at a time",
    ],
    "physics": [
        "Simulate a projectile motion trajectory with velocity vectors shown at each point",
        "Animate a simple pendulum swinging back and forth with a traced path",
        "Show two objects colliding elastically with momentum vectors before and after",
        "Create a visualization of electric field lines between a positive and negative charge",
        "Animate a wave interference pattern where two circular waves overlap",
    ],
    "value_trackers": [
        "Show a slider controlling the radius of a circle in real-time",
        "Create a counter that counts from 0 to 100 with the number displayed",
        "Animate a function graph where a parameter smoothly changes the curve shape",
        "Show a progress bar that fills up while a number tracks the percentage",
        "Create an interactive-style demo where a point on a curve moves and its coordinates update",
    ],
    "color_and_style": [
        "Create a rainbow gradient that flows across a row of circles",
        "Animate a shape cycling through multiple colors with smooth interpolation",
        "Show text with a gradient fill from blue to red",
        "Create a neon glow effect on geometric shapes against a dark background",
        "Animate objects changing opacity to create a fade-in fade-out sequence",
    ],
    "groups_and_composition": [
        "Arrange 12 circles in a 3x4 grid with equal spacing",
        "Create a fractal tree using recursive branching of line segments",
        "Show a deck of cards fanning out from a single point",
        "Build a molecule diagram by assembling atom circles and bond lines",
        "Create a tiling pattern of hexagons that fills the screen",
    ],
    "vector_fields": [
        "Show a 2D vector field for F(x,y) = (-y, x) representing rotation",
        "Animate particles flowing through a vector field with stream lines",
        "Create a gradient field visualization with arrows showing direction and magnitude",
        "Show the curl of a vector field by animating small paddle wheels",
        "Visualize a gravitational field around two massive objects",
    ],
    "tables_and_matrices": [
        "Create an animated truth table for AND, OR, and NOT operations",
        "Show a 3x3 matrix multiplication step by step, highlighting each dot product",
        "Display a comparison table of sorting algorithms with their complexities",
        "Animate a matrix being transposed with elements moving to new positions",
        "Create a periodic table section that highlights specific element groups",
    ],
    "svg_and_images": [
        "Create a scene with custom arrow styles pointing in different directions",
        "Show a floor plan layout using rectangles and lines with room labels",
        "Build a simple circuit diagram with resistor and battery symbols",
        "Create an organizational chart with boxes and connecting lines",
        "Animate a simple map with points of interest connected by dashed lines",
    ],
    "complex_scenes": [
        "Create a full lesson on the Pythagorean theorem with visual proof",
        "Animate the sorting process of bubble sort on an array of bars",
        "Show a neural network diagram with data flowing through layers",
        "Create a timeline animation showing key events in computer science history",
        "Animate the RSA encryption process with key generation, encryption, and decryption steps",
    ],
    "educational": [
        "Explain the concept of derivatives visually with a tangent line sliding along a curve",
        "Show how binary counting works from 0 to 15 with bit flips",
        "Animate the water cycle with clouds, rain, rivers, and evaporation",
        "Create a visual explanation of how recursion works with a factorial example",
        "Show the difference between linear and exponential growth with animated graphs",
    ],
}

# Total: 15 categories × 5 seeds = 75 seed prompts
TOTAL_SEEDS = sum(len(prompts) for prompts in CATEGORIES.values())
ALL_SEEDS = [(cat, prompt) for cat, prompts in CATEGORIES.items() for prompt in prompts]
