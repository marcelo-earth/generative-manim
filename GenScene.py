# Manim code generated with OpenAI GPT
# Command to generate animation: manim GenScene.py GenScene --format=mp4 --media_dir . --custom_folders video_dir

from manim import *
from math import *

# Force the output to be the scene name to guarantee uniqueness: https://github.com/360macky/generative-manim/issues/23
SceneFileWriter.force_output_as_scene_name = True

class GenScene(Scene):
    def construct(self):
        c = Circle(color=BLUE)
        self.play(Create(c))
