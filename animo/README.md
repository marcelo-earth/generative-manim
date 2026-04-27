<p align="center">
  <img
    src="https://raw.githubusercontent.com/marcelo-earth/generative-manim/main/.github/logo.png"
    align="center"
    width="100"
    alt="Animo"
    title="Animo"
  />
  <h1 align="center">Animo</h1>
</p>

<p align="center">
  🎨 Create animations from text using Manim under the hood ✨
</p>

<p align="center">
  <a href="https://animo.video">
    <img src="https://img.shields.io/static/v1?label=Platform&message=Animo&color=E11D48&logo=openai&style=flat" />
  </a>
  <a href="https://github.com/marcelo-earth/generative-manim">
    <img src="https://img.shields.io/static/v1?label=GitHub&message=Repository&color=181717&logo=github&style=flat" />
  </a>
  <a href="https://discord.com/invite/SNdbPU2AMM">
    <img src="https://img.shields.io/static/v1?label=Discord&message=Community&color=5865F2&logo=discord&style=flat" />
  </a>
  <a href="https://pypi.org/project/animo/">
    <img src="https://img.shields.io/pypi/v/animo.svg?color=blue&logo=python&logoColor=white" />
  </a>
</p>

---

## 🚀 What is Animo?

Animo is a Python package that allows you to create animations from text using Manim under the hood. It's part of the [Generative Manim](https://github.com/marcelo-earth/generative-manim) suite of tools that enables anyone to create wonderful animations from text.

Visit [animo.video](https://animo.video) to learn more about the platform and see examples of what you can create.

## 📦 Installation

```bash
pip install animo
```

## 🔑 API Key

To use Animo, you'll need an API key:

1. Visit [https://animo.video/account/developer](https://animo.video/account/developer)
2. Click on "Generate secret key"
3. Copy your API key and use it in your code

You'll also find quickstart guides and additional documentation on the developer page to help you get started.

## 🔧 Usage

### Basic Usage

```python
from animo import Animo

client = Animo(api_key="your_api_key")

# Create a single video
code = """
class GenScene(Scene):
    def construct(self):
        circle = Circle(color=BLUE)
        self.play(Create(circle))
"""

response = client.videos.create(
    code=code,
    file_class="GenScene",
    aspect_ratio="16:9"
)

# Response format
response = {
    "message": "Video generation completed",
    "video_url": "https://animovideo.blob.core.windows.net/animocontainer/video-xxx.mp4"
}

# Export multiple scenes
scenes = [
    {"videoUrl": "https://animovideo.blob.core.windows.net/animocontainer/scene1.mp4"},
    {"videoUrl": "https://animovideo.blob.core.windows.net/animocontainer/scene2.mp4"}
]

export_response = client.videos.export(
    scenes=scenes,
    title_slug="my-animation"
)
```

### Generating Videos from Text Prompts

```python
from animo import Animo
import time
import sys

# Initialize client
client = Animo(api_key="your_api_key")

# Define your prompt
prompt = "Create a blue square"

print(f"🚀 Generating video for: '{prompt}'")

# Start generation
try:
    generation = client.videos.generate(prompt=prompt)
    request_id = generation.get("requestId")
    
    if not request_id:
        print("❌ No request ID received")
        sys.exit(1)
        
    print(f"✅ Generation started with ID: {request_id}")
    
    # Poll for status with a simple progress indicator
    print("⏳ Waiting for completion", end="")
    
    while True:
        status_data = client.videos.retrieve(request_id=request_id)
        status = status_data.get("status")
        
        # Update progress indicator
        sys.stdout.write(".")
        sys.stdout.flush()
        
        # Check for completion or error
        if status == "SUCCEEDED":
            video_url = status_data.get("videoUrl")
            print(f"\n\n🎬 Video ready! URL: {video_url}")
            
            # Print processing time if available
            if processing_time := status_data.get("processingTime"):
                print(f"⏱️  Processing time: {processing_time} seconds")
                
            break
            
        elif status == "FAILED":
            error = status_data.get("error") or "Unknown error"
            print(f"\n\n❌ Generation failed: {error}")
            break
            
        # Wait before next check
        time.sleep(3)
        
except Exception as e:
    print(f"\n\n❌ Error: {str(e)}")
```

## 🤝 Contributing

We welcome contributions to Animo! Here's how to get started:

### Requirements
- Python 3.9.6 or later
- Poetry (recommended for dependency management)

### Setup for Development

1. Clone the repository
```bash
git clone https://github.com/marcelo-earth/generative-manim.git
cd generative-manim/animo
```

2. Install dependencies
```bash
poetry install
```

### Community

Join our [Discord community](https://discord.com/invite/SNdbPU2AMM) to connect with other Animo users, share your creations, and get help.

## 📄 License

[MIT License](LICENSE)
