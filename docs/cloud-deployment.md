# Cloud Deployment Guide

This guide covers a practical production deployment for the Generative Manim API on Render, Fly.io, Railway, Google Cloud Run, AWS ECS, or any platform that can run a Docker container.

## What You Deploy

Deploy the API container from the repository root. The API:

- listens on `PORT` and defaults to `8080`
- exposes endpoints such as `/v1/code/generation`, `/v1/chat/generation`, `/v1/video/rendering`, and `/v1/video/generation`
- runs Manim and FFmpeg inside the container
- stores generated videos either in the container filesystem or Azure Blob Storage, depending on environment variables

For production, Docker is the recommended deployment path because Manim needs system packages such as Cairo, Pango, FFmpeg, and LaTeX.

## Recommended Architecture

Use one web service for the API and external object storage for rendered videos.

```text
Client or app
  -> Generative Manim API container
      -> OpenAI or Anthropic API for code generation
      -> Manim plus FFmpeg for rendering
      -> Azure Blob Storage or another persistent public storage layer
```

For small experiments, local container storage is fine. For public or long-running deployments, do not rely on the container filesystem because many cloud platforms use ephemeral disks.

## Environment Variables

Set these variables in your cloud provider dashboard:

```bash
PORT=8080
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
USE_LOCAL_STORAGE=true
BASE_URL=https://your-api-domain.example
```

If you use Azure Blob Storage for rendered videos, set:

```bash
USE_LOCAL_STORAGE=false
AZURE_STORAGE_CONNECTION_STRING=your_azure_storage_connection_string
AZURE_STORAGE_CONTAINER_NAME=your_container_name
```

`BASE_URL` should be the public URL of your deployed API when `USE_LOCAL_STORAGE=true`. This lets the API return usable `/public/...` video URLs.

## Deploying on Render

1. Create a new **Web Service** in Render.
2. Connect your GitHub repository.
3. Select **Docker** as the runtime.
4. Use the repository root as the build context.
5. Set the environment variables listed above.
6. Choose an instance size with enough CPU and memory for Manim renders. Avoid free or very small instances for anything beyond quick smoke tests.
7. Deploy.

After deployment, verify the API:

```bash
curl https://your-render-service.onrender.com/
```

Expected response:

```text
Generative Manim Processor
```

Then test code generation:

```bash
curl -X POST https://your-render-service.onrender.com/v1/code/generation \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Create a Manim animation with a blue circle expanding from the center."}'
```

And test rendering:

```bash
curl -X POST https://your-render-service.onrender.com/v1/video/rendering \
  -H "Content-Type: application/json" \
  -d '{
    "code": "class DrawCircle(Scene):\n    def construct(self):\n        circle = Circle(color=BLUE)\n        self.play(Create(circle))",
    "file_class": "DrawCircle",
    "project_name": "cloud-smoke-test",
    "iteration": 1,
    "aspect_ratio": "16:9",
    "stream": false
  }'
```

## Deploying on Other Docker Hosts

Build the image:

```bash
docker build -t generative-manim-api .
```

Run it:

```bash
docker run --rm -p 8080:8080 \
  -e PORT=8080 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  -e USE_LOCAL_STORAGE=true \
  -e BASE_URL="http://localhost:8080" \
  generative-manim-api
```

On managed platforms, map the platform's public URL to `BASE_URL`.

## Dependencies

The Dockerfile installs the main native dependencies needed for rendering:

- `ffmpeg`
- `libcairo2-dev`
- `libpango1.0-dev`
- `pkg-config`
- TeX Live packages for LaTeX and MathTex rendering
- Python dependencies from `api/requirements.txt`

If you deploy without Docker, install the same native packages on the host before running:

```bash
pip install -r api/requirements.txt
python run.py
```

Docker is usually simpler because native Manim dependencies vary by operating system.

## Performance and Reliability

Video rendering is CPU-heavy and can run for minutes. Start with:

- at least 2 CPU cores for general use
- at least 2 GB memory for simple scenes
- more CPU and memory for LaTeX-heavy, 3D, or long animations
- request timeouts of 5 minutes or more

For production traffic, add a job queue instead of rendering every request inside the web process. A durable design is:

```text
API receives render request
  -> enqueue job
  -> worker renders video
  -> upload video to object storage
  -> client polls job status or receives a webhook
```

That avoids web request timeouts and lets you scale render workers separately from API traffic.

## Storage Notes

`USE_LOCAL_STORAGE=true` writes videos into the API's local `public` directory and returns URLs under `/public/...`. This is convenient, but it is not durable on many cloud hosts.

`USE_LOCAL_STORAGE=false` uploads rendered videos through Azure Blob Storage in the current API implementation. If you want S3, Cloudflare R2, or Google Cloud Storage, add a storage adapter with the same behavior as `upload_to_azure_storage`.

## Common Issues

- **Render fails with missing LaTeX or Cairo errors**: use Docker, or install the native packages from the Dockerfile on the host.
- **Returned video URL is localhost**: set `BASE_URL` to the public API URL.
- **Videos disappear after redeploy**: use external object storage instead of local storage.
- **Requests time out**: use a larger instance, shorter scenes, or a background worker queue.
- **Out-of-memory errors**: increase instance memory or reduce render quality/scene complexity.

## Security

Generated Manim code is Python code. Treat rendering as untrusted code execution unless all prompts and code come from trusted users. For public deployments, isolate render workers, restrict network access where possible, add authentication, rate-limit requests, and set per-job time and memory limits.
