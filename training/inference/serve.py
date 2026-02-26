"""vLLM server for serving the best trained model."""

import argparse
import os


def serve(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 4096,
):
    """Launch vLLM OpenAI-compatible server."""
    try:
        from vllm import LLM, SamplingParams
        from vllm.entrypoints.openai.api_server import run_server
    except ImportError:
        print("vLLM not installed. Install with: pip install vllm")
        print("\nAlternatively, use the LocalModelEngine in engine.py for direct inference.")
        return

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--host", host,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
    ]

    print(f"Starting vLLM server:")
    print(f"  Model: {model_path}")
    print(f"  Host: {host}:{port}")
    print(f"  API compatible with OpenAI format")
    print(f"\nUsage:")
    print(f'  curl http://{host}:{port}/v1/completions -d \'{{"model": "{model_path}", "prompt": "Create a circle animation"}}\'')

    os.execvp("python", cmd)


def main():
    parser = argparse.ArgumentParser(description="Serve model with vLLM")
    parser.add_argument("--model", type=str, required=True, help="Path to merged model or HF model ID")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-memory", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()

    serve(
        model_path=args.model,
        host=args.host,
        port=args.port,
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=args.max_model_len,
    )


if __name__ == "__main__":
    main()
