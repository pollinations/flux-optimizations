# Pollinations-Compatible Flux Schnell FP8 Server

This server replicates the Pollinations nunchaku service interface using our optimized Flux Schnell FP8 backend. It automatically registers with the Pollinations network and provides the same API interface.

## Features

- **Pollinations Compatible**: Same API interface as nunchaku service
- **Automatic Registration**: Heartbeat system registers with `https://image.pollinations.ai/register`
- **Multi-GPU Support**: Run multiple instances on different GPUs
- **FP8 Optimization**: ~3 second generation time per 1024x1024 image
- **Safety Checking**: NSFW content detection
- **Base64 Response**: Images returned as base64 encoded JPEG

## Quick Start

### Single GPU Server

```bash
# Start server on GPU 0, port 8765
./start_server.sh

# Or with custom settings
PORT=8080 GPU_ID=1 SERVICE_TYPE=flux ./start_server.sh
```

### Multi-GPU Servers

```bash
# Start 4 servers on GPUs 0-3, ports 8765-8768
./multi_gpu_server.sh

# Custom configuration
BASE_PORT=9000 NUM_GPUS=8 ./multi_gpu_server.sh

# Stop all servers
./stop_servers.sh
```

## API Usage

### Generate Images

```bash
curl -X POST "http://localhost:8765/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": ["a majestic dragon soaring through clouds at sunset"],
    "width": 1024,
    "height": 1024,
    "steps": 4,
    "seed": 12345
  }'
```

### Health Check

```bash
curl http://localhost:8765/health
```

## Environment Variables

- `PORT`: Server port (default: 8765)
- `GPU_ID`: GPU to use (default: 0)
- `SERVICE_TYPE`: Service type for registration (default: flux)

## API Endpoints

- `POST /generate`: Generate images (compatible with nunchaku)
- `GET /health`: Health check
- `GET /`: Service information

## Request Format

```json
{
  "prompts": ["text prompt"],
  "width": 1024,
  "height": 1024,
  "steps": 4,
  "seed": null,
  "safety_checker_adj": 0.5
}
```

## Response Format

```json
[{
  "image": "base64_encoded_jpeg",
  "has_nsfw_concept": false,
  "concept": [],
  "width": 1024,
  "height": 1024,
  "seed": 12345,
  "prompt": "text prompt"
}]
```

## Registration System

The server automatically:
1. Gets public IP address
2. Registers with Pollinations at `https://image.pollinations.ai/register`
3. Sends heartbeat every 30 seconds
4. Includes service URL and type in registration

## Performance

- **Generation Time**: ~3 seconds per 1024x1024 image
- **Memory Usage**: Optimized with FP8 quantization
- **Throughput**: Multiple parallel instances on different GPUs
- **Compatibility**: Works on H100 cluster environments

## Requirements

Install dependencies:
```bash
pip install -r server_requirements.txt
```

Ensure models are downloaded:
```bash
python3 flux_schnell_fp8_standalone.py --download-models
```

## Troubleshooting

- **Model Loading**: Ensure models are downloaded first
- **GPU Memory**: Each instance needs ~8GB VRAM
- **Network**: Server needs internet access for registration
- **Ports**: Ensure ports are not blocked by firewall
