# Pi 5 Digit AI

This project aims to design and deploy a lightweight embedded artificial intelligence application on the Raspberry Pi 5 equipped with the Raspberry Pi Camera Module v3, capable of detecting and recognizing handwritten digits (0–9) in real time from the camera feed. The camera stream is processed inside a Docker container using OpenCV and C++, and the resulting video feed is redistributed over a TCP port for remote viewing.


## Prerequisites

**Pi 5:** Docker, rpicam-vid  
**Client:** vlc

## Architecture

```
[Camera v3] → rpicam-vid:5000 → [Docker OpenCV]:8554 → [Client]
```

## How to use

### Installation

Ensure that your Raspberry Pi is connected to the same network as your host machine. Access it remotely using SSH and run the following commands :

```bash
git clone --no-checkout git@github.com:htalerfrais/edge-ai-cnn pi5-digit-ai
cd pi5-digit-ai

git sparse-checkout init --cone

git sparse-checkout set --skip-checks run.sh docker inference_c models

git checkout main

chmod +x run.sh
```

Only the essential files and folders for the application will be cloned, keeping the local repository lightweight. The resulting directory structure should look like this:

```
pi5-digit-ai
├── docker
│   └── Dockerfile
├── inference_c
│   ├── main.cpp
│   ├── Makefile
│   └── ...
├── models
│   ├── cnn_weights.txt
│   └── mlp_weights.txt
└── run.sh   
```

### Build and Start

```bash
./run.sh all            # Build + start
./run.sh logs           # Show logs
vlc "tcp://IP:8554"     # View from remote
```

### Other commands

```bash
./run.sh build          # Build
./run.sh start          # Start
./run.sh stop           # Stop
./run.sh view           # Show stream (in terminal)
```

## Changing or Adding Neural Network Models

The application currently supports pre-trained neural network models for digit recognition. You can modify the existing network architecture or add new models if needed.

**Important:** 

1. Any changes to the neural network structure require corresponding updates in the `neural_network.c` and `neural_network.h` files. Specifically, the `forward_pass_*`  and `load_*_model` functions must be adapted or implemented to match the new architecture.  
2. `main.cpp` must be updated to use the new model :
   - Add the model path and load it.
   - Update the call to the appropriate `forward_pass_*` function.

After updating, rebuild the Docker container using:

```bash
./run.sh build
```