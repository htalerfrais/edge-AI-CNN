# Pi 5 Camera Docker

Stream caméra v3 sur Pi 5 via Docker avec OpenCV C++.

## Architecture

```
[Caméra v3] → rpicam-vid:5000 → [Docker OpenCV]:8554 → [Client]
```

## Usage

```bash
./run.sh all          # Build + démarrer
./run.sh view         # Afficher le flux
./run.sh stop         # Arrêter
./run.sh logs         # Voir les logs
vlc "tcp://IP:8554"   # Affiche depuis un remote screen
```

## Prérequis

**Pi 5:** Docker, rpicam-vid  
**Client:** vlc

## Ajouter du traitement

Éditer `main.cpp` dans la boucle :

```cpp
// === TRAITEMENT OPENCV ICI ===
cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
cv::Canny(gray, edges, 50, 150);
```

Puis: `./run.sh build && ./run.sh start`
