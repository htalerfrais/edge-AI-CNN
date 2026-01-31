# Guide : Pipeline GStreamer, WSL et flux webcam

## Vue d'ensemble

L'app reçoit un flux vidéo H264 sur TCP, traite les images (détection de chiffres), et renvoie le flux annoté sur un autre port TCP.

```
[Webcam] → GStreamer (encode H264) → TCP :5000 → [App Docker] → TCP :8554 → [VLC / GStreamer]
```

---

## Formats de flux : MPEG-TS vs H264 brut

### Problème

- **H264 brut** (`byte-stream`) : fonctionne mal sur TCP entre Windows et WSL (erreurs `Internal data stream error`, `not-negotiated`).
- **MPEG-TS** : conteneur qui encapsule le H264 de façon fiable pour le streaming TCP.

### Solution retenue

Tous les pipelines (Windows, Linux natif) envoient du **MPEG-TS** sur le port 5000. L'app utilise `tsdemux` pour extraire le H264 avant décodage.

| Étape | Pipeline |
|-------|----------|
| **Émetteur** | `... ! x264enc ! h264parse config-interval=1 ! mpegtsmux ! tcpserversink port=5000` |
| **Récepteur (app)** | `tcpclientsrc ! tsdemux ! h264parse ! avdec_h264 ! ...` |

---

## Modes d'exécution (run_local.sh)

### Mode natif (Linux / Mac)

- Webcam locale (`/dev/video0`)
- GStreamer lance le pipeline sur la machine
- App connectée à `127.0.0.1:5000`

```bash
./run_local.sh start   # ou debug
```

### Mode WSL (Windows + WSL2)

- Webcam sur **Windows** (pas accessible depuis WSL)
- Pipeline GStreamer sur **Windows** (PowerShell)
- App dans **WSL** (Docker) se connecte à l’IP Windows

**Ordre de démarrage :**

1. **Windows** : lancer le pipeline webcam
   ```powershell
   .\run_pipeline_win.ps1
   ```

2. **WSL** : lancer l’app
   ```bash
   WIN_HOST=192.168.56.1 ./run_local.sh debug   # ou start
   ```

---

## Réseau WSL ↔ Windows

### IP de l’hôte Windows

- WSL2 utilise un réseau virtuel
- L’IP Windows n’est pas `127.0.0.1` depuis WSL
- Méthodes pour l’obtenir :
  - `grep nameserver /etc/resolv.conf` (souvent incorrect)
  - Variable `WIN_HOST` : `WIN_HOST=192.168.56.1 ./run_local.sh debug`
  - `ip route | grep default` ou l’IP affichée par VirtualBox/VMware si utilisé

### Vérifier la connectivité

```bash
nc -zv 192.168.56.1 5000   # doit afficher "succeeded"
```

---

## Visualisation du flux de sortie (port 8554)

L’app envoie le flux annoté en MPEG-TS sur le port 8554.

### Depuis WSL (même machine que l’app)

```bash
gst-launch-1.0 tcpclientsrc host=127.0.0.1 port=8554 ! tsdemux ! h264parse ! avdec_h264 ! autovideosink sync=false
```

### Depuis Windows (VLC ou GStreamer)

- **IP WSL** : `ip addr show eth0` → ex. `172.18.147.22`
- **VLC** : Média → Ouvrir un flux réseau → `tcp://172.18.147.22:8554`
- **GStreamer** :
  ```powershell
  gst-launch-1.0 tcpclientsrc host=172.18.147.22 port=8554 ! tsdemux ! h264parse ! avdec_h264 ! autovideosink
  ```

---

## GStreamer sur Windows

### Conda vs GStreamer système

- **Conda** : plugins incomplets (ex. `h264parse` manquant)
- **GStreamer officiel** : à installer depuis [gstreamer.freedesktop.org](https://gstreamer.freedesktop.org/download/)

Pour utiliser le GStreamer système : quitter Conda (`conda deactivate`) avant de lancer le pipeline.

---

## Récapitulatif des commandes

| Action | Commande |
|--------|----------|
| Build | `./run_local.sh build` |
| Démarrer (natif) | `./run_local.sh start` |
| Démarrer (WSL) | `.\run_pipeline_win.ps1` puis `WIN_HOST=192.168.56.1 ./run_local.sh start` |
| Debug (voir logs) | `./run_local.sh debug` |
| Arrêter | `./run_local.sh stop` |
| Voir le flux sortie | `gst-launch-1.0 tcpclientsrc host=127.0.0.1 port=8554 ! tsdemux ! h264parse ! avdec_h264 ! autovideosink` |

---

## Dépannage

| Symptôme | Piste |
|----------|------|
| `Erreur: input` | Pipeline Windows pas lancé, mauvaise IP, ou pare-feu |
| `Internal data stream error` | Format incompatible (vérifier MPEG-TS) ou GStreamer Conda |
| `no element "h264parse"` | Utiliser le GStreamer système, pas Conda |
| VLC cône orange | VLC sur Windows : utiliser l’IP WSL, pas 127.0.0.1 |
| App s’arrête seule | Déconnexion du flux d’entrée (pipeline Windows arrêté) |
