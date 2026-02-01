#!/bin/bash
# Local webcam → TCP :5000 → app (Docker)
# Mode: hybrid (WSL) | native (Linux/Mac)
# - Hybrid: run run_pipeline_win.ps1 on Windows first, then ./run_local.sh start
# - Native: pipeline starts here (v4l2src), app connects to 127.0.0.1

PIDFILE=/tmp/run_local_gst.pid

case "$1" in
  build)
    docker build -f docker/Dockerfile -t cam .
    ;;
  start)
    # WSL: pipeline on Windows, app connects to Windows host. Native: pipeline here, app to 127.0.0.1
    # WIN_HOST overrides auto-detection (e.g. WIN_HOST=192.168.56.1 ./run_local.sh start)
    if grep -qi microsoft /proc/version 2>/dev/null && [ -f /etc/resolv.conf ]; then
      IN_HOST="${WIN_HOST:-$(grep nameserver /etc/resolv.conf | awk '{print $2}')}"
      [ -z "$IN_HOST" ] && IN_HOST="127.0.0.1"
      echo "WSL mode: connecting to Windows host $IN_HOST (run run_pipeline_win.ps1 on Windows first)"
    else
      IN_HOST="127.0.0.1"
    fi
    kill $(cat "$PIDFILE" 2>/dev/null) 2>/dev/null
    rm -f "$PIDFILE"
    docker stop cam 2>/dev/null
    sleep 1
    if [ "$IN_HOST" = "127.0.0.1" ]; then
      gst-launch-1.0 \
        v4l2src device=/dev/video0 ! \
        video/x-raw,width=1280,height=720,framerate=30/1 ! \
        videoconvert ! \
        x264enc tune=zerolatency speed-preset=ultrafast ! \
        video/x-h264,streamformat=byte-stream ! \
        tcpserversink host=127.0.0.1 port=5000 &
      echo $! > "$PIDFILE"
      sleep 2
    fi
    docker run -d --rm --name cam --network host cam "$IN_HOST" 5000 8554 1280 720
    echo "Started - app reading from $IN_HOST:5000, output tcp://localhost:8554"
    ;;
  stop)
    kill $(cat "$PIDFILE" 2>/dev/null) 2>/dev/null
    rm -f "$PIDFILE"
    docker stop cam 2>/dev/null
    echo "Stopped"
    ;;
  all)
    $0 build && $0 start
    ;;
  debug)
    # Run app in foreground to see errors (no -d)
    # WIN_HOST overrides auto-detection (e.g. WIN_HOST=192.168.56.1 ./run_local.sh debug)
    if grep -qi microsoft /proc/version 2>/dev/null && [ -f /etc/resolv.conf ]; then
      IN_HOST="${WIN_HOST:-$(grep nameserver /etc/resolv.conf | awk '{print $2}')}"
      [ -z "$IN_HOST" ] && IN_HOST="127.0.0.1"
      echo "WSL mode: connecting to $IN_HOST:5000 (run run_pipeline_win.ps1 on Windows first)"
    else
      IN_HOST="127.0.0.1"
    fi
    docker stop cam 2>/dev/null
    docker run --rm --name cam --network host cam "$IN_HOST" 5000 8554 1280 720
    ;;
  *)
    echo "Usage: $0 {build|start|stop|all|debug}"
    ;;
esac
