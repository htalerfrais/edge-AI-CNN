#!/bin/bash
# Local webcam → TCP :5000 → app (Docker)

PIDFILE=/tmp/run_local_gst.pid

case "$1" in
  build)
    docker build -f docker/Dockerfile -t cam .
    ;;
  start)
    kill $(cat "$PIDFILE" 2>/dev/null) 2>/dev/null
    rm -f "$PIDFILE"
    docker stop cam 2>/dev/null
    sleep 1
    gst-launch-1.0 \
      v4l2src device=/dev/video0 ! \
      video/x-raw,width=1280,height=720,framerate=30/1 ! \
      videoconvert ! \
      x264enc tune=zerolatency speed-preset=ultrafast ! \
      video/x-h264,streamformat=byte-stream ! \
      tcpserversink host=127.0.0.1 port=5000 &
    echo $! > "$PIDFILE"
    sleep 2
    docker run -d --rm --name cam --network host cam
    echo "Started - app reading from local webcam, output tcp://localhost:8554"
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
  *)
    echo "Usage: $0 {build|start|stop|all}"
    ;;
esac
