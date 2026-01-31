# Webcam Windows -> TCP :5000 (MPEG-TS / H264)
# À lancer AVANT ./run_local.sh start ou debug (mode WSL)
# Utilise GStreamer officiel (pas Conda) - sortir de conda si besoin

Write-Host "Starting webcam pipeline on port 5000 (MPEG-TS)..."
Write-Host "Press Ctrl+C to stop."
Write-Host ""

gst-launch-1.0 mfvideosrc ! `
  video/x-raw,width=1280,height=720,framerate=30/1 ! `
  videoconvert ! queue ! `
  x264enc tune=zerolatency speed-preset=ultrafast key-int-max=15 ! `
  video/x-h264,profile=baseline ! `
  h264parse config-interval=1 ! `
  mpegtsmux ! `
  tcpserversink host=0.0.0.0 port=5000
