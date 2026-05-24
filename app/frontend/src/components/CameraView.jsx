import { useEffect, useRef, forwardRef } from 'react'
import { drawDetections } from '../utils/drawDetections'

/**
 * Shows the webcam feed with detection overlay.
 * The compositeCanvasRef is exposed so other hooks (useRecorder) can capture it.
 */
const CameraView = forwardRef(function CameraView({ videoRef, detections, active }, compositeCanvasRef) {
  const overlayRef = useRef(null)

  // Draw detections on the overlay canvas whenever they change
  useEffect(() => {
    const canvas = overlayRef.current
    const video = videoRef.current
    if (!canvas || !video) return

    // Keep overlay intrinsic size in sync with the video feed
    const vw = video.videoWidth || 640
    const vh = video.videoHeight || 480
    if (canvas.width !== vw || canvas.height !== vh) {
      canvas.width = vw
      canvas.height = vh
    }

    const ctx = canvas.getContext('2d')
    drawDetections(ctx, detections, 1, 1)
  }, [detections, videoRef])

  // Continuously composite video + overlay onto the recording canvas
  useEffect(() => {
    if (!active) return
    let rafId
    const composite = compositeCanvasRef?.current
    const video = videoRef.current
    const overlay = overlayRef.current
    if (!composite || !video || !overlay) return

    const draw = () => {
      composite.width = video.videoWidth || 1280
      composite.height = video.videoHeight || 720
      overlay.width = composite.width
      overlay.height = composite.height
      const ctx = composite.getContext('2d')
      ctx.drawImage(video, 0, 0, composite.width, composite.height)
      ctx.drawImage(overlay, 0, 0)
      rafId = requestAnimationFrame(draw)
    }
    rafId = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(rafId)
  }, [active, videoRef, compositeCanvasRef])

  return (
    <div className="camera-container">
      <video
        ref={videoRef}
        muted
        playsInline
        className="camera-video"
        style={{ display: active ? 'block' : 'none' }}
      />
      <canvas ref={overlayRef} className="camera-overlay" />
      {/* Hidden composite canvas used for recording */}
      <canvas ref={compositeCanvasRef} style={{ display: 'none' }} />
      {!active && (
        <div className="camera-placeholder">
          <span>Presiona "Iniciar" para activar la cámara</span>
        </div>
      )}
    </div>
  )
})

export default CameraView
