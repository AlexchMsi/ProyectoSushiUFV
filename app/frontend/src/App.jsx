import { useRef, useState, useCallback, useEffect } from 'react'
import CameraView from './components/CameraView'
import CountPanel from './components/CountPanel'
import RecordControls from './components/RecordControls'
import ImageAnalyzer from './components/ImageAnalyzer'
import { useCamera } from './hooks/useCamera'
import { useWebSocket } from './hooks/useWebSocket'
import { useRecorder } from './hooks/useRecorder'
import './App.css'

const FRAME_INTERVAL_MS = 66 // ~15 fps

export default function App() {
  const compositeCanvasRef = useRef(null)
  const captureCanvasRef = useRef(null)
  const frameTimerRef = useRef(null)

  const [mode, setMode] = useState('camera')
  const [detections, setDetections] = useState([])
  const [counts, setCounts] = useState({})
  const [total, setTotal] = useState(0)

  const { videoRef, active, error: camError, start: startCam, stop: stopCam } = useCamera()

  const handleResult = useCallback((data) => {
    setDetections(data.detections ?? [])
    setCounts(data.counts ?? {})
    setTotal(data.total ?? 0)
  }, [])

  const { connected, connect, disconnect, sendFrame } = useWebSocket(handleResult)
  const { recording, start: startRec, stop: stopRec } = useRecorder(compositeCanvasRef)

  // Capture and send frames when both camera and WS are ready
  useEffect(() => {
    if (!active || !connected) {
      clearInterval(frameTimerRef.current)
      return
    }

    if (!captureCanvasRef.current) {
      captureCanvasRef.current = document.createElement('canvas')
    }

    frameTimerRef.current = setInterval(() => {
      const video = videoRef.current
      const canvas = captureCanvasRef.current
      if (!video || video.readyState < 2) return

      canvas.width = video.videoWidth || 640
      canvas.height = video.videoHeight || 480
      canvas.getContext('2d').drawImage(video, 0, 0)
      canvas.toBlob(
        (blob) => { if (blob) sendFrame(blob) },
        'image/jpeg',
        0.8,
      )
    }, FRAME_INTERVAL_MS)

    return () => clearInterval(frameTimerRef.current)
  }, [active, connected, videoRef, sendFrame])

  const handleStart = async () => {
    connect()
    await startCam()
  }

  const handleStop = () => {
    clearInterval(frameTimerRef.current)
    if (recording) stopRec()
    stopCam()
    disconnect()
    setDetections([])
    setCounts({})
    setTotal(0)
  }

  const handleModeChange = (newMode) => {
    if (newMode === 'image' && active) handleStop()
    setMode(newMode)
  }

  return (
    <div className="app">
      <header className="app__header">
        <h1 className="app__title">Sushi Counter</h1>
        <div className="app__tabs">
          <button
            className={`tab-btn ${mode === 'camera' ? 'tab-btn--active' : ''}`}
            onClick={() => handleModeChange('camera')}
          >
            Cámara
          </button>
          <button
            className={`tab-btn ${mode === 'image' ? 'tab-btn--active' : ''}`}
            onClick={() => handleModeChange('image')}
          >
            Imagen
          </button>
        </div>
        {mode === 'camera' && (
          <div className="app__status">
            <span className={`status-dot ${connected ? 'status-dot--ok' : 'status-dot--off'}`} />
            {connected ? 'Conectado' : 'Desconectado'}
          </div>
        )}
      </header>

      <main className="app__main">
        {mode === 'image' ? (
          <ImageAnalyzer />
        ) : (
          <div className="app__body">
            <div className="app__video-wrap">
              <CameraView
                ref={compositeCanvasRef}
                videoRef={videoRef}
                detections={detections}
                active={active}
              />

              {camError && <p className="error-msg">Error de cámara: {camError}</p>}

              <div className="app__controls">
                {!active ? (
                  <button className="btn btn--primary" onClick={handleStart}>
                    Iniciar
                  </button>
                ) : (
                  <button className="btn btn--secondary" onClick={handleStop}>
                    Detener
                  </button>
                )}
                <RecordControls
                  recording={recording}
                  onStart={startRec}
                  onStop={stopRec}
                  disabled={!active}
                />
              </div>
            </div>

            <CountPanel counts={counts} total={total} />
          </div>
        )}
      </main>
    </div>
  )
}
