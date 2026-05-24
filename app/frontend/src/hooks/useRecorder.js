import { useRef, useState, useCallback } from 'react'

export function useRecorder(canvasRef) {
  const recorderRef = useRef(null)
  const chunksRef = useRef([])
  const [recording, setRecording] = useState(false)

  const start = useCallback(() => {
    if (!canvasRef.current) return
    const stream = canvasRef.current.captureStream(30)
    const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
      ? 'video/webm;codecs=vp9'
      : 'video/webm'
    const recorder = new MediaRecorder(stream, { mimeType })
    chunksRef.current = []

    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data)
    }

    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: mimeType })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `sushi-session-${Date.now()}.webm`
      a.click()
      URL.revokeObjectURL(url)
    }

    recorder.start(100) // collect data every 100ms
    recorderRef.current = recorder
    setRecording(true)
  }, [canvasRef])

  const stop = useCallback(() => {
    recorderRef.current?.stop()
    recorderRef.current = null
    setRecording(false)
  }, [])

  return { recording, start, stop }
}
