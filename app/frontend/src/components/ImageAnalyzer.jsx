import { useRef, useState, useCallback } from 'react'
import { drawDetections } from '../utils/drawDetections'
import CountPanel from './CountPanel'

export default function ImageAnalyzer() {
  const canvasRef = useRef(null)
  const fileInputRef = useRef(null)
  const [counts, setCounts] = useState({})
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(false)
  const [hasImage, setHasImage] = useState(false)

  const analyze = useCallback(async (file) => {
    if (!file || !file.type.startsWith('image/')) return

    setLoading(true)
    setCounts({})
    setTotal(0)

    const img = new Image()
    const url = URL.createObjectURL(file)

    img.onload = async () => {
      const canvas = canvasRef.current
      canvas.width = img.naturalWidth
      canvas.height = img.naturalHeight
      const ctx = canvas.getContext('2d')
      ctx.drawImage(img, 0, 0)
      setHasImage(true)

      try {
        const form = new FormData()
        form.append('file', file)
        const res = await fetch('/detect', { method: 'POST', body: form })
        const data = await res.json()
        ctx.drawImage(img, 0, 0)
        drawDetections(ctx, data.detections ?? [], 1, 1, false)
        setCounts(data.counts ?? {})
        setTotal(data.total ?? 0)
      } catch (err) {
        console.error('[ImageAnalyzer] fetch error:', err)
      } finally {
        URL.revokeObjectURL(url)
        setLoading(false)
      }
    }

    img.onerror = () => {
      URL.revokeObjectURL(url)
      setLoading(false)
    }

    img.src = url
  }, [])

  const handleFileInput = (e) => analyze(e.target.files[0])

  const handleDrop = (e) => {
    e.preventDefault()
    analyze(e.dataTransfer.files[0])
  }

  return (
    <div className="app__body">
      <div className="app__video-wrap">
        <div
          className="camera-container"
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
        >
          <canvas
            ref={canvasRef}
            className="image-canvas"
            style={{ display: hasImage ? 'block' : 'none' }}
          />
          {!hasImage && (
            <div className="camera-placeholder">
              <p>Arrastra una imagen aquí</p>
              <p style={{ fontSize: '0.75rem', marginTop: 4 }}>o usa el botón de abajo</p>
            </div>
          )}
          {loading && (
            <div className="image-analyzing">Analizando...</div>
          )}
        </div>

        <div className="app__controls">
          <button
            className="btn btn--primary"
            onClick={() => fileInputRef.current.click()}
            disabled={loading}
          >
            {loading ? 'Analizando…' : hasImage ? 'Cambiar imagen' : 'Seleccionar imagen'}
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={handleFileInput}
          />
        </div>
      </div>

      <CountPanel counts={counts} total={total} />
    </div>
  )
}
