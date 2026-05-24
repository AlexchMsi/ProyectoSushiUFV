import { useRef, useState, useCallback, useEffect } from 'react'

const WS_URL = '/ws/detect'

export function useWebSocket(onResult) {
  const wsRef = useRef(null)
  const [connected, setConnected] = useState(false)
  const onResultRef = useRef(onResult)
  useEffect(() => { onResultRef.current = onResult }, [onResult])

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const url = `${protocol}://${window.location.host}${WS_URL}`
    const ws = new WebSocket(url)
    ws.binaryType = 'arraybuffer'

    ws.onopen = () => setConnected(true)
    ws.onclose = () => setConnected(false)
    ws.onerror = () => setConnected(false)
    ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data)
        onResultRef.current?.(data)
      } catch (_) {}
    }

    wsRef.current = ws
  }, [])

  const disconnect = useCallback(() => {
    wsRef.current?.close()
    wsRef.current = null
  }, [])

  const sendFrame = useCallback((blob) => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) return
    blob.arrayBuffer().then(buf => wsRef.current?.send(buf))
  }, [])

  return { connected, connect, disconnect, sendFrame }
}
