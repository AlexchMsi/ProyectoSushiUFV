export default function RecordControls({ recording, onStart, onStop, disabled }) {
  return (
    <div className="record-controls">
      {recording ? (
        <button className="btn btn--danger" onClick={onStop}>
          <span className="rec-dot" /> Detener y Descargar
        </button>
      ) : (
        <button className="btn btn--record" onClick={onStart} disabled={disabled}>
          Grabar Sesión
        </button>
      )}
    </div>
  )
}
