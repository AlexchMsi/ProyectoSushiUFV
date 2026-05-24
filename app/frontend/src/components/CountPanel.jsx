export default function CountPanel({ counts, total }) {
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1])

  return (
    <aside className="count-panel">
      <h2 className="count-panel__title">Conteo</h2>

      {entries.length === 0 ? (
        <p className="count-panel__empty">Sin detecciones</p>
      ) : (
        <ul className="count-panel__list">
          {entries.map(([cls, count]) => (
            <li key={cls} className="count-panel__item">
              <span className="count-panel__name">{cls}</span>
              <span className="count-panel__badge">{count}</span>
            </li>
          ))}
        </ul>
      )}

      <div className="count-panel__total">
        <span>Total</span>
        <span className="count-panel__total-value">{total}</span>
      </div>
    </aside>
  )
}
