const CLASS_COLORS = {}

function colorForClass(className) {
  if (!CLASS_COLORS[className]) {
    let hash = 0
    for (let i = 0; i < className.length; i++) {
      hash = className.charCodeAt(i) + ((hash << 5) - hash)
    }
    const h = Math.abs(hash) % 360
    CLASS_COLORS[className] = { solid: `hsl(${h}, 85%, 55%)`, bg: `hsla(${h}, 85%, 55%, 0.75)` }
  }
  return CLASS_COLORS[className]
}

/**
 * Draw bounding boxes and labels on a canvas context.
 * @param {CanvasRenderingContext2D} ctx
 * @param {Array} detections  [{class, confidence, bbox:[x1,y1,x2,y2]}]
 * @param {number} scaleX     canvas.width / original frame width
 * @param {number} scaleY     canvas.height / original frame height
 */
export function drawDetections(ctx, detections, scaleX = 1, scaleY = 1, clearFirst = true) {
  if (clearFirst) ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)

  for (const det of detections) {
    const [x1, y1, x2, y2] = det.bbox
    const sx1 = x1 * scaleX
    const sy1 = y1 * scaleY
    const sw = (x2 - x1) * scaleX
    const sh = (y2 - y1) * scaleY

    const { solid, bg } = colorForClass(det.class)

    // Box
    ctx.strokeStyle = solid
    ctx.lineWidth = 1.5
    ctx.strokeRect(sx1, sy1, sw, sh)

    // Label overlaid at top-left corner of the box
    const label = `${det.class} ${det.confidence.toFixed(2)}`
    ctx.font = 'bold 12px monospace'
    const textWidth = ctx.measureText(label).width
    const labelH = 18
    ctx.fillStyle = bg
    ctx.fillRect(sx1, sy1, textWidth + 8, labelH)

    // Label text
    ctx.fillStyle = '#fff'
    ctx.fillText(label, sx1 + 4, sy1 + 13)
  }
}
