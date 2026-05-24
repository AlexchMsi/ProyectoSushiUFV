# Sushi Counter

Aplicación web de detección y conteo de sushi en tiempo real.
Detecta **8 tipos de sushi** a través de la cámara o mediante imagen estática.

**Clases:** `ball` · `gunkan` · `maki` · `nigiri` · `onigiri` · `roll` · `sashimi` · `wrap`

---

## Requisitos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado y en ejecución

---

## Instalación 
```bash
docker compose up -d
```

Abrir en el navegador: **http://localhost:8080**

---

## Uso

### Modo cámara

1. Pulsa **Iniciar** — el navegador pedirá permiso de cámara
2. Las detecciones aparecen en tiempo real con su clase y confianza
3. El panel derecho muestra el conteo por tipo y el total
4. Pulsa **Grabar** para guardar un vídeo `.webm` con las cajas superpuestas
5. Pulsa **Detener** para finalizar la sesión

### Modo imagen

1. Cambia a la pestaña **Imagen** (esquina superior derecha)
2. Pulsa **Seleccionar imagen** o arrastra una foto al área central
3. El modelo analiza la imagen y dibuja las cajas de detección
4. El panel derecho muestra el conteo de cada tipo detectado

---


## Rendimiento del modelo

Modelo: **YOLO11m** · mAP50 = 0.829

| Clase | mAP50 | Clase | mAP50 |
|-------|-------|-------|-------|
| maki | 0.923 | ball | 0.879 |
| onigiri | 0.906 | gunkan | 0.782 |
| wrap | 0.906 | nigiri | 0.739 |
| roll | 0.893 | sashimi | 0.602 |

Para más detalles sobre el entrenamiento y el dataset, ver [`MEMORIA.md`](MEMORIA.md).
