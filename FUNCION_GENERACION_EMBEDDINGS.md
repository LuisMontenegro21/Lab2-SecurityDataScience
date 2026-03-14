# Funcion para generar embeddings con Gemini (con rate limit y cache parcial)

Este documento contiene una funcion reutilizable para generar embeddings de secuencias de texto usando Gemini, respetando limites de requests y guardando progreso parcial para poder reanudar si ocurre un error o se corta la ejecucion.

## Funcion

```python
import os
import time
import numpy as np
import google.generativeai as genai


def generate_embeddings_with_rate_limit(
    texts,
    partial_cache_path,
    requests_per_minute=90,
    model_name="models/gemini-embedding-001"
):
    """
    Genera embeddings respetando rate limit y guardando progreso parcial.

    Args:
        texts: Lista o Serie de textos a embeber.
        partial_cache_path: Ruta del .npy para guardar avance parcial.
        requests_per_minute: Requests por minuto (90 recomendado para margen de seguridad).
        model_name: Modelo de embeddings de Gemini.

    Returns:
        np.ndarray con shape (n_samples, embedding_dim)
    """
    all_embeddings = []
    total_samples = len(texts)
    delay_between_requests = 60.0 / requests_per_minute

    print(f"Generando embeddings para {total_samples} muestras...")
    print(f"Rate limit configurado: {requests_per_minute} req/min")
    print(f"Delay entre requests: {delay_between_requests:.2f}s")

    # Reanudar desde cache parcial si existe
    start_idx = 0
    if os.path.exists(partial_cache_path):
        partial_embeddings = np.load(partial_cache_path)
        all_embeddings = partial_embeddings.tolist()
        start_idx = len(all_embeddings)
        print(f"Reanudando desde muestra {start_idx}/{total_samples}")

    for i in range(start_idx, total_samples):
        text = texts.iloc[i] if hasattr(texts, "iloc") else texts[i]

        try:
            result = genai.embed_content(
                model=model_name,
                content=text
            )
            all_embeddings.append(result["embedding"])

            # Guardar avance cada 100 muestras
            if (i + 1) % 100 == 0 or i == total_samples - 1:
                np.save(partial_cache_path, np.array(all_embeddings))
                print(f"Procesado {i + 1}/{total_samples} ({(i + 1) / total_samples * 100:.1f}%)")

            # Espera para respetar cuota por minuto
            if i < total_samples - 1:
                time.sleep(delay_between_requests)

        except Exception as e:
            error_msg = str(e)

            # Manejo basico de errores de cuota/rate limit
            if "429" in error_msg or "quota" in error_msg.lower():
                print(f"Rate limit/cuota alcanzado en muestra {i}. Guardando avance y esperando 60s...")
                np.save(partial_cache_path, np.array(all_embeddings))
                time.sleep(60)

                # Reintento unico
                try:
                    result = genai.embed_content(model=model_name, content=text)
                    all_embeddings.append(result["embedding"])
                except Exception as e2:
                    print(f"Error persistente en muestra {i}: {e2}")
                    np.save(partial_cache_path, np.array(all_embeddings))
                    raise
            else:
                print(f"Error en muestra {i}: {e}")
                np.save(partial_cache_path, np.array(all_embeddings))
                raise

    # Si completo todo, eliminar cache parcial
    if os.path.exists(partial_cache_path):
        os.remove(partial_cache_path)

    return np.array(all_embeddings)
```

## Explicacion

1. Control de velocidad (rate limit)
- `delay_between_requests = 60.0 / requests_per_minute` fuerza una pausa entre requests.
- Si usas `requests_per_minute=90`, cada request espera ~0.67 segundos.

2. Reanudacion de progreso
- Si existe `partial_cache_path`, carga lo ya generado.
- Retoma desde `start_idx` para no repetir trabajo.

3. Guardado parcial periodico
- Cada 100 muestras se guarda un `.npy` temporal.
- Si la ejecucion falla o se detiene, no pierdes todo el avance.

4. Manejo de errores de cuota (429)
- Detecta errores por cuota/rate limit.
- Guarda avance, espera 60 segundos y reintenta una vez.

5. Limpieza final
- Al completar exitosamente, elimina el cache parcial.
- Devuelve toda la matriz de embeddings en `np.ndarray`.

## Ejemplo de uso

```python
embeddings_cache_path = f"{current_directory}/processed_data/gemini_embeddings.npy"
labels_cache_path = f"{current_directory}/processed_data/gemini_labels.npy"
partial_cache_path = f"{current_directory}/processed_data/gemini_embeddings_partial.npy"

if os.path.exists(embeddings_cache_path) and os.path.exists(labels_cache_path):
    embeddings_matrix = np.load(embeddings_cache_path)
    labels_gemini = np.load(labels_cache_path)
else:
    embeddings_matrix = generate_embeddings_with_rate_limit(
        texts=documents,
        partial_cache_path=partial_cache_path,
        requests_per_minute=90,
        model_name="models/gemini-embedding-001"
    )
    labels_gemini = labels.copy()

    np.save(embeddings_cache_path, embeddings_matrix)
    np.save(labels_cache_path, labels_gemini)
```

## Nota importante para tu caso actual

Como en tu proyecto los embeddings ya estan generados (`gemini_embeddings.npy` y `gemini_labels.npy`), no necesitas ejecutar esta funcion en `Run All`. Esta guia queda como referencia o para regenerar embeddings en otro entorno.