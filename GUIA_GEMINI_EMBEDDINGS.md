# Guia: API Key de Gemini, Embeddings 001 y Cuota Gratis

Este documento resume el flujo que usaste en el laboratorio para:

1. Cargar la API key desde variables de entorno.
2. Conectar con Gemini.
3. Llamar embeddings con `models/gemini-embedding-001`.
4. Administrar la cuota gratis con rate limit y cache para reanudar.

## 1) Cargar variables de entorno

La idea es **no hardcodear** la key en el notebook. En su lugar, se guarda en un archivo `.env` y se lee con `python-dotenv`.

### Archivo `.env`

```env
GOOGLE_API_KEY=tu_api_key_aqui
```

### Codigo de carga

```python
import os
from dotenv import load_dotenv

load_dotenv()  # Carga variables desde .env
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("No se encontro GOOGLE_API_KEY en variables de entorno")
```

## 2) Configurar conexion a Gemini

```python
import google.generativeai as genai

genai.configure(api_key=GOOGLE_API_KEY)
print("API de Gemini configurada correctamente")
```

## 3) Llamado de embeddings con modelo 001

Tu llamado base fue con `embed_content` y el modelo `models/gemini-embedding-001`.

```python
import numpy as np

test_text = "NtClose NtOpenFile RegOpenKeyExW"

test_result = genai.embed_content(
    model="models/gemini-embedding-001",
    content=test_text
)

test_embedding = np.array(test_result["embedding"])
EMBEDDING_DIM = len(test_embedding)

print("Conexion exitosa con Gemini API")
print(f"Dimension del embedding: {test_embedding.shape}")
print(f"EMBEDDING_DIM = {EMBEDDING_DIM}")
```

## 4) Tecnica usada para la cuota gratis

Tu tecnica fue combinar tres cosas:

1. **Rate limit conservador**: usar ~90 requests/min (por debajo del maximo de 100/min en free tier).
2. **Cache final**: guardar embeddings y labels como `.npy` para no recalcular.
3. **Checkpoint parcial**: guardar avance intermedio para retomar desde el ultimo indice si se corta el proceso o cambias API key.

### Rutas de cache que usaste

```python
embeddings_cache_path = f"{current_directory}/processed_data/gemini_embeddings.npy"
labels_cache_path = f"{current_directory}/processed_data/gemini_labels.npy"
partial_cache_path = f"{current_directory}/processed_data/gemini_embeddings_partial.npy"
```

### Flujo de ejecucion

1. Si existen `gemini_embeddings.npy` y `gemini_labels.npy`, se cargan con `np.load` y no se vuelve a llamar a la API.
2. Si no existen, se generan embeddings con una funcion tipo `generate_embeddings_with_rate_limit(...)`.
3. Durante la generacion, se va guardando avance parcial para poder reanudar.
4. Al completar, se guardan archivos finales en `processed_data/`.

## 5) Reanudar y rotar API keys por rangos

La logica que describiste (ejemplo):

- API key 1: indices 0 a 999
- API key 2: indices 1000 a 1999
- API key 3: indices 2000 en adelante

Para eso, el patron recomendado es:

1. Definir `start_idx` y `end_idx` por key.
2. Procesar solo ese rango.
3. Guardar checkpoint cada N muestras.
4. Al llegar al limite/cuota, cambiar key y continuar desde el ultimo indice guardado.

## 6) Ejemplo compacto de funcion con rate limit

```python
import time
import numpy as np

def generate_embeddings_with_rate_limit(texts, model_name="models/gemini-embedding-001", requests_per_minute=90):
    delay = 60.0 / requests_per_minute
    vectors = []

    for i, txt in enumerate(texts):
        result = genai.embed_content(model=model_name, content=txt)
        vectors.append(np.array(result["embedding"], dtype=np.float32))

        # Respetar cuota gratis
        time.sleep(delay)

    return np.vstack(vectors)
```

## 7) Resumen rapido

- Leiste la key con `.env` + `load_dotenv()` + `os.getenv("GOOGLE_API_KEY")`.
- Configuraste Gemini con `genai.configure(api_key=...)`.
- Llamaste embeddings con `genai.embed_content(model="models/gemini-embedding-001", content=...)`.
- Evitaste gastar cuota de mas con rate limit + cache final + checkpoint parcial.
- La estrategia por rangos y cambio de API key te permite continuar desde el ultimo avance sin perder trabajo.
