
# DocumentRepository

Repositorio vectorial de documentos

## Descripción

Este proyecto implementa un sistema para el procesamiento y almacenamiento de documentos PDF en una base de datos vectorial utilizando la biblioteca Langchain y Chroma para el almacenamiento persistente. El sistema permite cargar documentos PDF desde un directorio, dividir el contenido en fragmentos más pequeños, y guardar estos fragmentos en una base de datos vectorial para una recuperación eficiente.

## Estructura del Proyecto

- `a_env_vars.py`: Archivo que contiene las variables de entorno utilizadas en el proyecto.
- `core.py`: Script principal que contiene las funciones para cargar, dividir y almacenar documentos en una base de datos vectorial.

## Dependencias

El proyecto utiliza las siguientes bibliotecas de Python:

- `langchain_huggingface`
- `langchain_community`
- `langchain`
- `os`
- `shutil`

## Configuración

1. Clona el repositorio en tu máquina local.
2. Crea un entorno virtual e instala las dependencias necesarias:

```bash
python -m venv env
source env/bin/activate  # En Windows usa `env\Scripts\activate`
pip install -r requirements.txt
```

3. Configura las variables de entorno en el archivo `a_env_vars.py`:

```python
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DATA_PATH = r"ruta/a/tus/documentos"
CHROMA_PATH = "../chroma"
```

## Uso

Para generar la base de datos vectorial a partir de los documentos PDF, ejecuta el script `core.py`:

```bash
python core.py
```

El proceso sigue los siguientes pasos:
1. Carga los documentos PDF desde el directorio especificado.
2. Divide el contenido de los documentos en fragmentos más pequeños.
3. Guarda los fragmentos de texto en una base de datos vectorial Chroma utilizando embeddings de HuggingFace.

## Ejemplo de Contenido

### a_env_vars.py

```python
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DATA_PATH = r"C:\Users\josel\OneDrive\Documents\Desktop\repo"
CHROMA_PATH = "../chroma"
```

### core.py

```python
# Dependencias de Langchain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
import os
import shutil
import a_env_vars

# Variables globales
EMBEDDING_MODEL_NAME = a_env_vars.EMBEDDING_MODEL_NAME
DATA_PATH = a_env_vars.DATA_PATH
CHROMA_PATH = a_env_vars.CHROMA_PATH

def load_documents() -> list[Document]:
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_text(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, length_function=len, add_start_index=True)
    chunks = text_splitter.split_documents(documents)
    return chunks

def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = Chroma.from_documents(chunks, embedding_function, persist_directory=CHROMA_PATH)
    print(f"Se guardaron {len(chunks)} fragmentos en {CHROMA_PATH}.")

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

generate_data_store()
```

## Contribuciones

Si puede deme dinero

## Licencia

Este proyecto está licenciado bajo los términos de la licencia de conducir
