# Dependencias de Langchain
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings  # Importar embeddings de HuggingFace de Langchain
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader  # Importar cargador de PDF de Langchain

from langchain.text_splitter import RecursiveCharacterTextSplitter  # Importar el cortador de texto de Langchain
from langchain.schema import Document  # Importar el esquema de Documento de Langchain
from langchain_community.vectorstores import Chroma  # DB vectorial Chroma de Langchain
import os  # Importar módulo os para funcionalidades del sistema operativo
import shutil  # Importar módulo shutil para operaciones de archivos de alto nivel
import a_env_vars  # Importar módulo para manejar variables de entorno
from langchain_openai import OpenAIEmbeddings
#from langchain_ollama import OllamaEmbeddings


# Variables globales
EMBEDDING_MODEL_NAME =  a_env_vars.EMBEDDING_MODEL_NAME
DATA_PATH = a_env_vars.DATA_PATH
CHROMA_PATH = a_env_vars.CHROMA_PATH

# Constante para el tamaño máximo del lote
MAX_BATCH_SIZE = 5461

os.environ["OPENAI_API_KEY"] = a_env_vars.OPENAI_API_KEY


def load_documents() -> list[Document]:
    """
    Cargar documentos PDF desde el directorio especificado utilizando PyPDFDirectoryLoader.
    Retorna:
        Lista de Documentos: Documentos PDF cargados representados como objetos Document de Langchain.
    """
    # Inicializar el cargador de PDF con el directorio especificado
    #document_loader = PyPDFDirectoryLoader(DATA_PATH)
    document_loader = DirectoryLoader(DATA_PATH, glob="**/*")
    
    # Cargar los documentos PDF y retornarlos como una lista de objetos Document
    print ("Documentos Leidos")
    return document_loader.load()

def split_text(documents: list[Document]) -> list[Document]:
    """
    Dividir el contenido de texto de la lista dada de objetos Document en fragmentos más pequeños.
    Args:
        documents (list[Document]): Lista de objetos Document que contienen el contenido de texto a dividir.
    Retorna:
        list[Document]: Lista de objetos Document que representan los fragmentos de texto divididos.
    """
    print ("Inicia splite")
    # Inicializar el divisor de texto con los parámetros especificados
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Tamaño de cada fragmento en caracteres
        chunk_overlap=200,  # Superposición entre fragmentos consecutivos
        length_function=len,  # Función para calcular la longitud del texto
        add_start_index=True  # Bandera para agregar índice de inicio a cada fragmento
    )

    # Dividir los documentos en fragmentos más pequeños utilizando el divisor de texto
    chunks = text_splitter.split_documents(documents)
    print(f"Se dividieron {len(documents)} documentos en {len(chunks)} fragmentos.")

    # Imprimir ejemplo de contenido de página y metadatos para un fragmento
    if chunks:
        document = chunks[0]
       # print(document.page_content)
        print(document.metadata)

    return chunks  # Retornar la lista de fragmentos de texto divididos


def save_to_chroma(chunks):
    """
    Guardar fragmentos de texto en una base de datos vectorial Chroma.
    Args:
        chunks (list[Document]): Lista de fragmentos de texto a guardar.
    """
    # Eliminar cualquier base de datos Chroma existente
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Crear una nueva base de datos Chroma a partir de los documentos utilizando embeddings de HuggingFace
    #embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    embedding_function = OpenAIEmbeddings()
    #embedding_function = OllamaEmbeddings(model="llama3.1:8b")
    
    # Procesar los fragmentos en lotes más pequeños
    for i in range(0, len(chunks), MAX_BATCH_SIZE):
        batch = chunks[i:i + MAX_BATCH_SIZE]
        db = Chroma.from_documents(
            batch,
            embedding_function,
            persist_directory=CHROMA_PATH
        )
        db.persist()
        print(f"Se guardaron {len(batch)} fragmentos en {CHROMA_PATH}.")

def generate_data_store():
    """
    Función para generar una base de datos vectorial en Chroma a partir de documentos.
    """
    print ("Buscando Documentos"  + str(datetime.now()))
    documents = load_documents()  # Cargar documentos desde una fuente
    print ("documentos cargados "  + str(datetime.now()))
    print ("inicio de CHUNKING "  + str(datetime.now()))
    chunks = split_text(documents)  # Dividir los documentos en fragmentos manejables
    print ("fin CHUNK "  + str(datetime.now()))
    print ("Guardar en DB "  + str(datetime.now()))
    save_to_chroma(chunks)  # Guardar los datos procesados en un almacén de datos
    print ("Fin Guardar en DB "  + str(datetime.now()))


generate_data_store()
