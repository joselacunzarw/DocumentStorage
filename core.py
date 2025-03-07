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
import win32com.client
from langchain_openai import OpenAIEmbeddings
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from odf.opendocument import load
from odf.text import P

#from langchain_ollama import OllamaEmbeddings


# Variables globales
EMBEDDING_MODEL_NAME =  a_env_vars.EMBEDDING_MODEL_NAME
DATA_PATH = a_env_vars.DATA_PATH
CHROMA_PATH = a_env_vars.CHROMA_PATH

# Constante para el tamaño máximo del lote
MAX_BATCH_SIZE = 5461

os.environ["OPENAI_API_KEY"] = a_env_vars.OPENAI_API_KEY


def read_doc(file_path: str) -> str:
    """Lee un archivo .doc en Windows con Word instalado (PyWin32)."""
    if not os.path.exists(file_path):
        print(f"El archivo {file_path} no existe.")
        return ""
    try:
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False

        abs_path = os.path.abspath(file_path)
        abs_path = abs_path.replace('/', '\\')  # Reemplazar barras normales por invertidas

        doc = word.Documents.Open(abs_path, ReadOnly=True)
        text = doc.Content.Text
        doc.Close()
        word.Quit()
        return text
    except Exception as e:
        print(f"Error al procesar el archivo .doc {file_path}: {e}")
        return ""

def read_pdf(file_path: str) -> str:
    """Lee un archivo PDF y devuelve su contenido como texto."""
    with open(file_path, "rb") as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def read_docx(file_path: str) -> str:
    """Lee un archivo .docx y devuelve su contenido como texto."""
    doc = DocxDocument(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def read_odt(file_path: str) -> str:
    """Lee un archivo .odt y devuelve su contenido como texto."""
    text = ""
    try:
        doc = load(file_path)
        for paragraph in doc.getElementsByType(P):
            if paragraph.firstChild is not None:
                text += paragraph.firstChild.nodeValue + "\n"
    except Exception as e:
        print(f"Error al procesar el archivo .odt {file_path}: {e}")
    return text

def read_txt(file_path: str) -> str:
    """Lee un archivo .txt y devuelve su contenido como texto."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def load_documents() -> list[Document]:
    """Cargar documentos desde un directorio, manejar diferentes tipos de archivos"""
    documents = []
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                print(f"Procesando archivo: {file}")
                content = ""
                # Leer el contenido según la extensión del archivo
                if file.endswith('.pdf'):
                    content = read_pdf(file_path)
                elif file.endswith('.doc'):
                    content = read_doc(file_path)    
                elif file.endswith('.docx'):
                    content = read_docx(file_path)
                elif file.endswith('.odt'):
                    content = read_odt(file_path)
                elif file.endswith('.txt'):
                    content = read_txt(file_path)
                else:
                    print(f"Formato de archivo no soportado: {file}")
                    continue

                # Crear un documento para agregarlo al proceso
                if content:
                    documents.append(Document(page_content=content, metadata={"source": file_path}))
            except Exception as e:
                print(f"Error al procesar el archivo {file_path}: {e}. Se omitirá este archivo.")
                continue  # Omitir el archivo y continuar con el siguiente

    print(f"Se cargaron {len(documents)} documentos.")
    return documents


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
    print ("Buscando Documentos "  + str(datetime.now()))
    documents = load_documents()  # Cargar documentos desde una fuente
    print ("documentos cargados "  + str(datetime.now()))
    print ("inicio de CHUNKING "  + str(datetime.now()))
    chunks = split_text(documents)  # Dividir los documentos en fragmentos manejables
    print ("fin CHUNK "  + str(datetime.now()))
    print ("Guardar en DB "  + str(datetime.now()))
    save_to_chroma(chunks)  # Guardar los datos procesados en un almacén de datos
    print ("Fin Guardar en DB "  + str(datetime.now()))


generate_data_store()
