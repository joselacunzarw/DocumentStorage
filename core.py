import argparse
import os
import shutil
from datetime import datetime
import win32com.client
from langchain_openai import OpenAIEmbeddings
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from odf.opendocument import load
from odf.text import P
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
import a_env_vars
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Variables globales
EMBEDDING_MODEL_NAME = a_env_vars.EMBEDDING_MODEL_NAME
DATA_PATH = a_env_vars.DATA_PATH
CHROMA_PATH = a_env_vars.CHROMA_PATH
MAX_BATCH_SIZE = 5461

os.environ["OPENAI_API_KEY"] = a_env_vars.OPENAI_API_KEY

# Función de lectura para .doc
def read_doc(file_path: str) -> str:
    if not os.path.exists(file_path):
        print(f"El archivo {file_path} no existe.")
        return ""
    try:
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        abs_path = os.path.abspath(file_path)
        abs_path = abs_path.replace('/', '\\')
        doc = word.Documents.Open(abs_path, ReadOnly=True)
        text = doc.Content.Text
        doc.Close()
        word.Quit()
        return text
    except Exception as e:
        print(f"Error al procesar el archivo .doc {file_path}: {e}")
        return ""

# Función de lectura para .pdf
def read_pdf(file_path: str) -> str:
    with open(file_path, "rb") as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Función de lectura para .docx
def read_docx(file_path: str) -> str:
    doc = DocxDocument(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

# Función de lectura para .odt
def read_odt(file_path: str) -> str:
    text = ""
    try:
        doc = load(file_path)
        for paragraph in doc.getElementsByType(P):
            if paragraph.firstChild is not None and hasattr(paragraph.firstChild, 'nodeValue'):
                text += paragraph.firstChild.nodeValue + "\n"
    except Exception as e:
        print(f"Error al procesar el archivo .odt {file_path}: {e}")
    return text

# Función de lectura para .txt
def read_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# Función para cargar documentos, procesando solo los tipos especificados
def load_documents(file_types=None) -> list[Document]:
    documents = []
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            try:
                file_path = os.path.join(root, file)  # <--- Definir file_path aquí
                content = ""
                # Verificar si el archivo tiene un tipo soportado y en la lista de tipos solicitados
                if file_types:                
                    # Comprobar si el archivo tiene una de las extensiones solicitadas
                    if not any(file.lower().endswith(ext) for ext in file_types):
                        continue  # Si no es del tipo solicitado, omitir archivo

                print(f"Procesando archivo: {file}")
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

# Función para dividir el texto en fragmentos más pequeños
def split_text(documents: list[Document]) -> list[Document]:
    print("Inicia splite")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Se dividieron {len(documents)} documentos en {len(chunks)} fragmentos.")
    return chunks

# Función para guardar los fragmentos en Chroma
def save_to_chroma(chunks):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    embedding_function = OpenAIEmbeddings()
    for i in range(0, len(chunks), MAX_BATCH_SIZE):
        batch = chunks[i:i + MAX_BATCH_SIZE]
        db = Chroma.from_documents(batch, embedding_function, persist_directory=CHROMA_PATH)
        db.persist()
        print(f"Se guardaron {len(batch)} fragmentos en {CHROMA_PATH}.")

# Función principal para generar la base de datos
def generate_data_store(file_types=None):
    print(f"Buscando Documentos {str(datetime.now())}")
    documents = load_documents(file_types=file_types)
    print(f"Documentos cargados {str(datetime.now())}")
    print(f"Inicio de CHUNKING {str(datetime.now())}")
    chunks = split_text(documents)
    print(f"Fin CHUNK {str(datetime.now())}")
    print(f"Guardar en DB {str(datetime.now())}")
    save_to_chroma(chunks)
    print(f"Fin Guardar en DB {str(datetime.now())}")

# Configuración de los parámetros de entrada con argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procesa documentos de diferentes formatos.")
    parser.add_argument('--formats', nargs='*', help="Especificar los formatos de archivo a procesar (por ejemplo: .odt .pdf .docx)")

    args = parser.parse_args()
    if args.formats:
        formats = [ext.strip().lower() for ext in args.formats]  # Remover espacios y convertir a minúsculas
        print(f"Formatos seleccionados: {formats}")
        generate_data_store(file_types=formats)
    else:
        generate_data_store()
