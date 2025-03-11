import argparse
import os
import shutil
from datetime import datetime
import win32com.client
from langchain_openai import OpenAIEmbeddings
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from odf.opendocument import load
from odf.text import P, H, List, Span
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
import a_env_vars
from langchain.text_splitter import RecursiveCharacterTextSplitter
import zipfile
import xml.etree.ElementTree as ET
import time

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



def extract_text_from_node(node):
    """Extrae texto de un nodo XML incluyendo sus hijos."""
    if node is None:
        return ""
    
    text_parts = []
    if node.text:
        text_parts.append(node.text.strip())

    for child in node:
        text_parts.append(extract_text_from_node(child))

    if node.tail:
        text_parts.append(node.tail.strip())

    return " ".join(filter(None, text_parts))


def read_odt(file_path):
    """Lee un archivo .odt y extrae su contenido de texto de manera robusta."""
    try:

        # Abrimos el archivo ODT como ZIP
        with zipfile.ZipFile(file_path, 'r') as z:
            # Extraemos el contenido principal donde está el texto
            with z.open('content.xml') as content_file:
                xml_content = content_file.read()


        # Parseamos el XML
        root = ET.fromstring(xml_content)

        # Espacios de nombres de OpenDocument
        ns = {
            'text': 'urn:oasis:names:tc:opendocument:xmlns:text:1.0',
            'office': 'urn:oasis:names:tc:opendocument:xmlns:office:1.0',
            'table': 'urn:oasis:names:tc:opendocument:xmlns:table:1.0'
        }

        extracted_text = []

        # Extraer texto de párrafos (<text:p>)
        for elem in root.findall('.//text:p', ns):
            text_content = extract_text_from_node(elem)
            if text_content:
                extracted_text.append(text_content)

        # Extraer texto de encabezados (<text:h>)
        for elem in root.findall('.//text:h', ns):
            header_content = extract_text_from_node(elem)
            if header_content:
                extracted_text.append(header_content)

        # Extraer texto de tablas (<table:table>)
        for table in root.findall('.//table:table', ns):
            for row in table.findall('.//table:table-row', ns):
                row_text = []
                for cell in row.findall('.//table:table-cell', ns):
                    cell_text = extract_text_from_node(cell)
                    if cell_text:
                        row_text.append(cell_text)
                if row_text:
                    extracted_text.append(" | ".join(row_text))

        # Si no encontramos texto, inspeccionamos otros nodos
        if not extracted_text:
            # Buscamos nodos adicionales que contengan texto
            for elem in root.iter():
                node_text = extract_text_from_node(elem)
                if node_text:
                    print(f"🔍 Nodo {elem.tag} -> {node_text}")
                    extracted_text.append(node_text)

        # Retornamos el texto completo
        return '\n'.join(extracted_text)

    except Exception as e:
        print(f"❌ Error al leer el archivo .odt {file_path}: {e}")
        return ""



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
                content = ""
                file_path = os.path.join(root, file)                  

                if file_types:                
                    # Comprobar si el archivo tiene una de las extensiones solicitadas
                    if not any(file.lower().endswith(ext) for ext in file_types):
                        continue  

                print(f"Procesando archivo: {file}")
                modification_time = os.path.getmtime(file_path)
                modification_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(modification_time))


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
                    documents.append(Document(page_content=content, metadata={
                        "source": file_path,
                        "modification_date": modification_date,
                        "file_name": file
                        }))
                else:
                    print(f"Error: El archivo {file_path} no tiene contenido válido.")
            except Exception as e:
                print(f"Error al procesar el archivo {file_path}: {e}. Se omitirá este archivo.")
                continue  # Omitir el archivo y continuar con el siguiente

    print(f"Se cargaron {len(documents)} documentos.")
    return documents

# Función para dividir el texto en fragmentos más pequeños
def split_text(documents: list[Document]) -> list[Document]:
    #print("Inicia splite")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    #print(f"Se dividieron {len(documents)} documentos en {len(chunks)} fragmentos.")
    return chunks

# Función para guardar los fragmentos en Chroma
def save_to_chromaOLD(chunks):
    #if os.path.exists(CHROMA_PATH):
    #    shutil.rmtree(CHROMA_PATH)
    embedding_function = OpenAIEmbeddings()
    for i in range(0, len(chunks), MAX_BATCH_SIZE):
        batch = chunks[i:i + MAX_BATCH_SIZE]
        db = Chroma.from_documents(batch, embedding_function, persist_directory=CHROMA_PATH)
        db.persist() 




def save_to_chroma(chunks):
    """
    Guarda los fragmentos de texto en la base de datos Chroma de manera eficiente.
    """
    BATCH_SIZE = 100  # Tamaño del lote para inserciones en la base

    # Eliminar el directorio existente para evitar datos obsoletos (opcional)
    #if os.path.exists(CHROMA_PATH):
    #    shutil.rmtree(CHROMA_PATH)

    # Inicializar el modelo de embeddings
    embedding_function = OpenAIEmbeddings()

    # Crear la base de datos Chroma solo una vez
    db = Chroma(embedding_function=embedding_function, persist_directory=CHROMA_PATH)

    batch = []  # Lista para almacenar los fragmentos en lotes

    for chunk in chunks:
        # Crear documento con embeddings
        batch.append(chunk)

        # Cuando alcanzamos el tamaño del lote, insertamos en la base de datos
        if len(batch) >= BATCH_SIZE:
            db.add_documents(batch)
            batch.clear()  # Limpiar el lote después de insertar

    # Insertar los documentos restantes si quedaron fuera del último lote
    if batch:
        db.add_documents(batch)

    # Persistimos todo al final para mejorar la eficiencia
    db.persist()
    print("Datos guardados en Chroma con éxito.")


# Función principal para generar la base de datos
def generate_data_store(file_types=None):
    documents = load_documents(file_types=file_types)
    chunks = split_text(documents)
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
