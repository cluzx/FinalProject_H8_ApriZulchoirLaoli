import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Handling penyesuaian versi modul chains LangChain
try:
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains import create_retrieval_chain
except ModuleNotFoundError:
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    from langchain_classic.chains import create_retrieval_chain

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Binjai Smart Travel Agent", page_icon="🌴")
st.title("🌴 Binjai Smart Travel & Culinary Agent (RAG)")
st.caption("Asisten Wisata & Kuliner Kota Binjai Berbasis Gemini API & LangChain RAG")

# Sidebar Konfigurasi API Key & Parameter
with st.sidebar:
    st.header("⚙️ Konfigurasi")
    # Mengambil API Key dari Secrets Streamlit atau Input User
    api_key = st.text_input("Gemini API Key:", type="password", value=st.secrets.get("GOOGLE_API_KEY", ""))
    temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.3, step=0.1)

# Inisialisasi System RAG (Cached agar cepat)
@st.cache_resource(show_spinner="Membangun Vector DB dari PDF...")
def init_rag(_api_key):
    os.environ["GOOGLE_API_KEY"] = _api_key
    
    # 1. Load PDF
    pdf_path = "Knowledge_Base_Wisata_Kuliner_Binjai.pdf"
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # 2. Split Teks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
    chunks = text_splitter.split_documents(documents)
    
    # 3. Embeddings & VectorStore (Eksplisit sertakan google_api_key)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        google_api_key=_api_key
    )
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

if not api_key:
    st.warning("⚠️ Masukkan Gemini API Key di sidebar untuk melanjutkan.")
    st.stop()

# Load Retriever & Chain
retriever = init_rag(api_key)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=temperature, google_api_key=api_key)

system_prompt = """
Kamu adalah 'Si Rambutan', asisten lokal ramah & cerdas khusus wisata dan kuliner Kota Binjai.

Aturan Penting:
1. Jawab pertanyaan pengguna HANYA berdasarkan konteks dokumen RAG berikut.
2. Jika informasi tidak ada di dokumen, katakan jujur bahwa informasi tersebut belum ada di basis pengetahuan kamu.

Konteks Dokumen:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# UI Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Tanyakan sesuatu seputar wisata atau kuliner di Binjai..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    with st.chat_message("assistant"):
        with st.spinner("Mencari jawaban dari dokumen..."):
            res = rag_chain.invoke({"input": user_input})
            answer = res["answer"]
            st.markdown(answer)
            
            with st.expander("📚 Lihat Sumber Dokumen RAG"):
                for i, doc in enumerate(res["context"]):
                    st.caption(f"**Chunk {i+1} (Halaman {doc.metadata['page'] + 1}):**")
                    st.text(doc.page_content[:200] + "...")
                    
    st.session_state.messages.append({"role": "assistant", "content": answer})
