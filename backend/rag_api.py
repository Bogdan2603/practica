# rag_api.py
import os
import shutil
import time
import subprocess
import requests
import torch
from pathlib import Path
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware # Important pentru comunicarea cu Next.js
from pydantic import BaseModel
import uvicorn

# Importuri Langchain - asigură-te că sunt corecte și complete
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.documents import Document # Necesar pentru serializare și tipare

# === CONFIG === (păstrează configurațiile tale din scriptul original)
DOCUMENT_FOLDER = Path("./fisiere") # Presupune că folderul 'fisiere' e lângă acest script
CHROMA_DB_PATH = Path("./chroma_db_api") # Poți folosi un alt nume pentru a nu suprascrie DB-ul original dacă vrei
CHUNK_SIZE = 256
CHUNK_OVERLAP = 25
K_CONTEXT_CHUNKS = 10
OLLAMA_MODEL_NAME = "llama3.1:8b-instruct-q4_K_S" # Asigură-te că modelul e corect
OLLAMA_API_URL = "http://localhost:11434"

# --- Inițializare Globală (se execută o singură dată la pornirea API-ului) ---
print("--- Inițializare API RAG Cod Rutier ---")

# ȘTERGE DB EXISTENTĂ (poate vrei să controlezi asta diferit într-un API, ex. doar la prima rulare)
if CHROMA_DB_PATH.exists():
    print(f"🗑️ Șterge baza Chroma existentă din '{CHROMA_DB_PATH}'...")
    shutil.rmtree(CHROMA_DB_PATH)

# PORNEȘTE/VERIFICĂ OLLAMA
def ensure_ollama_model_running(model_name):
    print(f"🔍 Verific Ollama și modelul `{model_name}`...")
    try:
        r = requests.post(f"{OLLAMA_API_URL}/api/generate",
                          json={"model": model_name, "prompt": "Hi", "stream": False, "options": {"num_predict": 1}}, timeout=3)
        if r.status_code == 200:
            try:
                r.json()
                print("✅ Modelul Ollama rulează.")
                return
            except requests.exceptions.JSONDecodeError:
                print("⚠️ Modelul a răspuns, dar nu cu JSON valid. Verificăm pornirea...")
        elif r.status_code == 404:
            print(f"❌ Modelul '{model_name}' nu a fost găsit în Ollama. Încerc să îl descarc/rulez (acest lucru poate dura)...")
        else:
            print(f"⚠️ Serverul Ollama a răspuns cu status {r.status_code}. Încerc pornirea modelului...")
    except requests.exceptions.RequestException as e:
        print(f"🔌 Serverul Ollama nu răspunde ({e}). Se încearcă pornirea...")

    print(f"🚀 Pornește modelul '{model_name}' în Ollama (dacă nu este deja pornit)...")
    process_flags = 0
    if os.name == 'nt': # Pentru Windows, rulează într-o consolă nouă (dacă e necesar)
        process_flags = subprocess.CREATE_NEW_CONSOLE
    
    # Încercăm să pornim modelul. Dacă Ollama nu e pornit, comanda 'ollama run' îl va porni.
    # Această parte poate fi problematică dacă Ollama nu e în PATH sau necesită privilegii.
    # Ideal, Ollama ar trebui să ruleze deja ca un serviciu separat.
    try:
        # Comanda `ollama run` este blocantă dacă modelul nu e descărcat,
        # sau dacă Ollama server nu rulează.
        # Pentru un API, e mai bine ca Ollama să fie deja un serviciu pornit.
        # Aici, doar verificăm, presupunând că utilizatorul gestionează pornirea Ollama.
        print("   (Presupunând că serviciul Ollama este activ și modelul este disponibil sau se descarcă)")
        # subprocess.Popen(["ollama", "run", model_name], creationflags=process_flags)
    except FileNotFoundError:
        print("❌ Comanda 'ollama' nu a fost găsită. Asigură-te că Ollama este instalat și în PATH.")
        raise RuntimeError("Ollama command not found. Please start Ollama manually.")
    
    # Așteaptă ca modelul să devină disponibil
    for i in range(60): # Așteaptă până la 1 minut
        print(f"⏳ Aștept ca modelul '{model_name}' să fie gata în Ollama... ({i+1}/60)")
        try:
            r = requests.post(f"{OLLAMA_API_URL}/api/generate",
                              json={"model": model_name, "prompt": "Hi", "stream": False, "options": {"num_predict": 1}}, timeout=2)
            if r.status_code == 200 and "error" not in r.text.lower():
                try:
                    r.json() # Verifică dacă răspunsul e JSON valid
                    print(f"✅ Modelul '{model_name}' este gata în Ollama.")
                    return
                except requests.exceptions.JSONDecodeError:
                    pass # Continuă să încerci dacă JSON e invalid dar status e 200
            elif r.status_code == 404 or "error" in r.text.lower() : # Model not found or other error
                 print(f"   Modelul '{model_name}' încă nu e gata sau e o eroare: {r.text[:100]}")

        except requests.exceptions.RequestException:
            print("   Serviciul Ollama nu răspunde încă...")
        time.sleep(2) # Așteaptă 2 secunde între verificări
    
    print(f"❌ Timeout la pornirea/verificarea modelului Ollama '{model_name}'. Verifică manual Ollama.")
    raise RuntimeError(f"Ollama model '{model_name}' not ready after timeout. Please check Ollama logs.")

ensure_ollama_model_running(OLLAMA_MODEL_NAME)

# EMBEDDING
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔄 Inițializează embedding BGE-M3 pe dispozitivul: {device}...")
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': device}
)

# VERIFICĂ FOLDER DOCUMENTE
if not DOCUMENT_FOLDER.is_dir() or not any(DOCUMENT_FOLDER.iterdir()):
    print(f"❌ Directorul '{DOCUMENT_FOLDER}' nu a fost găsit sau este gol. Creați-l și adăugați fișiere .txt și .pdf.")
    raise FileNotFoundError(f"Document folder '{DOCUMENT_FOLDER}' not found or is empty.")

# ÎNCARCĂ DOCUMENTELE
docs_loaded = []
print(f"📂 Se încarcă documente din '{DOCUMENT_FOLDER}'...")
for fname in os.listdir(DOCUMENT_FOLDER):
    path_str = str(DOCUMENT_FOLDER / fname)
    file_specific_docs = []
    try:
        if fname.lower().endswith(".txt"):
            loader = TextLoader(path_str, encoding="utf-8")
            file_specific_docs = loader.load()
            print(f"  👍 Încărcat (TXT): {fname}")
        elif fname.lower().endswith(".pdf"):
            loader = PyPDFLoader(path_str)
            file_specific_docs = loader.load()
            print(f"  👍 Încărcat (PDF): {fname} ({len(file_specific_docs)} pagini procesate)")

        if file_specific_docs:
            for doc_item in file_specific_docs:
                doc_item.metadata["source"] = fname # Adaugă numele fișierului în metadate
            docs_loaded.extend(file_specific_docs)
        elif not (fname.lower().endswith(".txt") or fname.lower().endswith(".pdf")):
            if not os.path.isdir(DOCUMENT_FOLDER / fname): # Ignoră subfolderele
                print(f"  ⚠️ Ignorat (tip fișier neacceptat): {fname}")
    except Exception as e:
        print(f"  ❌ Eroare la încărcarea fișierului {fname}: {e}")

if not docs_loaded:
    print("❌ Nu s-au găsit surse de documente valide (.txt, .pdf) în folder.")
    raise ValueError("No valid documents found to load.")
print(f"✅ Total documente (inclusiv pagini PDF) încărcate: {len(docs_loaded)}")

# SPLIT
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
texts = splitter.split_documents(docs_loaded)
print(f"📄 Fragmente create: {len(texts)}")

# VECTORSTORE
print(f"💾 Se creează VectorStore Chroma în '{CHROMA_DB_PATH}'...")
vectorstore = Chroma(
    embedding_function=embedding_model,
    persist_directory=str(CHROMA_DB_PATH) # Chroma va crea directorul dacă nu există
)
BATCH_SIZE_CHROMA = 500 # Poți ajusta mărimea batch-ului dacă ai multe fragmente
for i in range(0, len(texts), BATCH_SIZE_CHROMA):
    batch = texts[i:i + BATCH_SIZE_CHROMA]
    print(f"📦 Se adaugă batch de fragmente {i//BATCH_SIZE_CHROMA + 1}/{(len(texts) -1)//BATCH_SIZE_CHROMA + 1} ({len(batch)} fragmente) în ChromaDB...")
    vectorstore.add_documents(batch)
print("✅ VectorStore creat și fragmentele adăugate.")

# LLM
print(f"🤖 Inițializează LLM: {OLLAMA_MODEL_NAME}...")
llm = OllamaLLM(
    model=OLLAMA_MODEL_NAME,
    temperature=0.3,
    top_p=0.95,
    repeat_penalty=1.1,
    num_ctx=4096, # Context window size
    num_predict=512 # Max tokens to predict
)

# PROMPTURI (copiază-le exact din scriptul tău original)
map_prompt_template_str = """
Ai mai jos un fragment de text și o întrebare principală.
Sarcina ta este să analizezi CU ATENȚIE fragmentul și să extragi ORICE informație din ACEST FRAGMENT care ar putea contribui la răspunsul final pentru Întrebarea principală.
Concentrează-te pe extragerea exactă a detaliilor relevante din fragmentul curent. Nu adăuga interpretări care nu sunt direct susținute de textul fragmentului. Nu adăuga informații externe.
Dacă fragmentul curent nu conține absolut nicio informație relevantă pentru Întrebarea principală, răspunde foarte scurt, de exemplu: "-fragment irelevant pentru întrebare-".

Fragment:
{context}

Întrebare principală (pentru care acest fragment ar putea conține o parte din răspuns):
{question}

Informații extrase din fragmentul de mai sus, relevante pentru întrebarea principală (sau "-fragment irelevant pentru întrebare-" dacă este cazul):
"""
map_prompt = PromptTemplate.from_template(map_prompt_template_str)

reduce_prompt_template_str = """
CONTEXT: Ai mai jos o serie de extrageri informaționale ("Răspunsuri Parțiale") provenite din diferite fragmente ale Codului Rutier, menite să răspundă la "Întrebarea Inițială". Unele extrageri pot fi etichetate ca "-fragment irelevant-" sau pot conține negații despre prezența informației.

SARCINA TA PRINCIPALĂ: Sintetizează un răspuns comparativ FINAL la "Întrebarea Inițială", bazându-te STRICT pe informațiile AFIRMATIVE și FACTUALE extrase în "Răspunsurile Parțiale".

PAȘI DE URMAT:
1.  Ignoră complet orice "Răspuns Parțial" care este marcat explicit ca "-fragment irelevant-" sau care doar neagă prezența informației fără a oferi date concrete.
2.  Pentru FIECARE ASPECT al "Întrebării Inițiale" (ex. vârstă minimă, locuri permise):
    a.  Adună TOATE informațiile afirmative și factuale specifice acelui aspect, extrase din "Răspunsurile Parțiale" rămase, CHIAR DACĂ informația apare într-un singur răspuns parțial.
    b.  Notează dacă pentru un vehicul (ex. bicicletă) există informații despre un aspect, iar pentru celălalt (ex. trotinetă) nu există informații specifice despre ACELAȘI aspect în extragerile valide.
3.  Compară informațiile adunate pentru fiecare subiect (ex. trotinete vs. biciclete) și pentru fiecare aspect.
4.  Formulează un "Răspuns Comparativ Final" care:
    a.  Prezintă clar asemănările și DEOSEBIRILE specifice pentru fiecare aspect cerut în "Întrebarea Inițială". Include toate detaliile extrase.
    b.  Se bazează EXCLUSIV pe detaliile concrete furnizate în "Răspunsurile Parțiale" valide.
    c.  Dacă pentru un anumit aspect specific (ex. 'locuri permise pentru biciclete în lipsa pistei') nu există informații concrete afirmative în niciun "Răspuns Parțial" valid, menționează explicit acest lucru pentru acel aspect și vehicul.
    d.  Fii cât mai complet și detaliat posibil pe baza extragerilor.

Răspunsuri Parțiale (rezultate din analiza individuală a fragmentelor):
{context}

Întrebare Inițială:
{question}

Răspuns Comparativ Final (bazat pe analiza și sinteza informațiilor factuale din răspunsurile parțiale de mai sus):
"""
reduce_prompt = PromptTemplate.from_template(reduce_prompt_template_str)

prompt_stuff_template_str = """
Folosind strict informațiile din contextul de mai jos, oferă un răspuns cât mai complet și detaliat posibil la întrebare, bazându-te pe toate detaliile relevante găsite.
Nu inventa nimic.

Context:
{context}

Întrebare: {question}
Răspuns:
"""
prompt_stuff = PromptTemplate.from_template(prompt_stuff_template_str)

# LANȚURI RAG (RetrievalQA și MapReduce)
print("🔗 Se configurează lanțurile RAG...")
retriever = vectorstore.as_retriever(search_kwargs={"k": K_CONTEXT_CHUNKS})
map_chain = map_prompt | llm
reduce_chain = reduce_prompt | llm

qa_chain_stuff = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff", # Folosește direct contextul recuperat
    chain_type_kwargs={"prompt": prompt_stuff},
    return_source_documents=True # Important pentru a putea afișa sursele
)
print("✅ Inițializare completă. API-ul este gata să primească cereri.")
# --- Sfârșit Inițializare Globală ---


# Inițializează aplicația FastAPI
app = FastAPI(title="RAG Cod Rutier API")

# Configurare CORS (Cross-Origin Resource Sharing)
# Permite cereri de la frontend-ul tău Next.js care rulează pe localhost:3000
origins = [
    "http://localhost:3000", # Adresa standard pentru Next.js în mod dezvoltare
    "http://127.0.0.1:3000", # Uneori e nevoie și de aceasta
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Permite toate metodele (GET, POST, etc.)
    allow_headers=["*"], # Permite toate headerele
)

# Modele Pydantic pentru validarea datelor de request și response
class QuestionRequest(BaseModel):
    question: str

class SourceDocumentResponse(BaseModel): # Nume schimbat pentru claritate
    page_content: str
    metadata: dict
    score: float | None = None # Scorul de similaritate, poate fi None

class QAResponse(BaseModel):
    answer: str
    source_documents: list[SourceDocumentResponse]
    processing_time: float

# Endpoint-ul API pentru a primi întrebări
@app.post("/ask", response_model=QAResponse)
async def ask_question_endpoint(request: QuestionRequest = Body(...)):
    question = request.question
    if not question:
        raise HTTPException(status_code=400, detail="Întrebarea nu poate fi goală.")

    start_time_request = time.time()
    print(f"\n🔍 Primit întrebare prin API: '{question}'")

    try:
        # Recuperare documente și scoruri
        print("   🔍 Recuperez fragmente și scoruri...")
        docs_and_scores_tuples = vectorstore.similarity_search_with_score(question, k=K_CONTEXT_CHUNKS)
        
        documents_for_processing = [doc_score[0] for doc_score in docs_and_scores_tuples]
        scores_for_processing = [doc_score[1] for doc_score in docs_and_scores_tuples]

        if not documents_for_processing:
            print("   ⚠️ Nu s-au găsit fragmente relevante.")
            # Chiar dacă nu găsim documente, putem încerca să răspundem (LLM-ul ar putea avea cunoștințe generale)
            # Sau putem returna un mesaj specific. Aici, lăsăm LLM-ul să încerce cu context gol, sau returnăm mesaj.
            # Pentru RAG, e mai bine să indicăm că nu s-a găsit context.
            processing_time = time.time() - start_time_request
            return QAResponse(
                answer="Nu am găsit informații relevante în documentele mele pentru a răspunde la această întrebare.",
                source_documents=[],
                processing_time=processing_time
            )

        final_answer_text = ""
        processed_source_docs_response = []

        # Decide ce lanț să folosești: map_reduce pentru comparații, stuff pentru altele
        if any(keyword in question.lower() for keyword in ["diferența", "comparație", "vs", "comparativ", "compară"]):
            print(f"   ⚙️ Procesez ca întrebare comparativă (map_reduce) cu {len(documents_for_processing)} fragmente...")
            partials = []
            for i, doc_content in enumerate(documents_for_processing):
                print(f"      🗺️ Map Pasul {i+1}/{len(documents_for_processing)} pentru sursa '{doc_content.metadata.get('source', 'necunoscut')}'...")
                partial_response_text = map_chain.invoke({"context": doc_content.page_content, "question": question})
                partials.append(partial_response_text)
            
            joined_partials = "\n\n---\n\n".join(partials)
            print("      ⚙️ Reduce: Sintetizez răspunsul final...")
            final_answer_text = reduce_chain.invoke({"context": joined_partials, "question": question})
            
            # Pentru map_reduce, toate documentele recuperate sunt considerate surse
            for i, doc in enumerate(documents_for_processing):
                processed_source_docs_response.append(SourceDocumentResponse(
                    page_content=doc.page_content,
                    metadata=doc.metadata,
                    score=scores_for_processing[i] if i < len(scores_for_processing) else None
                ))
        else:
            print(f"   ⚙️ Procesez ca întrebare directă (stuff) cu {len(documents_for_processing)} fragmente...")
            # Pentru RetrievalQA cu chain_type="stuff", contextul este format din toate documentele recuperate
            # și trimis direct la LLM împreună cu întrebarea.
            # 'result' conține răspunsul, 'source_documents' conține documentele folosite de chain.
            qa_result = qa_chain_stuff.invoke({"query": question})
            final_answer_text = qa_result['result']
            
            # Asociază scorurile cu documentele returnate de qa_chain_stuff
            # Presupunem că ordinea și numărul documentelor returnate de qa_chain_stuff
            # corespund celor din docs_and_scores_tuples (ceea ce ar trebui să fie adevărat
            # dacă retriever-ul este același și k este același)
            returned_sources_from_chain = qa_result['source_documents']
            for i, doc in enumerate(returned_sources_from_chain):
                 # Caută scorul original bazat pe conținutul paginii, ca o măsură de siguranță
                 # sau pur și simplu folosește scorurile în ordinea primită.
                 # Aici folosim ordinea, presupunând că este consistentă.
                original_score = scores_for_processing[i] if i < len(scores_for_processing) else None

                # Verificăm dacă putem găsi documentul original pentru a lua scorul mai sigur
                # Aceasta e o potrivire mai robustă.
                matching_original_doc_score = None
                for k_doc, k_score in docs_and_scores_tuples:
                    if k_doc.page_content == doc.page_content and k_doc.metadata.get('source') == doc.metadata.get('source'):
                        matching_original_doc_score = k_score
                        break
                
                processed_source_docs_response.append(SourceDocumentResponse(
                    page_content=doc.page_content,
                    metadata=doc.metadata,
                    score=matching_original_doc_score if matching_original_doc_score is not None else original_score
                ))


        processing_time = time.time() - start_time_request
        print(f"   ✅ Răspuns generat în {processing_time:.2f} secunde.")
        
        return QAResponse(
            answer=final_answer_text,
            source_documents=processed_source_docs_response,
            processing_time=processing_time
        )

    except Exception as e:
        print(f"❌ Eroare majoră în timpul procesării API: {e}")
        # Aici poți adăuga logging mai detaliat al excepției dacă e nevoie
        # import traceback
        # print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Eroare internă la procesarea întrebării: {str(e)}")


# Această parte permite rularea serverului direct cu "python rag_api.py"
# Dar este recomandat să folosești "uvicorn rag_api:app --reload --port 8000" pentru dezvoltare
if __name__ == "__main__":
    print("🚀 Pornesc serverul API FastAPI cu Uvicorn pe http://localhost:8000")
    print("   Pentru dezvoltare, este recomandat să rulezi cu: uvicorn rag_api:app --reload --port 8000")
    # Asigură-te că Ollama rulează și modelul e descărcat.
    # Asigură-te că folderul ./fisiere există și conține documente.
    uvicorn.run(app, host="0.0.0.0", port=8000)