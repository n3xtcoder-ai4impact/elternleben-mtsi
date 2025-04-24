import streamlit as st
import os
import csv
from datetime import datetime
import pandas as pd
from newspaper import Article

from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core import VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.schema import Document
from llama_index.readers.web import SimpleWebPageReader

# ----- Sidebar -----
with st.sidebar:
    st.image(os.path.join(os.path.dirname(__file__), "Images", "elternleben_holding_hands.jpg"), use_container_width=True)
    
    st.title("Hello! I'm ParentingBot")
    st.markdown("Here to help you in your journey with parenting.")
    st.markdown("""
        <style>
            .stButton button {
                background-color: #80A331;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 18px;
                font-weight: bold;
            }
            .stButton button:hover {
                background-color: #0056b3;
            }
        </style>
    """, unsafe_allow_html=True)

    if st.button("🔄 Want to restart the conversation?"):
        st.session_state["chat_history"] = []
        st.success("Conversation history has been cleared. We can start fresh 😊")

    st.markdown("---")
    st.markdown("### Our resources:")

    st.markdown("""
        [Join a free live webinar](https://www.elternleben.de/elternsprechstunde/)
    """)

    st.markdown("""
        [Get in touch with us](https://www.elternleben.de/ueber-stell-uns-deine-frage/)
    """)


# ----- Load article data from CSV -----
@st.cache_resource
def load_article_urls(csv_path):
    df = pd.read_csv(csv_path)
    return df.iloc[:, 0].dropna().tolist()  # Pulls from first column only, where the urls are listed


def download_articles(urls):
    docs = []
    for url in urls:
        try:
            article = Article(url)
            article.download()
            article.parse()
            text = article.text.strip()
            if text:
                docs.append(Document(text=text))
        except Exception as e:
            print(f"Failed to download {url}: {e}")
    return docs

@st.cache_resource
def get_documents_from_urls(csv_path):
    urls = load_article_urls(csv_path)
    return download_articles(urls)

# ----- Search Engine Setup -----
csv_path = os.path.join(os.path.dirname(__file__), "data", "metadata.csv")
documents = get_documents_from_urls(csv_path)

hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceInferenceAPI(
    model_name=hf_model,
    task="text-generation",
    token="[insert_your_Hugging_Face_toke]" #to get a free token, visit https://huggingface.co/, create an account, select "Settings" from dropdown menu, in left sidebar, within "Access Tokens" click on "New Token", select "read", and generate. Copy and save token! 
)

embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
embeddings = HuggingFaceEmbedding(
    model_name=embedding_model,
    cache_folder=os.path.join(os.path.dirname(__file__), "Embeddings")
)

text_splitter = SentenceSplitter(chunk_size=800, chunk_overlap=150)

# Uncomment for first run-------
vector_index = VectorStoreIndex.from_documents(
    documents,
    transformations=[text_splitter],
    embed_model=embeddings
)
vector_index.storage_context.persist(persist_dir=os.path.join(os.path.dirname(__file__), "Vector_index"))

# Uncomment these lines after first run (and comment out the first-run lines, 104-109)-------
#storage_context = StorageContext.from_defaults(persist_dir=os.path.join(os.path.dirname(__file__), "Vector_index"))
#vector_index = load_index_from_storage(storage_context, embed_model=embeddings)

# ---------------ChatBot Setup -----------
retriever = vector_index.as_retriever(similarity_top_k=2)
memory = ChatMemoryBuffer.from_defaults()

prompts = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content=(
            "OBJECTIVE: Du bist ein herausragender Kundenservice-Mitarbeiter von ElternLeben.de, "
            "einer Plattform, die Eltern mit Inhalten und Online-Beratung zu Erziehung, Baby- und Kindesentwicklung, Ernährung und mehr unterstützt. "
            "Dein Ziel ist es, die Bedürfnisse der Nutzer zu verstehen und sie zu passenden Produkten oder Informationen auf ElternLeben.de zu führen.\n\n"

            "VORGEHEN:\n"
            "1. Frage nach der aktuellen Herausforderung des Nutzers (z. B. Babyschlaf, Ernährung, Erziehung, Mediennutzung etc.).\n"
            "2. Ermittle IMMER die gewünschte Unterstützung:\n"
            "- Braucht die Person schnelle, personalisierte Beratung?\n"
            "- Sucht sie allgemeine Informationen oder hat sie konkrete Fragen?\n"
            "- Möchte sie tiefer in ein Thema einsteigen?\n"
            "Stelle immer 1–2 Rückfragen, bevor du etwas empfiehlst.\n"
            "3. Empfehle passende Services basierend auf dem Bedarf:\n"
            "- Artikel: Nur wenn du eine relevante Seite auf ElternLeben.de kennst. Gib keine allgemeinen Tipps. Immer mit Link zur Seite.\n"
            "- E-Mail-Beratung: Bei konkreten Fragen. Link: https://www.elternleben.de/ueber-stell-uns-deine-frage/\n"
            "- Elternsprechstunde: Bei Babyschlaf, Erziehung, Sauberkeit, Kindergarten, Mediennutzung. Link: https://www.elternleben.de/elternsprechstunde/\n"
            "- Hebammenberatung: Bei Babyschlaf oder Stillen (nur für Babys unter 1 Jahr). Link: https://www.elternleben.de/hebammensprechstunde/\n"
            "- Kurse & eBooks: Wenn die Person tiefer einsteigen möchte. Verwende nur bekannte Links aus dem Shop (shop_1.csv).\n\n"

            "WICHTIG:\n"
            "- Am Ende jeder Unterhaltung: immer E-Mail-Beratung anbieten.\n"
            "- Sei freundlich, professionell, mobilfreundlich & emoji-freundlich.\n"
            "- Nutze nur bekannte ElternLeben.de-Links. Niemals andere Quellen.\n"
            "- Wenn du keine passende Information im Kontext findest, sag: "
            "'Ich kann das auf Basis der vorliegenden Informationen nicht beantworten.'\n"
            "- Antworte nur auf Themen, für die ElternLeben.de zuständig ist.\n"
        )
    )
]



@st.cache_resource
def init_bot():
    return ContextChatEngine(llm=llm, retriever=retriever, memory=memory, prefix_messages=prompts)

rag_bot = init_bot()

# ----- Page presentation -----
st.title("Parenting Bot: Help at Your Fingertips")
st.markdown("""
Ask me anything about parenting.  My answers are based on articles available in the [Elternleben website.](https://www.elternleben.de)
""")

st.markdown("**Sample Questions:**")
st.markdown("- My child does not stop crying; how can I help him?")
st.markdown("- My baby has red marks all over his body; what do I do?")

# ----- Display chat history -----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("What can I help with today?"):
    st.chat_message("human", avatar=os.path.join(os.path.dirname(__file__),"Images","parent.jpg")).markdown(user_input)
    st.session_state.chat_history.append({"role": "human", "content": user_input})

    with st.spinner("📂 Searching parenting articles..."):
        try:
            answer = rag_bot.chat(user_input).response
        except Exception as e:
            answer = f"Sorry, I had trouble processing your question: {e}"

    with st.chat_message("assistant", avatar=os.path.join(os.path.dirname(__file__),"Images","helping_hands.jpg")):
        st.markdown(answer)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})

# ----- Feedback logging -----
def log_feedback(feedback_data):
    folder_path = os.path.join(os.path.dirname(__file__),"ChatBot_Feedback")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, "chat_feedback_log.csv")

    file_exists = os.path.exists(file_path)
    with open(file_path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["timestamp", "question", "answer", "feedback"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(feedback_data)

# ----- Collect feedback -----
if st.session_state.chat_history:
    if len(st.session_state.chat_history) > 1:
        last_message = st.session_state.chat_history[-1]
        if last_message["role"] == "assistant":
            feedback_key = f"feedback_radio_{len(st.session_state.chat_history)}"
            st.markdown("### Was this answer helpful?")
            feedback = st.radio(" ", ("👍 Yes", "👎 No"), index=None, key=feedback_key)

            if feedback:
                st.write("Thank you for your feedback!")
                user_question = st.session_state.chat_history[-2]["content"]
                assistant_answer = last_message["content"]
                feedback_data = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "question": user_question,
                    "answer": assistant_answer,
                    "feedback": feedback,
                }
                log_feedback(feedback_data)
                del st.session_state[feedback_key]
