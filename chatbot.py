import streamlit as st
import os
import tempfile

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredImageLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer

import chromadb


load_dotenv()


# =========================================================
# LLM
# =========================================================

model = ChatGoogleGenerativeAI(
    api_key="",
    model="gemini-2.5-flash",
    temperature=0.7
)


# =========================================================
# EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "Qwen/Qwen3-Embedding-0.6B"
    )


embedding_model = load_embedding_model()


# =========================================================
# FRONTEND
# =========================================================

st.title("RAG-APPLICATION")


uploaded_file = st.file_uploader(
    "Choose a PDF or Image",
    type=["pdf", "jpg", "jpeg"]
)


# =========================================================
# EVALUATION QUESTIONS
# =========================================================

questions = [

    {
        "question": "Where did Shivanshu work as a Data Science Intern?",
        "relevant_chunks": [0]
    },

    {
        "question": "What did Shivanshu do at CBRE?",
        "relevant_chunks": [0, 1]
    },

    {
        "question": "Which states did Shivanshu scrape RERA data for?",
        "relevant_chunks": [0, 2]
    },

    {
        "question": "What technologies did Shivanshu use for RERA data extraction?",
        "relevant_chunks": [0, 2]
    },

    {
        "question": "What fuzzy lookup tool did Shivanshu build?",
        "relevant_chunks": [0]
    },

    {
        "question": "What was the purpose of the Google News scraper?",
        "relevant_chunks": [0, 1, 2]
    },

    {
        "question": "Which technologies were used to build the Google News scraper?",
        "relevant_chunks": [0, 1, 2]
    },

    {
        "question": "How did Shivanshu validate the extracted RERA data?",
        "relevant_chunks": [1]
    },

    {
        "question": "How did Shivanshu use KD-Tree at CBRE?",
        "relevant_chunks": [1]
    },

    {
        "question": "How many Pune projects and schools were involved in the geospatial matching?",
        "relevant_chunks": [1]
    },

    {
        "question": "Where did Shivanshu work as a Quality Analyst Intern?",
        "relevant_chunks": [1]
    },

    {
        "question": "What did Shivanshu test during his Quality Analyst internship?",
        "relevant_chunks": [1, 2]
    },

    {
        "question": "What is Shivanshu's AI-Powered iPad Calculator project?",
        "relevant_chunks": [2, 3]
    },

    {
        "question": "Which LLM was used in the AI-Powered iPad Calculator?",
        "relevant_chunks": [2, 3]
    },

    {
        "question": "What is the Multi-Agent AI Research System?",
        "relevant_chunks": [2, 3, 4]
    },

    {
        "question": "What are the four stages of the Multi-Agent AI Research System?",
        "relevant_chunks": [3]
    },

    {
        "question": "Which tools and frameworks were used in the Multi-Agent AI Research System?",
        "relevant_chunks": [3]
    },

    {
        "question": "What technical skills does Shivanshu have?",
        "relevant_chunks": [4]
    },

    {
        "question": "What is Shivanshu's B.Tech CGPA?",
        "relevant_chunks": [5]
    },

    {
        "question": "Where did Shivanshu complete his B.Tech?",
        "relevant_chunks": [5]
    }

]


# =========================================================
# FILE PROCESSING
# =========================================================

if uploaded_file is not None:

    st.write(
        f"Selected file: {uploaded_file.name}"
    )


    # -----------------------------------------------------
    # GET FILE EXTENSION
    # -----------------------------------------------------

    file_extension = os.path.splitext(
        uploaded_file.name
    )[1].lower()


    # -----------------------------------------------------
    # DISPLAY IMAGE
    # -----------------------------------------------------

    if file_extension in [".jpg", ".jpeg"]:

        st.image(
            uploaded_file,
            caption="Uploaded Image",
            use_container_width=True
        )


    # -----------------------------------------------------
    # SAVE FILE TEMPORARILY
    # -----------------------------------------------------

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=file_extension
    ) as temp_file:

        temp_file.write(
            uploaded_file.getvalue()
        )

        temp_file_path = temp_file.name


    # -----------------------------------------------------
    # DOCUMENT LOADER
    # -----------------------------------------------------

    if file_extension == ".pdf":

        loader = PyMuPDFLoader(
            temp_file_path
        )

    else:

        loader = UnstructuredImageLoader(
            temp_file_path
        )


    documents = loader.load()


    # =====================================================
    # DISPLAY EXTRACTED TEXT
    # =====================================================

    st.write(
        f"Number of documents: {len(documents)}"
    )

    st.write(
        documents[0].page_content
    )


    # =====================================================
    # DOCUMENT SPLITTER
    # =====================================================

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )


    chunks = text_splitter.split_documents(
        documents
    )


    # =====================================================
    # DISPLAY CHUNKS
    # =====================================================

    st.write(
        f"Number of documents: {len(documents)}"
    )

    st.write(
        f"Number of chunks: {len(chunks)}"
    )

    st.write("## Chunks")


    for i in range(len(chunks)):

        st.write(
            f"### Chunk ID: {i}"
        )

        st.write("Content:")

        st.write(
            chunks[i].page_content
        )

        st.write("Metadata:")

        st.write(
            chunks[i].metadata
        )


    # =====================================================
    # VECTOR EMBEDDINGS
    # =====================================================

    texts = [
        chunk.page_content
        for chunk in chunks
    ]


    embeddings = embedding_model.encode(
        texts
    )


    st.write(
        "### Embedding Shape"
    )

    st.write(
        embeddings.shape
    )


    # =====================================================
    # CHROMA VECTOR DATABASE
    # =====================================================

    client = chromadb.Client()


    collection = client.get_or_create_collection(
        name="rag_documents"
    )


    collection.add(

        ids=[
            str(i)
            for i in range(len(chunks))
        ],

        documents=texts,

        embeddings=embeddings.tolist(),

        metadatas=[
            chunk.metadata
            for chunk in chunks
        ]
    )


    st.write(
        "Documents stored in Chroma:"
    )

    st.write(
        collection.count()
    )


    # =====================================================
    # RETRIEVAL EVALUATION FUNCTION
    # =====================================================

    def evaluate_retrieval(
        collection,
        embedding_model,
        questions,
        k=5
    ):

        evaluation_results = []


        # -------------------------------------------------
        # LOOP THROUGH ALL QUESTIONS
        # -------------------------------------------------

        for item in questions:

            question = item["question"]


            # Ground truth
            relevant_chunks = set(
                item["relevant_chunks"]
            )


            # -------------------------------------------------
            # EMBED QUESTION
            # -------------------------------------------------

            query_embedding = embedding_model.encode(
                question
            ).tolist()


            # -------------------------------------------------
            # SEARCH CHROMA
            # -------------------------------------------------

            search_results = collection.query(

                query_embeddings=[
                    query_embedding
                ],

                n_results=k,

                include=[
                    "documents",
                    "metadatas",
                    "distances"
                ]
            )


            # -------------------------------------------------
            # GET RETRIEVED CHUNK IDS
            # -------------------------------------------------

            retrieved_ids = [

                int(chunk_id)

                for chunk_id in search_results["ids"][0]

            ]


            # -------------------------------------------------
            # CALCULATE RECALL@K
            # -------------------------------------------------

            found_relevant = (

                set(retrieved_ids)
                & relevant_chunks

            )


            recall = (

                len(found_relevant)
                /
                len(relevant_chunks)

            )


            # -------------------------------------------------
            # CALCULATE RECIPROCAL RANK
            # -------------------------------------------------

            reciprocal_rank = 0


            for rank, chunk_id in enumerate(
                retrieved_ids,
                start=1
            ):

                if chunk_id in relevant_chunks:

                    reciprocal_rank = 1 / rank

                    break


            # -------------------------------------------------
            # STORE RESULT
            # -------------------------------------------------

            evaluation_results.append({

                "question": question,

                "relevant_chunks":
                    list(relevant_chunks),

                "retrieved_chunks":
                    retrieved_ids,

                "recall":
                    recall,

                "reciprocal_rank":
                    reciprocal_rank

            })


        return evaluation_results


    # =====================================================
    # RUN EVALUATION
    # =====================================================

    evaluation_results = evaluate_retrieval(

        collection,

        embedding_model,

        questions,

        k=5

    )


    # =====================================================
    # CALCULATE RECALL@K
    # =====================================================

    def calculate_recall_at_k(
        evaluation_results,
        k
    ):

        recalls = []


        for result in evaluation_results:

            relevant = set(
                result["relevant_chunks"]
            )


            retrieved = set(
                result["retrieved_chunks"][:k]
            )


            found = (
                relevant & retrieved
            )


            recall = (

                len(found)
                /
                len(relevant)

            )


            recalls.append(
                recall
            )


        return (
            sum(recalls)
            /
            len(recalls)
        )


    # -----------------------------------------------------
    # RECALL@1
    # -----------------------------------------------------

    recall_at_1 = calculate_recall_at_k(

        evaluation_results,

        1

    )


    # -----------------------------------------------------
    # RECALL@3
    # -----------------------------------------------------

    recall_at_3 = calculate_recall_at_k(

        evaluation_results,

        3

    )


    # -----------------------------------------------------
    # RECALL@5
    # -----------------------------------------------------

    recall_at_5 = calculate_recall_at_k(

        evaluation_results,

        5

    )


    # =====================================================
    # CALCULATE MRR
    # =====================================================

    mrr = (

        sum(
            result["reciprocal_rank"]
            for result in evaluation_results
        )

        /

        len(evaluation_results)

    )


    # =====================================================
    # DISPLAY OVERALL METRICS
    # =====================================================

    st.write(
        "## Retrieval Metrics"
    )


    st.write(
        f"Recall@1: {recall_at_1:.3f}"
    )


    st.write(
        f"Recall@3: {recall_at_3:.3f}"
    )


    st.write(
        f"Recall@5: {recall_at_5:.3f}"
    )


    st.write(
        f"MRR: {mrr:.3f}"
    )


    # =====================================================
    # QUESTION-LEVEL RESULTS
    # =====================================================

    st.write(
        "## Question-Level Results"
    )


    for result in evaluation_results:

        st.write(
            f"### {result['question']}"
        )


        st.write(
            f"Ground Truth Chunks: "
            f"{result['relevant_chunks']}"
        )


        st.write(
            f"Retrieved Chunks: "
            f"{result['retrieved_chunks']}"
        )


        st.write(
            f"Recall@5: "
            f"{result['recall']:.3f}"
        )


        st.write(
            f"Reciprocal Rank: "
            f"{result['reciprocal_rank']:.3f}"
        )


    # =====================================================
    # NORMAL MANUAL RETRIEVER
    # =====================================================

    st.write(
        "## Manual Retrieval"
    )


    query = st.text_input(
        "Ask a question about your document"
    )


    if query:

        # -------------------------------------------------
        # EMBED USER QUERY
        # -------------------------------------------------

        query_embedding = embedding_model.encode(
            query
        ).tolist()


        # -------------------------------------------------
        # SEARCH CHROMA
        # -------------------------------------------------

        results = collection.query(

            query_embeddings=[
                query_embedding
            ],

            n_results=6,

            include=[
                "documents",
                "metadatas",
                "distances"
            ]

        )


        # -------------------------------------------------
        # DISPLAY RETRIEVED CHUNKS
        # -------------------------------------------------

        for i in range(
            len(results["documents"][0])
        ):

            st.write(
                f"### Result {i + 1}"
            )


            st.write(
                f"Distance: "
                f"{results['distances'][0][i]}"
            )


            st.write(
                results["documents"][0][i]
            )


            st.write(
                results["metadatas"][0][i]
            )
