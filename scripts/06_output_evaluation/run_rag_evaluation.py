"""
RAG Evaluation Runner
Comprehensive RAGAS evaluation across embedding models, chunking methods, and search modes.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import shared pipeline
try:
    # Use importlib for modules starting with numbers
    import importlib
    rag_pipeline_module = importlib.import_module('scripts.00_chunking.rag_pipeline')
    load_documents_from_directory = rag_pipeline_module.load_documents_from_directory
    ingest_documents_to_qdrant = rag_pipeline_module.ingest_documents_to_qdrant
    PIPELINE_AVAILABLE = True
except ImportError as e:
    PIPELINE_AVAILABLE = False
    logger.error(f"Pipeline not available: {e}")

# Import Qdrant client
try:
    from qdrant_client import QdrantClient
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.error("Qdrant client not available")

# Import RAGAS components
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_similarity
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning("RAGAS not available. Install with: pip install ragas datasets")

# Import RAGAS TestsetGenerator components
try:
    from ragas.testset.graph import KnowledgeGraph, Node, NodeType
    from ragas.testset.transforms import default_transforms, apply_transforms
    from ragas.testset import TestsetGenerator
    from ragas.testset.synthesizers import default_query_distribution
    TESTSET_GENERATOR_AVAILABLE = True
except ImportError:
    TESTSET_GENERATOR_AVAILABLE = False
    logger.warning("RAGAS TestsetGenerator not available. Using fallback question generation.")

# Import LangChain components for document loading
try:
    try:
        from langchain_community.document_loaders import (
            DirectoryLoader,
            PyPDFLoader,
            TextLoader
        )
    except ImportError:
        from langchain.document_loaders import (
            DirectoryLoader,
            PyPDFLoader,
            TextLoader
        )
    
    # Try DOCX loader
    try:
        from langchain_community.document_loaders import Docx2txtLoader
    except ImportError:
        try:
            from langchain.document_loaders import Docx2txtLoader
        except ImportError:
            Docx2txtLoader = None
    
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.error("LangChain not available")

# Import legal chunker integration
try:
    # Use importlib for modules starting with numbers
    legal_chunker_module = importlib.import_module('scripts.00_chunking.legal_chunker_integration')
    get_distinct_chunking_methods = legal_chunker_module.get_distinct_chunking_methods
    query_legal_documents = legal_chunker_module.query_legal_documents
    LEGAL_INTEGRATION_AVAILABLE = True
except ImportError:
    LEGAL_INTEGRATION_AVAILABLE = False
    logger.warning("Legal chunker integration not available")

# Helper functions for Qdrant client and collection name
def get_qdrant_client():
    """Get Qdrant client instance."""
    if not QDRANT_AVAILABLE:
        raise ImportError("Qdrant client not available")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
    return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

def get_legal_collection_name():
    """Get default legal collection name."""
    return os.getenv("QDRANT_COLLECTION_NAME", "legal_documents")

def get_legal_embedding_model():
    """Get default legal embedding model."""
    return os.getenv("LEGAL_EMBEDDING_MODEL", "ollama/llama3.1:latest")

# Helper functions for LLM and embedding initialization
def initialize_llm(model: str = "llama3.1:latest"):
    """Initialize LLM using Ollama."""
    try:
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, temperature=0.7)
    except ImportError:
        logger.warning("LangChain Ollama not available")
        return None

def initialize_embeddings(embedding_model: str):
    """Initialize embeddings based on model name."""
    try:
        if embedding_model.startswith("ollama/"):
            from langchain_ollama import OllamaEmbeddings
            model_name = embedding_model.replace("ollama/", "")
            return OllamaEmbeddings(model=model_name)
        elif embedding_model.startswith("text-embedding-"):
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model=embedding_model)
        elif "sentence-transformers" in embedding_model or "/" in embedding_model:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=embedding_model)
        else:
            # Default to sentence-transformers
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except ImportError as e:
        logger.warning(f"Embeddings not available for {embedding_model}: {e}")
        return None

    LLM_AVAILABLE = True


def get_save_root() -> Path:
    """Get the root directory for saving evaluation results."""
    # Get project root (two levels up from scripts/06_output_evaluation/)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    save_root = project_root / "RAG_performatnece"
    save_root.mkdir(parents=True, exist_ok=True)
    return save_root


def load_documents_for_question_generation(source_dir: str) -> List[Any]:
    """
    Load documents using LangChain loaders for question generation.
    Returns LangChain Document objects.
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain not available for document loading")
    
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    documents = []
    
    # Load PDFs
    pdf_files = list(source_path.rglob("*.pdf"))
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            documents.extend(docs)
            logger.info(f"Loaded {len(docs)} pages from {pdf_file.name}")
        except Exception as e:
            logger.warning(f"Error loading {pdf_file}: {e}")
    
    # Load DOCX files
    if Docx2txtLoader:
        docx_files = list(source_path.rglob("*.docx"))
        for docx_file in docx_files:
            try:
                loader = Docx2txtLoader(str(docx_file))
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} sections from {docx_file.name}")
            except Exception as e:
                logger.warning(f"Error loading {docx_file}: {e}")
    
    # Load MD/TXT files
    md_files = list(source_path.rglob("*.md")) + list(source_path.rglob("*.txt"))
    for md_file in md_files:
        try:
            loader = TextLoader(str(md_file))
            docs = loader.load()
            documents.extend(docs)
            logger.info(f"Loaded {len(docs)} documents from {md_file.name}")
        except Exception as e:
            logger.warning(f"Error loading {md_file}: {e}")
    
    logger.info(f"Total documents loaded for question generation: {len(documents)}")
    return documents


def generate_evaluation_dataset(
    documents: List[Any],
    llm: Any,
    embeddings: Any,
    num_questions: int = 100
) -> List[Dict[str, str]]:
    """
    Generate evaluation questions using RAGAS KnowledgeGraph and TestsetGenerator.
    
    Args:
        documents: List of LangChain Document objects
        llm: LLM instance for question generation
        embeddings: Embeddings instance
        num_questions: Number of questions to generate
        
    Returns:
        List of dicts with 'query' and 'expected_answer' (ground_truth)
    """
    if not TESTSET_GENERATOR_AVAILABLE:
        logger.warning("TestsetGenerator not available. Using fallback question generation.")
        # Fallback: Generate simple questions from document content
        questions = []
        for doc in documents[:min(50, len(documents))]:
            content = doc.page_content[:500]
            if len(content) > 100:
                sentences = content.split('.')
                if sentences:
                    first_sentence = sentences[0].strip()
                    if len(first_sentence) > 20:
                        questions.append({
                            "query": f"What does the document say about {first_sentence[:50]}?",
                            "expected_answer": first_sentence
                        })
        return questions[:num_questions]
    
    try:
        logger.info(f"Creating KnowledgeGraph from {len(documents)} documents...")
        
        # Step 1: Create KnowledgeGraph
        kg = KnowledgeGraph()
        
        # Step 2: Add documents as nodes
        for doc in documents:
            kg.nodes.append(
                Node(
                    type=NodeType.DOCUMENT,
                    properties={
                        "page_content": doc.page_content,
                        "document_metadata": doc.metadata
                    }
                )
            )
        
        logger.info(f"KnowledgeGraph created with {len(kg.nodes)} nodes")
        
        # Step 3: Enrich knowledge graph with transformations
        logger.info("Applying transformations to enrich knowledge graph...")
        trans = default_transforms(
            documents=documents,
            llm=llm,
            embedding_model=embeddings
        )
        apply_transforms(kg, trans)
        
        logger.info(f"KnowledgeGraph enriched: {len(kg.nodes)} nodes, {len(kg.relationships)} relationships")
        
        # Step 4: Create TestsetGenerator
        logger.info("Creating TestsetGenerator...")
        generator = TestsetGenerator(
            llm=llm,
            embedding_model=embeddings,
            knowledge_graph=kg
        )
        
        # Step 5: Define query distribution
        query_distribution = default_query_distribution(llm)
        logger.info(f"Query distribution: {query_distribution}")
        
        # Step 6: Generate testset
        logger.info(f"Generating {num_questions} questions...")
        testset = generator.generate(
            testset_size=num_questions,
            query_distribution=query_distribution
        )
        
        # Step 7: Convert to our format
        questions = []
        if hasattr(testset, 'to_pandas'):
            df = testset.to_pandas()
            for _, row in df.iterrows():
                questions.append({
                    "query": str(row.get("question", "")),
                    "expected_answer": str(row.get("ground_truth", ""))
                })
        elif hasattr(testset, '__iter__'):
            for item in testset:
                if isinstance(item, dict):
                    questions.append({
                        "query": str(item.get("question", "")),
                        "expected_answer": str(item.get("ground_truth", ""))
                    })
                else:
                    # Try to extract from object attributes
                    query = getattr(item, "question", None) or getattr(item, "query", None)
                    ground_truth = getattr(item, "ground_truth", None) or getattr(item, "expected_answer", None)
                    if query:
                        questions.append({
                            "query": str(query),
                            "expected_answer": str(ground_truth) if ground_truth else ""
                        })
        
        logger.info(f"Generated {len(questions)} questions from RAGAS TestsetGenerator")
        
        # If we didn't get enough questions, use fallback
        if len(questions) < num_questions:
            logger.warning(f"Only generated {len(questions)} questions, using fallback for remaining")
            for doc in documents[len(questions):min(len(questions) + num_questions, len(documents))]:
                content = doc.page_content[:500]
                if len(content) > 100:
                    sentences = content.split('.')
                    if sentences:
                        first_sentence = sentences[0].strip()
                        if len(first_sentence) > 20:
                            questions.append({
                                "query": f"What does the document say about {first_sentence[:50]}?",
                                "expected_answer": first_sentence
                            })
                            if len(questions) >= num_questions:
                                break
        
        return questions[:num_questions]
        
    except Exception as e:
        logger.error(f"Error generating questions with RAGAS: {e}", exc_info=True)
        # Fallback to simple generation
        logger.info("Using fallback question generation...")
        questions = []
        for doc in documents[:min(50, len(documents))]:
            content = doc.page_content[:500]
            if len(content) > 100:
                sentences = content.split('.')
                if sentences:
                    first_sentence = sentences[0].strip()
                    if len(first_sentence) > 20:
                        questions.append({
                            "query": f"What does the document say about {first_sentence[:50]}?",
                            "expected_answer": first_sentence
                        })
        return questions[:num_questions]


def evaluate_combination(
    qdrant_client: QdrantClient,
    collection_name: str,
    embedding_model: str,
    chunking_method: str,
    search_mode: str,
    evaluation_questions: List[Dict[str, str]],
    llm: Any,
    progress_callback: Optional[Callable[[str, float], None]] = None
) -> List[Dict[str, Any]]:
    """
    Evaluate a single combination of chunking method and search mode.
    
    Returns:
        List of result dictionaries
    """
    combination_name = f"{chunking_method}_{search_mode}"
    results = []
    num_queries = len(evaluation_questions)
    
    logger.info(f"Evaluating {combination_name} with {num_queries} queries...")
    
    for q_idx, question_item in enumerate(evaluation_questions):
        query = question_item.get("query", "")
        expected_answer = question_item.get("expected_answer", "")
        
        if not query:
            continue
        
        if progress_callback:
            progress = (q_idx + 1) / num_queries
            progress_callback(f"Evaluating query {q_idx + 1}/{num_queries} for {combination_name}...", progress)
        
        try:
            # Query using the specific chunking method and search mode
            retrieved_chunks = query_legal_documents(
                qdrant_client=qdrant_client,
                collection_name=collection_name,
                query_text=query,
                embedding_model=embedding_model,
                chunking_method=chunking_method,
                top_k=5,
                search_mode=search_mode
            )
            
            # Extract contexts
            contexts = [chunk.get("text", "") for chunk in retrieved_chunks if chunk.get("text")]
            
            # Generate answer using LLM with retrieved contexts
            if contexts:
                context_text = "\n\n".join(contexts[:3])  # Use top 3 contexts
                prompt = f"""Based on the following context, answer the question. If the answer is not in the context, say "The information is not available in the provided documents."

Context:
{context_text}

Question: {query}

Answer:"""
                
                try:
                    if hasattr(llm, 'invoke'):
                        response = llm.invoke(prompt)
                        generated_answer = response.content if hasattr(response, 'content') else str(response)
                    elif hasattr(llm, 'predict'):
                        generated_answer = llm.predict(prompt)
                    else:
                        generated_answer = str(llm(prompt))
                except Exception as e:
                    logger.warning(f"Error generating answer: {e}")
                    generated_answer = f"Error generating answer: {str(e)}"
            else:
                generated_answer = "No relevant context found."
            
            # Evaluate with RAGAS
            if contexts and generated_answer and RAGAS_AVAILABLE:
                try:
                    eval_data = {
                        "question": [query],
                        "answer": [generated_answer],
                        "contexts": [contexts],
                        "ground_truth": [expected_answer] if expected_answer else [""]
                    }
                    
                    eval_dataset = Dataset.from_dict(eval_data)
                    
                    result = evaluate(
                        dataset=eval_dataset,
                        metrics=[
                            faithfulness,
                            answer_relevancy,
                            context_precision,
                            answer_similarity
                        ],
                        llm=llm,
                        embeddings=None
                    )
                    
                    metrics_dict = result.to_dict()
                    faithfulness_score = metrics_dict.get("faithfulness", [0])[0] if metrics_dict.get("faithfulness") else 0
                    answer_relevancy_score = metrics_dict.get("answer_relevancy", [0])[0] if metrics_dict.get("answer_relevancy") else 0
                    context_precision_score = metrics_dict.get("context_precision", [0])[0] if metrics_dict.get("context_precision") else 0
                    answer_similarity_score = metrics_dict.get("answer_similarity", [0])[0] if metrics_dict.get("answer_similarity") else 0
                    
                except Exception as e:
                    logger.debug(f"Ragas evaluation failed: {e}")
                    faithfulness_score = 0
                    answer_relevancy_score = 0
                    context_precision_score = 0
                    answer_similarity_score = 0
            else:
                faithfulness_score = 0
                answer_relevancy_score = 0
                context_precision_score = 0
                answer_similarity_score = 0
            
            # Simple match check
            match = expected_answer.lower() in generated_answer.lower() if expected_answer else False
            
            results.append({
                "chunking_method": chunking_method,
                "search_mode": search_mode,
                "combination": combination_name,
                "query": query,
                "expected_answer": expected_answer,
                "generated_answer": generated_answer,
                "match": "✅ Yes" if match else "❌ No",
                "faithfulness": f"{faithfulness_score:.3f}" if faithfulness_score > 0 else "N/A",
                "answer_relevancy": f"{answer_relevancy_score:.3f}" if answer_relevancy_score > 0 else "N/A",
                "context_precision": f"{context_precision_score:.3f}" if context_precision_score > 0 else "N/A",
                "answer_similarity": f"{answer_similarity_score:.3f}" if answer_similarity_score > 0 else "N/A",
                "num_contexts": len(contexts),
                "question_index": q_idx + 1
            })
            
        except Exception as e:
            logger.error(f"Error evaluating query '{query}' with {combination_name}: {e}")
            results.append({
                "chunking_method": chunking_method,
                "search_mode": search_mode,
                "combination": combination_name,
                "query": query,
                "expected_answer": expected_answer,
                "generated_answer": f"Error: {str(e)}",
                "match": "❌ Error",
                "faithfulness": "N/A",
                "answer_relevancy": "N/A",
                "context_precision": "N/A",
                "answer_similarity": "N/A",
                "num_contexts": 0,
                "question_index": q_idx + 1
            })
    
    logger.info(f"Completed {combination_name}: {len(results)} queries evaluated")
    return results


def run_rag_evaluation(
    source_dir: str,
    embedding_models: List[str],
    chunking_methods: List[str],
    search_modes: List[str],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    num_questions: int = 100,
    progress_callback: Optional[Callable[[str, float], None]] = None
) -> pd.DataFrame:
    """
    Run comprehensive RAG evaluation across all combinations.
    
    Args:
        source_dir: Directory containing source documents
        embedding_models: List of embedding model names to test
        chunking_methods: List of chunking methods to test
        search_modes: List of search modes to test
        chunk_size: Chunk size (for chunkers that use it)
        chunk_overlap: Chunk overlap (for chunkers that use it)
        num_questions: Number of questions to generate for evaluation
        progress_callback: Optional callback(message, progress) for progress updates
        
    Returns:
        DataFrame with evaluation results
    """
    if not PIPELINE_AVAILABLE:
        raise ImportError("Pipeline not available")
    
    if not QDRANT_AVAILABLE:
        raise ImportError("Qdrant client not available")
    
    if not LEGAL_INTEGRATION_AVAILABLE:
        raise ImportError("Legal chunker integration not available")
    
    # Get Qdrant client and collection name
    qdrant_client = get_qdrant_client()
    collection_name = get_legal_collection_name()
    
    # Load documents for question generation (using LangChain loaders)
    if progress_callback:
        progress_callback("Loading documents for question generation...", 0.01)
    
    documents_for_questions = load_documents_for_question_generation(source_dir)
    
    if not documents_for_questions:
        raise ValueError(f"No documents found in {source_dir}")
    
    logger.info(f"Loaded {len(documents_for_questions)} documents for question generation")
    
    # Get file paths for ingestion
    file_paths = load_documents_from_directory(source_dir)
    
    if not file_paths:
        raise ValueError(f"No files found in {source_dir}")
    
    logger.info(f"Found {len(file_paths)} files to ingest")
    
    all_results = []
    
    # Process each embedding model
    for emb_idx, embedding_model in enumerate(embedding_models):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing embedding model: {embedding_model} ({emb_idx + 1}/{len(embedding_models)})")
        logger.info(f"{'='*60}")
        
        # Ingest documents for this embedding model
        if progress_callback:
            progress = 0.05 + (0.20 * emb_idx / len(embedding_models))
            progress_callback(f"Ingesting documents with {embedding_model}...", progress)
        
        try:
            ingestion_result = ingest_documents_to_qdrant(
                file_paths=file_paths,
                qdrant_client=qdrant_client,
                collection_name=collection_name,
                embedding_model=embedding_model,
                chunking_methods=chunking_methods
            )
            
            logger.info(f"Ingestion complete: {ingestion_result['total_chunks']} chunks stored")
            
        except Exception as e:
            logger.error(f"Error ingesting documents with {embedding_model}: {e}")
            continue
        
        # Initialize LLM and embeddings for question generation
        if progress_callback:
            progress = 0.25 + (0.10 * emb_idx / len(embedding_models))
            progress_callback(f"Initializing LLM and embeddings...", progress)
        
        try:
            llm = initialize_llm()
            embeddings = initialize_embeddings(embedding_model)
        except Exception as e:
            logger.error(f"Error initializing LLM/embeddings: {e}")
            continue
        
        # Generate evaluation questions
        if progress_callback:
            progress = 0.35 + (0.10 * emb_idx / len(embedding_models))
            progress_callback(f"Generating {num_questions} evaluation questions...", progress)
        
        try:
            evaluation_questions = generate_evaluation_dataset(
                documents=documents_for_questions,
                llm=llm,
                embeddings=embeddings,
                num_questions=num_questions
            )
            
            logger.info(f"Generated {len(evaluation_questions)} evaluation questions")
            
            if not evaluation_questions:
                logger.warning("No questions generated, skipping evaluation")
                continue
                
        except Exception as e:
            logger.error(f"Error generating questions: {e}", exc_info=True)
            continue
        
        # Evaluate all combinations for this embedding model
        total_combinations = len(chunking_methods) * len(search_modes)
        combination_num = 0
        
        for chunking_method in chunking_methods:
            for search_mode in search_modes:
                combination_num += 1
                
                if progress_callback:
                    base_progress = 0.45 + (0.50 * emb_idx / len(embedding_models))
                    combo_progress = (combination_num / total_combinations) * (0.50 / len(embedding_models))
                    progress = base_progress + combo_progress
                    progress_callback(
                        f"Evaluating {chunking_method}_{search_mode} ({combination_num}/{total_combinations})...",
                        progress
                    )
                
                try:
                    combination_results = evaluate_combination(
                        qdrant_client=qdrant_client,
                        collection_name=collection_name,
                        embedding_model=embedding_model,
                        chunking_method=chunking_method,
                        search_mode=search_mode,
                        evaluation_questions=evaluation_questions,
                        llm=llm,
                        progress_callback=None  # Don't nest callbacks
                    )
                    
                    # Add embedding model to results
                    for result in combination_results:
                        result["embedding_model"] = embedding_model
                    
                    all_results.extend(combination_results)
                    
                    logger.info(f"Completed {chunking_method}_{search_mode}: {len(combination_results)} queries evaluated")
                    
                except Exception as e:
                    logger.error(f"Error evaluating {chunking_method}_{search_mode}: {e}", exc_info=True)
                    continue
    
    if progress_callback:
        progress_callback("Evaluation complete!", 1.0)
    
    if not all_results:
        logger.warning("No evaluation results generated")
        return pd.DataFrame()
    
    results_df = pd.DataFrame(all_results)
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluation complete: {len(results_df)} total results")
    logger.info(f"{'='*60}")
    
    return results_df

