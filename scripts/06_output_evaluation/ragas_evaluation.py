"""
RAGAS Evaluation Module
Provides RAG evaluation using Ragas framework with Qdrant and LangChain.
"""

import logging
import os
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import Ragas
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
    # Try to import TestsetGenerator for automatic question generation
    try:
        from ragas.testset import TestsetGenerator
        TESTSET_GENERATOR_AVAILABLE = True
    except ImportError:
        TESTSET_GENERATOR_AVAILABLE = False
        logger.debug("Ragas TestsetGenerator not available. Question generation will use manual dataset.")
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    TESTSET_GENERATOR_AVAILABLE = False
    logger.warning("Ragas not available. Install with: pip install ragas datasets")

# Try to import LangChain components
try:
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from langchain.chains import RetrievalQA
    # Try langchain_community first (newer versions), fallback to langchain.vectorstores
    try:
        from langchain_community.vectorstores import Qdrant
    except ImportError:
        from langchain.vectorstores import Qdrant
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    # Try langchain_community first (newer versions), fallback to langchain.document_loaders
    try:
        from langchain_community.document_loaders import (
            PyPDFLoader,
            TextLoader,
            Docx2txtLoader
        )
    except ImportError:
        from langchain.document_loaders import (
            PyPDFLoader,
            TextLoader,
            Docx2txtLoader
        )
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain components not available. Install with: pip install langchain langchain-ollama langchain-community")


def load_documents_from_directory(
    source_dir: str,
    file_extensions: List[str] = [".pdf", ".md", ".docx"]
) -> List[Any]:
    """
    Load all documents from source directory with specified extensions.
    
    Args:
        source_dir: Directory path containing source files
        file_extensions: List of file extensions to load
        
    Returns:
        List of LangChain Document objects
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain not available. Install required packages.")
    
    documents = []
    source_path = Path(source_dir)
    file_count = 0
    file_stats = {}  # Track documents per file
    
    if not source_path.exists():
        logger.warning(f"Source directory {source_dir} does not exist. Creating it.")
        source_path.mkdir(parents=True, exist_ok=True)
        return documents
    
    for ext in file_extensions:
        pattern = f"*{ext}"
        files = list(source_path.glob(pattern))
        
        for file_path in files:
            try:
                if ext == ".pdf":
                    loader = PyPDFLoader(str(file_path))
                elif ext == ".md" or ext == ".txt":
                    loader = TextLoader(str(file_path), encoding='utf-8')
                elif ext == ".docx":
                    loader = Docx2txtLoader(str(file_path))
                else:
                    continue
                
                loaded_docs = loader.load()
                # Add source file metadata
                for doc in loaded_docs:
                    doc.metadata["source_file"] = file_path.name
                    doc.metadata["file_path"] = str(file_path)
                
                documents.extend(loaded_docs)
                file_count += 1
                file_stats[file_path.name] = len(loaded_docs)
                
                # Log with explanation
                if ext == ".pdf":
                    logger.info(f"Loaded {len(loaded_docs)} pages (documents) from PDF: {file_path.name}")
                elif ext == ".docx":
                    logger.info(f"Loaded {len(loaded_docs)} sections (documents) from DOCX: {file_path.name}")
                else:
                    logger.info(f"Loaded {len(loaded_docs)} document(s) from {file_path.name}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
    
    # Log summary
    logger.info(f"Summary: Loaded {file_count} file(s) → {len(documents)} document(s)/page(s)")
    for filename, doc_count in file_stats.items():
        logger.info(f"  - {filename}: {doc_count} document(s)/page(s)")
    
    return documents


def chunk_documents(
    documents: List[Any],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    chunking_method: str = "RecursiveCharacterTextSplitter"
) -> List[Any]:
    """
    Chunk documents using RecursiveCharacterTextSplitter.
    
    Args:
        documents: List of LangChain Document objects
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        chunking_method: Name of chunking method used
        
    Returns:
        List of chunked Document objects
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain not available.")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Add chunking metadata
    upload_timestamp = datetime.now().isoformat()
    for chunk in chunks:
        chunk.metadata["chunking_method"] = chunking_method
        chunk.metadata["upload_timestamp"] = upload_timestamp
    
    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks


def store_chunks_in_qdrant(
    chunks: List[Any],
    qdrant_client: Any,
    collection_name: str,
    embeddings: Any
) -> bool:
    """
    Store chunks in Qdrant vector database.
    
    Args:
        chunks: List of chunked Document objects
        qdrant_client: Qdrant client instance
        collection_name: Name of Qdrant collection
        embeddings: Embeddings model instance
        
    Returns:
        True if successful
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain not available.")
    
    try:
        # Get Qdrant connection details from the client
        # Try to get URL from client, or use default
        qdrant_url = None
        qdrant_api_key = None
        
        # Check if client has url attribute (for QdrantClient)
        if hasattr(qdrant_client, 'url'):
            qdrant_url = qdrant_client.url
        elif hasattr(qdrant_client, '_client') and hasattr(qdrant_client._client, 'url'):
            qdrant_url = qdrant_client._client.url
        else:
            # Fallback to environment variable or default
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        
        # Check for API key
        if hasattr(qdrant_client, 'api_key'):
            qdrant_api_key = qdrant_client.api_key
        else:
            qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
        
        # Create Qdrant vectorstore using URL instead of client
        # This works with all LangChain versions
        vectorstore = Qdrant.from_documents(
            documents=chunks,
            embedding=embeddings,
            url=qdrant_url,  # Use URL instead of client
            api_key=qdrant_api_key,  # Include API key if available
            collection_name=collection_name,
            force_recreate=False  # Don't overwrite existing collection - allows multiple files
        )
        
        logger.info(f"Stored {len(chunks)} chunks in Qdrant collection '{collection_name}'")
        return True
    except Exception as e:
        logger.error(f"Error storing chunks in Qdrant: {e}", exc_info=True)
        raise


def create_rag_chain(
    qdrant_client: Any,
    collection_name: str,
    embeddings: Any,
    llm: Any
) -> Any:
    """
    Create a RAG chain using LangChain RetrievalQA.
    
    Args:
        qdrant_client: Qdrant client instance
        collection_name: Name of Qdrant collection
        embeddings: Embeddings model instance
        llm: LLM instance for answer generation
        
    Returns:
        RetrievalQA chain
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain not available.")
    
    try:
        # Extract connection details from the client
        qdrant_url = None
        qdrant_api_key = None
        
        # Check if client has url attribute (for QdrantClient)
        if hasattr(qdrant_client, 'url'):
            qdrant_url = qdrant_client.url
        elif hasattr(qdrant_client, '_client') and hasattr(qdrant_client._client, 'url'):
            qdrant_url = qdrant_client._client.url
        else:
            # Fallback to environment variable or default
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        
        # Check for API key
        if hasattr(qdrant_client, 'api_key'):
            qdrant_api_key = qdrant_client.api_key
        else:
            qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
        
        # Parse URL to get host and port
        from urllib.parse import urlparse
        parsed = urlparse(qdrant_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 6333
        
        # Create embedding function wrapper
        def embedding_fn(text: str) -> List[float]:
            """Wrapper for embedding function."""
            if hasattr(embeddings, 'embed_query'):
                return embeddings.embed_query(text)
            elif hasattr(embeddings, 'embed_documents'):
                # embed_documents expects a list
                return embeddings.embed_documents([text])[0]
            else:
                raise ValueError("Embeddings object must have embed_query or embed_documents method")
        
        # Try different initialization methods based on LangChain version
        vectorstore = None
        
        # Method 1: Try with location tuple and embeddings object (most common)
        try:
            vectorstore = Qdrant(
                location=(host, port),
                collection_name=collection_name,
                embeddings=embeddings,
                api_key=qdrant_api_key
            )
        except (TypeError, AttributeError) as e1:
            logger.debug(f"Method 1 (embeddings object) failed: {e1}")
            
            # Method 2: Try with embedding_function (callable)
            try:
                vectorstore = Qdrant(
                    location=(host, port),
                    collection_name=collection_name,
                    embedding_function=embedding_fn,
                    api_key=qdrant_api_key
                )
            except (TypeError, AttributeError) as e2:
                logger.debug(f"Method 2 (embedding_function) failed: {e2}")
                
                # Method 3: Try with client parameter directly
                try:
                    vectorstore = Qdrant(
                        client=qdrant_client,
                        collection_name=collection_name,
                        embeddings=embeddings
                    )
                except (TypeError, AttributeError) as e3:
                    logger.debug(f"Method 3 (client) failed: {e3}")
                    
                    # Method 4: Try using from_existing_collection
                    try:
                        vectorstore = Qdrant.from_existing_collection(
                            embedding=embedding_fn,
                            path=collection_name,
                            location=(host, port),
                            api_key=qdrant_api_key
                        )
                    except (TypeError, AttributeError) as e4:
                        logger.error(f"All Qdrant initialization methods failed.")
                        logger.error(f"Method 1 error: {e1}")
                        logger.error(f"Method 2 error: {e2}")
                        logger.error(f"Method 3 error: {e3}")
                        logger.error(f"Method 4 error: {e4}")
                        raise ValueError(f"Could not initialize Qdrant vectorstore. Tried 4 methods. Check logs for details.")
        
        if vectorstore is None:
            raise ValueError("Failed to initialize Qdrant vectorstore with any method")
        
        # Create retrieval chain with increased k for better context
        # Increased from 5 to 10 to get more relevant chunks for better answers
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
            return_source_documents=True
        )
        
        return qa_chain
    except Exception as e:
        logger.error(f"Error creating RAG chain: {e}", exc_info=True)
        raise


def generate_questions_from_documents(
    documents: List[Any],
    llm: Any,
    embeddings: Any,
    num_questions: int = 100
) -> List[Dict[str, str]]:
    """
    Generate evaluation questions automatically from documents using RAGAS TestsetGenerator.
    
    Args:
        documents: List of LangChain Document objects
        llm: LLM instance for question generation
        embeddings: Embeddings instance
        num_questions: Number of questions to generate (default: 100)
        
    Returns:
        List of dicts with 'query' and 'expected_answer' (ground_truth)
    """
    if not RAGAS_AVAILABLE:
        raise ImportError("Ragas not available. Install with: pip install ragas datasets")
    
    if not TESTSET_GENERATOR_AVAILABLE:
        logger.warning("TestsetGenerator not available. Using fallback question generation.")
        # Fallback: Generate simple questions from document content
        questions = []
        for doc in documents[:min(50, len(documents))]:  # Limit to 50 docs for fallback
            content = doc.page_content[:500]  # First 500 chars
            if len(content) > 100:
                # Simple extraction: use first sentence as question basis
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
        # Create testset generator
        generator = TestsetGenerator(
            generator_llm=llm,
            critic_llm=llm,
            embeddings=embeddings
        )
        
        # Convert documents to Dataset format
        from datasets import Dataset as HFDataset
        
        # Prepare documents for testset generation
        doc_texts = [doc.page_content for doc in documents]
        doc_metadata = [doc.metadata for doc in documents]
        
        # Create dataset
        doc_dataset = HFDataset.from_dict({
            "text": doc_texts,
            "metadata": doc_metadata
        })
        
        # Generate testset
        logger.info(f"Generating {num_questions} questions from {len(documents)} documents...")
        testset = generator.generate(
            doc_dataset,
            num_questions=num_questions,
            distributions={
                "simple": 0.25,
                "reasoning": 0.25,
                "multi_context": 0.25,
                "conditional": 0.25
            }
        )
        
        # Convert to our format
        questions = []
        for item in testset:
            questions.append({
                "query": item.get("question", ""),
                "expected_answer": item.get("ground_truth", "")
            })
        
        logger.info(f"Generated {len(questions)} questions")
        return questions
        
    except Exception as e:
        logger.error(f"Error generating questions: {e}", exc_info=True)
        # Fallback to simple generation
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


def evaluate_rag_with_ragas(
    qa_chain: Any,
    evaluation_dataset: List[Dict[str, str]],
    llm: Any
) -> pd.DataFrame:
    """
    Evaluate RAG system using Ragas metrics.
    
    Args:
        qa_chain: LangChain RetrievalQA chain
        evaluation_dataset: List of dicts with 'query' and 'expected_answer'
        llm: LLM instance for Ragas evaluation
        
    Returns:
        DataFrame with evaluation results
    """
    if not RAGAS_AVAILABLE:
        raise ImportError("Ragas not available. Install with: pip install ragas datasets")
    
    results = []
    
    for item in evaluation_dataset:
        query = item["query"]
        expected_answer = item.get("expected_answer", "")
        
        try:
            # Get answer from RAG chain
            response = qa_chain({"query": query})
            generated_answer = response.get("result", "")
            source_documents = response.get("source_documents", [])
            
            # Extract contexts from source documents
            contexts = [doc.page_content for doc in source_documents]
            
            # Prepare dataset for Ragas evaluation
            if contexts and generated_answer:
                # Create evaluation dataset
                eval_data = {
                    "question": [query],
                    "answer": [generated_answer],
                    "contexts": [contexts],
                    "ground_truth": [expected_answer]
                }
                
                eval_dataset = Dataset.from_dict(eval_data)
                
                # Evaluate with Ragas
                try:
                    result = evaluate(
                        dataset=eval_dataset,
                        metrics=[
                            faithfulness,
                            answer_relevancy,
                            context_precision,
                            answer_similarity
                        ],
                        llm=llm,
                        embeddings=None  # Will use default
                    )
                    
                    # Extract metrics
                    metrics_dict = result.to_dict()
                    faithfulness_score = metrics_dict.get("faithfulness", [0])[0] if metrics_dict.get("faithfulness") else 0
                    answer_relevancy_score = metrics_dict.get("answer_relevancy", [0])[0] if metrics_dict.get("answer_relevancy") else 0
                    context_precision_score = metrics_dict.get("context_precision", [0])[0] if metrics_dict.get("context_precision") else 0
                    answer_similarity_score = metrics_dict.get("answer_similarity", [0])[0] if metrics_dict.get("answer_similarity") else 0
                    
                except Exception as e:
                    logger.warning(f"Ragas evaluation failed for query '{query}': {e}")
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
                "query": query,
                "expected_answer": expected_answer,
                "generated_answer": generated_answer,
                "match": "✅ Yes" if match else "❌ No",
                "faithfulness": f"{faithfulness_score:.3f}" if faithfulness_score > 0 else "N/A",
                "answer_relevancy": f"{answer_relevancy_score:.3f}" if answer_relevancy_score > 0 else "N/A",
                "context_precision": f"{context_precision_score:.3f}" if context_precision_score > 0 else "N/A",
                "answer_similarity": f"{answer_similarity_score:.3f}" if answer_similarity_score > 0 else "N/A",
                "num_contexts": len(contexts)
            })
            
        except Exception as e:
            logger.error(f"Error evaluating query '{query}': {e}", exc_info=True)
            results.append({
                "query": query,
                "expected_answer": expected_answer,
                "generated_answer": f"Error: {str(e)}",
                "match": "❌ Error",
                "faithfulness": "N/A",
                "answer_relevancy": "N/A",
                "context_precision": "N/A",
                "answer_similarity": "N/A",
                "num_contexts": 0
            })
    
    return pd.DataFrame(results)


def evaluate_all_combinations(
    qdrant_client: Any,
    collection_name: str,
    embedding_model: str,
    embeddings: Any,
    llm: Any,
    documents: List[Any],
    num_questions: int = 100,
    search_modes: List[str] = ["semantic", "bm25", "mixed"],
    progress_callback: Optional[Callable[[str, float], None]] = None
) -> pd.DataFrame:
    """
    Comprehensive RAGAS evaluation across all combinations of chunking methods and search modes.
    
    Args:
        qdrant_client: Qdrant client instance
        collection_name: Collection name
        embedding_model: Embedding model name
        embeddings: Embeddings instance
        llm: LLM instance for answer generation and evaluation
        documents: List of documents for question generation
        num_questions: Number of questions to generate (default: 100)
        search_modes: List of search modes to test (default: ["semantic", "bm25", "mixed"])
        progress_callback: Optional callback for progress updates
        
    Returns:
        DataFrame with evaluation results for all combinations
    """
    if not RAGAS_AVAILABLE:
        raise ImportError("Ragas not available. Install with: pip install ragas datasets")
    
    try:
        # Use importlib for modules starting with numbers
        import importlib
        legal_chunker_module = importlib.import_module('scripts.00_chunking.legal_chunker_integration')
        get_distinct_chunking_methods = legal_chunker_module.get_distinct_chunking_methods
        query_legal_documents = legal_chunker_module.query_legal_documents
    except ImportError:
        raise ImportError("Legal chunker integration not available")
    
    # Get available chunking methods from Qdrant
    available_chunking_methods = get_distinct_chunking_methods(
        qdrant_client,
        collection_name
    )
    
    if not available_chunking_methods:
        raise ValueError("No chunking methods found in Qdrant. Please upload documents first.")
    
    logger.info(f"Found {len(available_chunking_methods)} chunking methods: {available_chunking_methods}")
    logger.info(f"Testing {len(search_modes)} search modes: {search_modes}")
    
    total_combinations = len(available_chunking_methods) * len(search_modes)
    logger.info(f"Total combinations to test: {total_combinations}")
    
    # Generate questions from documents
    if progress_callback:
        progress_callback("Generating questions...", 0.05)
    
    evaluation_questions = generate_questions_from_documents(
        documents=documents,
        llm=llm,
        embeddings=embeddings,
        num_questions=num_questions
    )
    
    logger.info(f"Generated {len(evaluation_questions)} questions for evaluation")
    
    if progress_callback:
        progress_callback(f"Generated {len(evaluation_questions)} questions", 0.10)
    
    # Evaluate all combinations
    all_results = []
    combination_num = 0
    
    for chunking_method in available_chunking_methods:
        for search_mode in search_modes:
            combination_num += 1
            combination_name = f"{chunking_method}_{search_mode}"
            
            if progress_callback:
                progress = 0.10 + (0.85 * combination_num / total_combinations)
                progress_callback(
                    f"Evaluating {combination_name} ({combination_num}/{total_combinations})...",
                    progress
                )
            
            logger.info(f"[{combination_num}/{total_combinations}] Evaluating: {combination_name}")
            
            # Query documents for each question using this combination
            combination_results = []
            
            for q_idx, question_item in enumerate(evaluation_questions):
                query = question_item["query"]
                expected_answer = question_item.get("expected_answer", "")
                
                try:
                    # Query using the specific chunking method and search mode
                    retrieved_chunks = query_legal_documents(
                        qdrant_client=qdrant_client,
                        collection_name=collection_name,
                        query_text=query,
                        embedding_model=embedding_model,
                        chunking_method=chunking_method,
                        top_k=10,  # Increased for better context retrieval
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
                    if contexts and generated_answer:
                        try:
                            eval_data = {
                                "question": [query],
                                "answer": [generated_answer],
                                "contexts": [contexts],
                                "ground_truth": [expected_answer]
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
                    
                    combination_results.append({
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
                    combination_results.append({
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
            
            all_results.extend(combination_results)
            logger.info(f"Completed {combination_name}: {len(combination_results)} questions evaluated")
    
    if progress_callback:
        progress_callback("Evaluation complete!", 1.0)
    
    return pd.DataFrame(all_results)


def get_qdrant_statistics(
    qdrant_client: Any,
    collection_name: str
) -> pd.DataFrame:
    """
    Get statistics about files stored in Qdrant.
    
    Args:
        qdrant_client: Qdrant client instance
        collection_name: Name of Qdrant collection
        
    Returns:
        DataFrame with file statistics
    """
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        # Scroll through all points
        all_files_stats = {}
        offset = None
        
        while True:
            scroll_result = qdrant_client.scroll(
                collection_name=collection_name,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            points = scroll_result[0]
            if not points:
                break
            
            for point in points:
                payload = point.payload or {}
                source_file = payload.get("source_file", "unknown")
                
                if source_file not in all_files_stats:
                    all_files_stats[source_file] = {
                        "chunk_count": 0,
                        "chunking_method": payload.get("chunking_method", "unknown"),
                        "upload_timestamp": payload.get("upload_timestamp", "unknown")
                    }
                
                all_files_stats[source_file]["chunk_count"] += 1
            
            offset = scroll_result[1]
            if offset is None:
                break
        
        # Format as DataFrame
        stats_list = []
        for file_name, stats in sorted(all_files_stats.items()):
            upload_time = stats["upload_timestamp"]
            if upload_time != "unknown":
                try:
                    dt = datetime.fromisoformat(upload_time.replace('Z', '+00:00'))
                    upload_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    pass
            
            stats_list.append({
                "file_name": file_name,
                "num_chunks": stats["chunk_count"],
                "chunking_method": stats["chunking_method"],
                "upload_time": upload_time
            })
        
        return pd.DataFrame(stats_list)
        
    except Exception as e:
        logger.error(f"Error getting Qdrant statistics: {e}", exc_info=True)
        return pd.DataFrame(columns=["file_name", "num_chunks", "chunking_method", "upload_time"])

