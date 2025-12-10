"""
RAG (Retrieval-Augmented Generation) Module for Fraud Detection
Implements vector embeddings and semantic search for fraud case retrieval
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import pickle
import os

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformers not installed. Install with: pip install sentence-transformers")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️  faiss not installed. Install with: pip install faiss-cpu")


class FraudRAG:
    """
    RAG system for fraud detection using vector embeddings and semantic search
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize RAG system
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.index = None
        self.claims_data = None
        self.embeddings = None
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
        
        if not FAISS_AVAILABLE:
            raise ImportError("faiss is required. Install with: pip install faiss-cpu")
        
        print(f"🤖 Initializing RAG system with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"✅ Model loaded successfully")
    
    def create_claim_text(self, row: pd.Series) -> str:
        """
        Create searchable text representation of a claim
        
        Args:
            row: DataFrame row containing claim data
            
        Returns:
            Text representation of the claim
        """
        text_parts = []
        
        # Basic claim info
        text_parts.append(f"Claim {row.get('claim_id', 'Unknown')}")
        text_parts.append(f"Specialty: {row.get('specialty', 'Unknown')}")
        text_parts.append(f"Diagnosis: {row.get('diagnosis_code', 'Unknown')}")
        text_parts.append(f"Procedure: {row.get('procedure_code', 'Unknown')}")
        text_parts.append(f"Medication: {row.get('medication_code', 'Unknown')}")
        text_parts.append(f"Amount: ${row.get('amount', 0):.2f}")
        
        # Fraud information
        if 'rule_based_fraud' in row and row['rule_based_fraud']:
            text_parts.append(f"FRAUD DETECTED: {row.get('fraud_reasons', 'Unknown reason')}")
        
        if 'prediction_label' in row and row['prediction_label'] == 'Suspicious':
            text_parts.append("ML Model flagged as suspicious")
        
        # Provider info if available
        if 'name' in row:
            text_parts.append(f"Provider: {row['name']}")
        if 'location' in row:
            text_parts.append(f"Location: {row['location']}")
        
        return " | ".join(text_parts)
    
    def build_index(self, claims_df: pd.DataFrame, save_path: str = 'rag_index'):
        """
        Build FAISS index from claims data
        
        Args:
            claims_df: DataFrame containing claims data
            save_path: Path to save the index and embeddings
        """
        print(f"\n🔨 Building RAG index from {len(claims_df)} claims...")
        
        self.claims_data = claims_df.copy()
        
        # Create text representations
        print("📝 Creating text representations...")
        claim_texts = [self.create_claim_text(row) for _, row in claims_df.iterrows()]
        
        # Generate embeddings
        print("🧠 Generating embeddings (this may take a minute)...")
        self.embeddings = self.model.encode(
            claim_texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Build FAISS index
        print("🔍 Building FAISS index...")
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"✅ Index built successfully!")
        print(f"  • Dimension: {dimension}")
        print(f"  • Total vectors: {self.index.ntotal}")
        
        # Save index
        if save_path:
            self.save_index(save_path)
    
    def save_index(self, save_path: str):
        """Save FAISS index and associated data"""
        print(f"\n💾 Saving RAG index to {save_path}...")
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(save_path, 'faiss.index'))
        
        # Save claims data
        self.claims_data.to_pickle(os.path.join(save_path, 'claims_data.pkl'))
        
        # Save embeddings
        np.save(os.path.join(save_path, 'embeddings.npy'), self.embeddings)
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'num_claims': len(self.claims_data),
            'embedding_dim': self.embeddings.shape[1]
        }
        with open(os.path.join(save_path, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✅ Index saved successfully!")
    
    def load_index(self, load_path: str):
        """Load FAISS index and associated data"""
        print(f"\n📂 Loading RAG index from {load_path}...")
        
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(load_path, 'faiss.index'))
        
        # Load claims data
        self.claims_data = pd.read_pickle(os.path.join(load_path, 'claims_data.pkl'))
        
        # Load embeddings
        self.embeddings = np.load(os.path.join(load_path, 'embeddings.npy'))
        
        # Load metadata
        with open(os.path.join(load_path, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        print(f"✅ Index loaded successfully!")
        print(f"  • Model: {metadata['model_name']}")
        print(f"  • Claims: {metadata['num_claims']}")
        print(f"  • Dimension: {metadata['embedding_dim']}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar fraud cases using semantic search
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            
        Returns:
            List of dictionaries containing search results
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Format results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            claim = self.claims_data.iloc[idx].to_dict()
            claim['similarity_score'] = float(1 / (1 + dist))  # Convert distance to similarity
            claim['rank'] = i + 1
            results.append(claim)
        
        return results
    
    def search_fraud_cases(self, query: str, top_k: int = 5, fraud_only: bool = True) -> List[Dict]:
        """
        Search for fraud cases with optional filtering
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            fraud_only: If True, only return fraudulent claims
            
        Returns:
            List of fraud cases matching the query
        """
        # Search more results to account for filtering
        search_k = top_k * 3 if fraud_only else top_k
        results = self.search(query, top_k=search_k)
        
        # Filter for fraud cases if requested
        if fraud_only:
            results = [
                r for r in results 
                if r.get('rule_based_fraud', False) or r.get('prediction_label') == 'Suspicious'
            ]
        
        return results[:top_k]
    
    def get_fraud_summary(self, results: List[Dict]) -> str:
        """
        Generate a summary of fraud cases
        
        Args:
            results: List of fraud case dictionaries
            
        Returns:
            Text summary of the fraud cases
        """
        if not results:
            return "No fraud cases found."
        
        summary_parts = [f"Found {len(results)} relevant fraud cases:\n"]
        
        for i, case in enumerate(results, 1):
            summary_parts.append(f"\n{i}. Claim {case.get('claim_id', 'Unknown')}")
            summary_parts.append(f"   • Specialty: {case.get('specialty', 'Unknown')}")
            summary_parts.append(f"   • Amount: ${case.get('amount', 0):,.2f}")
            summary_parts.append(f"   • Fraud Reasons: {case.get('fraud_reasons', 'ML flagged')}")
            summary_parts.append(f"   • Similarity: {case.get('similarity_score', 0):.2%}")
        
        return "\n".join(summary_parts)


def build_rag_index_from_file(claims_file: str = 'processed_claims_etl.csv', 
                                save_path: str = 'rag_index'):
    """
    Convenience function to build RAG index from a CSV file
    
    Args:
        claims_file: Path to processed claims CSV
        save_path: Path to save the index
    """
    print("=" * 60)
    print("🚀 Building RAG Index for Fraud Detection")
    print("=" * 60)
    
    # Load data
    print(f"\n📂 Loading claims data from {claims_file}...")
    df = pd.read_csv(claims_file)
    print(f"✅ Loaded {len(df)} claims")
    
    # Initialize RAG
    rag = FraudRAG()
    
    # Build index
    rag.build_index(df, save_path=save_path)
    
    print("\n" + "=" * 60)
    print("✅ RAG Index Built Successfully!")
    print("=" * 60)
    
    # Test search
    print("\n🧪 Testing semantic search...")
    test_query = "duplicate billing in cardiology"
    results = rag.search_fraud_cases(test_query, top_k=3)
    print(f"\nQuery: '{test_query}'")
    print(rag.get_fraud_summary(results))
    
    return rag


if __name__ == "__main__":
    # Build index from processed claims
    rag = build_rag_index_from_file()
    
    print("\n💡 Next Steps:")
    print("  1. Import FraudRAG in your application")
    print("  2. Load the index with: rag = FraudRAG(); rag.load_index('rag_index')")
    print("  3. Search with: results = rag.search_fraud_cases('your query')")
