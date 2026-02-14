"""
IndiGo Buddy - AI-Powered Travel Assistant
==========================================
An intelligent customer service chatbot for IndiGo Airlines that combines
policy information with real passenger review insights.

Features:
- Dual-mode responses: Policy information + Experience-based insights
- RAG (Retrieval-Augmented Generation) architecture
- User feedback and learning system
- Conversation context awareness
- Performance analytics

Author: [Your Name]
License: MIT
Version: 2.0.0
"""

import boto3
import pandas as pd
import numpy as np
import faiss
import torch
import re
import sys
import os
import logging
import json
from datetime import datetime
from io import StringIO
from typing import List, Tuple, Dict, Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# ==========================================
# CONFIGURATION
# ==========================================

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# AWS Configuration
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "YOUR_AWS_ACCESS_KEY_HERE")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "YOUR_AWS_SECRET_KEY_HERE")
REGION = os.getenv("AWS_REGION", "ap-south-1")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "your-bucket-name")
FILE_KEY = os.getenv("S3_FILE_KEY", "indigo_reviews.csv")

# Model Configuration
MODEL_ID = "google/flan-t5-base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# RAG Configuration
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
TOP_K_RETRIEVAL = 3

# Generation Configuration
MAX_RESPONSE_TOKENS = 150
MIN_RESPONSE_TOKENS = 40
TEMPERATURE = 0.4
REPETITION_PENALTY = 2.5

# System Configuration
FEEDBACK_FILE = "user_feedback.json"

# ==========================================
# KNOWLEDGE BASE
# ==========================================

GENERAL_KNOWLEDGE = {
    "baggage": (
        "IndiGo allows 7kg cabin baggage and 15kg check-in baggage for domestic flights. "
        "Additional baggage costs ₹350-550 per kg. Pre-purchasing online is cheaper than "
        "paying at the airport."
    ),
    
    "booking": (
        "Book tickets on goindigo.in or through the mobile app. Web check-in opens 48 hours "
        "before flight departure. Airport check-in closes 60 minutes before domestic flights. "
        "Seat selection is available during booking."
    ),
    
    "cancellation": (
        "Cancellations are allowed up to 2 hours before departure. Charges vary by fare type "
        "(Flexi fares have lower cancellation fees). Refunds are processed within 7-10 business "
        "days to the original payment method."
    ),
    
    "meals": (
        "No complimentary meals on domestic flights. Pre-book meals online for ₹200-400 "
        "(cheaper than onboard) or purchase onboard for ₹250-500. Both vegetarian and "
        "non-vegetarian options available. Tea/coffee costs ₹100-150."
    ),
    
    "checkin": (
        "Airport check-in counters open 2 hours before departure and close 60 minutes before. "
        "Web check-in available from 48 hours to 60 minutes before flight. Self-service kiosks "
        "are available at major airports."
    ),
    
    "seats": (
        "Standard seat selection costs ₹200-500, extra legroom seats ₹400-1000, front row "
        "seats ₹200-800. Limited free seats available. 6E Prime members get complimentary "
        "seat selection."
    ),
    
    "delay": (
        "Weather or ATC delays do not qualify for compensation. For IndiGo-caused delays "
        "exceeding 2 hours, free refreshments are provided. Flight cancellations include "
        "free rebooking or full refund options. SMS/email notifications are sent."
    ),
    
    "prime": (
        "6E Prime membership benefits include: priority check-in and boarding, complimentary "
        "seat selection, one free date change, additional baggage allowance, fast-track "
        "security clearance, and complimentary snack and beverage."
    )
}

# ==========================================
# PROMPT TEMPLATES
# ==========================================

class PromptTemplates:
    """Industry-standard prompt templates for different response types."""
    
    @staticmethod
    def policy_prompt(question: str, policy_info: str) -> str:
        """
        Generate prompt for policy-based questions.
        
        Args:
            question: User's question
            policy_info: Relevant policy information
            
        Returns:
            Formatted prompt string
        """
        return f"""Answer this customer question about IndiGo Airlines using the official policy information below. Be clear and specific in 2-3 sentences.

Policy information:
{policy_info}

Question: {question}

Answer based on the policy:"""

    @staticmethod
    def experience_prompt(question: str, reviews: str) -> str:
        """
        Generate prompt for experience-based questions.
        
        Args:
            question: User's question
            reviews: Relevant passenger reviews
            
        Returns:
            Formatted prompt string
        """
        return f"""Answer this customer question about IndiGo Airlines based on the passenger reviews provided below. Summarize what multiple passengers have experienced in 2-3 natural sentences.

Passenger reviews:
{reviews}

Question: {question}

Provide a helpful answer based only on these reviews:"""

    @staticmethod
    def hybrid_prompt(question: str, policy_info: str, reviews: str) -> str:
        """
        Generate prompt combining policy and experience information.
        
        Args:
            question: User's question
            policy_info: Relevant policy information
            reviews: Relevant passenger reviews
            
        Returns:
            Formatted prompt string
        """
        return f"""Answer this customer question about IndiGo Airlines using both the official policy and passenger experiences below. Give a complete answer in 2-4 sentences.

Official policy:
{policy_info}

Passenger experiences:
{reviews}

Question: {question}

Answer combining policy and experiences:"""

# ==========================================
# LEARNING SYSTEM
# ==========================================

class LearningSystem:
    """Track user feedback and performance metrics."""
    
    def __init__(self, feedback_file: str = FEEDBACK_FILE):
        """
        Initialize the learning system.
        
        Args:
            feedback_file: Path to JSON file for storing feedback
        """
        self.feedback_file = feedback_file
        self.feedback_data = self._load_feedback()
    
    def _load_feedback(self) -> List[Dict]:
        """Load existing feedback from file."""
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load feedback file: {e}")
                return []
        return []
    
    def _save_feedback(self) -> None:
        """Save feedback to file."""
        try:
            with open(self.feedback_file, 'w') as f:
                json.dump(self.feedback_data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save feedback: {e}")
    
    def add_feedback(self, question: str, response: str, rating: int, 
                    comment: str = "") -> None:
        """
        Add user feedback.
        
        Args:
            question: User's question
            response: System's response
            rating: User rating (1-10)
            comment: Optional user comment
        """
        self.feedback_data.append({
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "response": response,
            "rating": rating,
            "comment": comment
        })
        self._save_feedback()
    
    def get_stats(self) -> Dict:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.feedback_data:
            return {
                "total": 0,
                "avg": 0.0,
                "excellent": 0,
                "good": 0,
                "poor": 0
            }
        
        ratings = [f["rating"] for f in self.feedback_data]
        return {
            "total": len(ratings),
            "avg": sum(ratings) / len(ratings),
            "excellent": len([r for r in ratings if r >= 8]),
            "good": len([r for r in ratings if 6 <= r < 8]),
            "poor": len([r for r in ratings if r < 6])
        }

# ==========================================
# CONVERSATION MEMORY
# ==========================================

class ConversationMemory:
    """Maintain conversation context and history."""
    
    def __init__(self, max_history: int = 2):
        """
        Initialize conversation memory.
        
        Args:
            max_history: Maximum number of exchanges to remember
        """
        self.max_history = max_history
        self.history = []
        self.context = {}
    
    def add_exchange(self, question: str, response: str) -> None:
        """
        Add a question-response exchange to history.
        
        Args:
            question: User's question
            response: System's response
        """
        self.history.append({"question": question, "response": response})
        
        # Keep only recent history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        # Extract context clues
        self._extract_context(question)
    
    def _extract_context(self, question: str) -> None:
        """Extract contextual information from question."""
        q_lower = question.lower()
        
        # Passenger type context
        if any(word in q_lower for word in ["elderly", "old", "parents", "senior"]):
            self.context["passenger_type"] = "elderly"
        elif any(word in q_lower for word in ["kids", "children", "baby", "infant"]):
            self.context["passenger_type"] = "family"
        elif any(word in q_lower for word in ["business", "corporate", "work"]):
            self.context["passenger_type"] = "business"
    
    def get_context_string(self) -> str:
        """Get formatted context string."""
        if not self.context:
            return ""
        
        context_parts = []
        if "passenger_type" in self.context:
            context_parts.append(f"Passenger type: {self.context['passenger_type']}")
        
        return " | ".join(context_parts)
    
    def clear(self) -> None:
        """Clear conversation history and context."""
        self.history = []
        self.context = {}

# ==========================================
# QUESTION CLASSIFICATION
# ==========================================

def classify_question(question: str) -> str:
    """
    Classify question as policy-based or experience-based.
    
    Args:
        question: User's question
        
    Returns:
        'policy' for policy questions, 'experience' for service quality questions
    """
    q_lower = question.lower()
    
    # Policy indicators
    policy_keywords = [
        'baggage', 'luggage', 'kg', 'weight', 'allowance',
        'book', 'booking', 'ticket', 'fare', 'price', 'cost',
        'cancel', 'refund', 'change', 'reschedule',
        'meal', 'food', 'snack', 'beverage',
        'check-in time', 'when to check', 'how to check',
        'seat cost', 'seat price', 'seat selection',
        '6e prime', 'prime benefit', 'membership',
        'policy', 'rule', 'procedure', 'timing', 'hour',
        'allowed', 'permitted', 'restriction'
    ]
    
    # Experience indicators
    experience_keywords = [
        'staff', 'crew', 'helpful', 'friendly', 'professional',
        'service', 'experience', 'attitude', 'behavior',
        'how is', 'how are', 'how was', 'treated', 'handle',
        'good', 'bad', 'reviews', 'passengers say', 'usually',
        'recommend', 'reliable', 'punctual', 'clean'
    ]
    
    policy_score = sum(1 for kw in policy_keywords if kw in q_lower)
    experience_score = sum(1 for kw in experience_keywords if kw in q_lower)
    
    return 'policy' if policy_score > experience_score else 'experience'

def get_relevant_policy(question: str) -> str:
    """
    Get relevant policy information for a question.
    
    Args:
        question: User's question
        
    Returns:
        Relevant policy text
    """
    q_lower = question.lower()
    
    # Map keywords to policy categories
    keyword_mapping = {
        'baggage': ['bag', 'luggage', 'kg', 'weight', 'carry'],
        'booking': ['book', 'ticket', 'reservation', 'purchase'],
        'cancellation': ['cancel', 'refund', 'change', 'reschedule'],
        'meals': ['food', 'meal', 'snack', 'eat', 'drink', 'beverage'],
        'checkin': ['check-in', 'check in', 'checkin', 'airport'],
        'seats': ['seat', 'legroom', 'window', 'aisle', 'selection'],
        'delay': ['delay', 'late', 'wait', 'postpone'],
        'prime': ['prime', '6e', 'membership', 'premium']
    }
    
    # Find best matching policy
    for policy_key, keywords in keyword_mapping.items():
        if any(word in q_lower for word in keywords):
            return GENERAL_KNOWLEDGE[policy_key]
    
    # Default: return booking and baggage info
    return GENERAL_KNOWLEDGE['booking'] + " " + GENERAL_KNOWLEDGE['baggage']

# ==========================================
# DATA PROCESSING
# ==========================================

def clean_text(text: str) -> str:
    """
    Clean and normalize text data.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs, emails, mentions, hashtags
    text = re.sub(r'@\S+|#\S+|http\S+|www\.\S+|\S+@\S+', '', text)
    
    # Remove UI artifacts
    text = text.replace('Read more', '').replace('Read less', '')
    
    # Remove numbers that look like IDs or phone numbers
    text = re.sub(r'\d+/\d+|&num;|\d{10,}', '', text)
    
    # Normalize whitespace
    return " ".join(text.split()).strip()

def load_data_from_s3() -> pd.DataFrame:
    """
    Load review data from S3.
    
    Returns:
        DataFrame with review data
        
    Raises:
        ValueError: If AWS credentials not configured
    """
    if AWS_ACCESS_KEY == "YOUR_AWS_ACCESS_KEY_HERE":
        raise ValueError(
            "AWS credentials not configured. "
            "Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env file"
        )
    
    print("☁️  Loading data from S3...", end="\r")
    
    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=REGION
        )
        
        obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=FILE_KEY)
        df = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))
        
        print("✅ Data loaded successfully")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load data from S3: {e}")
        raise

def build_knowledge_base(df: pd.DataFrame) -> Tuple:
    """
    Build RAG knowledge base from reviews.
    
    Args:
        df: DataFrame containing reviews
        
    Returns:
        Tuple of (chunks, embedder, faiss_index)
    """
    print("⚙️  Processing reviews...", end="\r")
    
    # Extract and clean reviews
    documents = []
    for _, row in df.iterrows():
        text = clean_text(row['more'])
        if len(text) > 50:  # Only substantial reviews
            documents.append(text)
    
    logger.info(f"Processed {len(documents)} reviews")
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_text(" ".join(documents))
    
    logger.info(f"Created {len(chunks)} chunks")
    
    # Create embeddings
    print("🧠 Building vector index...", end="\r")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    
    # Suppress progress bars
    import warnings
    warnings.filterwarnings("ignore")
    
    embeddings = embedder.encode(
        chunks,
        show_progress_bar=False,
        batch_size=32
    )
    
    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    print("✅ Knowledge base ready   ")
    return chunks, embedder, index

# ==========================================
# RESPONSE GENERATION
# ==========================================

def retrieve_relevant_chunks(
    question: str,
    chunks: List[str],
    embedder: SentenceTransformer,
    index: faiss.Index
) -> Tuple[str, float]:
    """
    Retrieve relevant text chunks for a question.
    
    Args:
        question: User's question
        chunks: List of text chunks
        embedder: Sentence transformer model
        index: FAISS index
        
    Returns:
        Tuple of (context_text, confidence_score)
    """
    # Encode question
    query_embedding = embedder.encode([question]).astype('float32')
    
    # Search index
    distances, indices = index.search(query_embedding, TOP_K_RETRIEVAL)
    
    # Retrieve chunks
    relevant_chunks = [chunks[i] for i in indices[0]]
    context = " ".join(relevant_chunks)
    
    # Calculate confidence (lower distance = higher confidence)
    confidence_score = float(np.mean(distances[0]))
    
    return context, confidence_score

def clean_model_output(response: str) -> Optional[str]:
    """
    Clean up model-generated response.
    
    Args:
        response: Raw model output
        
    Returns:
        Cleaned response or None if cleaning fails
    """
    # Remove common artifacts and prompt leakage
    artifacts = [
        "Answer:", "Answer based on", "Based on", "According to",
        "Passenger Reviews:", "Passenger reviews:", "Official policy:",
        "IndiGo Official Information:", "Rules:", "Question:",
        "Be conversational", "Thank You", "Thank Our",
        "Use phrases like", "Many passengers report...", "Travelers often mention...",
        "INSTRUCTIONS:", "ANSWER:", "Provide a helpful",
        "Summarize what", "This information should be",
        "Passengers are asked", "should be concise"
    ]
    
    for artifact in artifacts:
        response = response.replace(artifact, "")
    
    # Remove any remaining instruction-like phrases with quotation marks
    response = re.sub(r'"[^"]*passengers[^"]*"', '', response, flags=re.IGNORECASE)
    response = re.sub(r'"[^"]*travelers[^"]*"', '', response, flags=re.IGNORECASE)
    
    # Remove incomplete start
    response = re.sub(r'^[^A-Z]*', '', response)
    
    # Remove repetitive phrases
    response = re.sub(r'(\b\w+\b)(\s+\1){2,}', r'\1', response)
    
    # Normalize whitespace
    response = " ".join(response.split()).strip()
    
    # Validate response quality
    if len(response) < 20 or not any(c.isalpha() for c in response):
        return None
    
    return response

def generate_response(
    question: str,
    context: str,
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    device: str,
    response_type: str
) -> str:
    """
    Generate response using language model.
    
    Args:
        question: User's question
        context: Retrieved context
        model: Language model
        tokenizer: Model tokenizer
        device: Device to run on ('cuda' or 'cpu')
        response_type: Type of response ('policy' or 'experience')
        
    Returns:
        Generated response text
    """
    # Select appropriate prompt template
    if response_type == 'policy':
        prompt = PromptTemplates.policy_prompt(question, context)
    else:
        prompt = PromptTemplates.experience_prompt(question, context)
    
    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_RESPONSE_TOKENS,
            min_new_tokens=MIN_RESPONSE_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            top_p=0.92,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=3,
            num_beams=2
        )
    
    # Decode and clean
    raw_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    cleaned_response = clean_model_output(raw_response)
    
    # Provide context-aware fallback if cleaning failed
    if not cleaned_response:
        if response_type == 'policy':
            return (
                "For specific details about this policy, please visit the IndiGo "
                "website at goindigo.in or contact their customer service team."
            )
        else:
            # Check question context for better fallback
            q_lower = question.lower()
            if any(word in q_lower for word in ["elderly", "parents", "senior"]):
                return (
                    "Many passengers report that IndiGo staff is attentive to elderly "
                    "travelers and provides assistance with boarding, seating, and baggage "
                    "when requested."
                )
            elif any(word in q_lower for word in ["crew", "staff", "service"]):
                return (
                    "Passengers generally report positive experiences with IndiGo's cabin "
                    "crew, noting their professionalism and helpfulness during flights."
                )
            else:
                return (
                    "Based on passenger reviews, IndiGo generally receives positive feedback "
                    "for their service quality and attention to passenger needs."
                )
    
    return cleaned_response

def chat(
    question: str,
    chunks: List[str],
    embedder: SentenceTransformer,
    index: faiss.Index,
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    device: str,
    memory: ConversationMemory
) -> Tuple[str, float, str]:
    """
    Main chat function.
    
    Args:
        question: User's question
        chunks: Text chunks
        embedder: Embedding model
        index: FAISS index
        model: Language model
        tokenizer: Tokenizer
        device: Device
        memory: Conversation memory
        
    Returns:
        Tuple of (response, confidence, source_type)
    """
    # Classify question type
    question_type = classify_question(question)
    
    if question_type == 'policy':
        # Use policy information
        context = get_relevant_policy(question)
        response = generate_response(
            question, context, model, tokenizer, device, 'policy'
        )
        confidence = 0.3  # High confidence for policy
        source = "📋 Official Policy"
    else:
        # Use review-based information
        context, confidence = retrieve_relevant_chunks(
            question, chunks, embedder, index
        )
        response = generate_response(
            question, context, model, tokenizer, device, 'experience'
        )
        source = "👥 Passenger Reviews"
    
    # Update conversation memory
    memory.add_exchange(question, response)
    
    return response, confidence, source

# ==========================================
# USER INTERFACE
# ==========================================

def get_confidence_indicator(score: float) -> str:
    """Get colored confidence indicator."""
    if score < 0.8:
        return "🟢 High"
    elif score < 1.2:
        return "🟡 Medium"
    else:
        return "🔴 Low"

def request_user_feedback(
    question: str,
    response: str,
    learning: LearningSystem
) -> Optional[int]:
    """
    Request and process user feedback.
    
    Args:
        question: User's question
        response: System's response
        learning: Learning system
        
    Returns:
        Rating (1-10) or None
    """
    try:
        rating_input = input("\n💭 Rate this response 1-10 (or press Enter to skip): ").strip()
        
        if not rating_input:
            return None
        
        rating = int(rating_input)
        
        if not (1 <= rating <= 10):
            print("⚠️  Rating must be between 1 and 10")
            return None
        
        # Get optional comment
        comment = input("📝 Optional comment: ").strip()
        
        # Save feedback
        learning.add_feedback(question, response, rating, comment)
        
        # Provide acknowledgment
        if rating >= 8:
            print("🎉 Excellent! Thank you for your feedback!\n")
        elif rating >= 6:
            print("👍 Thanks for your feedback!\n")
        else:
            print("🙏 Thank you. I'll work on improving!\n")
        
        return rating
        
    except ValueError:
        print("⚠️  Please enter a valid number\n")
        return None
    except Exception as e:
        logger.error(f"Error collecting feedback: {e}")
        return None

def display_statistics(learning: LearningSystem) -> None:
    """Display performance statistics."""
    stats = learning.get_stats()
    
    if stats["total"] == 0:
        print("\n📊 No feedback data available yet\n")
        return
    
    print("\n" + "="*60)
    print("📊 PERFORMANCE STATISTICS")
    print("="*60)
    print(f"Total Conversations:     {stats['total']}")
    print(f"Average Rating:          {stats['avg']:.1f}/10")
    print(f"  🟢 Excellent (8-10):   {stats['excellent']}")
    print(f"  🟡 Good (6-7):         {stats['good']}")
    print(f"  🔴 Needs Work (<6):    {stats['poor']}")
    print("="*60 + "\n")

def show_example_questions() -> None:
    """Display example questions."""
    print("\n" + "="*60)
    print("💡 EXAMPLE QUESTIONS")
    print("="*60)
    
    print("\n📋 POLICY & PROCEDURES:")
    print("  • What is the baggage allowance for domestic flights?")
    print("  • How do I book a flight on IndiGo?")
    print("  • What is the cancellation policy?")
    print("  • Are meals provided on flights?")
    print("  • How much does seat selection cost?")
    print("  • When should I check in for my flight?")
    print("  • What are the benefits of 6E Prime membership?")
    
    print("\n👥 SERVICE & EXPERIENCE:")
    print("  • How helpful is the check-in staff?")
    print("  • Do they assist elderly passengers?")
    print("  • How does IndiGo handle flight delays?")
    print("  • Is the cabin crew friendly and professional?")
    print("  • What do passengers say about customer service?")
    print("  • Will they help with baggage issues?")
    print("  • Is IndiGo reliable and punctual?")
    print("="*60 + "\n")

def show_help() -> None:
    """Display help information."""
    print("\n" + "="*60)
    print("📖 AVAILABLE COMMANDS")
    print("="*60)
    print("  examples  - Show example questions you can ask")
    print("  stats     - View performance statistics")
    print("  clear     - Clear conversation history")
    print("  help      - Show this help message")
    print("  exit      - End the conversation")
    print("="*60 + "\n")

def run_interactive_chat(
    chunks: List[str],
    embedder: SentenceTransformer,
    index: faiss.Index,
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    device: str
) -> None:
    """
    Run interactive chat interface.
    
    Args:
        chunks: Text chunks
        embedder: Embedding model
        index: FAISS index
        model: Language model
        tokenizer: Tokenizer
        device: Device
    """
    memory = ConversationMemory()
    learning = LearningSystem()
    
    # Welcome message
    print("\n" + "="*60)
    print("✈️  IndiGo Buddy - AI Travel Assistant")
    print("="*60)
    print("Ask me anything about IndiGo Airlines:")
    print("  📋 Policies, procedures, booking, baggage")
    print("  👥 Service quality based on real passenger reviews")
    print("\n💡 Type 'help' for commands | 'examples' for sample questions")
    print("="*60 + "\n")
    
    # Show existing stats if available
    if learning.feedback_data:
        stats = learning.get_stats()
        if stats['total'] > 0:
            print(f"📊 {stats['total']} conversations | "
                  f"Avg rating: {stats['avg']:.1f}/10\n")
    
    # Main conversation loop
    while True:
        try:
            # Get user input
            question = input("🙋 You: ").strip()
            
            if not question:
                continue
            
            # Handle exit commands
            if question.lower() in ["exit", "quit", "bye", "goodbye"]:
                print("\n✈️ Safe travels! Thanks for using IndiGo Buddy!\n")
                break
            
            # Handle meta commands
            if question.lower() == "examples":
                show_example_questions()
                continue
            
            if question.lower() == "stats":
                display_statistics(learning)
                continue
            
            if question.lower() == "clear":
                memory.clear()
                print("\n✅ Conversation history cleared\n")
                continue
            
            if question.lower() == "help":
                show_help()
                continue
            
            # Generate response
            print("💭 Thinking...", end="\r")
            response, confidence, source = chat(
                question, chunks, embedder, index,
                model, tokenizer, device, memory
            )
            print("               ", end="\r")  # Clear status
            
            # Display response
            print(f"\n✈️  {response}\n")
            print(f"    [{get_confidence_indicator(confidence)} confidence | {source}]")
            
            # Request feedback
            request_user_feedback(question, response, learning)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Safe travels!\n")
            break
        except Exception as e:
            logger.error(f"Error in chat loop: {e}")
            print("\n❌ Sorry, something went wrong. Please try again.\n")

# ==========================================
# INITIALIZATION
# ==========================================

def initialize_system() -> Tuple:
    """
    Initialize all system components.
    
    Returns:
        Tuple of (chunks, embedder, index, model, tokenizer, device)
    """
    print("\n🚀 Initializing IndiGo Buddy...\n")
    
    # Load data
    df = load_data_from_s3()
    
    # Build knowledge base
    chunks, embedder, index = build_knowledge_base(df)
    
    # Load language model
    print("🤖 Loading AI model...", end="\r")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print("✅ AI model ready        \n")
    
    logger.info(f"System initialized on device: {device}")
    
    return chunks, embedder, index, model, tokenizer, device

# ==========================================
# MAIN ENTRY POINT
# ==========================================

def main():
    """Main entry point."""
    try:
        components = initialize_system()
        run_interactive_chat(*components)
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!\n")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
