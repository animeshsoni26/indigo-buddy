# IndiGo Buddy - AI-Powered Travel Assistant ✈️

An intelligent customer service chatbot for IndiGo Airlines that combines official policy information with insights from real passenger reviews using RAG (Retrieval-Augmented Generation) architecture.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)

## 🌟 Features

- **Dual-Mode Intelligence**: Answers both policy-based and experience-based questions
- **RAG Architecture**: Uses semantic search to retrieve relevant information
- **Context Awareness**: Maintains conversation history and user context
- **Learning System**: Collects and analyzes user feedback for continuous improvement
- **Performance Analytics**: Track satisfaction metrics and response quality
- **Industry-Ready**: Professional prompts optimized for production use

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Architecture](#-architecture)
- [Prompt Engineering](#-prompt-engineering)
- [API Reference](#-api-reference)
- [Examples](#-examples)
- [Contributing](#-contributing)
- [License](#-license)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- AWS account with S3 access
- 4GB+ RAM recommended
- (Optional) CUDA-capable GPU for faster processing

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/indigo-buddy.git
   cd indigo-buddy
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your AWS credentials
   ```

## ⚙️ Configuration

Create a `.env` file in the project root:

```env
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_REGION=ap-south-1
S3_BUCKET_NAME=your-bucket-name
S3_FILE_KEY=indigo_reviews.csv

# Model Configuration (Optional - defaults provided)
MODEL_ID=google/flan-t5-base
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Generation Parameters (Optional)
MAX_RESPONSE_TOKENS=150
TEMPERATURE=0.4
```

### Data Format

Your S3 CSV file should contain a `more` column with passenger reviews:

```csv
more
"Great service! The staff was very helpful..."
"Flight was on time. Clean cabin..."
...
```

## 🎯 Usage

### Basic Usage

```bash
python indigo_buddy.py
```

### Interactive Commands

Once running, you can use these commands:

- **examples** - View sample questions
- **stats** - Display performance statistics
- **clear** - Clear conversation history
- **help** - Show help information
- **exit** - End the conversation

### Example Interaction

```
🙋 You: What's the baggage allowance?

✈️  IndiGo allows 7kg cabin baggage and 15kg check-in baggage for 
    domestic flights. Additional baggage costs ₹350-550 per kg, with 
    online pre-purchase being cheaper than airport rates.

    [🟢 High confidence | 📋 Official Policy]

💭 Rate this response 1-10 (or press Enter to skip): 9
🎉 Excellent! Thank you for your feedback!
```

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                     │
│  (Interactive CLI with feedback collection)                  │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                   Question Classification                    │
│  (Policy-based vs Experience-based routing)                  │
└─────────────┬───────────────────────────────┬───────────────┘
              │                               │
    ┌─────────▼─────────┐         ┌──────────▼──────────┐
    │  Policy Knowledge │         │   RAG System        │
    │      Base         │         │  (FAISS + Embeddings)│
    └─────────┬─────────┘         └──────────┬──────────┘
              │                               │
              └───────────┬───────────────────┘
                          │
              ┌───────────▼──────────┐
              │  LLM (FLAN-T5)       │
              │  + Prompt Templates  │
              └───────────┬──────────┘
                          │
              ┌───────────▼──────────┐
              │   Response Cleaning  │
              │   & Quality Control  │
              └───────────┬──────────┘
                          │
              ┌───────────▼──────────┐
              │  Learning System     │
              │  (Feedback Storage)  │
              └──────────────────────┘
```

### Key Technologies

- **Language Model**: Google FLAN-T5-Base
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Framework**: LangChain for text splitting
- **Cloud Storage**: AWS S3

## 🎨 Prompt Engineering

### Prompt Template Design

The system uses three specialized prompt templates:

#### 1. Policy Prompts
```python
"""You are a helpful IndiGo Airlines customer service assistant. 
Answer the customer's question using the official policy information provided.

POLICY INFORMATION:
{policy_info}

CUSTOMER QUESTION:
{question}

INSTRUCTIONS:
- Provide a clear, accurate answer based on the policy information
- Use 2-3 complete sentences
- Include specific details (prices, timings, limits) when relevant
"""
```

#### 2. Experience Prompts
```python
"""You are a helpful IndiGo Airlines assistant. 
Answer the customer's question based on actual passenger experiences.

PASSENGER REVIEWS:
{reviews}

CUSTOMER QUESTION:
{question}

INSTRUCTIONS:
- Summarize what passengers have experienced
- Be balanced and honest
- Use phrases like "Many passengers report..."
"""
```

#### 3. Hybrid Prompts
Combines both policy and experience information for comprehensive answers.

### Prompt Optimization Features

- **Clear Role Definition**: Establishes assistant identity
- **Structured Format**: Separates context, question, and instructions
- **Explicit Guidelines**: Detailed instructions for response style
- **Length Control**: Specifies desired response length
- **Tone Guidance**: Maintains professional, helpful demeanor
- **Attribution Phrases**: Guides proper sourcing of information

## 📚 API Reference

### Core Functions

#### `chat(question, chunks, embedder, index, model, tokenizer, device, memory)`

Main chat function that processes user questions.

**Parameters:**
- `question` (str): User's question
- `chunks` (List[str]): Pre-processed text chunks
- `embedder` (SentenceTransformer): Embedding model
- `index` (faiss.Index): FAISS index
- `model` (AutoModelForSeq2SeqLM): Language model
- `tokenizer` (AutoTokenizer): Model tokenizer
- `device` (str): Computing device ('cuda' or 'cpu')
- `memory` (ConversationMemory): Conversation context

**Returns:**
- `response` (str): Generated response
- `confidence` (float): Confidence score
- `source` (str): Information source type

#### `classify_question(question)`

Classifies questions as policy or experience-based.

**Parameters:**
- `question` (str): User's question

**Returns:**
- `str`: 'policy' or 'experience'

#### `get_relevant_policy(question)`

Retrieves relevant policy information.

**Parameters:**
- `question` (str): User's question

**Returns:**
- `str`: Relevant policy text

### Classes

#### `LearningSystem`

Manages user feedback and performance tracking.

**Methods:**
- `add_feedback(question, response, rating, comment)`: Store feedback
- `get_stats()`: Get performance statistics
- `_load_feedback()`: Load feedback from file
- `_save_feedback()`: Save feedback to file

#### `ConversationMemory`

Maintains conversation context and history.

**Methods:**
- `add_exchange(question, response)`: Add Q&A to history
- `get_context_string()`: Get formatted context
- `clear()`: Clear history
- `_extract_context(question)`: Extract contextual clues

#### `PromptTemplates`

Static prompt template generators.

**Methods:**
- `policy_prompt(question, policy_info)`: Generate policy prompt
- `experience_prompt(question, reviews)`: Generate experience prompt
- `hybrid_prompt(question, policy_info, reviews)`: Generate hybrid prompt

## 💡 Examples

### Policy Questions

```python
Q: "What's the baggage allowance?"
A: "IndiGo allows 7kg cabin baggage and 15kg check-in baggage 
    for domestic flights. Additional baggage costs ₹350-550 per kg."
Source: 📋 Official Policy
```

### Experience Questions

```python
Q: "Is the cabin crew helpful?"
A: "Many passengers report that IndiGo's cabin crew is professional 
    and helpful. Travelers often mention the staff's friendly attitude 
    and willingness to assist with passenger needs."
Source: 👥 Passenger Reviews
```

### Context-Aware Questions

```python
Q: "I'm traveling with elderly parents. Will they get assistance?"
A: "Passengers frequently mention that IndiGo staff is attentive to 
    elderly travelers. The crew typically provides boarding assistance 
    and helps with seating arrangements."
Source: 👥 Passenger Reviews
Context: Passenger type - elderly
```

## 📊 Performance Metrics

The system tracks:

- **Total Conversations**: Number of interactions
- **Average Rating**: Mean user satisfaction (1-10)
- **Rating Distribution**: 
  - Excellent (8-10)
  - Good (6-7)
  - Needs Work (<6)

View with: `stats` command in chat

## 🛠️ Development

### Project Structure

```
indigo-buddy/
├── indigo_buddy.py        # Main application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment template
├── .gitignore           # Git ignore rules
├── README.md            # This file
├── LICENSE              # MIT License
├── docs/                # Additional documentation
│   ├── API.md          # API documentation
│   └── PROMPTS.md      # Prompt engineering guide
└── tests/              # Unit tests (future)
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Document all public functions and classes
- Maximum line length: 88 characters (Black formatter)

### Testing

```bash
# Run tests (when available)
pytest tests/

# Check code style
flake8 indigo_buddy.py

# Format code
black indigo_buddy.py
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines

- Write clear commit messages
- Add tests for new features
- Update documentation
- Follow existing code style
- Ensure all tests pass

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** for transformer models
- **Facebook Research** for FAISS
- **Sentence Transformers** for embedding models
- **IndiGo Airlines** for inspiration

## 📧 Contact


Project Link: [https://github.com/animeshsoni26/indigo-buddy](https://github.com/animeshsoni26/indigo-buddy)

## 🗺️ Roadmap

- [ ] Web interface (Flask/Streamlit)
- [ ] Multi-language support
- [ ] Voice interface
- [ ] Integration with booking systems
- [ ] Advanced analytics dashboard
- [ ] Fine-tuned domain-specific model
- [ ] Real-time feedback analysis
- [ ] A/B testing framework

---

Made with ❤️ for better customer service
