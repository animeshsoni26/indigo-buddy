# Prompt Engineering Guide

This document explains the prompt engineering strategy used in IndiGo Buddy and provides best practices for customization.

## Table of Contents

- [Overview](#overview)
- [Prompt Architecture](#prompt-architecture)
- [Template Design Principles](#template-design-principles)
- [Prompt Types](#prompt-types)
- [Optimization Techniques](#optimization-techniques)
- [Customization Guide](#customization-guide)
- [Testing & Evaluation](#testing--evaluation)

## Overview

IndiGo Buddy uses carefully engineered prompts to generate high-quality, contextually appropriate responses. The prompt system is designed to:

- **Minimize hallucination** by grounding responses in provided context
- **Control output format** through explicit instructions
- **Maintain consistency** across different question types
- **Enable easy customization** for different use cases

## Prompt Architecture

### Three-Part Structure

Every prompt follows this structure:

```
1. ROLE DEFINITION
   ↓
2. CONTEXT PROVISION
   ↓
3. INSTRUCTION SET
   ↓
4. OUTPUT MARKER
```

This structure ensures the model understands:
- **Who it is** (role)
- **What it knows** (context)
- **How to respond** (instructions)
- **Where to start** (output marker)

## Template Design Principles

### 1. Clear Role Assignment

```python
"You are a helpful IndiGo Airlines customer service assistant."
```

**Why it works:**
- Establishes clear identity
- Sets behavioral expectations
- Frames response perspective

### 2. Explicit Context Labeling

```python
POLICY INFORMATION:
{policy_info}

CUSTOMER QUESTION:
{question}
```

**Why it works:**
- Clear separation of information types
- Easy for model to parse
- Reduces confusion

### 3. Detailed Instructions

```python
INSTRUCTIONS:
- Provide a clear, accurate answer based on the policy information
- Use 2-3 complete sentences
- Be friendly and professional
- Include specific details (prices, timings, limits) when relevant
```

**Why it works:**
- Explicit behavioral guidelines
- Specific output requirements
- Quality control criteria

### 4. Output Marker

```python
ANSWER:
```

**Why it works:**
- Signals where to begin generation
- Reduces preamble/meta-text
- Improves response cleanliness

## Prompt Types

### Policy Prompt Template

**Use Case:** Questions about official rules, procedures, pricing

```python
def policy_prompt(question: str, policy_info: str) -> str:
    return f"""Answer this customer question about IndiGo Airlines using the official policy information below. Be clear and specific in 2-3 sentences.

Policy information:
{policy_info}

Question: {question}

Answer based on the policy:"""
```

**Design Rationale:**
- **Minimal instructions**: Reduces prompt leakage risk
- **Natural language**: Avoids formal INSTRUCTION blocks
- **Clear task framing**: Model understands what to do without explicit rules
- **Direct output**: No "ANSWER:" markers that might appear in response

**Example:**

```
Input: "What's the baggage allowance?"

Output: "IndiGo allows 7kg cabin baggage and 15kg check-in 
baggage for domestic flights. Additional baggage costs ₹350-550 
per kg, with online pre-purchase being cheaper than airport rates."
```

### Experience Prompt Template

**Use Case:** Questions about service quality, passenger experiences

```python
def experience_prompt(question: str, reviews: str) -> str:
    return f"""Answer this customer question about IndiGo Airlines based on the passenger reviews provided below. Summarize what multiple passengers have experienced in 2-3 natural sentences.

Passenger reviews:
{reviews}

Question: {question}

Provide a helpful answer based only on these reviews:"""
```

**Design Rationale:**
- **Conversational framing**: "Summarize what passengers have experienced" instead of bullet-point instructions
- **Embedded guidance**: Length requirement integrated into main instruction
- **No example phrases**: Avoids "Use phrases like..." which can leak into output
- **Simple structure**: Reduces cognitive load on model

**Example:**

```
Input: "Is the cabin crew helpful?"

Output: "Many passengers report that IndiGo's cabin crew is 
professional and attentive. Travelers frequently mention the staff's 
willingness to assist with requests and their friendly demeanor 
during flights."
```

### Hybrid Prompt Template

**Use Case:** Complex questions requiring both policy and experience

```python
def hybrid_prompt(question: str, policy_info: str, reviews: str) -> str:
    return f"""Answer this customer question about IndiGo Airlines using both the official policy and passenger experiences below. Give a complete answer in 2-4 sentences.

Official policy:
{policy_info}

Passenger experiences:
{reviews}

Question: {question}

Answer combining policy and experiences:"""
```

**Design Rationale:**
- **Unified instruction**: Single sentence tells model what to do
- **Natural flow**: Model decides how to blend sources
- **Flexible length**: 2-4 sentences for complex topics
- **Clear labels**: Simple section headers without formal structure

**Example:**

```
Input: "Do they help elderly passengers with boarding?"

Output: "IndiGo provides wheelchair assistance and priority 
boarding for elderly passengers upon request at no extra charge. 
Many travelers report that the staff is particularly attentive 
to elderly passengers, often helping with carry-on luggage and 
ensuring comfortable seating."
```

## Optimization Techniques

### 1. Length Control

**Problem:** Responses too long or too short

**Solution:**
```python
# In generation parameters
max_new_tokens=150
min_new_tokens=40

# In prompt
"Use 2-3 complete sentences"  # Explicit length guidance
```

### 2. Repetition Prevention

**Problem:** Model repeats phrases or gets stuck in loops

**Solution:**
```python
# In generation parameters
repetition_penalty=2.5
no_repeat_ngram_size=3  # Increased from 2
num_beams=2  # Added beam search

# In post-processing
response = re.sub(r'(\b\w+\b)(\s+\1){2,}', r'\1', response)
```

### 3. Prompt Leakage Prevention

**Problem:** Instructions or prompt structure appears in model output

**Root Causes:**
- Explicit instruction lists (bullets, numbered steps)
- Phrases with quotation marks ("Use phrases like...")
- Formal section headers (INSTRUCTIONS:, ANSWER:)
- Example phrases that model copies literally

**Solution - Simplified Prompts:**
```python
# ❌ BAD - Prone to leakage
"""INSTRUCTIONS:
- Use phrases like "Many passengers report..."
- Be concise and helpful
ANSWER:"""

# ✅ GOOD - Natural language
"""Summarize what passengers have experienced in 2-3 sentences.

Provide a helpful answer:"""
```

**Solution - Enhanced Cleaning:**
```python
def clean_model_output(response: str) -> Optional[str]:
    # Remove instruction phrases that leaked
    artifacts = [
        "Use phrases like", "Many passengers report...",
        "Travelers often mention...", "This information should be",
        "Passengers are asked", "Be conversational"
    ]
    
    for artifact in artifacts:
        response = response.replace(artifact, "")
    
    # Remove quoted instruction examples
    response = re.sub(r'"[^"]*passengers[^"]*"', '', response, flags=re.IGNORECASE)
    
    return response
```

**Best Practices:**
1. **Avoid bullet lists in prompts** - Use natural sentences
2. **No example phrases** - Don't show what to say
3. **Integrated instructions** - Weave guidance into main text
4. **Simple structure** - Fewer headers and markers
5. **Post-processing** - Always clean output aggressively

### 4. Artifact Removal

**Problem:** Prompt instructions leak into output

**Solution:**
```python
artifacts = [
    "Answer:", "Based on", "According to", 
    "Passenger Reviews:", "INSTRUCTIONS:", "Question:"
]

for artifact in artifacts:
    response = response.replace(artifact, "")
```

### 4. Temperature Tuning

**Problem:** Responses too generic or too random

**Solution:**
```python
temperature=0.4  # Lower for factual, higher for creative

# 0.1-0.3: Very factual, minimal variation
# 0.4-0.6: Balanced (recommended)
# 0.7-1.0: More creative, higher variation
```

### 5. Context Window Management

**Problem:** Too much context confuses the model

**Solution:**
```python
# Limit context length
max_context_tokens = 400

# Prioritize most relevant chunks
top_k = 3  # Only use top 3 most relevant passages
```

## Customization Guide

### For Different Airlines

**Modify:**
1. Role definition: `"You are a [Airline] customer service assistant"`
2. Knowledge base: Update `GENERAL_KNOWLEDGE` dictionary
3. Attribution: Update source labels

### For Different Languages

**Approach:**
```python
def policy_prompt_hindi(question: str, policy_info: str) -> str:
    return f"""आप IndiGo Airlines के सहायक ग्राहक सेवा प्रतिनिधि हैं।

नीति जानकारी:
{policy_info}

ग्राहक का प्रश्न:
{question}

निर्देश:
- स्पष्ट और सटीक उत्तर दें
- 2-3 वाक्यों में जवाब दें
- विशिष्ट विवरण शामिल करें

उत्तर:"""
```

### For Different Domains (Hotels, Railways, etc.)

**Template:**
```python
def domain_prompt(question: str, context: str, domain: str) -> str:
    return f"""You are a helpful {domain} customer service assistant.

INFORMATION:
{context}

CUSTOMER QUESTION:
{question}

INSTRUCTIONS:
- [Domain-specific instructions]
- Use 2-3 complete sentences
- Be helpful and professional

ANSWER:"""
```

### For Different Tone

**Formal:**
```python
"Please provide a professional response using complete sentences."
```

**Casual:**
```python
"Give a friendly, conversational answer in 2-3 sentences."
```

**Empathetic:**
```python
"Respond with understanding and empathy. Acknowledge the customer's concern."
```

## Testing & Evaluation

### Prompt Testing Checklist

- [ ] **Accuracy**: Does it use provided information correctly?
- [ ] **Completeness**: Does it address all parts of the question?
- [ ] **Conciseness**: Is it appropriately brief (2-4 sentences)?
- [ ] **Relevance**: Does it stay on topic?
- [ ] **Tone**: Is it professional and helpful?
- [ ] **Attribution**: Does it cite sources appropriately?
- [ ] **Hallucination**: Does it avoid making up information?

### A/B Testing Framework

```python
def test_prompts(questions, prompt_v1, prompt_v2):
    results = {"v1": [], "v2": []}
    
    for question in questions:
        response_v1 = generate(question, prompt_v1)
        response_v2 = generate(question, prompt_v2)
        
        # Collect user ratings
        rating_v1 = get_user_rating(response_v1)
        rating_v2 = get_user_rating(response_v2)
        
        results["v1"].append(rating_v1)
        results["v2"].append(rating_v2)
    
    # Compare averages
    return {
        "v1_avg": np.mean(results["v1"]),
        "v2_avg": np.mean(results["v2"])
    }
```

### Evaluation Metrics

1. **Response Quality (1-10 scale)**
   - Accuracy
   - Helpfulness
   - Clarity

2. **Technical Metrics**
   - Average response length
   - Confidence scores
   - Processing time

3. **User Satisfaction**
   - Direct ratings
   - Conversation completion rate
   - Follow-up question rate

## Best Practices

### DO:
✅ Use clear, explicit instructions
✅ Provide structured context
✅ Include length guidelines
✅ Specify desired tone
✅ Add anti-hallucination instructions
✅ Test with real user questions
✅ Iterate based on feedback

### DON'T:
❌ Use vague instructions
❌ Mix different types of context
❌ Assume model understands implicit requirements
❌ Use overly complex language
❌ Include contradictory instructions
❌ Forget to handle edge cases

## Advanced Techniques

### Chain-of-Thought Prompting

```python
INSTRUCTIONS:
- First, identify the key aspects of the question
- Then, locate relevant information in the context
- Finally, formulate a clear 2-3 sentence answer
```

### Few-Shot Examples

```python
EXAMPLE 1:
Question: "What's the cancellation fee?"
Answer: "Cancellation fees vary by fare type..."

EXAMPLE 2:
Question: "Can I change my flight?"
Answer: "Flight changes are allowed up to..."

NOW ANSWER:
Question: {question}
Answer:
```

### Dynamic Instruction Adaptation

```python
if question_complexity == "high":
    instructions += "Take your time and provide detailed explanation."
elif passenger_sentiment == "frustrated":
    instructions += "Show empathy and acknowledge concerns."
```

## Conclusion

Effective prompt engineering is crucial for chatbot quality. The templates provided here are optimized for IndiGo Buddy but can be adapted for various use cases. Remember to:

1. Test thoroughly with real users
2. Collect feedback systematically
3. Iterate based on data
4. Monitor for drift over time
5. Update prompts as the model or domain changes

For questions or contributions, please open an issue on GitHub.

---

**Last Updated:** February 2025
**Version:** 2.0.0
