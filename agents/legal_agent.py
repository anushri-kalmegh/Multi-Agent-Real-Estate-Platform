import os
import sys
import json
import re
from agents.config import RAG_DIR

def answer_legal_query(question):
    """
    Retrieves the most relevant legal text chunk from rag/legal_chunks.json
    using simple keyword matching.

    Parameters:
    - question (str): The legal question queried by the user.

    Returns:
    - dict: A dictionary containing the answer text, source document, and chunk ID.
    """
    try:
        if not question or not isinstance(question, str):
            raise ValueError("Question must be a non-empty string.")

        chunks_path = RAG_DIR / "legal_chunks.json"
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"Legal chunks database not found at {chunks_path}. Please run build_legal_index.py first.")

        # 1. Load the chunks
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        if not chunks:
            raise ValueError("Legal chunks database is empty.")

        # 2. Extract keywords from the question (ignoring common stop words)
        stop_words = {
            "what", "is", "a", "the", "in", "to", "of", "and", "for", "on",
            "with", "at", "by", "from", "an", "about", "are", "how", "does",
            "do", "can", "why", "who", "where", "which"
        }
        words = re.findall(r'\b\w+\b', question.lower())
        keywords = [w for w in words if w not in stop_words]

        # Fallback to using all words if all were stop words
        if not keywords:
            keywords = words

        # 3. Find the chunk with the highest keyword matching score using composite rule-based weights
        best_chunk = None
        best_score = -1
        best_matches_count = 0

        # Deduplicate keywords to compute unique match count
        unique_keywords = list(set(keywords))
        total_keywords = len(unique_keywords)

        # Check if query is a definition query
        q_lower = question.lower()
        is_definition_query = any(phrase in q_lower for phrase in ["what is", "define", "meaning of"])

        # Helper to generate flexible regex pattern allowing optional spaces between letters
        def get_flex_pattern(word):
            return r"\s*".join(re.escape(c) for c in word)

        # Common real estate acronym expansions to boost retrieval quality for definition queries
        acronym_expansions = {
            "rera": [
                r"real\s*estat\s*e\s*\(?\s*regulation\s*and\s*dev\s*elopment\s*\)?\s*act",
                r"real\s*estat\s*e\s*regulatory\s*authority"
            ],
            "maharera": [
                r"maharashtra\s*real\s*estat\s*e\s*regulatory\s*authority"
            ]
        }

        for chunk in chunks:
            text_lower = chunk.get("text", "").lower()

            # A. Keyword frequency (1.0 weight per occurrence, using flexible spacing pattern)
            freq_score = 0
            for keyword in keywords:
                pattern = get_flex_pattern(keyword)
                freq_score += len(re.findall(pattern, text_lower))

            # B. Keyword presence bonus (5.0 weight per present keyword)
            presence_score = 0
            for keyword in unique_keywords:
                pattern = get_flex_pattern(keyword)
                if re.search(pattern, text_lower):
                    presence_score += 5.0

            # C. Bonus if keywords appear in first 200 characters (10.0 weight per keyword)
            first_200 = text_lower[:200]
            first_200_bonus = 0
            for keyword in unique_keywords:
                pattern = get_flex_pattern(keyword)
                if re.search(pattern, first_200):
                    first_200_bonus += 10.0

            # D. Definition query prioritization (20.0 weight per matching term with word boundaries)
            definition_bonus = 0
            if is_definition_query:
                def_patterns = [r"\bmeans\b", r"\bdefined as\b", r"\bact\b", r"\bdefinition\b"]
                for pattern in def_patterns:
                    # Clean boundary regex characters to generate flexible pattern
                    raw_pattern = pattern.replace(r"\b", "")
                    flex_pattern = r"\b" + r"\s*".join(re.escape(c) for c in raw_pattern) + r"\b"
                    if re.search(flex_pattern, text_lower):
                        definition_bonus += 20.0

                # Acronym expansions bonus for definition queries
                for keyword in unique_keywords:
                    if keyword in acronym_expansions:
                        for exp_pattern in acronym_expansions[keyword]:
                            if re.search(exp_pattern, text_lower):
                                definition_bonus += 40.0

            # Calculate final composite score
            chunk_score = freq_score + presence_score + first_200_bonus + definition_bonus

            if chunk_score > best_score:
                best_score = chunk_score
                best_chunk = chunk
                # Count how many unique query keywords are present in the chunk
                best_matches_count = sum(1 for kw in unique_keywords if re.search(get_flex_pattern(kw), text_lower))

        # 4. Format and return output
        if best_chunk is None or best_score == 0:
            return {
                "answer": "No highly relevant legal text could be identified for your query.",
                "source_document": "N/A",
                "chunk_id": -1,
                "confidence_score": 0.0
            }

        # Compute confidence score
        confidence_score = 0.0
        if total_keywords > 0:
            confidence_score = (best_matches_count / total_keywords) * 100
        confidence_score = round(confidence_score, 2)

        # Rule-based summarization: clean whitespace, extract 3-5 meaningful sentences, limit to 400-500 chars
        raw_text = best_chunk.get("text", "")
        cleaned_text = re.sub(r'\s+', ' ', raw_text).strip()

        # Split text into sentences using simple regex punctuation bounds
        sentence_ends = re.compile(r'(?<=[.!?])\s+')
        raw_sentences = sentence_ends.split(cleaned_text)

        meaningful_sentences = []
        for s in raw_sentences:
            s = s.strip()
            # Define meaningful sentence: >15 characters with alphanumeric content
            if len(s) > 15 and re.search(r'[a-zA-Z]', s):
                meaningful_sentences.append(s)

        # Target 3-5 sentences
        if len(meaningful_sentences) < 3:
            selected_sentences = meaningful_sentences
        else:
            selected_sentences = meaningful_sentences[:5]

        summary = " ".join(selected_sentences)

        # Dynamically scale sentence count down to fit within the 400-500 characters range
        for limit in [4, 3]:
            if len(summary) > 500 and len(meaningful_sentences) >= limit:
                selected_sentences = meaningful_sentences[:limit]
                summary = " ".join(selected_sentences)

        # Hard cap truncation at word boundary if still above 500 characters
        if len(summary) > 500:
            truncated = summary[:497]
            last_space = truncated.rfind(' ')
            if last_space > 350:
                summary = truncated[:last_space] + "..."
            else:
                summary = truncated + "..."

        return {
            "answer": summary,
            "source_document": best_chunk.get("source_file", "Unknown"),
            "chunk_id": best_chunk.get("chunk_id", -1),
            "confidence_score": confidence_score
        }

    except Exception as e:
        print(f"Error in answer_legal_query: {e}", file=sys.stderr)
        return {
            "answer": f"Error processing query: {str(e)}",
            "source_document": "N/A",
            "chunk_id": -1
        }

if __name__ == "__main__":
    print("Testing Legal Agent...")

    question = "What is RERA?"
    print(f"\nQuestion: {question}")

    result = answer_legal_query(question)

    print("\nResult:")
    print(json.dumps(result, indent=4))
