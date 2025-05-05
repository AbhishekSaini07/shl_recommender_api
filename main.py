# main.py
import json
import re
import numpy as np
import os
import time
import logging
from fastapi import FastAPI, HTTPException, Depends # Added Depends for potential future use
from pydantic import BaseModel, Field
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from dotenv import load_dotenv

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
ASSESSMENTS_FILE = "assessments_180.json" # Make sure this matches your filename
SBERT_MODEL_NAME = 'all-mpnet-base-v2'
INITIAL_RETRIEVE_COUNT = 25 # How many candidates for Gemini to re-rank
FINAL_RECOMMENDATION_COUNT = 10

# --- Load Environment Variables & Configure Gemini ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_configured = False
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_configured = True
        logger.info("Gemini API configured successfully.")
    except Exception as e:
        logger.error(f"Failed to configure Gemini API: {e}", exc_info=True)
else:
    logger.warning("GEMINI_API_KEY not found in environment variables. Re-ranking will be skipped.")

# --- Global Variables (to hold loaded models/data) ---
_assessments_data: List[dict] = []
_sbert_model: Optional[SentenceTransformer] = None
_app_ready: bool = False # Flag to indicate successful startup

# --- Helper Functions (Copied/Adapted from Colab) ---

def extract_minutes(duration_str):
    if not isinstance(duration_str, str): return None
    match = re.search(r'(\d+)\s*minutes?', duration_str, re.IGNORECASE)
    return int(match.group(1)) if match else None

def parse_duration_constraint(query_text):
    max_duration = None
    match = re.search(r'\b(less than|under|at most|max|maximum|within|no more than)\s+(\d+)\s*(minutes?|mins?)\b', query_text, re.IGNORECASE)
    if match: return int(match.group(2))
    match = re.search(r'\b(\d+)\s*(minutes?|mins?)\b', query_text, re.IGNORECASE)
    if match: return int(match.group(1)) # Less strict match
    return None

def generate_llm_relevance_prompt(query_text, candidate_assessments):
    max_candidates_in_prompt = INITIAL_RETRIEVE_COUNT
    candidates_to_prompt = candidate_assessments[:max_candidates_in_prompt]
    prompt = f"You are an expert assistant helping hiring managers find relevant pre-employment assessments.\n"
    prompt += f"Based *only* on the provided assessment names and descriptions, please re-rank the following candidate assessments according to their relevance to the user's query.\n\n"
    prompt += f"User Query: \"{query_text}\"\n\n"
    prompt += "Candidate Assessments:\n----------------------\n"
    for i, assessment in enumerate(candidates_to_prompt):
        prompt += f"{i+1}. Name: {assessment.get('assignment', 'N/A')}\n"
        desc = assessment.get('description', 'N/A')
        prompt += f"   Description: {desc[:300]}{'...' if len(desc) > 300 else ''}\n"
    prompt += "----------------------\n\n"
    prompt += f"Instructions: Return ONLY the re-ranked list of the {len(candidates_to_prompt)} assessment names provided above, one name per line, starting with the most relevant assessment based on the query and the descriptions. Do not include numbers, introductory text, explanations, or assessments not listed above."
    return prompt

def call_gemini_api(prompt):
    if not gemini_configured: return None
    try:
        # Consider making model configurable (e.g., 'gemini-1.5-flash')
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(max_output_tokens=1024, temperature=0.1)
            # safety_settings=... # Optional
        )
        if response.parts: return response.text
        else:
            logger.warning(f"Gemini response blocked or empty. Reason: {response.prompt_feedback}")
            return None
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}", exc_info=True)
        return None

def parse_llm_ranking(response_text, candidate_names):
    if not response_text: return []
    ranked_names = []
    lines = response_text.strip().split('\n')
    candidate_names_set = set(candidate_names)
    processed_candidates = set() # Track added candidate names

    for line in lines:
        name = line.strip().lstrip('0123456789.- ').rstrip()
        if not name: continue

        matched_candidate = None
        if name in candidate_names_set:
             matched_candidate = name
        else:
            # Attempt partial matching
            for candidate_name in candidate_names_set:
                 if name.lower() in candidate_name.lower() or candidate_name.lower() in name.lower():
                    matched_candidate = candidate_name
                    logger.info(f"Matched LLM output '{name}' to candidate '{candidate_name}'")
                    break # Take first partial match
            if not matched_candidate:
                logger.warning(f"LLM returned name not matched to candidates: '{name}'")
                continue # Skip if no match found

        # Add the matched candidate name if not already added
        if matched_candidate and matched_candidate not in processed_candidates:
            ranked_names.append(matched_candidate)
            processed_candidates.add(matched_candidate)

    # Add any original candidates missed by the LLM/parsing to the end
    missing_names = [name for name in candidate_names if name not in processed_candidates]
    if missing_names:
        logger.info(f"Appending {len(missing_names)} candidates potentially missed by LLM.")
        ranked_names.extend(missing_names)

    return ranked_names

# --- Main Recommendation Logic ---
def get_recommendations_gemini_rerank(query_text: str, assessments_data: List[dict], sbert_model: SentenceTransformer) -> List[dict]:
    """Internal function to generate recommendations."""
    global _app_ready # Access global flag

    if not _app_ready or not assessments_data or not sbert_model:
        logger.error("Recommendation function called before app is ready.")
        return [] # Should not happen if endpoint checks are correct

    start_time = time.time()
    logger.info(f"Starting recommendation for query: '{query_text[:50]}...'")

    # 1. Initial Candidate Retrieval (SentenceBERT)
    try:
        query_embedding = sbert_model.encode([query_text])[0]
        all_assessment_embeddings = np.array([item['embedding'] for item in assessments_data])
        similarities = cosine_similarity(query_embedding.reshape(1, -1), all_assessment_embeddings)[0]
    except Exception as e:
        logger.error(f"Error during SBERT embedding/similarity: {e}", exc_info=True)
        return []

    num_candidates = min(INITIAL_RETRIEVE_COUNT, len(assessments_data))
    if num_candidates <= 0: return []
    semantic_ranked_indices = np.argsort(similarities)[::-1][:num_candidates]
    candidate_assessments = [assessments_data[i] for i in semantic_ranked_indices]
    candidate_names = [item['assignment'] for item in candidate_assessments]
    retrieval_time = time.time()
    logger.info(f"Initial retrieval ({len(candidate_assessments)} candidates) took {retrieval_time - start_time:.2f}s")

    # 2. LLM Re-ranking Section
    reranked_candidates = candidate_assessments # Default to semantic order
    if gemini_configured and candidate_assessments:
        prompt = generate_llm_relevance_prompt(query_text, candidate_assessments)
        logger.info(f"Sending prompt for {len(candidate_assessments)} candidates to Gemini...")
        llm_response_text = call_gemini_api(prompt)
        llm_call_time = time.time()
        logger.info(f"Gemini API call took {llm_call_time - retrieval_time:.2f}s")

        if llm_response_text:
            logger.info("Parsing LLM response...")
            ranked_names_from_llm = parse_llm_ranking(llm_response_text, candidate_names)
            assessment_map = {item['assignment']: item for item in candidate_assessments}
            temp_reranked_list = []
            processed_names = set()
            for name in ranked_names_from_llm:
                if name in assessment_map and name not in processed_names:
                    temp_reranked_list.append(assessment_map[name])
                    processed_names.add(name)
            if temp_reranked_list:
                 reranked_candidates = temp_reranked_list
                 logger.info(f"Successfully re-ranked {len(reranked_candidates)} candidates using LLM.")
            else:
                 logger.warning("LLM parsing resulted in empty list. Falling back to semantic ranking.")
        else:
            logger.warning("Gemini did not return a valid response. Falling back to semantic ranking.")
    else:
        logger.info("Skipping Gemini re-ranking (API key not configured or no candidates). Using semantic ranking.")
    parsing_time = time.time()

    # 3. Apply Duration Filter
    max_duration_filter = parse_duration_constraint(query_text)
    final_recommendations = []
    for assessment in reranked_candidates:
        passes_filter = True
        duration_val = assessment.get('duration_minutes')
        if max_duration_filter is not None and (duration_val is None or duration_val > max_duration_filter):
            passes_filter = False
        if passes_filter:
            final_recommendations.append(assessment)
        if len(final_recommendations) >= FINAL_RECOMMENDATION_COUNT:
            break
    filtering_time = time.time()
    logger.info(f"Filtering & final selection took {filtering_time - parsing_time:.2f}s")
    total_time = time.time() - start_time
    logger.info(f"Total processing time: {total_time:.2f}s. Returning {len(final_recommendations)} recommendations.")

    return final_recommendations

# --- Pydantic Models for API ---
class QueryRequest(BaseModel):
    query_text: str = Field(..., min_length=3, description="Natural language query or job description text.")

class RecommendationItem(BaseModel):
    assignment: str
    url: str
    remote_support: str
    adaptive_support: str
    duration: str
    test_type: List[str]

class RecommendationResponse(BaseModel):
    recommendations: List[RecommendationItem]

# --- Formatting Function for API Output ---
def format_recommendations_for_api(raw_recommendations: List[dict]) -> dict:
    """Formats the recommendation list to match the API specification."""
    output_recommendations = []
    required_keys = ['assignment', 'url', 'remote_support', 'adaptive_support', 'duration', 'test_type']
    for item in raw_recommendations:
        formatted_item = {}
        for key in required_keys:
            value = item.get(key)
            if value is None: value = [] if key == 'test_type' else ('#' if key == 'url' else 'N/A')
            formatted_item[key] = value
        try:
             # Validate with Pydantic model before adding (optional but good)
             pydantic_item = RecommendationItem(**formatted_item)
             output_recommendations.append(pydantic_item.dict())
        except Exception as e:
             logger.warning(f"Skipping item due to formatting/validation error: {item.get('assignment')}. Error: {e}")
    return {"recommendations": output_recommendations}

# --- FastAPI Application ---
app = FastAPI(
    title="SHL Assessment Recommender API",
    description="Recommends SHL assessments based on natural language queries or job descriptions, using SBERT retrieval and optional Gemini re-ranking.",
    version="1.1.0" # Incremented version
)

# --- Startup Event Handler ---
@app.on_event("startup")
async def startup_event():
    """Load models and data when the application starts."""
    global _assessments_data, _sbert_model, _app_ready
    logger.info("Application startup: Loading models and data...")
    start_time = time.time()
    try:
        # 1. Load Assessment Data
        logger.info(f"Loading assessment data from '{ASSESSMENTS_FILE}'...")
        if not os.path.exists(ASSESSMENTS_FILE):
             raise FileNotFoundError(f"Assessment data file not found: {ASSESSMENTS_FILE}")
        with open(ASSESSMENTS_FILE, 'r', encoding='utf-8') as f: # Added encoding
            assessments_data_raw_list = json.load(f)
        logger.info(f"Loaded {len(assessments_data_raw_list)} raw assessment entries.")

        # 2. Load SBERT Model
        logger.info(f"Loading Sentence Transformer model '{SBERT_MODEL_NAME}'...")
        _sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
        logger.info("SBERT Model loaded successfully.")

        # 3. Preprocess Data & Generate Embeddings
        logger.info("Preprocessing data and generating embeddings...")
        texts_to_embed = []
        valid_assessments_temp = []
        for item in assessments_data_raw_list:
            item['duration_minutes'] = extract_minutes(item.get('duration'))
            if not all(k in item for k in ('assignment', 'description')) or item['duration_minutes'] is None:
                logger.warning(f"Skipping assessment due to missing critical data: {item.get('assignment', 'Unknown')}")
                continue
            text = f"Assessment: {item['assignment']}. Type: {', '.join(item.get('test_type', []))}. Description: {item['description']}"
            texts_to_embed.append(text)
            valid_assessments_temp.append(item)

        if not valid_assessments_temp:
             raise ValueError("No valid assessments found after preprocessing.")

        assessment_embeddings = _sbert_model.encode(texts_to_embed, show_progress_bar=False) # Turn off bar in prod
        for i, item in enumerate(valid_assessments_temp):
            item['embedding'] = assessment_embeddings[i]

        _assessments_data = valid_assessments_temp # Assign globally
        _app_ready = True # Set flag only after successful loading
        end_time = time.time()
        logger.info(f"Application ready. Loaded {len(_assessments_data)} assessments. Startup took {end_time - start_time:.2f} seconds.")

    except FileNotFoundError as e:
         logger.error(f"Startup failed: {e}", exc_info=True)
    except ValueError as e:
         logger.error(f"Startup failed: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error during startup: {e}", exc_info=True)

# --- API Endpoints ---
@app.get("/health", tags=["Status"])
async def health_check():
    """Provides a status check indicating if the API is ready."""
    if _app_ready:
        # Check if Gemini is intended to be used and if it's configured
        gemini_status = "configured" if gemini_configured else ("not_configured" if GEMINI_API_KEY else "key_missing")
        return {
            "status": "ok",
            "message": "Service is ready.",
            "sbert_model_loaded": _sbert_model is not None,
            "assessments_loaded": len(_assessments_data),
            "gemini_status": gemini_status
        }
    else:
        # Use 503 Service Unavailable if startup failed
        raise HTTPException(status_code=503, detail="Service Unavailable: Application not ready.")

@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
async def recommend_assessments_endpoint(request: QueryRequest):
    """
    Accepts a job description or Natural language query and returns
    recommended relevant assessments (At most 10, minimum 1) based on the input.
    """
    if not _app_ready or not _sbert_model or not _assessments_data:
        logger.warning("Recommendation request received before application is ready.")
        raise HTTPException(status_code=503, detail="Service Unavailable: Application not ready.")
    # If Gemini is required for the primary logic and not configured, maybe raise 503?
    # if not gemini_configured:
    #     logger.warning("Recommendation request failed: Gemini API Key not configured for re-ranking.")
    #     raise HTTPException(status_code=503, detail="Service Configuration Incomplete: LLM API key missing.")

    try:
        # === Call the Main Logic Function ===
        raw_recs = get_recommendations_gemini_rerank(
            query_text=request.query_text,
            assessments_data=_assessments_data, # Use globally loaded data
            sbert_model=_sbert_model           # Use globally loaded model
        )
        # ===================================

        formatted_response = format_recommendations_for_api(raw_recs)
        return formatted_response

    except Exception as e:
        logger.error(f"Error processing recommendation request for query '{request.query_text[:50]}...': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error: Could not process recommendations.")

@app.get("/", tags=["Status"], include_in_schema=False) # Hide from docs by default
async def read_root():
    return {"message": "Welcome to the SHL Assessment Recommender API. Visit /docs for details."}

# --- Optional: Add code to run with uvicorn directly for simple testing ---
# import uvicorn
# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
# Note: Using uvicorn command line is generally preferred over this.