import json
import logging
import time
import uuid

from flask import Blueprint, current_app, jsonify, render_template, request

from .extensions import cache, limiter
from .llm.cache import clear_project_cache, compute_llm_cache_key, get_cached_projects, set_cached_projects
from .services.pdf import extract_resume_text
from .services.projects import (
    CuratedProjectLibrary,
    build_prompt,
    generate_fallback_response,
    infer_role_level,
    validate_and_fix_projects,
)
from .services.roadmap import build_learning_roadmap
from .services.scoring import calculate_weighted_score, extract_jd_priority_keywords
from .services.skills import fuzzy_skill_match

logger = logging.getLogger(__name__)
bp = Blueprint("main", __name__)


@bp.route('/')
def home():
    return render_template('index.html')


@bp.route('/analyze', methods=['POST'])
@limiter.limit("10 per minute")
def analyze():
    request_id = str(uuid.uuid4())
    request.request_id = request_id
    start_time = time.time()
    nlp_processor = current_app.extensions["nlp_processor"]

    logger.info(f"[{request_id}] Resume analysis request received.")

    try:
        file = request.files.get('resume')
        pasted_text = request.form.get('resume_text', '').strip()
        job_desc = request.form.get('job_desc', '').strip()

        if not job_desc or len(job_desc) < 50:
            return jsonify({
                "error": "Job description is required and must be at least 50 characters.",
                "request_id": request_id
            }), 400

        # Validate that either a file or pasted text is provided
        has_file = file is not None and file.filename != ''
        has_text = bool(pasted_text)

        if not has_file and not has_text:
            return jsonify({
                "error": "Please provide your resume by uploading a file (PDF/DOCX) or pasting the text.",
                "request_id": request_id
            }), 400

        # Extract resume text safely
        try:
            resume_text = extract_resume_text(uploaded_file=file if has_file else None, pasted_text=pasted_text if has_text else None)
        except (ValueError, RuntimeError) as e:
            logger.warning(f"[{request_id}] Validation error during extraction: {e}")
            return jsonify({"error": str(e), "request_id": request_id}), 400

        logger.info(f"[{request_id}] Extracted {len(resume_text)} characters from resume input.")

        # Sequential NLP extraction
        t_nlp_start = time.perf_counter()
        try:
            resume_skills = nlp_processor.extract_skills_nlp(resume_text)
            jd_skills = nlp_processor.extract_skills_nlp(job_desc)
            text_similarity = nlp_processor.calculate_semantic_similarity(resume_text, job_desc)
            elapsed_nlp_ms = (time.perf_counter() - t_nlp_start) * 1000
            logger.info(f"[{request_id}] NLP extraction in {elapsed_nlp_ms:.2f}ms: {len(resume_skills)} resume skills, {len(jd_skills)} JD skills.")
        except Exception as e:
            logger.error(f"[{request_id}] NLP processing error: {e}", exc_info=True)
            resume_skills, jd_skills, text_similarity = set(), set(), 0.0

        match_result = fuzzy_skill_match(resume_skills, jd_skills)
        matched_skills_list = match_result['exact'] + match_result['partial']
        missing_skills_list = match_result['missing']

        # Scoring v2: Weighted required vs preferred matching
        score_result = calculate_weighted_score(
            resume_skills,
            jd_skills,
            job_desc,
            skill_categories=getattr(nlp_processor, "tech_skills_patterns", None)
        )
        base_score = score_result["score"]
        semantic_boost = min(text_similarity * 0.05, 5)
        final_match_score = min(round(base_score + semantic_boost), 100)

        logger.info(
            f"[{request_id}] Match: {len(matched_skills_list)}/{len(jd_skills)} | "
            f"Required: {score_result['required']['matched']}/{score_result['required']['total']} | "
            f"Score: {final_match_score}%"
        )

        # Multi-Tier Project Recommendations: Cache -> Library -> LLM Chain (Groq/Gemini/OpenRouter) -> Template
        level = infer_role_level(job_desc)
        cache_key = compute_llm_cache_key(missing_skills_list, level=level, prompt_version="v3")
        cached_res = get_cached_projects(cache, cache_key)

        provider_used = "template"
        projects = []

        if cached_res:
            projects = cached_res.get("projects", [])
            provider_used = "cache"
            logger.info(f"[{request_id}] Projects served from 7-day cache.")
        elif current_app.config.get("TESTING"):
            projects = generate_fallback_response(missing_skills_list, matched_skills_list)
            provider_used = "template"
            logger.info(f"[{request_id}] Testing mode active: using deterministic project response.")
        else:
            # Primary: Call LLM Chain directly (Groq -> OpenRouter -> Gemini)
            jd_priority_keywords = extract_jd_priority_keywords(jd_skills, job_desc, top_k=10)
            prompt = build_prompt(missing_skills_list, matched_skills_list, job_desc, jd_keywords=jd_priority_keywords)
            llm_chain = current_app.extensions.get("llm_chain")

            if llm_chain:
                try:
                    chain_res = llm_chain.execute(
                        prompt,
                        missing_skills=missing_skills_list,
                        matched_skills=matched_skills_list,
                        jd_keywords=jd_priority_keywords,
                        providers=["groq", "openrouter", "gemini"]
                    )
                    provider_used = chain_res.get("provider", "template")

                    if provider_used in ("template", "offline-curated"):
                        # Remote LLMs failed; fall back to offline curated library
                        logger.info(f"[{request_id}] LLM providers unavailable or exhausted. Falling back to offline curated library.")
                        projects = generate_fallback_response(missing_skills_list, matched_skills_list)
                        provider_used = "offline_library" if projects else "template"
                    else:
                        projects = chain_res["data"].get("ProjectIdeas") or chain_res["data"].get("Projects", [])
                        logger.info(f"[{request_id}] Projects generated via LLMChain ({provider_used}).")
                except Exception as e:
                    logger.warning(f"[{request_id}] LLMChain execution failed: {e}. Falling back to offline curated library.")
                    projects = generate_fallback_response(missing_skills_list, matched_skills_list)
                    provider_used = "offline_library"
            else:
                logger.info(f"[{request_id}] No LLMChain configured. Using offline curated library.")
                projects = generate_fallback_response(missing_skills_list, matched_skills_list)
                provider_used = "offline_library"

            # Store generated projects in cache for 7 days
            if projects:
                set_cached_projects(cache, cache_key, {"projects": projects, "provider": provider_used})

        projects = validate_and_fix_projects(projects, missing_skills_list, matched_skills_list)

        # Build prioritized learning roadmap with topological prerequisite ordering
        roadmap = build_learning_roadmap(missing_skills_list)

        data = {
            "JD Match": str(final_match_score),
            "MatchedSkills": matched_skills_list,
            "MissingSkills": missing_skills_list,
            "ExactMatches": match_result['exact'],
            "PartialMatches": match_result['partial'],
            "SkillMatchScore": round(base_score),
            "SemanticScore": round(text_similarity),
            "ScoreBreakdown": score_result,
            "ScoreVersion": 2,
            "ProjectIdeas": projects,
            "ProviderUsed": provider_used,
            "Roadmap": roadmap,
            "ProcessingTime": round(time.time() - start_time, 2),
            "NLPEnhanced": True,
            "AIUsed": provider_used not in ("template", "library", "cache"),
            "request_id": request_id
        }

        logger.info(f"[{request_id}] Request completed in {data['ProcessingTime']}s")
        return jsonify(data)

    except Exception as e:
        from werkzeug.exceptions import HTTPException
        if isinstance(e, HTTPException):
            raise e
        elapsed = round(time.time() - start_time, 2)
        logger.error(f"[{request_id}] Unhandled error after {elapsed}s: {e}", exc_info=True)
        return jsonify({
            "error": "An unexpected server error occurred. Please try again.",
            "request_id": request_id,
            "ProcessingTime": elapsed
        }), 500

@bp.route('/api/cache/clear', methods=['POST', 'GET'])
def clear_cache_endpoint():
    cache_inst = current_app.extensions.get("cache")
    success = clear_project_cache(cache_inst)
    return jsonify({
        "status": "success",
        "message": "Project cache has been emptied."
    })


@bp.route('/health', methods=['GET'])
def health_check():
    llm_chain = current_app.extensions.get("llm_chain")
    providers = [p["name"] for p in getattr(llm_chain, "providers", [])] if llm_chain else []
    return jsonify({
        "status": "healthy",
        "nlp_loaded": current_app.extensions.get("nlp_processor") is not None,
        "llm_providers": providers,
        "groq_configured": bool(current_app.config.get("GROQ_API_KEY")),
        "gemini_configured": bool(current_app.config.get("GOOGLE_API_KEY")),
    })
