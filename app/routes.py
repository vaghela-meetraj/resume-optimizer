import concurrent.futures
import json
import time

from flask import Blueprint, current_app, jsonify, render_template, request

from .extensions import limiter
from .services.llm import GEMINI_MODELS, get_gemini_response_with_retry
from .services.pdf import extract_pdf_text
from .services.projects import build_prompt, generate_fallback_response, validate_and_fix_projects
from .services.skills import fuzzy_skill_match


bp = Blueprint("main", __name__)


@bp.route('/')
def home():
    return render_template('index.html')

@bp.route('/analyze', methods=['POST'])
@limiter.limit("10 per minute")
def analyze():
    nlp_processor = current_app.extensions["nlp_processor"]
    start_time = time.time()
    print("[*] Resume analysis request received.")

    try:
        if 'resume' not in request.files:
            return jsonify({"error": "No resume file uploaded."}), 400
        file = request.files['resume']
        job_desc = request.form.get('job_desc', '').strip()
        if not file or file.filename == '':
            return jsonify({"error": "No file selected."}), 400
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "Only PDF files are supported. Please upload a PDF."}), 400
        if not job_desc or len(job_desc) < 50:
            return jsonify({"error": "Job description is required and must be detailed."}), 400

        print("[*] Extracting PDF text...")
        try:
            resume_text = extract_pdf_text(file)
        except (ValueError, RuntimeError) as e:
            return jsonify({"error": str(e)}), 400

        print("[*] Running NLP analysis...")
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                f_resume = executor.submit(nlp_processor.extract_skills_nlp, resume_text)
                f_jd = executor.submit(nlp_processor.extract_skills_nlp, job_desc)
                f_sim = executor.submit(nlp_processor.calculate_semantic_similarity, resume_text, job_desc)
                resume_skills = f_resume.result(timeout=5)
                jd_skills = f_jd.result(timeout=5)
                text_similarity = f_sim.result(timeout=5)
        except Exception as e:
            print(f"[*] NLP error: {e}")
            resume_skills, jd_skills, text_similarity = set(), set(), 0.0

        match_result = fuzzy_skill_match(resume_skills, jd_skills)
        matched_skills_list = match_result['exact'] + match_result['partial']
        missing_skills_list = match_result['missing']
        
        if len(jd_skills) > 0:
            base_score = (len(matched_skills_list) / len(jd_skills)) * 100
        else:
            base_score = 0.0
            
        semantic_boost = min(text_similarity * 0.05, 5)
        final_match_score = min(round(base_score + semantic_boost), 95)
        
        print(f"[*] JD Skills: {len(jd_skills)} | Matched: {len(matched_skills_list)} | Missing: {len(missing_skills_list)}")
        print(f"[*] Base Skill Match: {base_score:.1f}% | Semantic Boost: {semantic_boost:.1f}% | Final: {final_match_score}%")

        print("[*] Building prompt...")
        prompt = build_prompt(missing_skills_list, matched_skills_list, job_desc)

        print("[*] Calling Gemini API...")
        ai_used = True
        try:
            json_string = get_gemini_response_with_retry(prompt, max_retries=2)
            project_data = json.loads(json_string)
            projects = project_data.get("ProjectIdeas", [])
            print("[*] Gemini analysis complete.")
        except Exception as e:
            print(f"[*] Gemini unavailable: {str(e)[:80]}...")
            print("[*] Falling back to intelligent NLP project generation...")
            projects = generate_fallback_response(missing_skills_list, matched_skills_list)
            ai_used = False

        projects = validate_and_fix_projects(projects, missing_skills_list, matched_skills_list)

        data = {
            "JD Match": str(final_match_score),
            "MatchedSkills": matched_skills_list,
            "MissingSkills": missing_skills_list,
            "ExactMatches": match_result['exact'],
            "PartialMatches": match_result['partial'],
            "SkillMatchScore": round(base_score),
            "SemanticScore": round(text_similarity),
            "ProjectIdeas": projects,
            "ProcessingTime": round(time.time() - start_time, 2),
            "NLPEnhanced": True,
            "AIUsed": ai_used
        }

        print(f"[*] Done in {data['ProcessingTime']}s | AI: {ai_used}")
        return jsonify(data)

    except Exception as e:
        elapsed = round(time.time() - start_time, 2)
        print(f"[*] Unhandled error after {elapsed}s: {e}")
        return jsonify({"error": "Server error. Please try again.", "detail": str(e), "ProcessingTime": elapsed}), 500

@bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "nlp_loaded": current_app.extensions.get("nlp_processor") is not None,
        "gemini_configured": bool(current_app.config.get("GOOGLE_API_KEY")),
        "primary_model": GEMINI_MODELS[0][0],
    })
