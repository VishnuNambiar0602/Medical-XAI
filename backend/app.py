from flask import Flask, request, jsonify
from flask_cors import CORS
from model import initialize_model
import logging
import os

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model on startup
try:
    model = initialize_model()
    logger.info("Medical XAI model initialized successfully")
except Exception as e:
    logger.error(f"Error initializing model: {e}")
    model = None


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    }), 200


@app.route('/diagnose', methods=['POST'])
def diagnose():
    """
    Diagnose possible diseases based on symptoms
    Expected JSON: {"symptoms": "symptom1, symptom2, ..."}
    """
    if model is None:
        return jsonify({"error": "Model not initialized"}), 500
    
    try:
        data = request.get_json()
        symptoms = data.get('symptoms', '').strip()
        
        if not symptoms:
            return jsonify({"error": "Please provide symptoms"}), 400
        
        # Get diagnosis
        result = model.diagnose(symptoms)
        
        if result.get("error"):
            return jsonify(result), 400
        
        # Format response
        response = {
            "input_symptoms": result.get("input_symptoms", []),
            "total_matches": result.get("total_matched", 0),
            "diseases": []
        }
        
        for disease in result.get("possible_diseases", []):
            response["diseases"].append({
                "name": disease.get("disease_name"),
                "disease_id": disease.get("disease_id"),
                "confidence": round(disease.get("confidence_score", 0) * 100, 1),
                "explanation": disease.get("explanation"),
                "matched_symptoms": disease.get("matched_symptoms", []),
                "all_symptoms": disease.get("all_symptoms", [])
            })
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in diagnose endpoint: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/recommend', methods=['POST'])
def get_recommendation():
    """
    Get recommendation based on symptoms
    Expected JSON: {"symptoms": "symptom1, symptom2, ..."}
    """
    if model is None:
        return jsonify({"error": "Model not initialized"}), 500
    
    try:
        data = request.get_json()
        symptoms = data.get('symptoms', '').strip()
        
        if not symptoms:
            return jsonify({"error": "Please provide symptoms"}), 400
        
        recommendation = model.get_recommendation(symptoms)
        return jsonify(recommendation), 200
        
    except Exception as e:
        logger.error(f"Error in recommend endpoint: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/explain/<disease_id>', methods=['GET'])
def explain_disease(disease_id):
    """
    Get detailed explanation for a disease
    """
    if model is None:
        return jsonify({"error": "Model not initialized"}), 500
    
    try:
        explanation = model.explain_diagnosis(disease_id)
        
        if explanation.get("error"):
            return jsonify(explanation), 404
        
        return jsonify(explanation), 200
        
    except Exception as e:
        logger.error(f"Error in explain endpoint: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/symptoms', methods=['GET'])
def get_all_symptoms():
    """Get list of all recognized symptoms"""
    if model is None:
        return jsonify({"error": "Model not initialized"}), 500
    
    try:
        symptom_keywords = model.knowledge_base.get("symptom_keywords", {})
        symptoms = list(symptom_keywords.keys())
        return jsonify({
            "total_symptoms": len(symptoms),
            "symptoms": sorted(symptoms)
        }), 200
    except Exception as e:
        logger.error(f"Error in symptoms endpoint: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/diseases', methods=['GET'])
def get_all_diseases():
    """Get list of all diseases in knowledge base"""
    if model is None:
        return jsonify({"error": "Model not initialized"}), 500
    
    try:
        diseases = model.knowledge_base.get("diseases", {})
        disease_list = [
            {
                "id": disease_id,
                "name": disease_info.get("name", disease_id),
                "symptom_count": len(disease_info.get("symptoms", []))
            }
            for disease_id, disease_info in diseases.items()
        ]
        return jsonify({
            "total_diseases": len(disease_list),
            "diseases": sorted(disease_list, key=lambda x: x["name"])
        }), 200
    except Exception as e:
        logger.error(f"Error in diseases endpoint: {e}")
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
