import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from difflib import SequenceMatcher
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MedicalXAIModel:
    """TinyLM-based Medical Explainable AI Model for symptom diagnosis"""
    
    def __init__(self, knowledge_base_path: str):
        """Initialize the medical model with knowledge base"""
        self.kb_path = knowledge_base_path
        self.knowledge_base = self.load_knowledge_base()
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
        self.disease_vectors = None
        self.initialization_complete = False
        self._initialize_model()
        
    def load_knowledge_base(self) -> Dict:
        """Load medical knowledge base from JSON file"""
        try:
            with open(self.kb_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            return {"diseases": {}, "symptom_keywords": {}}
    
    def _initialize_model(self):
        """Initialize vectorizer with disease symptoms"""
        diseases = self.knowledge_base.get("diseases", {})
        disease_texts = []
        
        for disease_id, disease_info in diseases.items():
            symptoms_str = " ".join(disease_info.get("symptoms", []))
            disease_texts.append(symptoms_str)
        
        if disease_texts:
            self.disease_vectors = self.vectorizer.fit_transform(disease_texts)
            self.initialization_complete = True
    
    def preprocess_symptoms(self, symptoms_input: str) -> List[str]:
        """Preprocess and normalize symptom input"""
        # Convert to lowercase and split by common delimiters
        symptoms_raw = symptoms_input.lower().split(',')
        processed = []
        
        symptom_keywords = self.knowledge_base.get("symptom_keywords", {})
        
        for symptom in symptoms_raw:
            symptom = symptom.strip()
            if symptom:
                # Try to match with known symptoms
                matched = False
                for key_symptom, keywords in symptom_keywords.items():
                    if any(kw in symptom for kw in keywords):
                        if key_symptom not in processed:
                            processed.append(key_symptom)
                        matched = True
                        break
                
                # If no match, keep the symptom as is (fuzzy matching)
                if not matched:
                    processed.append(symptom)
        
        return processed
    
    def _calculate_similarity(self, symptoms_input: str, disease_symptoms: List[str]) -> float:
        """Calculate similarity between input symptoms and disease symptoms"""
        input_str = symptoms_input.lower()
        disease_str = " ".join(disease_symptoms).lower()
        
        # TF-IDF similarity
        try:
            vectors = self.vectorizer.transform([input_str, disease_str])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        except:
            similarity = 0.0
        
        # Bonus for exact symptom matches
        matched_count = 0
        for symptom in disease_symptoms:
            if symptom.lower() in input_str:
                matched_count += 1
        
        if len(disease_symptoms) > 0:
            match_bonus = (matched_count / len(disease_symptoms)) * 0.3
            similarity = min(1.0, similarity + match_bonus)
        
        return similarity
    
    def diagnose(self, symptoms_input: str) -> Dict:
        """
        Diagnose possible diseases based on symptoms
        
        Args:
            symptoms_input: Comma-separated symptoms or free text
            
        Returns:
            Dictionary with diagnosis results and explanations
        """
        if not self.initialization_complete:
            return {"error": "Model not initialized", "diseases": []}
        
        processed_symptoms = self.preprocess_symptoms(symptoms_input)
        
        if not processed_symptoms:
            return {"error": "No valid symptoms provided", "diseases": []}
        
        diseases = self.knowledge_base.get("diseases", {})
        results = []
        
        # Score each disease
        for disease_id, disease_info in diseases.items():
            disease_symptoms = disease_info.get("symptoms", [])
            similarity_score = self._calculate_similarity(
                symptoms_input,
                disease_symptoms
            )
            
            # Count matching symptoms
            matched_symptoms = [s for s in processed_symptoms if s in disease_symptoms]
            match_ratio = len(matched_symptoms) / len(disease_symptoms) if disease_symptoms else 0
            
            # Combine scores
            combined_score = (similarity_score * 0.6 + match_ratio * 0.4)
            
            if combined_score > 0.1:  # Only include if there's meaningful similarity
                results.append({
                    "disease_id": disease_id,
                    "disease_name": disease_info.get("name", disease_id),
                    "confidence_score": float(combined_score),
                    "matched_symptoms": matched_symptoms,
                    "disease_symptoms": disease_symptoms,
                    "explanation": disease_info.get("explanation", "No explanation available"),
                    "all_symptoms": disease_info.get("symptoms", [])
                })
        
        # Sort by confidence score
        results.sort(key=lambda x: x["confidence_score"], reverse=True)
        
        # Return top 5 results
        top_results = results[:5]
        
        return {
            "input_symptoms": processed_symptoms,
            "possible_diseases": top_results,
            "total_matched": len(top_results)
        }
    
    def explain_diagnosis(self, disease_id: str) -> Dict:
        """Get detailed explanation for a specific disease"""
        diseases = self.knowledge_base.get("diseases", {})
        
        if disease_id not in diseases:
            return {"error": f"Disease {disease_id} not found"}
        
        disease = diseases[disease_id]
        
        return {
            "disease_name": disease.get("name", disease_id),
            "symptoms": disease.get("symptoms", []),
            "explanation": disease.get("explanation", ""),
            "disease_id": disease_id
        }
    
    def get_recommendation(self, symptoms_input: str) -> Dict:
        """Get diagnostic recommendation based on symptoms"""
        diagnosis = self.diagnose(symptoms_input)
        
        if diagnosis.get("error"):
            return {"recommendation": "Please provide valid symptoms", "urgency": "low"}
        
        if not diagnosis.get("possible_diseases"):
            return {
                "recommendation": "No matching diseases found. Please consult a healthcare provider.",
                "urgency": "normal"
            }
        
        top_disease = diagnosis["possible_diseases"][0]
        confidence = top_disease["confidence_score"]
        disease_name = top_disease["disease_name"]
        explanation = top_disease["explanation"]
        
        # Determine urgency based on symptoms
        urgent_keywords = ["fever", "chest pain", "difficulty breathing", "severe", "emergency"]
        symptoms_lower = symptoms_input.lower()
        is_urgent = any(keyword in symptoms_lower for keyword in urgent_keywords)
        
        urgency = "high" if is_urgent else ("medium" if confidence > 0.5 else "low")
        
        return {
            "top_disease": disease_name,
            "confidence": f"{confidence * 100:.1f}%",
            "explanation": explanation,
            "urgency": urgency,
            "matched_symptoms": top_disease["matched_symptoms"],
            "recommendation": f"Based on your symptoms, {disease_name} is likely. {explanation}"
        }


def initialize_model(kb_path: str = None) -> MedicalXAIModel:
    """Initialize and return the medical XAI model"""
    if kb_path is None:
        kb_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "medical_knowledge_base.json"
        )
    
    return MedicalXAIModel(kb_path)
