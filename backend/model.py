import json
import os
import numpy as np
from typing import Dict, List, Tuple, Any

class MedicalXAIModel:
    """TinyLM-based Medical Explainable AI Model for symptom diagnosis using a simple Neural Network"""
    
    def __init__(self, knowledge_base_path: str):
        """Initialize the medical model with knowledge base"""
        self.kb_path = knowledge_base_path
        self.knowledge_base = self.load_knowledge_base()

        # Model parameters
        self.input_size = 0
        self.hidden_size = 32
        self.output_size = 0
        self.learning_rate = 0.01
        self.epochs = 2000

        # Mappings
        self.symptom_to_idx = {}
        self.idx_to_symptom = {}
        self.disease_to_idx = {}
        self.idx_to_disease = {}

        # Weights
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

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
        """Initialize and train the Neural Network on knowledge base data"""
        diseases = self.knowledge_base.get("diseases", {})
        symptom_keywords = self.knowledge_base.get("symptom_keywords", {})
        
        # 1. Build Vocabulary (Symptoms)
        unique_symptoms = set()
        # Add all canonical symptoms from keywords
        unique_symptoms.update(symptom_keywords.keys())
        # Add any symptoms in diseases that might not be in keywords (just in case)
        for d in diseases.values():
            unique_symptoms.update(d.get("symptoms", []))

        self.symptom_to_idx = {s: i for i, s in enumerate(sorted(unique_symptoms))}
        self.idx_to_symptom = {i: s for s, i in self.symptom_to_idx.items()}
        self.input_size = len(unique_symptoms)
        
        # 2. Build Labels (Diseases)
        disease_ids = sorted(diseases.keys())
        self.disease_to_idx = {d: i for i, d in enumerate(disease_ids)}
        self.idx_to_disease = {i: d for d, i in self.disease_to_idx.items()}
        self.output_size = len(disease_ids)
        
        if self.input_size == 0 or self.output_size == 0:
            print("Warning: Knowledge base is empty or malformed.")
            return

        # 3. Prepare Training Data
        X = []
        y = []
        
        for d_id, d_info in diseases.items():
            # Input vector
            d_symptoms = d_info.get("symptoms", [])
            x_vec = np.zeros((1, self.input_size))
            for s in d_symptoms:
                if s in self.symptom_to_idx:
                    x_vec[0, self.symptom_to_idx[s]] = 1.0

            # Target vector (one-hot)
            y_vec = np.zeros((1, self.output_size))
            y_vec[0, self.disease_to_idx[d_id]] = 1.0

            X.append(x_vec)
            y.append(y_vec)

        if not X:
            return

        X = np.vstack(X)
        y = np.vstack(y)
        
        # 4. Initialize Weights (Xavier/Glorot initialization)
        np.random.seed(42) # For reproducibility
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2.0 / self.input_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2.0 / self.hidden_size)
        self.b2 = np.zeros((1, self.output_size))
        
        # 5. Train Model
        self._train(X, y)
        self.initialization_complete = True
        print(f"Model trained on {len(X)} diseases with {self.input_size} unique symptoms.")

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _relu(self, x):
        return np.maximum(0, x)

    def _softmax(self, x):
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def _train(self, X, y):
        """Train the neural network using backpropagation"""
        for epoch in range(self.epochs):
            # Forward Pass
            z1 = np.dot(X, self.W1) + self.b1
            a1 = self._relu(z1) # ReLU activation
            z2 = np.dot(a1, self.W2) + self.b2
            a2 = self._softmax(z2) # Output probabilities

            # Compute Loss (Cross-Entropy) - just for monitoring if needed
            # loss = -np.mean(np.sum(y * np.log(a2 + 1e-8), axis=1))

            # Backward Pass
            # Gradient of Loss w.r.t z2 (output error)
            # For Softmax + CrossEntropy, dL/dz2 = a2 - y
            dz2 = a2 - y

            # Gradients for W2, b2
            dW2 = np.dot(a1.T, dz2)
            db2 = np.sum(dz2, axis=0, keepdims=True)

            # Gradients for hidden layer
            da1 = np.dot(dz2, self.W2.T)
            dz1 = da1.copy()
            dz1[z1 <= 0] = 0 # Derivative of ReLU

            # Gradients for W1, b1
            dW1 = np.dot(X.T, dz1)
            db1 = np.sum(dz1, axis=0, keepdims=True)

            # Update Weights
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2

    def predict(self, x_vec):
        """Forward pass for inference"""
        z1 = np.dot(x_vec, self.W1) + self.b1
        a1 = self._relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self._softmax(z2)
        return a2[0]

    def preprocess_symptoms(self, symptoms_input: str) -> List[str]:
        """Preprocess input string to canonical symptoms"""
        symptoms_input = symptoms_input.lower()
        processed = set()
        
        symptom_keywords = self.knowledge_base.get("symptom_keywords", {})
        
        # Check for keywords in input
        for canonical_symptom, keywords in symptom_keywords.items():
            for kw in keywords:
                # Simple containment check
                # Add word boundary check if needed, but simple containment is okay for "Tiny"
                if kw in symptoms_input:
                    processed.add(canonical_symptom)
                    break # Found one variation, move to next symptom
        
        # Also check strict matches against canonical names directly if not in keywords
        if not processed:
             # Fallback: simple split if no keywords match (rare)
             words = symptoms_input.replace(',', ' ').split()
             for w in words:
                 if w in self.symptom_to_idx:
                     processed.add(w)

        return list(processed)

    def diagnose(self, symptoms_input: str) -> Dict:
        """
        Diagnose possible diseases based on symptoms
        """
        if not self.initialization_complete:
            return {"error": "Model not initialized", "diseases": []}
        
        processed_symptoms = self.preprocess_symptoms(symptoms_input)
        
        if not processed_symptoms:
            return {
                "input_symptoms": [],
                "possible_diseases": [],
                "total_matched": 0,
                "message": "No known symptoms identified."
            }
        
        # Create input vector
        x_vec = np.zeros((1, self.input_size))
        for s in processed_symptoms:
            if s in self.symptom_to_idx:
                x_vec[0, self.symptom_to_idx[s]] = 1.0

        # Predict
        probs = self.predict(x_vec)

        # Get results
        results = []
        diseases = self.knowledge_base.get("diseases", {})
        
        for i, prob in enumerate(probs):
            if prob > 0.01: # Filter very low probability
                d_id = self.idx_to_disease[i]
                d_info = diseases.get(d_id, {})
                d_symptoms = d_info.get("symptoms", [])

                # Calculate explicit matches for explanation
                matched = [s for s in processed_symptoms if s in d_symptoms]

                # Hybrid score: Model Probability + explicit match ratio
                # This ensures that even if model is slightly off, exact matches boost confidence
                match_ratio = len(matched) / len(d_symptoms) if d_symptoms else 0
                final_score = (prob * 0.7) + (match_ratio * 0.3)

                if final_score > 0.1:
                    results.append({
                        "disease_id": d_id,
                        "disease_name": d_info.get("name", d_id),
                        "confidence_score": float(final_score),
                        "matched_symptoms": matched,
                        "explanation": d_info.get("explanation", "No explanation available"),
                        "all_symptoms": d_symptoms
                    })
        
        # Sort by confidence
        results.sort(key=lambda x: x["confidence_score"], reverse=True)
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
        urgent_keywords = ["fever", "chest pain", "difficulty breathing", "severe", "emergency", "confusion"]
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
