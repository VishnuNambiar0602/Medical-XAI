import sys
import os
import json

# Add backend to path so we can import model
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from model import initialize_model

def test_model():
    print("Initializing model...")
    # Point to the correct data path
    kb_path = os.path.join(os.getcwd(), 'data', 'medical_knowledge_base.json')
    try:
        model = initialize_model(kb_path)
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return

    print("\n--- Test Case 1: Simple Keywords ---")
    input_text = "I have a fever and a cough"
    print(f"Input: '{input_text}'")
    result = model.diagnose(input_text)
    print_result(result)

    # Assertions for Case 1
    # Should detect fever and cough, and likely suggest Flu, Common Cold, COVID-19
    symptoms = result.get("input_symptoms", [])
    assert "fever" in symptoms, "Failed to detect fever"
    assert "cough" in symptoms, "Failed to detect cough"
    top_disease = result["possible_diseases"][0]["disease_name"]
    print(f"Top prediction: {top_disease}")

    print("\n--- Test Case 2: Sentence with variations ---")
    input_text = "My head hurts really bad and I feel very hot"
    # "head hurts" -> headache, "hot" -> fever
    print(f"Input: '{input_text}'")
    result = model.diagnose(input_text)
    print_result(result)

    # Assertions for Case 2
    symptoms = result.get("input_symptoms", [])
    print(f"Detected symptoms: {symptoms}")
    assert "headache" in symptoms, "Failed to detect headache from 'head hurts'"
    assert "fever" in symptoms, "Failed to detect fever from 'hot'"

    print("\n--- Test Case 3: Explanation Retrieval ---")
    if result["possible_diseases"]:
        disease_id = result["possible_diseases"][0]["disease_id"]
        explanation = model.explain_diagnosis(disease_id)
        print(f"Explanation for {disease_id}: {explanation.get('explanation')}")
        assert explanation.get("explanation"), "Failed to retrieve explanation"

    print("\n--- Test Case 4: Recommendation ---")
    rec = model.get_recommendation("chest pain and difficulty breathing")
    print(f"Recommendation: {rec}")
    assert rec["urgency"] == "high", "Failed to detect high urgency"

    print("\nAll tests passed!")

def print_result(result):
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"Detected Symptoms: {result['input_symptoms']}")
    print("Top Diseases:")
    for d in result["possible_diseases"][:3]:
        print(f"  - {d['disease_name']} ({d['confidence_score']:.2f}): {d['explanation'][:50]}...")

if __name__ == "__main__":
    test_model()
