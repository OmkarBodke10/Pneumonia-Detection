import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def generate_llm_medical_report(prediction: str, confidence: float, normal_prob: float, pneumonia_prob: float) -> str:
    """
    Generates a structured medical preliminary report based on DL model outputs.
    Uses Groq API with llama-3.3-70b-versatile.
    """
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        return "LLM Report Pending: GROQ_API_KEY environment variable not configured."

    try:
        client = Groq(api_key=api_key)
        
        prompt = f"""
        You are an AI Radiology Assistant. Analyze these classification outputs for a Chest X-ray scan:
        
        - Primary Classification: {prediction}
        - Model Confidence: {confidence:.2f}%
        - Normal Class Probability: {normal_prob:.2f}%
        - Pneumonia Class Probability: {pneumonia_prob:.2f}%

        Generate a concise, professional diagnostic summary for an attending doctor with the following 3 sections:
        1. IMPRESSION SUMMARY
        2. RADIOLOGICAL INDICATORS TO LOOK FOR
        3. RECOMMENDED CLINICAL NEXT STEPS

        Keep it brief, under 180 words. Include a brief medical disclaimer at the end.
        """

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=300
        )
        
        return response.choices[0].message.content

    except Exception as e:
        return f"LLM Generation Error: {str(e)}"