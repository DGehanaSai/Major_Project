from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class Question(BaseModel):
    question: str

@router.post("/ask")
def ask(q: Question):
    question_text = q.question.lower()
    
    if "water" in question_text or "irrigation" in question_text:
        answer = "For optimal yield, ensure consistent moisture levels. Drip irrigation is recommended for water conservation."
    elif "pest" in question_text or "disease" in question_text:
        answer = "Monitor crops daily. Early detection using our image analysis tool can prevent spread. Consider organic pesticides like neem oil."
    elif "fertilizer" in question_text or "nitrogen" in question_text:
        answer = "Balanced fertilization is key. Our model suggests Nitrogen levels around 60kg/acre for this crop type. Avoid over-fertilizing."
    else:
        answer = "I can help with irrigation, pests, and soil health. Please ask specifically about those topics."

    return {
        "answer": answer
    }
