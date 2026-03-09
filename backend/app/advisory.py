from fastapi import APIRouter

router = APIRouter()

@router.post("/recommend")
def recommend(crop: str, temp: float, moisture: float):
    alternatives = []
    if crop == "Cotton":
        alternatives = [
            {"crop": "Wheat", "yield": 52, "gain": "+37%"},
            {"crop": "Corn", "yield": 48, "gain": "+26%"},
        ]
    return {
        "alternatives": alternatives,
        "reason": {
            "soil": 85,
            "water": 70,
            "sunlight": 88
        }
    }
