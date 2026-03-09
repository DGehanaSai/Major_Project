from fastapi import APIRouter

router = APIRouter()

users = {}

@router.post("/register")
def register(username: str, password: str):
    users[username] = password
    return {"message": "User registered successfully"}

@router.post("/login")
def login(username: str, password: str):
    if users.get(username) == password:
        return {"message": "Login successful"}
    return {"error": "Invalid credentials"}
