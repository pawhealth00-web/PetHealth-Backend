from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "PawHealth API is running"}