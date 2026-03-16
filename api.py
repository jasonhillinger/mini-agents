from fastapi import FastAPI

app = FastAPI()

# To run the server, use the command: uvicorn api:app --reload


@app.get("/")
def read_root():
    return {"message": "Hello World"}
