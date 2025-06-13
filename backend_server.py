from fastapi import FastAPI, HTTPException, openapi
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
from sat_chain import chain


app = FastAPI(title="SAT Reading Tutor Backend", version="1.0.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MCQRequest(BaseModel):
    text: str
    question: str
    choices: List[str]


class MCQResponse(BaseModel):
    answer: str


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "SAT Reading Tutor Backend is running!"}


@app.post("/submit_mcq", response_model=MCQResponse)
async def submit_fn(request: MCQRequest):
    """
    Process the submission of SAT text, question, and choices and return the correct answer.
    args:
        request (MCQRequest): The request containing the SAT text, question, and choices.
    returns:
        MCQResponse: The response containing the correct answer.
    """

    try:
        answer = chain.invoke(
            {
                "passage": request.text,
                "question": request.question,
                "choices": request.choices,
            }
        )

        response = MCQResponse(answer=answer)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "service": "SAT Reading Tutor Backend",
        "version": "1.0.0",
    }


if __name__ == "__main__":
    uvicorn.run(
        "backend_server:app",
        host="0.0.0.0",
        port=8090,
        reload=True,
        log_level="info",
    )
