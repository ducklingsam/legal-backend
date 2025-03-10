from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

app = FastAPI()


@app.post('/search')
async def search(request: Request):
    return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content={"detail": "No content"})


@app.post('/scan')
async def scan(request: Request):
    return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content={"detail": "No content"})