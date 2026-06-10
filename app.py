import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response, status
import uvicorn
from uvicorn.config import LOGGING_CONFIG

import inference


def init_model():
    """
    Initialize any model or resources here.
    This runs once at startup, not on every invocation.
    For this algorithm we don't have a heavy model to load,
    but this is where you'd put model loading code.
    """
    print("=== Initializing model / resources")
    # No heavy model needed for sphere generation
    return None


MODELS = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model at startup
    MODELS["model"] = init_model()
    print("=== Model initialized, ready for invocations")
    yield
    # Clean up
    MODELS.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    try:
        _ = MODELS["model"]
        return Response(status_code=status.HTTP_200_OK)
    except KeyError:
        return Response(status_code=status.HTTP_404_NOT_FOUND)


@app.post("/invoke")
async def invoke():
    t0 = time.perf_counter()
    model = MODELS["model"]
    inference.run(model)
    elapsed = time.perf_counter() - t0
    print(f"[app.py] /invoke handler total: {elapsed:.4f} s", flush=True)
    return Response(status_code=status.HTTP_201_CREATED)


if __name__ == "__main__":
    log_config = LOGGING_CONFIG.copy()
    log_config["handlers"]["default"]["stream"] = "ext://sys.stdout"
    uvicorn.run(app, host="0.0.0.0", port=4743, log_config=log_config)
