from fastapi import APIRouter

from api.models import all_model_status
from api.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def health_check():
    models = all_model_status()
    db_ok = True
    try:
        from api.config import settings
        import pymongo

        client = pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=2000)
        client.admin.command("ping")
        client.close()
    except Exception:
        db_ok = False
    return HealthResponse(
        status="ok" if any(models.values()) or db_ok else "degraded",
        models=models,
        database=db_ok,
    )
