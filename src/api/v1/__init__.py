"""FastAPI v1 Modular API Routers Aggregate."""

from fastapi import APIRouter

from src.api.v1.nlp import router as nlp_router
from src.api.v1.dyadic import router as dyadic_router
from src.api.v1.sessions import router as sessions_router
from src.api.v1.longitudinal import router as longitudinal_router
from src.api.v1.edge import router as edge_router
from src.api.v1.agent import router as agent_router
from src.api.v1.biometrics import router as biometrics_router
from src.api.v1.cognitive import router as cognitive_router
from src.api.v1.somatosensory import router as somatosensory_router
from src.api.v1.forensic import router as forensic_router
from src.api.v1.forecasting import router as forecasting_router

api_v1_router = APIRouter()

api_v1_router.include_router(nlp_router)
api_v1_router.include_router(dyadic_router)
api_v1_router.include_router(sessions_router)
api_v1_router.include_router(longitudinal_router)
api_v1_router.include_router(edge_router)
api_v1_router.include_router(agent_router)
api_v1_router.include_router(biometrics_router)
api_v1_router.include_router(cognitive_router)
api_v1_router.include_router(somatosensory_router)
api_v1_router.include_router(forensic_router)
api_v1_router.include_router(forecasting_router)
