from .auth import router as auth_router
from .checklist import router as checklist_router

routers = [
    auth_router,
    checklist_router
]