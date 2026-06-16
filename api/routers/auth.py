from __future__ import annotations

import logging

import pymongo
from fastapi import APIRouter, HTTPException
from datetime import datetime

from api.config import settings
from api.schemas import (
    ErrorResponse,
    LoginRequest,
    LoginResponse,
    RegisterRequest,
    RegisterResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


def _get_db():
    client = pymongo.MongoClient(settings.MONGO_URI)
    return client[settings.MONGO_DB], client


@router.post(
    "/login",
    response_model=LoginResponse,
    responses={401: {"model": ErrorResponse}},
    summary="Authenticate a user",
)
def login(body: LoginRequest):
    try:
        db, client = _get_db()
        users = db["users"]
        user = users.find_one({"customer_name": body.username})
        client.close()
    except Exception as exc:
        logger.error("MongoDB error during login: %s", exc)
        raise HTTPException(503, detail="Database connection failed")

    if user is None or user.get("password") != body.password:
        raise HTTPException(401, detail="Invalid username or password")

    return LoginResponse(
        success=True,
        customer_name=user.get("customer_name"),
        role=user.get("role"),
        plan=user.get("plan"),
        email=user.get("email"),
    )


@router.post(
    "/register",
    response_model=RegisterResponse,
    responses={409: {"model": ErrorResponse}},
    summary="Register a new user",
)
def register(body: RegisterRequest):
    try:
        db, client = _get_db()
        users = db["users"]

        if users.find_one({
            "$or": [
                {"customer_name": body.customer_name},
                {"email": body.email},
            ]
        }):
            client.close()
            raise HTTPException(409, detail="Username or email already exists")

        users.insert_one({
            "customer_name": body.customer_name,
            "password": body.password,
            "email": body.email,
            "role": body.role,
            "plan": body.plan,
            "created_at": datetime.now(),
        })
        client.close()
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("MongoDB error during register: %s", exc)
        raise HTTPException(503, detail="Database connection failed")

    return RegisterResponse(success=True, message="User registered successfully")
