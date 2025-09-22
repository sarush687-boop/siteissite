from fastapi import APIRouter, HTTPException, Depends, Request, status
from fastapi.responses import RedirectResponse
from supabase import create_client, Client
import os
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

router = APIRouter()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")

if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
    logger.critical("ERROR: Missing Supabase environment variables")
    raise RuntimeError("Missing Supabase configuration")

# Initialize Supabase client
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
    if hasattr(supabase, 'postgrest'):
        supabase.postgrest.auth(SUPABASE_SECRET_KEY)
    logger.info("Supabase client initialized successfully")
except Exception as e:
    logger.error(f"Failed to create Supabase client: {e}")
    raise RuntimeError(f"Failed to initialize Supabase client: {e}")

# Existing get_current_user function
async def get_current_user(request: Request) -> Dict[str, Any]:
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        logger.warning("Authentication header missing")
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        token = auth_header.split(" ")[1]
        user_response = supabase.auth.get_user(token)
        if hasattr(user_response, 'error') and user_response.error:
            raise HTTPException(status_code=401, detail=user_response.error.message)
        
        user_data = user_response.user
        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid user data")

        try:
            profile_response = supabase.table("users") \
                .select("plan") \
                .eq("id", user_data.id) \
                .maybe_single() \
                .execute()
            
            user_plan = profile_response.data.get("plan", "free") if profile_response.data else "free"
        except Exception as profile_error:
            logger.error(f"Error fetching user profile: {profile_error}")
            user_plan = "free"

        return {
            "id": user_data.id,
            "email": user_data.email,
            "plan": user_plan
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Authentication error:")
        raise HTTPException(status_code=500, detail="Internal authentication error")

# New endpoint to initiate Google OAuth
@router.get("/auth/google")
async def auth_google():
    try:
        # Redirect to Supabase's Google OAuth URL
        redirect_url = supabase.auth.sign_in_with_provider({
            "provider": "google",
            "options": {
                "redirect_to": "https://localhost:3000/auth/callback"  # Replace with your frontend callback URL
            }
        })
        return RedirectResponse(url=redirect_url)
    except Exception as e:
        logger.error(f"Error initiating Google OAuth: {e}")
        raise HTTPException(status_code=500, detail="Failed to initiate Google OAuth")

# New endpoint to handle OAuth callback (optional, if you need custom handling)
@router.get("/auth/callback")
async def auth_callback(request: Request):
    try:
        code = request.query_params.get("code")
        if not code:
            raise HTTPException(status_code=400, detail="Authorization code missing")

        # Exchange code for session
        session = supabase.auth.exchange_code_for_session({"provider": "google", "code": code})
        if session.error:
            raise HTTPException(status_code=401, detail=session.error.message)

        user = session.user
        token = session.access_token

        # Optionally, store or update user profile in your users table
        try:
            supabase.table("users").upsert({
                "id": user.id,
                "email": user.email,
                "plan": "free",  # Default plan
                "updated_at": datetime.now(timezone.utc).isoformat()
            }).execute()
        except Exception as e:
            logger.error(f"Error updating user profile: {e}")

        # Redirect to frontend with token (or handle as needed)
        return RedirectResponse(url=f"http://localhost:3000/app?token={token}")
    except Exception as e:
        logger.error(f"Error in OAuth callback: {e}")
        raise HTTPException(status_code=500, detail="OAuth callback error")