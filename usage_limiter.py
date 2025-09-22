from fastapi import Request, HTTPException, Depends
from datetime import datetime, date
from supabase import create_client, Client
import os
import logging
from auth import get_current_user

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.critical("ERROR: Missing Supabase configuration")
    raise RuntimeError("Missing Supabase configuration")

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    # Manually set auth for postgrest if needed
    if hasattr(supabase, 'postgrest'):
        supabase.postgrest.auth(SUPABASE_KEY)
    logger.info("Supabase client initialized successfully")
except Exception as e:
    logger.error(f"Failed to create Supabase client: {e}")
    raise RuntimeError(f"Failed to initialize Supabase client: {e}")

def enforce_usage_limit(feature: str):
    async def _enforce(current_user: dict = Depends(get_current_user)):
        user_id = current_user["id"]
        user_plan = current_user.get("plan", "free")
        
        try:
            # Get current usage with proper error handling
            usage_response = supabase.table("usage_limits") \
                .select("*") \
                .eq("user_id", user_id) \
                .eq("feature", feature) \
                .maybe_single() \
                .execute()

            # Handle response
            if not usage_response.data:
                # Initialize usage record if none exists
                supabase.table("usage_limits").insert({
                    "user_id": user_id,
                    "feature": feature,
                    "used_count": 0,
                    "reset_at": datetime.now().date().isoformat()
                }).execute()
                used_count = 0
            else:
                used_count = usage_response.data.get("used_count", 0)

            # Increment and update usage
            supabase.table("usage_limits").upsert({
                "user_id": user_id,
                "feature": feature,
                "used_count": used_count + 1,
                "reset_at": datetime.now().date().isoformat()
            }, on_conflict='user_id,feature').execute()

            return user_id

        except Exception as e:
            logger.error(f"Usage tracking error: {e}")
            # Allow request to proceed even if usage tracking fails
            return user_id

    return Depends(_enforce)