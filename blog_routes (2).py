from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import os
import logging
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from supabase import create_client
from supabase import create_client
from supabase.lib.client_options import ClientOptions
client_options = ClientOptions(
    headers={
        "Authorization": f"Bearer {os.getenv('SUPABASE_SERVICE_ROLE_KEY')}",
        "apikey": os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
)

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
    options=client_options
)
router = APIRouter()
security = HTTPBearer()

class BlogPostCreate(BaseModel):
    title: str
    content: str
    author: str = "Admin"
    tags: List[str] = []
    cover_image: Optional[str] = None

@router.post("/create")
async def create_blog_post(
    post: BlogPostCreate,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    user_id = credentials.credentials if credentials else None
    if not user_id:
        raise HTTPException(status_code=403, detail="Authentication required")

    data = {
        "title": post.title,
        "content": post.content,
        "author": post.author,
        "tags": post.tags,
        "cover_image": post.cover_image,
        "updated_at": datetime.utcnow().isoformat()
    }

    result = supabase.table("blogs").insert(data).execute()
    if result.error:
        raise HTTPException(status_code=500, detail=str(result.error))
    return result.data[0]
