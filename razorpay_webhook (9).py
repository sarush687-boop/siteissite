from fastapi import FastAPI, Request, HTTPException, APIRouter
from fastapi.responses import JSONResponse
import hmac
import hashlib
import json
import os
from datetime import datetime
from supabase import create_client
from supabase.lib.client_options import ClientOptions

app = FastAPI()
router = APIRouter()

# ---------- Supabase Client ----------
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

# ---------- Razorpay Credentials ----------
RAZORPAY_WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET")

# ---------- Plan to Credit Mapping ----------
CREDIT_LOOKUP = {
    "free": 50,
    "basic": 500,
    "premium": 1000,
    "pro": 999999  # practically unlimited
}

# ---------- Feature Costs ----------
FEATURE_COST = {
    "summary": 5,
    "quiz": 10,
    "humanize": 5,
    "mindmap": 10,
    "diagram": 10,
    "flashcards": 5,
    "vocabulary": 5,
    "handwritten": 5,
    "solve_questions": 5,
    "ppt": 10,
    "study_plan": 5,
    "grammar_correction": 5,
    "knowledge_graph": 10
}

# ---------- Map Razorpay Plan IDs ----------
def map_plan_id_to_internal(plan_id: str):
    mapping = {
        "plan_BASIC_ID": "basic",
        "plan_PREMIUM_ID": "premium",
        "plan_PRO_ID": "pro"
    }
    return mapping.get(plan_id, "free")

# ---------- Verify Razorpay Signature ----------
def verify_signature(payload: bytes, signature: str):
    generated_signature = hmac.new(
        RAZORPAY_WEBHOOK_SECRET.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    if generated_signature != signature:
        raise ValueError("Invalid Razorpay webhook signature")

# ---------- Razorpay Webhook ----------
@app.post("/razorpay/webhook")
async def razorpay_webhook(request: Request):
    payload = await request.body()
    signature = request.headers.get("x-razorpay-signature")
    event = json.loads(payload)

    # Verify signature
    try:
        verify_signature(payload, signature)
    except ValueError:
        return JSONResponse({"status": "error", "message": "Invalid signature"}, status_code=400)

    subscription = event.get("payload", {}).get("subscription", {}).get("entity", {})
    user_id = subscription.get("customer_notify_id") or subscription.get("notes", {}).get("user_id")
    plan_id = subscription.get("plan_id")

    internal_plan = map_plan_id_to_internal(plan_id)

    # Update user in Supabase
    supabase.table("users").update({
        "plan": internal_plan,
        "plan_updated_at": datetime.utcnow().isoformat()
    }).eq("id", user_id).execute()

    # Reset credits based on plan
    credits_to_add = CREDIT_LOOKUP.get(internal_plan, 0)
    supabase.table("credits").upsert({
        "user_id": user_id,
        "credits": credits_to_add,
        "last_updated": datetime.utcnow().isoformat()
    }).execute()

    return JSONResponse({"status": "success", "plan": internal_plan, "credits": credits_to_add})

# ---------- Deduct Credit API ----------
@app.post("/deduct-credit")
async def deduct_credit(request: Request):
    body = await request.json()
    user_id = body.get("user_id")
    feature = body.get("feature")

    if not user_id or not feature:
        raise HTTPException(status_code=400, detail="Missing user_id or feature")

    # Fetch credits
    credits_data = supabase.table("credits").select("credits").eq("user_id", user_id).execute()
    if not credits_data.data:
        raise HTTPException(status_code=404, detail="User not found")

    current_credits = credits_data.data[0]["credits"]
    cost = FEATURE_COST.get(feature, 1)

    if current_credits < cost:
        raise HTTPException(status_code=402, detail="Not enough credits")

    new_credits = current_credits - cost

    # Update credits in Supabase
    supabase.table("credits").update({
        "credits": new_credits,
        "last_updated": datetime.utcnow().isoformat()
    }).eq("user_id", user_id).execute()

    return {"success": True, "remaining_credits": new_credits}
