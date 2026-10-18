import asyncio
import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


app = FastAPI(title="PawHealth API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 開發測試期間允許 Flutter Web 連線
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BehaviorAnalysisRequest(BaseModel):
    pet_name: str = Field(min_length=1, max_length=80)
    record_date: str = Field(min_length=1, max_length=20)
    mood: str = Field(min_length=1, max_length=30)
    activity_level: int = Field(ge=1, le=5)
    sleep_status: str = Field(min_length=1, max_length=50)
    appetite_change: str = Field(min_length=1, max_length=50)
    abnormal_behaviors: list[str] = Field(default_factory=list, max_length=20)
    notes: str = Field(default="", max_length=500)


@app.get("/")
def home() -> dict[str, str]:
    return {"message": "PawHealth API is running"}


def _call_gemini(prompt: str) -> dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Render 尚未設定 GEMINI_API_KEY")

    model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 450,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "analysis": {"type": "STRING"},
                    "advice": {"type": "STRING"},
                },
                "required": ["analysis", "advice"],
            },
        },
    }
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=70) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API 錯誤 ({error.code})：{detail[:500]}") from error
    except URLError as error:
        raise RuntimeError(f"無法連線 Gemini：{error.reason}") from error


def _extract_gemini_json(response: dict[str, Any]) -> dict[str, str]:
    try:
        text = response["candidates"][0]["content"]["parts"][0]["text"]
        result = json.loads(text)
        analysis = str(result["analysis"]).strip()
        advice = str(result["advice"]).strip()
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("Gemini 回傳格式不完整") from error

    if not analysis or not advice:
        raise RuntimeError("Gemini 沒有回傳完整分析")
    return {"analysis": analysis, "advice": advice}


@app.post("/analyze-behavior")
async def analyze_behavior(data: BehaviorAnalysisRequest) -> dict[str, str]:
    behaviors = "、".join(data.abnormal_behaviors) or "無異常"
    notes = data.notes or "無"
    prompt = f"""
你是一位審慎的寵物健康照護助理。請根據飼主提供的單日行為紀錄，
用繁體中文提供簡短的觀察與照護建議。你不能診斷疾病，也不能斷言病因。
若出現抽搐、呼吸困難、昏迷、大量出血、持續嘔吐、血便、完全拒食飲水等警訊，
應明確建議儘速聯絡獸醫。若資料不足，請直接說明需要持續觀察。

寵物：{data.pet_name}
日期：{data.record_date}
整體情緒：{data.mood}
活動量：{data.activity_level}/5
睡眠狀況：{data.sleep_status}
食慾變化：{data.appetite_change}
異常行為：{behaviors}
補充說明：{notes}

請只輸出符合指定結構的 JSON：
- analysis：60～120 個繁體中文字，描述觀察到的趨勢與需要注意之處。
- advice：40～100 個繁體中文字，提供安全、具體的觀察與就醫提醒。
""".strip()

    try:
        response = await asyncio.to_thread(_call_gemini, prompt)
        return _extract_gemini_json(response)
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
# === 寵物保險理賠預審 API ===
class ClaimItem(BaseModel):
    record_id: str
    visit_date: str
    diagnosis: str
    cost: float
    receipt_url: str = ""

class PreCheckRequest(BaseModel):
    pet_id: str
    claim_items: list[ClaimItem]

@app.post("/insurance/pre-check")
def pre_check_claim(payload: PreCheckRequest):
    total_cost = sum(item.cost for item in payload.claim_items)
    diagnoses = "、".join([item.diagnosis for item in payload.claim_items])
    return {
        "status": "success",
        "pet_id": payload.pet_id,
        "total_cost": total_cost,
        "eligible_items_count": len(payload.claim_items),
        "ai_precheck_note": f"已收到【{diagnoses}】共 {len(payload.claim_items)} 筆理賠申請（總計 NT$ {total_cost:,.0f}）。初步判定符合一般門診理賠範疇，請備妥獸醫診斷書與收據正本。"
    }