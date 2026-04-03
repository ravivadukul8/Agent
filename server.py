import asyncio
import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from vision_agents.core.agents import Agent
from vision_agents.core.edge.types import User
from vision_agents.plugins import gemini, getstream
from pathlib import Path

# Load env
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

app = FastAPI()

class AgentRequest(BaseModel):
    callId: str
    agentUserId: str
    instructions: str

# 🔥 CORE AGENT FUNCTION (unchanged)
async def run_agent(callId: str, agentUserId: str, instructions: str):
    try:
        agent = Agent(
            edge=getstream.Edge(),
            agent_user=User(id=agentUserId, name="AI Agent"),
            instructions=instructions,
            llm=gemini.Realtime(
                api_key=os.getenv("GEMINI_API_KEY"),
                model="gemini-2.5-flash-native-audio-latest",
            ),
        )

        call = agent.edge.client.video.call("default", callId)

        async with agent.join(call):
            print(f"✅ Agent joined call: {callId}")
            await agent.finish()

        print(f"✅ Agent finished call: {callId}")

    except Exception as e:
        print(f"❌ Agent error: {e}")

# ✅ WORKER ENDPOINT (called by Inngest)
@app.post("/run-agent")
async def run_agent_worker(req: AgentRequest):
    print(f"🚀 Worker received job: {req.callId}")

    # 🔥 Run directly (NO background task)
    await run_agent(req.callId, req.agentUserId, req.instructions)

    return {"status": "done"}

# Health check
@app.get("/health")
async def health():
    return {"status": "ok"}