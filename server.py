import asyncio
import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from vision_agents.core.agents import Agent
from vision_agents.core.edge.types import User
from vision_agents.plugins import gemini, getstream
from pathlib import Path

# ✅ Load .env from agent folder
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# ✅ Force set as system env vars
os.environ["STREAM_API_KEY"] = os.getenv("STREAM_API_KEY", "")
os.environ["STREAM_API_SECRET"] = os.getenv("STREAM_API_SECRET", "")



app = FastAPI()

class StartAgentRequest(BaseModel):
    callId: str
    agentUserId: str
    instructions: str

async def run_agent(callId: str, agentUserId: str, instructions: str):
    """Runs the Gemini agent - joins call and waits for it to end."""
    try:
        agent = Agent(
            edge=getstream.Edge(),
            agent_user=User(
                id=agentUserId,
                name="AI Agent"
            ),
            instructions=instructions,
            llm=gemini.Realtime(
                api_key=os.getenv("GEMINI_API_KEY"),
                model="gemini-2.5-flash-native-audio-latest",
                config={
                    "response_modalities": ["AUDIO"],
                    "speech_config": {
                        "voice_config": {
                            "prebuilt_voice_config": {
                                "voice_name": "Leda"
                            }
                        }
                    },
                },
            ),
        )

        # ✅ New API: get call object then join it
        call = agent.edge.client.video.call("default", callId)

        async with agent.join(call):
            print(f"✅ Agent joined call: {callId}")
            await agent.finish()  # Wait until call ends

        print(f"✅ Agent finished call: {callId}")

    except Exception as e:
        print(f"❌ Agent error: {e}")

@app.post("/start-agent")
async def start_agent(req: StartAgentRequest):
    print(f"✅ Starting Gemini agent for call: {req.callId}")
    # Run agent in background so API responds immediately
    asyncio.create_task(run_agent(req.callId, req.agentUserId, req.instructions))
    return {"status": "agent started", "callId": req.callId}

@app.get("/health")
async def health():
    return {"status": "ok"}