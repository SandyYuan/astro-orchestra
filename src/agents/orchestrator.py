"""Orchestrator agent that coordinates the multi-agent astronomy research workflow."""

from typing import Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from src.agents.base import BaseAgent
from src.state.agent_state import AgentState
from config.settings import settings
import json


class OrchestratorAgent(BaseAgent):
    """Main orchestrator that breaks down tasks and delegates to specialists."""
    
    def __init__(self, llm: ChatGoogleGenerativeAI = None):
        super().__init__(
            name="orchestrator",
            mcp_tools=[],  # Orchestrator doesn't use external tools directly
            description="Breaks down research tasks and delegates to specialist agents"
        )
        self.llm = llm or ChatGoogleGenerativeAI(
            model=settings.orchestrator_model,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            google_api_key=settings.google_api_key
        )
        
    async def process(self, state: AgentState) -> AgentState:
        """Look at current state and decide what to do next."""
        
        # Build context from current state
        context = f"""Current astronomy research state:
        
Task: {state.get('current_task', 'No task specified')}

Progress:
- Data gathered: {len(state.get('data_artifacts', {}))} datasets
- Analysis complete: {len(state.get('analysis_results', {}))} results  
- Literature reviewed: {len(state.get('literature_context', {}))} papers
- Simulations run: {len(state.get('simulation_outputs', {}))} simulations

Available agents:
- data_gathering: Access DESI, LSST, CMB databases
- analysis: Statistical analysis and computations
- theorist_simulation: Run cosmological simulations
- literature_reviewer: Search and synthesize papers

Decide the next action. Return JSON:
{{
    "next_agent": "agent_name" or null (if done),
    "instructions": "specific instructions for the next agent",
    "is_complete": boolean,
    "summary": "brief update for the user about what you're doing"
}}"""

        messages = [
            SystemMessage(content=context),
            *state["messages"]
        ]
        
        response = await self.llm.ainvoke(messages)
        
        try:
            decision = json.loads(response.content)
            
            # Add orchestrator's update to conversation history
            if decision.get("is_complete", False):
                # Final summary
                final_message = decision.get("summary", "Research complete. Here's what I found...")
                state["messages"].append(AIMessage(content=final_message))
                state["final_response"] = final_message
                state["next_agent"] = None  # Routes to END
            else:
                # Progress update
                update_message = decision.get("summary", f"Routing to {decision['next_agent']}...")
                state["messages"].append(AIMessage(content=update_message))
                state["next_agent"] = decision["next_agent"]
                state["current_task"] = decision.get("instructions", state["current_task"])
                
        except json.JSONDecodeError:
            # If parsing fails, still update messages
            state["messages"].append(
                AIMessage(content="I'll gather some initial data to get started.")
            )
            state["next_agent"] = "data_gathering"
                
        return state 