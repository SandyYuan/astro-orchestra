"""Theorist simulation agent for running cosmological simulations."""

from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AIMessage
from src.agents.base import BaseAgent
from src.state.agent_state import AgentState
from config.settings import settings


class TheoristSimulationAgent(BaseAgent):
    """Agent specialized in running cosmological simulations."""
    
    def __init__(self, llm: ChatGoogleGenerativeAI = None):
        super().__init__(
            name="theorist_simulation",
            mcp_tools=["nbody-server", "camb-server"],
            description="Runs cosmological simulations and theoretical calculations"
        )
        self.llm = llm or ChatGoogleGenerativeAI(
            model=settings.specialist_model,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            google_api_key=settings.google_api_key
        )
    
    async def process(self, state: AgentState) -> AgentState:
        """Run simulations based on the current task."""
        
        self.log_message(state, "Starting simulation work")
        
        current_task = state.get("current_task", "")
        
        # Placeholder simulation - in real implementation, would use MCP tools
        simulation_key = "cosmological_sim"
        state["simulation_outputs"][simulation_key] = {
            'description': 'Cosmological N-body simulation',
            'filename': 'nbody_simulation.hdf5',
            'summary': 'Simulated dark matter halo formation',
            'preview_command': 'preview_simulation("cosmological_sim")'
        }
        
        state["messages"].append(
            AIMessage(content="Completed cosmological simulation to model the theoretical framework.")
        )
        
        self.log_message(state, "Simulation complete")
        state["next_agent"] = "orchestrator"
        return state 