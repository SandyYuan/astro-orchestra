"""Theorist simulation agent for running cosmological simulations."""

from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AIMessage, HumanMessage
from src.agents.base import BaseAgent
from src.state.agent_state import AgentState
from config.settings import settings
import json


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
        
        # Single LLM call to determine both: sufficient info + fast track eligibility
        decision = await self._analyze_task_requirements(current_task)
        
        if decision["has_sufficient_info"]:
            # Proceed with simulation
            return await self._run_simulation(state, current_task)
        else:
            # Need more info - set fast track if applicable
            if decision["should_fast_track"]:
                state["fast_track"] = "theorist_simulation"
            
            # Request more information
            clarification_msg = decision["clarification_message"]
            state["messages"].append(AIMessage(content=clarification_msg))
            state["next_agent"] = None
            self.log_message(state, "Requested more simulation parameters from user")
            return state
    
    async def _analyze_task_requirements(self, task: str) -> dict:
        """Single LLM call to analyze task and determine next steps."""
        prompt = f"""
        Analyze this simulation task: "{task}"
        
        Determine:
        1. Do I have sufficient parameters to run a meaningful simulation?
        2. If not, is this a straightforward parameter request that should fast-track back to me?
        
        For sufficient parameters, I need at least 2 of: mass info, box size, particle count, redshift range, cosmological parameters.
        
        Fast track if:
        - This is clearly a simulation request, just missing technical details
        - User will likely provide parameters and want simulation to proceed
        
        Don't fast track if:
        - Request is ambiguous about what type of work to do
        - User might want to change direction entirely
        
        Return JSON:
        {{
            "has_sufficient_info": true/false,
            "should_fast_track": true/false,
            "clarification_message": "specific message if more info needed"
        }}
        """
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        
        try:
            # Strip markdown code block formatting if present
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            result = json.loads(content)
            return result
        except:
            # Fallback if JSON parsing fails
            return {
                "has_sufficient_info": False,
                "should_fast_track": False,
                "clarification_message": "I need more specific parameters to proceed with the simulation."
            }
    
    async def _run_simulation(self, state: AgentState, current_task: str) -> AgentState:
        """Execute the simulation with sufficient parameters."""
        # Placeholder simulation - in real implementation, would use MCP tools
        simulation_key = "cosmological_sim"
        state["simulation_outputs"][simulation_key] = {
            'description': 'Cosmological N-body simulation',
            'filename': 'nbody_simulation.hdf5',
            'summary': 'Simulated dark matter halo formation',
            'preview_command': 'preview_simulation("cosmological_sim")',
            'parameters': current_task  # Store the specific parameters used
        }
        
        state["messages"].append(
            AIMessage(content=f"Completed cosmological simulation with parameters: {current_task}")
        )
        
        self.log_message(state, "Simulation complete")
        state["next_agent"] = "orchestrator"
        return state 