"""Analysis agent for statistical analysis of astronomy data."""

from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AIMessage, SystemMessage, HumanMessage
from src.agents.base import BaseAgent
from src.state.agent_state import AgentState
from config.settings import settings
import json


class AnalysisAgent(BaseAgent):
    """Agent specialized in analyzing astronomy data."""
    
    def __init__(self, llm: ChatGoogleGenerativeAI = None):
        super().__init__(
            name="analysis",
            mcp_tools=["statistics-server", "correlation-server", "power-spectrum-server"],
            description="Performs statistical analysis on astronomy data"
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
        """Analyze data using appropriate MCP tools."""
        
        self.log_message(state, "Starting data analysis")
        
        current_task = state.get("current_task", "")
        data_artifacts = state.get("data_artifacts", {})
        
        # Single LLM call to determine both: sufficient info + fast track eligibility
        decision = await self._analyze_task_requirements(current_task, data_artifacts)
        
        if not decision["has_sufficient_info"]:
            # Need more info - set fast track if applicable
            if decision["should_fast_track"]:
                state["fast_track"] = "analysis"
            
            # Request more information
            clarification_msg = decision["clarification_message"]
            state["messages"].append(AIMessage(content=clarification_msg))
            state["next_agent"] = None
            self.log_message(state, "Requested more analysis parameters from user")
            return state
        
        # We have sufficient parameters - proceed with analysis
        if not data_artifacts:
            state["messages"].append(
                AIMessage(content="No data available to analyze. Need to gather data first.")
            )
        else:
            # Placeholder analysis - in real implementation, would use MCP tools
            data_summary = f"Available data files for analysis: {len(data_artifacts)} datasets"
            for key, file_info in data_artifacts.items():
                data_summary += f"\n- {file_info['filename']}"
                if 'total_records' in file_info:
                    data_summary += f" ({file_info['total_records']} records)"
            
            # Mock analysis result
            analysis_key = "statistical_summary"
            state["analysis_results"][analysis_key] = {
                'description': 'Statistical summary analysis',
                'filename': 'statistical_analysis.json',
                'summary': 'Basic statistical analysis completed',
                'preview_command': 'preview_analysis("statistical_summary")',
                'parameters': current_task  # Store the specific parameters used
            }
            
            state["messages"].append(
                AIMessage(content=f"{data_summary}\n\nCompleted analysis with parameters: {current_task}")
            )
            self.log_message(state, "Analysis complete")
        
        state["next_agent"] = "orchestrator"
        return state
    
    async def _analyze_task_requirements(self, task: str, data_artifacts: dict) -> dict:
        """Single LLM call to analyze task and determine next steps."""
        data_summary = f"Available data: {len(data_artifacts)} datasets" if data_artifacts else "No data available"
        
        prompt = f"""
        Analyze this analysis task: "{task}"
        
        Data context: {data_summary}
        
        Determine:
        1. Do I have sufficient information to perform meaningful data analysis?
        2. If not, is this a straightforward analysis request that should fast-track back to me?
        
        For sufficient information, I need specifics like: analysis type (correlation, power spectrum, clustering), specific variables to analyze, statistical methods, or analysis parameters.
        
        Fast track if:
        - This is clearly an analysis request, just missing technical details
        - User will likely provide specifics and want analysis to proceed
        
        Don't fast track if:
        - Request is ambiguous about what type of work to do
        - User might want to change direction entirely
        - Unclear if they want analysis vs data gathering vs simulation
        
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
                "clarification_message": "I need more specific information about what analysis to perform. Please specify: analysis type (correlation, power spectrum, clustering), variables to analyze, or statistical methods."
            } 