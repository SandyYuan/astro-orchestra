"""Analysis agent for statistical analysis of astronomy data."""

from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage
from src.agents.base import BaseAgent
from src.state.agent_state import AgentState
from config.settings import settings


class AnalysisAgent(BaseAgent):
    """Agent specialized in analyzing astronomy data."""
    
    def __init__(self, llm: ChatOpenAI = None):
        super().__init__(
            name="analysis",
            mcp_tools=["statistics-server", "correlation-server"],
            description="Performs statistical analysis on astronomy data"
        )
        self.llm = llm or ChatOpenAI(model=settings.specialist_model, temperature=0)
    
    async def process(self, state: AgentState) -> AgentState:
        """Analyze data using appropriate MCP tools."""
        
        self.log_message(state, "Starting data analysis")
        
        # Get available data from state (file references only)
        data_artifacts = state.get("data_artifacts", {})
        current_task = state.get("current_task", "")
        
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
                'preview_command': 'preview_analysis("statistical_summary")'
            }
            
            state["messages"].append(
                AIMessage(content=f"{data_summary}\n\nCompleted statistical analysis of the available data.")
            )
            self.log_message(state, "Analysis complete")
        
        state["next_agent"] = "orchestrator"
        return state 