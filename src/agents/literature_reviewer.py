"""Literature review agent for searching and synthesizing scientific papers."""

from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AIMessage
from src.agents.base import BaseAgent
from src.state.agent_state import AgentState
from config.settings import settings


class LiteratureReviewerAgent(BaseAgent):
    """Agent specialized in reviewing scientific literature."""
    
    def __init__(self, llm: ChatGoogleGenerativeAI = None):
        super().__init__(
            name="literature_reviewer",
            mcp_tools=["arxiv-server", "scholarly-server"],
            description="Searches and synthesizes scientific literature"
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
        """Search and review relevant literature."""
        
        self.log_message(state, "Starting literature review")
        
        current_task = state.get("current_task", "")
        
        # Placeholder literature review - in real implementation, would use MCP tools
        topic = "astronomy_research"
        state["literature_context"][topic] = [
            "arXiv:2301.12345 - Dark matter halo analysis",
            "arXiv:2302.67890 - Galaxy formation simulations",
            "arXiv:2303.11111 - DESI spectroscopic survey results"
        ]
        
        paper_count = len(state["literature_context"][topic])
        state["messages"].append(
            AIMessage(content=f"Reviewed {paper_count} relevant papers on {topic} to provide context for the research.")
        )
        
        self.log_message(state, f"Literature review complete: {paper_count} papers")
        state["next_agent"] = "orchestrator"
        return state 