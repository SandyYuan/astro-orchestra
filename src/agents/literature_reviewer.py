"""Literature review agent for searching and synthesizing scientific papers."""

from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AIMessage, HumanMessage
from src.agents.base import BaseAgent
from src.state.agent_state import AgentState
from config.settings import settings
import json


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
        
        # Single LLM call to determine both: sufficient info + fast track eligibility
        decision = await self._analyze_task_requirements(current_task)
        
        if not decision["has_sufficient_info"]:
            # Need more info - set fast track if applicable
            if decision["should_fast_track"]:
                state["fast_track"] = "literature_reviewer"
            
            # Request more information
            clarification_msg = decision["clarification_message"]
            state["messages"].append(AIMessage(content=clarification_msg))
            state["next_agent"] = None
            self.log_message(state, "Requested more literature search parameters from user")
            return state
        
        # We have sufficient parameters - proceed with literature review
        # Placeholder literature review - in real implementation, would use MCP tools
        topic = "astronomy_research"
        state["literature_context"][topic] = [
            "arXiv:2301.12345 - Dark matter halo analysis",
            "arXiv:2302.67890 - Galaxy formation simulations",
            "arXiv:2303.11111 - DESI spectroscopic survey results"
        ]
        
        paper_count = len(state["literature_context"][topic])
        state["messages"].append(
            AIMessage(content=f"Reviewed {paper_count} relevant papers on {topic} with parameters: {current_task}")
        )
        
        self.log_message(state, f"Literature review complete: {paper_count} papers")
        state["next_agent"] = "orchestrator"
        return state
    
    async def _analyze_task_requirements(self, task: str) -> dict:
        """Single LLM call to analyze task and determine next steps."""
        prompt = f"""
        Analyze this literature review task: "{task}"
        
        Determine:
        1. Do I have sufficient information to perform a meaningful literature search?
        2. If not, is this a straightforward literature request that should fast-track back to me?
        
        For sufficient information, I need specifics like: research topic, specific keywords, author names, time periods, journal preferences, or specific questions to answer.
        
        Fast track if:
        - This is clearly a literature search request, just missing specific details
        - User will likely provide specifics and want literature review to proceed
        
        Don't fast track if:
        - Request is ambiguous about what type of work to do
        - User might want to change direction entirely
        - Unclear if they want literature review vs data gathering vs analysis
        
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
                "clarification_message": "I need more specific information about what literature to search. Please specify: research topic, keywords, author names, time period, or specific questions to answer."
            } 