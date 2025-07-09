"""Orchestrator agent that coordinates the multi-agent astronomy research workflow."""

from typing import Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from src.agents.base import BaseAgent
from src.state.agent_state import AgentState
from src.models import OrchestratorDecision
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
        """Process state - pause after specialist completion or route to next agent."""
        
        # Check if a specialist just completed
        last_action = state["action_log"][-1] if state["action_log"] else None
        
        if (last_action and 
            last_action["agent"] != "orchestrator" and 
            last_action.get("tool_result")):
            
            # Specialist completed - format results and pause
            agent_name = last_action["agent"]
            
            # Build result summary
            summary = f"## {agent_name.replace('_', ' ').title()} Complete\n\n"
            
            # Add specific results based on agent type
            if agent_name == "data_gathering" and state.get("data_artifacts"):
                for key, file_info in state["data_artifacts"].items():
                    summary += f"- Saved: {file_info['filename']} ({file_info['size_bytes']:,} bytes)\n"
                    if 'total_records' in file_info:
                        summary += f"  Records: {file_info['total_records']}\n"
            
            elif agent_name == "analysis" and state.get("analysis_results"):
                for key, result in state["analysis_results"].items():
                    summary += f"- Analysis: {result.get('description', key)}\n"
                    if 'summary' in result:
                        summary += f"  Result: {result['summary']}\n"
            
            # Add similar handling for other agents...
            
            summary += "\nProvide instructions for next steps:"
            
            # Update state and pause
            state["messages"].append(AIMessage(content=summary))
            state["next_agent"] = None  # This triggers pause
            return state
        
        # Not paused - determine next agent
        # Include human feedback in decision
        human_feedback = state.get("human_feedback", [])
        latest_feedback = human_feedback[-1]["content"] if human_feedback else None
        
        context = f"""Current research state:
Task: {state.get('current_task')}
Data files: {len(state.get('data_artifacts', {}))}
Analyses: {len(state.get('analysis_results', {}))}
Recent feedback: {latest_feedback}

Determine next agent (data_gathering, analysis, theorist_simulation, literature_reviewer) or null if complete.
Return JSON: {{"next_agent": "...", "instructions": "...", "reasoning": "..."}}"""
        
        messages = [SystemMessage(content=context)]
        response = await self.llm.ainvoke(messages)
        
        try:
            decision = json.loads(response.content)
            state["next_agent"] = decision.get("next_agent")
            if decision.get("next_agent"):
                state["current_task"] = decision.get("instructions", state["current_task"])
                state["messages"].append(AIMessage(content=decision["reasoning"]))
        except:
            state["next_agent"] = "data_gathering"  # Default
        
        return state
    
    def _format_recent_messages(self, messages: List) -> str:
        """Format the most recent messages for context."""
        if not messages:
            return "No previous conversation"
        
        # Get last 3 messages
        recent = messages[-3:] if len(messages) > 3 else messages
        formatted = []
        
        for msg in recent:
            if hasattr(msg, 'content'):
                role = "Human" if not isinstance(msg, AIMessage) else "Assistant"
                content = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
                formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted) 