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
        
        # Check if we just came from a specialist agent (regardless of tool_result)
        if (last_action and 
            last_action["agent"] != "orchestrator" and 
            last_action["agent"] in ["data_gathering", "analysis", "theorist_simulation", "literature_reviewer"]):
            
            # Specialist completed - format results and pause for human review
            agent_name = last_action["agent"]
            
            # Build comprehensive result summary
            summary = f"## {agent_name.replace('_', ' ').title()} Agent Results\n\n"
            
            # Add specific results based on agent type
            if agent_name == "data_gathering":
                if state.get("data_artifacts"):
                    summary += "**Data Gathered:**\n"
                    for key, file_info in state["data_artifacts"].items():
                        summary += f"- {file_info.get('filename', key)}"
                        if 'size_bytes' in file_info:
                            summary += f" ({file_info['size_bytes']:,} bytes)"
                        if 'total_records' in file_info:
                            summary += f" - {file_info['total_records']} records"
                        summary += f"\n  Description: {file_info.get('description', 'N/A')}\n"
                else:
                    summary += "**Status:** No data artifacts created (development mode)\n"
            
            elif agent_name == "analysis":
                if state.get("analysis_results"):
                    summary += "**Analysis Results:**\n"
                    for key, result in state["analysis_results"].items():
                        summary += f"- {result.get('description', key)}\n"
                        if 'summary' in result:
                            summary += f"  Result: {result['summary']}\n"
                else:
                    summary += "**Status:** No analysis results yet\n"
            
            elif agent_name == "theorist_simulation":
                if state.get("simulation_outputs"):
                    summary += "**Simulation Results:**\n"
                    for key, sim in state["simulation_outputs"].items():
                        summary += f"- {sim.get('description', key)}\n"
                        if 'summary' in sim:
                            summary += f"  Result: {sim['summary']}\n"
                else:
                    summary += "**Status:** No simulation outputs yet\n"
            
            elif agent_name == "literature_reviewer":
                if state.get("literature_context"):
                    summary += "**Literature Found:**\n"
                    for key, lit in state["literature_context"].items():
                        summary += f"- {lit.get('title', key)}\n"
                        if 'summary' in lit:
                            summary += f"  Summary: {lit['summary']}\n"
                else:
                    summary += "**Status:** No literature context yet\n"
            
            # Add session info
            summary += f"\n**Session:** {state.get('metadata', {}).get('session_id', 'unknown')}\n"
            summary += f"**Total Steps:** {len(state.get('action_log', []))}\n\n"
            
            # Request human feedback
            summary += "**Next Steps:** Please provide feedback or instructions for continuing the research.\n\n"
            summary += "Options:\n"
            summary += "- Request more data gathering\n"
            summary += "- Proceed to analysis\n"
            summary += "- Run simulations\n"
            summary += "- Search literature\n"
            summary += "- Complete the research\n"
            
            # Update state and pause for human input
            state["messages"].append(AIMessage(content=summary))
            state["next_agent"] = None  # This triggers pause - system waits for human feedback
            return state
        
        # Not paused - determine next agent based on current state and human feedback
        # Get the most recent human message to understand their intent
        latest_human_msg = None
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                latest_human_msg = msg.content
                break
        
        # Check if this is initial task routing or follow-up after human feedback
        data_count = len(state.get("data_artifacts", {}))
        analysis_count = len(state.get("analysis_results", {}))
        simulation_count = len(state.get("simulation_outputs", {}))
        literature_count = len(state.get("literature_context", {}))
        
        system_prompt = """You are an orchestrator for astronomy research. Based on the current research state and human input, determine the next specialist agent to route to.

Available agents:
- data_gathering: Access astronomy databases (DESI, LSST, CMB)
- analysis: Statistical analysis, correlations, power spectra  
- theorist_simulation: N-body simulations, cosmological modeling
- literature_reviewer: Paper search, citation analysis

Return JSON: {"next_agent": "agent_name", "instructions": "specific task", "reasoning": "why this agent"}

If the research is complete or the human wants to end, return: {"next_agent": null, "instructions": null, "reasoning": "explanation"}"""
        
        user_query = f"""Current research state:
Human's request: {latest_human_msg}
Data files: {data_count}
Analysis results: {analysis_count}
Simulation outputs: {simulation_count}
Literature items: {literature_count}

Based on the human's request, what should be the next step?"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query)
        ]
        response = await self.llm.ainvoke(messages)
        
        try:
            # Strip markdown code block formatting if present
            response_content = response.content.strip()
            if response_content.startswith("```json"):
                response_content = response_content[7:]  # Remove ```json
            if response_content.startswith("```"):
                response_content = response_content[3:]   # Remove ```
            if response_content.endswith("```"):
                response_content = response_content[:-3]  # Remove closing ```
            response_content = response_content.strip()
            
            decision = json.loads(response_content)
            next_agent = decision.get("next_agent")
            
            # Debug output - show the LLM's reasoning
            print(f"\n--- ORCHESTRATOR DEBUG ---")
            print(f"User request: {latest_human_msg}")
            print(f"LLM response: {response.content}")
            print(f"Cleaned response: {response_content}")
            print(f"Parsed next_agent: {next_agent}")
            print(f"Parsed reasoning: {decision.get('reasoning', 'No reasoning provided')}")
            print("--- END DEBUG ---\n")
            
            state["next_agent"] = next_agent
            
            # Create response
            reasoning = decision.get("reasoning", "Routing to specialist agent")
            instructions = decision.get("instructions", latest_human_msg)
            
            if next_agent:
                state["current_task"] = instructions
                response_msg = f"I'll route this to the **{next_agent.replace('_', ' ').title()} Agent**.\n\n"
                response_msg += f"**Task**: {instructions}\n\n"
                response_msg += f"**Reasoning**: {reasoning}"
                state["messages"].append(AIMessage(content=response_msg))
            else:
                state["messages"].append(AIMessage(content=f"Research complete. {reasoning}"))
                
        except Exception as e:
            # Debug output for errors
            print(f"\n--- ORCHESTRATOR ERROR DEBUG ---")
            print(f"User request: {latest_human_msg}")
            print(f"LLM raw response: {response.content}")
            print(f"Error: {str(e)}")
            print("--- END ERROR DEBUG ---\n")
            
            # Simple response when tools aren't available or routing fails
            state["messages"].append(AIMessage(content=f"I don't have the appropriate tools to handle this request: {latest_human_msg}"))
            state["next_agent"] = None  # Pause for human to provide different request
        
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