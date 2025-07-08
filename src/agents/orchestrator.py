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
        
        # Build detailed context about available agents
        agent_capabilities = """
AVAILABLE SPECIALIST AGENTS:

1. DATA_GATHERING AGENT:
   - Purpose: Collect observational data from astronomical databases
   - Capabilities: Access DESI spectroscopic data, LSST imaging, CMB maps from ACT/Planck
   - Use when: You need to retrieve existing observational data or measurements
   - MCP Tools: desi-server, lsst-server, cmb-server

2. ANALYSIS AGENT:
   - Purpose: Perform statistical analysis and data processing
   - Capabilities: Correlation studies, clustering analysis, power spectrum calculations, parameter fitting
   - Use when: You have data and need to analyze it statistically
   - MCP Tools: statistics-server, correlation-server, power-spectrum-server

3. THEORIST_SIMULATION AGENT:
   - Purpose: Run theoretical simulations and cosmological modeling
   - Capabilities: N-body simulations, dark matter halo modeling, cosmological predictions, matter power spectra
   - Use when: You need to run simulations, predict theoretical results, or model cosmological scenarios
   - MCP Tools: nbody-server, camb-server

4. LITERATURE_REVIEWER AGENT:
   - Purpose: Search and synthesize scientific literature
   - Capabilities: ArXiv paper search, citation analysis, research context generation
   - Use when: You need background research or want to understand existing work
   - MCP Tools: arxiv-server, scholarly-server
"""
        
        # Build context from current state
        current_progress = f"""
CURRENT RESEARCH STATE:

Task: {state.get('current_task', 'No task specified')}

Progress so far:
- Data gathered: {len(state.get('data_artifacts', {}))} datasets
- Analysis completed: {len(state.get('analysis_results', {}))} results  
- Literature reviewed: {len(state.get('literature_context', {}))} papers
- Simulations run: {len(state.get('simulation_outputs', {}))} simulations

Recent conversation:
{self._format_recent_messages(state.get('messages', []))}
"""

        prompt = f"""You are the Orchestrator for an astronomy research system. Your job is to analyze the user's request and decide which specialist agent should handle it.

{agent_capabilities}

{current_progress}

INSTRUCTIONS:
1. Carefully analyze the user's request
2. Determine which agent is best suited for this specific task
3. Consider what has already been done vs what still needs to be done
4. Provide clear reasoning for your decision

CRITICAL: Return ONLY valid JSON in your response. No explanation text before or after.

Return your response as JSON:
{{
    "reasoning": "Detailed explanation of your thought process and why you chose this agent",
    "next_agent": "agent_name" or null (if task is complete),
    "instructions": "Specific instructions for the chosen agent",
    "is_complete": boolean,
    "summary": "Brief user-friendly explanation of what you're doing next"
}}

ROUTING EXAMPLES:
- "run simulation" or "N-body" or "predict" → theorist_simulation
- "get data" or "download" or "access database" → data_gathering  
- "analyze data" or "statistics" or "correlation" → analysis
- "search papers" or "literature review" → literature_reviewer"""

        messages = [
            SystemMessage(content=prompt)
        ]
        
        # Add only the most recent user message for decision making
        if state.get("messages"):
            latest_human_msg = None
            for msg in reversed(state["messages"]):
                if hasattr(msg, 'content') and not isinstance(msg, AIMessage):
                    latest_human_msg = msg
                    break
            if latest_human_msg:
                messages.append(latest_human_msg)
        
        self.log_message(state, "Making routing decision...")
        
        try:
            response = await self.llm.ainvoke(messages)
            
            # Debug: Print the complete response
            print(f"\n=== DEBUGGING LLM RESPONSE ===")
            print(f"Response type: {type(response)}")
            print(f"Response content type: {type(response.content)}")
            print(f"Response content length: {len(response.content) if response.content else 'None'}")
            print(f"Response content (full): '{response.content}'")
            print(f"Response content repr: {repr(response.content)}")
            print(f"=== END DEBUG ===\n")
            
            # Log the raw LLM response for debugging
            self.log_message(state, f"LLM Response Length: {len(response.content) if response.content else 'None'}")
            self.log_message(state, f"LLM Response: {response.content}")
            
            if not response.content or not response.content.strip():
                raise ValueError("Empty response from LLM")
            
            decision = json.loads(response.content.strip())
            
            # Validate the decision
            if not isinstance(decision, dict):
                raise ValueError("Response is not a JSON object")
                
            required_fields = ["reasoning", "next_agent", "summary"]
            for field in required_fields:
                if field not in decision:
                    decision[field] = f"Missing {field}"
            
            # Show the orchestrator's reasoning
            reasoning = decision.get("reasoning", "No reasoning provided")
            summary = decision.get("summary", "Processing request...")
            
            # Create a detailed response that shows the thinking
            orchestrator_response = f"{summary}\n\nMy reasoning: {reasoning}"
            
            # Add orchestrator's response to conversation history
            state["messages"].append(AIMessage(content=orchestrator_response))
            
            # Log the decision
            next_agent = decision.get("next_agent")
            self.log_message(state, f"Decision: Route to {next_agent} - {reasoning[:100]}...")
            
            if decision.get("is_complete", False):
                # Final summary
                state["final_response"] = orchestrator_response
                state["next_agent"] = None  # Routes to END
            else:
                # Route to specialist
                state["next_agent"] = next_agent
                state["current_task"] = decision.get("instructions", state["current_task"])
                
        except json.JSONDecodeError as e:
            # Better error handling
            error_msg = f"I had trouble parsing my decision (JSON error: {str(e)}). Let me try a different approach."
            self.log_message(state, f"JSON parsing failed: {str(e)}")
            
            # Try to make a simple decision based on keywords
            current_task = state.get("current_task", "").lower()
            if any(word in current_task for word in ["simulation", "simulate", "nbody", "n-body", "predict", "model", "theoretical"]):
                next_agent = "theorist_simulation"
                reasoning = "Detected simulation/modeling keywords in request"
            elif any(word in current_task for word in ["data", "get", "download", "access", "desi", "lsst"]):
                next_agent = "data_gathering"
                reasoning = "Detected data access keywords in request"
            elif any(word in current_task for word in ["analyze", "analysis", "statistics", "correlation"]):
                next_agent = "analysis"
                reasoning = "Detected analysis keywords in request"
            elif any(word in current_task for word in ["paper", "literature", "search", "arxiv"]):
                next_agent = "literature_reviewer"
                reasoning = "Detected literature search keywords in request"
            else:
                next_agent = "data_gathering"
                reasoning = "Default fallback - starting with data gathering"
            
            fallback_response = f"{error_msg}\n\nBased on keywords in your request, I'll route to the {next_agent} agent. {reasoning}"
            state["messages"].append(AIMessage(content=fallback_response))
            state["next_agent"] = next_agent
                
        except Exception as e:
            # Handle other errors
            error_msg = f"I encountered an error while making my decision: {str(e)}"
            self.log_message(state, f"Orchestrator error: {str(e)}")
            state["messages"].append(AIMessage(content=error_msg))
            state["next_agent"] = "data_gathering"  # Safe fallback
                
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