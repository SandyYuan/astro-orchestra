"""Orchestrator agent that coordinates the multi-agent astronomy research workflow."""

from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from src.agents.base import BaseAgent
from src.state.agent_state import AgentState
from config.settings import settings
import json


class OrchestratorAgent(BaseAgent):
    """Main orchestrator that breaks down tasks and delegates to specialists."""
    
    def __init__(self, llm: ChatOpenAI = None):
        super().__init__(
            name="orchestrator",
            mcp_tools=[],  # Orchestrator doesn't use external tools directly
            description="Breaks down research tasks and delegates to specialist agents"
        )
        self.llm = llm or ChatOpenAI(
            model=settings.orchestrator_model, 
            temperature=0
        )
        
    async def process(self, state: AgentState) -> AgentState:
        """Analyze current state and decide what to do next."""
        
        self.log_message(state, "Orchestrator analyzing current state and planning next action")
        
        # Build context from current state
        context = self._build_context_prompt(state)
        
        messages = [
            SystemMessage(content=context),
            *state["messages"]
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            decision = self._parse_decision(response.content)
            
            # Update state based on decision
            if decision.get("is_complete", False):
                # Research is complete
                final_message = decision.get("summary", "Research complete. Here's what I found...")
                final_response = self._generate_final_response(state, decision)
                
                state["messages"].append(AIMessage(content=final_response))
                state["final_response"] = final_response
                state["next_agent"] = None  # Routes to END
                
                self.log_message(state, "Research workflow complete")
                
            else:
                # Continue with next agent
                update_message = decision.get("summary", f"Routing to {decision['next_agent']}...")
                state["messages"].append(AIMessage(content=update_message))
                state["next_agent"] = decision["next_agent"]
                state["current_task"] = decision.get("instructions", state["current_task"])
                
                self.log_message(
                    state, 
                    f"Routing to {decision['next_agent']} with task: {decision.get('instructions', 'continue current task')}"
                )
                
        except Exception as e:
            # Fallback behavior if LLM fails
            self.log_message(state, f"Error in orchestrator decision making: {str(e)}")
            
            # Default to data gathering if we haven't started yet
            if not state.get("data_artifacts") and not state.get("analysis_results"):
                state["messages"].append(
                    AIMessage(content="I'll start by gathering some initial data to get started.")
                )
                state["next_agent"] = "data_gathering"
            else:
                # Default to completing the workflow
                state["messages"].append(
                    AIMessage(content="Let me summarize what we've found so far.")
                )
                state["final_response"] = self._generate_final_response(state)
                state["next_agent"] = None
                
        return state
    
    def _build_context_prompt(self, state: AgentState) -> str:
        """Build the context prompt for the orchestrator's decision making."""
        
        task = state.get('current_task', 'No task specified')
        
        # Count progress
        data_count = len(state.get('data_artifacts', {}))
        analysis_count = len(state.get('analysis_results', {}))
        literature_count = sum(len(papers) for papers in state.get('literature_context', {}).values())
        simulation_count = len(state.get('simulation_outputs', {}))
        
        # Recent actions summary
        recent_actions = state.get('action_log', [])[-3:]  # Last 3 actions
        actions_summary = ""
        if recent_actions:
            actions_summary = "Recent actions:\n"
            for action in recent_actions:
                actions_summary += f"- {action['agent']}: {action.get('message', 'No message')}\n"
        
        context = f"""You are the orchestrator for Astro Orchestra, a multi-agent astronomy research system.

Current research task: {task}

Progress summary:
- Data gathered: {data_count} datasets
- Analysis completed: {analysis_count} results  
- Literature reviewed: {literature_count} papers
- Simulations run: {simulation_count} simulations
- Total actions taken: {len(state.get('action_log', []))}

{actions_summary}

Available specialist agents:
- data_gathering: Access astronomical databases (DESI, LSST, CMB)
- analysis: Statistical analysis and computations  
- theorist_simulation: Run cosmological simulations
- literature_reviewer: Search and synthesize scientific papers

Based on the current state, decide the next action. Return JSON with this exact structure:
{{
    "next_agent": "agent_name" or null (if research is complete),
    "instructions": "specific instructions for the next agent",
    "is_complete": boolean,
    "summary": "brief update for the user about what you're doing",
    "reasoning": "explanation of why you chose this action"
}}

Guidelines:
1. Start with data gathering if no data has been collected
2. Move to analysis once relevant data is available
3. Consider literature review to provide context
4. Use simulations for theoretical questions
5. Complete when sufficient progress has been made to answer the original question"""

        return context
    
    def _parse_decision(self, response_content: str) -> Dict[str, Any]:
        """Parse the LLM's decision response."""
        try:
            # Try to extract JSON from the response
            decision = json.loads(response_content)
            
            # Validate required fields
            if "next_agent" not in decision:
                decision["next_agent"] = "data_gathering"
            if "is_complete" not in decision:
                decision["is_complete"] = False
            if "summary" not in decision:
                decision["summary"] = f"Proceeding with {decision['next_agent']}"
                
            return decision
            
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "next_agent": "data_gathering",
                "instructions": "Begin data gathering",
                "is_complete": False,
                "summary": "Starting research workflow",
                "reasoning": "JSON parsing failed, using default action"
            }
    
    def _generate_final_response(self, state: AgentState, decision: Dict[str, Any] = None) -> str:
        """Generate a comprehensive final response summarizing the research."""
        
        # Start with the summary from decision if available
        if decision and decision.get("summary"):
            response = decision["summary"] + "\n\n"
        else:
            response = "Research completed. Here's a summary of what was accomplished:\n\n"
        
        # Add progress summary
        data_artifacts = state.get("data_artifacts", {})
        analysis_results = state.get("analysis_results", {})
        literature_context = state.get("literature_context", {})
        simulation_outputs = state.get("simulation_outputs", {})
        
        if data_artifacts:
            response += f"**Data Gathering ({len(data_artifacts)} datasets):**\n"
            for key, artifact in data_artifacts.items():
                response += f"- {artifact.get('filename', key)}: {artifact.get('description', 'Data file')}\n"
            response += "\n"
        
        if analysis_results:
            response += f"**Analysis Results ({len(analysis_results)} analyses):**\n"
            for key, result in analysis_results.items():
                response += f"- {result.get('description', key)}: {result.get('summary', 'Analysis completed')}\n"
            response += "\n"
        
        if literature_context:
            paper_count = sum(len(papers) for papers in literature_context.values())
            response += f"**Literature Review ({paper_count} papers):**\n"
            for topic, papers in literature_context.items():
                response += f"- {topic}: {len(papers)} papers reviewed\n"
            response += "\n"
        
        if simulation_outputs:
            response += f"**Simulations ({len(simulation_outputs)} runs):**\n"
            for key, sim in simulation_outputs.items():
                response += f"- {sim.get('description', key)}: {sim.get('summary', 'Simulation completed')}\n"
            response += "\n"
        
        # Add workflow statistics
        total_actions = len(state.get("action_log", []))
        total_tool_calls = state.get("total_tool_calls", 0)
        start_time = state.get("start_time", "unknown")
        
        response += f"**Workflow Statistics:**\n"
        response += f"- Total actions: {total_actions}\n"
        response += f"- Tool calls made: {total_tool_calls}\n"
        response += f"- Started: {start_time}\n"
        
        return response 