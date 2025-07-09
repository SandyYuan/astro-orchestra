"""LangGraph workflow builder for the multi-agent astronomy research system."""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.state.agent_state import AgentState
from src.agents.orchestrator import OrchestratorAgent
from src.agents.planning import PlanningAgent
from src.agents.data_gathering import DataGatheringAgent
from src.agents.analysis import AnalysisAgent
from src.agents.theorist_simulation import TheoristSimulationAgent
from src.agents.literature_reviewer import LiteratureReviewerAgent
from config.tool_configs import MCP_TOOL_SERVERS
from typing import Dict, Any


async def build_astronomy_graph():
    """Build the LangGraph workflow for the multi-agent system."""
    
    # Initialize agents
    orchestrator = OrchestratorAgent()
    planning_agent = PlanningAgent()
    data_agent = DataGatheringAgent()
    analysis_agent = AnalysisAgent()
    simulation_agent = TheoristSimulationAgent()
    literature_agent = LiteratureReviewerAgent()
    
    # Initialize MCP connections for each agent
    agents = [orchestrator, planning_agent, data_agent, analysis_agent, simulation_agent, literature_agent]
    for agent in agents:
        try:
            await agent.initialize_mcp_clients(MCP_TOOL_SERVERS)
        except Exception as e:
            print(f"Warning: Failed to initialize MCP clients for {agent.name}: {e}")
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes with error handling wrappers
    def safe_process(agent):
        """Create a safe wrapper for agent processing."""
        async def process(state):
            try:
                return await agent.process(state)
            except Exception as e:
                # Log error and route back to orchestrator
                error_msg = f"Error in {agent.name}: {str(e)}"
                state["messages"].append(f"System error: {error_msg}")
                state["action_log"].append({
                    "timestamp": "error",
                    "agent": agent.name,
                    "action_type": "error",
                    "tool_call": None,
                    "tool_result": None,
                    "message": error_msg
                })
                state["next_agent"] = "orchestrator"
                return state
        return process
    
    workflow.add_node("orchestrator", safe_process(orchestrator))
    workflow.add_node("planning", safe_process(planning_agent))
    workflow.add_node("data_gathering", safe_process(data_agent))
    workflow.add_node("analysis", safe_process(analysis_agent))
    workflow.add_node("theorist_simulation", safe_process(simulation_agent))
    workflow.add_node("literature_reviewer", safe_process(literature_agent))
    
    # Define routing logic
    def route_from_orchestrator(state: AgentState) -> str:
        """Route based on orchestrator's decision."""
        next_agent = state.get("next_agent")
        if next_agent and next_agent in ["planning", "data_gathering", "analysis", "theorist_simulation", "literature_reviewer"]:
            return next_agent
        return END
    
    # Add edges
    workflow.set_entry_point("orchestrator")
    workflow.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator
    )
    
    # Add edges back to orchestrator from each specialist
    for agent_name in ["planning", "data_gathering", "analysis", "theorist_simulation", "literature_reviewer"]:
        workflow.add_edge(agent_name, "orchestrator")
    
    # Add before compile:
    checkpointer = MemorySaver()
    
    # Compile the workflow
    compiled_workflow = workflow.compile(checkpointer=checkpointer)
    
    # Store agents for cleanup
    compiled_workflow._agents = agents
    
    return compiled_workflow


async def cleanup_workflow(workflow):
    """Clean up MCP connections for all agents."""
    if hasattr(workflow, '_agents'):
        for agent in workflow._agents:
            try:
                await agent.cleanup()
            except Exception as e:
                print(f"Warning: Error cleaning up {agent.name}: {e}")


async def create_workflow_runner():
    """Create a workflow runner that handles initialization and cleanup."""
    return await build_astronomy_graph()

# Server can send progress notifications during tool execution
async def handle_call_tool(self, name: str, arguments: Dict[str, Any]):
    # Start tool execution
    await self.send_progress("Starting tool execution...")
    
    # During execution
    await self.send_progress("50% complete...")
    
    # Return final result
    return final_result 