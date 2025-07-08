"""LangGraph workflow builder for the multi-agent astronomy research system."""

from langgraph.graph import StateGraph, END
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
    async def safe_process(agent):
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
    
    workflow.add_node("orchestrator", await safe_process(orchestrator))
    workflow.add_node("planning", await safe_process(planning_agent))
    workflow.add_node("data_gathering", await safe_process(data_agent))
    workflow.add_node("analysis", await safe_process(analysis_agent))
    workflow.add_node("theorist_simulation", await safe_process(simulation_agent))
    workflow.add_node("literature_reviewer", await safe_process(literature_agent))
    
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
        route_from_orchestrator,
        {
            "planning": "planning",
            "data_gathering": "data_gathering",
            "analysis": "analysis",
            "theorist_simulation": "theorist_simulation",
            "literature_reviewer": "literature_reviewer",
            END: END
        }
    )
    
    # Add edges back to orchestrator from each specialist
    for agent_name in ["planning", "data_gathering", "analysis", "theorist_simulation", "literature_reviewer"]:
        workflow.add_edge(agent_name, "orchestrator")
    
    # Compile the workflow
    compiled_workflow = workflow.compile()
    
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


def create_workflow_runner():
    """Create a workflow runner that handles initialization and cleanup."""
    
    class WorkflowRunner:
        def __init__(self):
            self.workflow = None
        
        async def __aenter__(self):
            self.workflow = await build_astronomy_graph()
            return self.workflow
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            if self.workflow:
                await cleanup_workflow(self.workflow)
    
    return WorkflowRunner() 