"""Agent state definitions for the multi-agent astronomy research system.

This module defines the state structures that flow between agents, maintaining
a complete audit trail of research activities while keeping large datasets as
file references rather than in-memory objects.
"""

from typing import TypedDict, List, Dict, Any, Optional
from langchain.schema import BaseMessage
from datetime import datetime


class ToolCall(TypedDict):
    """Record of a tool call made by an agent."""
    timestamp: str
    agent: str  # Which agent made the call
    mcp_server: str  # Which MCP server was called
    tool: str  # Tool name
    arguments: Dict[str, Any]  # Tool arguments
    duration_ms: Optional[float]  # How long the call took


class ToolResult(TypedDict):
    """Result from a tool call (metadata only for large data)."""
    status: str  # 'success' or 'error'
    error: Optional[str]  # Error message if failed
    data_saved: Optional[Dict[str, Any]]  # File metadata if data was saved
    summary: Optional[str]  # Brief summary of results
    preview: Optional[Any]  # Small preview of data (not full dataset)


class AgentAction(TypedDict):
    """Complete record of an agent action."""
    timestamp: str
    agent: str
    action_type: str  # 'tool_call', 'analysis', 'decision', 'message'
    tool_call: Optional[ToolCall]
    tool_result: Optional[ToolResult]
    message: Optional[str]  # What the agent communicated


class AgentState(TypedDict):
    """Shared state passed between agents with full audit trail.
    
    This state maintains a complete history of the research process including:
    - Full conversation history (messages)
    - All agent actions and tool calls (action_log)
    - File references without storing large datasets
    - Metadata about saved data for later retrieval
    """
    # Conversation history
    messages: List[BaseMessage]  # Full conversation history
    
    # Action audit trail
    action_log: List[AgentAction]  # Complete log of all agent actions
    
    # Current task context
    current_task: str  # Current instructions for the active agent
    task_breakdown: List[Dict[str, Any]]  # Optional task decomposition
    
    # Data references (not actual data)
    data_artifacts: Dict[str, Dict[str, Any]]  # File metadata from data gathering
    analysis_results: Dict[str, Dict[str, Any]]  # Analysis output metadata
    literature_context: Dict[str, List[str]]  # Paper references/citations
    simulation_outputs: Dict[str, Dict[str, Any]]  # Simulation file metadata
    
    # Workflow control
    next_agent: Optional[str]  # Where to route next
    final_response: Optional[str]  # Final answer when complete
    human_feedback: List[Dict[str, Any]]  # Track all human inputs
    
    # Additional context
    metadata: Dict[str, Any]  # User-provided context/parameters
    start_time: str  # When the research started
    total_tool_calls: int  # Counter for tool calls made


def create_initial_state(
    user_query: str, 
    context: Optional[Dict[str, Any]] = None
) -> AgentState:
    """Create initial state for a new research session."""
    from langchain.schema import HumanMessage
    
    return AgentState(
        messages=[HumanMessage(content=user_query)],
        action_log=[],
        current_task=user_query,
        task_breakdown=[],
        data_artifacts={},
        analysis_results={},
        literature_context={},
        simulation_outputs={},
        next_agent=None,
        final_response=None,
        human_feedback=[],
        metadata=context or {},
        start_time=datetime.now().isoformat(),
        total_tool_calls=0
    )


def log_agent_action(
    state: AgentState,
    agent_name: str,
    action_type: str,
    message: Optional[str] = None,
    tool_call: Optional[ToolCall] = None,
    tool_result: Optional[ToolResult] = None
) -> None:
    """Helper function to log an agent action to the state."""
    action = AgentAction(
        timestamp=datetime.now().isoformat(),
        agent=agent_name,
        action_type=action_type,
        tool_call=tool_call,
        tool_result=tool_result,
        message=message
    )
    state["action_log"].append(action)


def get_research_summary(state: AgentState) -> Dict[str, Any]:
    """Generate a summary of the research session from the state."""
    return {
        "start_time": state["start_time"],
        "total_actions": len(state["action_log"]),
        "total_tool_calls": state["total_tool_calls"],
        "data_files_gathered": len(state["data_artifacts"]),
        "analyses_completed": len(state["analysis_results"]),
        "papers_reviewed": sum(len(papers) for papers in state["literature_context"].values()),
        "simulations_run": len(state["simulation_outputs"]),
        "current_status": "complete" if state.get("final_response") else "in_progress"
    } 