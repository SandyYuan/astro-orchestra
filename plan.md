# Multi-Agent Astronomy Research System - Python Scaffolding

## Quick Start

1. **Clone and Setup**:
```bash
git init astronomy-research-agent
cd astronomy-research-agent
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

2. **Configure Environment**:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. **Run MCP Server**:
```bash
python -m src.mcp.server
```

4. **Configure Cursor**:
Add to `.cursor/mcp_config.json` in your workspace.

## Project Structure

```
astronomy-research-agent/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .env.example
├── mcp_config.json
├── config/
│   ├── __init__.py
│   ├── settings.py
│   └── tool_configs.py
├── src/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── orchestrator.py
│   │   ├── data_gathering.py
│   │   ├── analysis.py
│   │   ├── theorist_simulation.py
│   │   └── literature_reviewer.py
│   ├── state/
│   │   ├── __init__.py
│   │   └── agent_state.py
│   ├── workflow/
│   │   ├── __init__.py
│   │   ├── graph_builder.py
│   │   └── routing.py
│   ├── mcp/
│   │   ├── __init__.py
│   │   └── server.py
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       └── data_formats.py
├── tests/
│   ├── __init__.py
│   ├── test_agents/
│   ├── test_tools/
│   ├── test_workflow/
│   └── test_mcp/
└── examples/
    ├── basic_research_task.py
    └── complex_workflow.py
```

## Architecture Overview

### How MCP Integration Works

1. **External Tool Servers**: All astronomy tools (DESI, LSST, etc.) run as separate MCP servers
2. **Agents as MCP Clients**: Each specialist agent connects to relevant MCP servers to access tools
3. **Dynamic Tool Discovery**: Agents query MCP servers for available tools and their schemas
4. **LLM-Driven Tool Selection**: Agents use LLMs to decide which tools to call based on the task

### Improved Workflow with Planning Phase

```
High-Level Research Idea → Orchestrator Agent → Planning Agent
                                                      ↓
                                              Detailed Research Plan
                                                      ↓
                                            Back to Orchestrator
                                                      ↓
                                              Specialist Agents
                                                      ↓
                                             Query MCP Servers
                                             for Available Tools
                                                      ↓
                                             LLM Decides Which
                                             Tools to Call
                                                      ↓
                                             Execute Tool Calls
                                             via MCP Protocol
                                                      ↓
                                             Process Results
                                                      ↓
                                            Return to Orchestrator
```

The system now intelligently distinguishes between:
- **High-level research ideas** (e.g., "I want to model DESI LRG BAO measurements") → Planning Agent
- **Specific tasks** (e.g., "Download DESI galaxy data") → Direct to specialist agents

## Core Component Implementations

### 1. Base Agent Class (`src/agents/base.py`)

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from langchain.schema import BaseMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from datetime import datetime
import time

class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""
    
    def __init__(self, name: str, mcp_tools: List[str], description: str):
        self.name = name
        self.mcp_tools = mcp_tools  # List of MCP server names this agent can use
        self.description = description
        self.mcp_clients = {}  # Will store active MCP client sessions
    
    async def initialize_mcp_clients(self, mcp_configs: Dict[str, Dict[str, Any]]):
        """Initialize MCP client connections for this agent's tools."""
        for tool_name in self.mcp_tools:
            if tool_name in mcp_configs:
                config = mcp_configs[tool_name]
                # Create MCP client for this tool
                server_params = StdioServerParameters(
                    command=config["command"],
                    args=config.get("args", []),
                    env=config.get("env", {})
                )
                client = stdio_client(server_params)
                self.mcp_clients[tool_name] = await client.__aenter__()
    
    async def call_mcp_tool(
        self, 
        server_name: str, 
        tool_name: str, 
        arguments: Dict[str, Any],
        state: Dict[str, Any]  # Pass state to log the action
    ) -> Any:
        """Call a tool on a specific MCP server and log the action."""
        if server_name not in self.mcp_clients:
            raise ValueError(f"MCP server '{server_name}' not initialized")
        
        # Record the tool call
        start_time = time.time()
        tool_call = {
            "timestamp": datetime.now().isoformat(),
            "agent": self.name,
            "mcp_server": server_name,
            "tool": tool_name,
            "arguments": arguments,
            "duration_ms": None
        }
        
        try:
            # Make the actual tool call
            client = self.mcp_clients[server_name]
            result = await client.call_tool(tool_name, arguments)
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            tool_call["duration_ms"] = duration_ms
            
            # Process result to extract metadata (not full data)
            tool_result = self._extract_result_metadata(result)
            
            # Log the complete action
            action = {
                "timestamp": datetime.now().isoformat(),
                "agent": self.name,
                "action_type": "tool_call",
                "tool_call": tool_call,
                "tool_result": tool_result,
                "message": f"Called {server_name}.{tool_name}"
            }
            state["action_log"].append(action)
            state["total_tool_calls"] += 1
            
            return result
            
        except Exception as e:
            # Log the error
            tool_result = {
                "status": "error",
                "error": str(e),
                "data_saved": None,
                "summary": None,
                "preview": None
            }
            
            action = {
                "timestamp": datetime.now().isoformat(),
                "agent": self.name,
                "action_type": "tool_call",
                "tool_call": tool_call,
                "tool_result": tool_result,
                "message": f"Error calling {server_name}.{tool_name}: {str(e)}"
            }
            state["action_log"].append(action)
            state["total_tool_calls"] += 1
            
            raise
    
    def _extract_result_metadata(self, result: Any) -> Dict[str, Any]:
        """Extract metadata from tool result without storing large datasets."""
        # This is a simplified version - in practice, you'd parse based on tool type
        metadata = {
            "status": "success",
            "error": None,
            "data_saved": None,
            "summary": None,
            "preview": None
        }
        
        # Look for common patterns in tool responses
        if isinstance(result, dict):
            # Check for file save information
            if 'save_result' in result and result['save_result'].get('status') == 'success':
                metadata['data_saved'] = {
                    'file_id': result['save_result'].get('file_id'),
                    'filename': result['save_result'].get('filename'),
                    'size_bytes': result['save_result'].get('size_bytes'),
                    'file_type': result['save_result'].get('file_type')
                }
            
            # Extract summary info
            if 'total_found' in result:
                metadata['summary'] = f"Found {result['total_found']} objects"
            elif 'num_results' in result:
                metadata['summary'] = f"Retrieved {result['num_results']} results"
            
            # Get small preview if available (not full data)
            if 'results' in result and isinstance(result['results'], list) and len(result['results']) > 0:
                metadata['preview'] = result['results'][:3]  # Just first 3 items
        
        return metadata
    
    def log_message(self, state: Dict[str, Any], message: str):
        """Log an agent message/action to the state."""
        action = {
            "timestamp": datetime.now().isoformat(),
            "agent": self.name,
            "action_type": "message",
            "tool_call": None,
            "tool_result": None,
            "message": message
        }
        state["action_log"].append(action)
    
    @abstractmethod
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the current state and return updated state."""
        pass
    
    async def cleanup(self):
        """Clean up MCP client connections."""
        for client in self.mcp_clients.values():
            await client.__aexit__(None, None, None)
```

### 2. Agent State (`src/state/agent_state.py`)

```python
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
    action_type: str  # 'tool_call', 'analysis', 'decision'
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
    
    # Additional context
    metadata: Dict[str, Any]  # User-provided context/parameters
    start_time: str  # When the research started
    total_tool_calls: int  # Counter for tool calls made
```

### 3. Planning Agent (`src/agents/planning.py`)

```python
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from src.agents.base import BaseAgent
from src.state.agent_state import AgentState
import json

class PlanningAgent(BaseAgent):
    """Agent specialized in expanding research ideas into detailed execution plans."""
    
    def __init__(self, llm: ChatGoogleGenerativeAI = None):
        super().__init__(
            name="planning",
            mcp_tools=["taskmaster-server"],
            description="Expands research ideas into detailed, actionable execution plans"
        )
        self.llm = llm or ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.1,  # Lower temperature for structured planning
            google_api_key=settings.google_api_key
        )
    
    async def process(self, state: AgentState) -> AgentState:
        """Expand a research idea into a detailed execution plan."""
        
        current_task = state.get("current_task", "")
        
        # Create detailed planning prompt
        planning_prompt = f\"\"\"You are a Research Planning Specialist for astronomy.
        
CURRENT RESEARCH IDEA: {current_task}

AVAILABLE SPECIALIST AGENTS:
1. DATA_GATHERING: Access DESI, LSST, CMB databases
2. ANALYSIS: Statistical analysis, correlations, power spectra  
3. THEORIST_SIMULATION: N-body simulations, cosmological modeling
4. LITERATURE_REVIEWER: ArXiv search, paper synthesis

Create a detailed execution plan with specific steps, dependencies, 
and expected outputs. Return structured JSON with the plan.\"\"\"

        # Process with LLM and store structured plan in state
        response = await self.llm.ainvoke([SystemMessage(content=planning_prompt)])
        
        # Parse plan and update state
        plan_data = json.loads(response.content)
        state["task_breakdown"] = plan_data.get("detailed_steps", [])
        state["metadata"]["research_plan"] = plan_data.get("research_plan", {})
        state["metadata"]["planning_complete"] = True
        
        # Route to first execution step
        next_step = plan_data.get("next_step", {})
        state["next_agent"] = next_step.get("agent", "orchestrator")
        
        return state
```

### 4. Orchestrator Agent (`src/agents/orchestrator.py`)

```python
from typing import Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from src.agents.base import BaseAgent
from src.state.agent_state import AgentState
import json

class OrchestratorAgent(BaseAgent):
    """Main orchestrator that routes between planning and specialist agents."""
    
    def __init__(self, llm: ChatGoogleGenerativeAI = None):
        super().__init__(
            name="orchestrator", 
            mcp_tools=[],  # Orchestrator doesn't use external tools directly
            description="Routes between planning and specialist agents based on task type"
        )
        self.llm = llm or ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
        
            async def process(self, state: AgentState) -> AgentState:
        """Analyze request type and route to appropriate agent."""
        
        current_task = state.get('current_task', '')
        planning_complete = state.get('metadata', {}).get('planning_complete', False)
        
        # Build context for routing decision
        context = f"""Current astronomy research state:
        
Task: {current_task}
Planning Complete: {planning_complete}

Available agents:
- planning: Expand research ideas into detailed execution plans
- data_gathering: Access DESI, LSST, CMB databases  
- analysis: Statistical analysis and computations
- theorist_simulation: Run cosmological simulations
- literature_reviewer: Search and synthesize papers

ROUTING LOGIC:
1. HIGH-LEVEL IDEAS (no planning yet) → planning
2. PLANNED PROJECTS (planning complete) → follow the plan
3. SPECIFIC TASKS → direct to specialist agents

Analyze the task and decide routing. Return JSON:
{{
    "reasoning": "why you chose this agent",
    "next_agent": "agent_name" or null (if done),
    "instructions": "specific instructions for the next agent", 
    "is_complete": boolean,
    "summary": "brief update for the user"
}}"""

        messages = [
            SystemMessage(content=context),
            *state["messages"]
        ]
        
        response = await self.llm.ainvoke(messages)
        
        try:
            decision = json.loads(response.content)
            
            # Add orchestrator's update to conversation history
            if decision.get("is_complete", False):
                # Final summary
                final_message = decision.get("summary", "Research complete. Here's what I found...")
                state["messages"].append(AIMessage(content=final_message))
                state["final_response"] = final_message
                state["next_agent"] = None  # Routes to END
            else:
                # Progress update
                update_message = decision.get("summary", f"Routing to {decision['next_agent']}...")
                state["messages"].append(AIMessage(content=update_message))
                state["next_agent"] = decision["next_agent"]
                state["current_task"] = decision.get("instructions", state["current_task"])
                
        except json.JSONDecodeError:
            # If parsing fails, still update messages
            state["messages"].append(
                AIMessage(content="I'll gather some initial data to get started.")
            )
            state["next_agent"] = "data_gathering"
                
        return state
```

### 4. Tool Base Class - REMOVED
(All tools are now external MCP servers, not internal implementations)

### 4. Graph Builder (`src/workflow/graph_builder.py`)

```python
from langgraph.graph import StateGraph, END
from src.state.agent_state import AgentState
from src.agents import (
    OrchestratorAgent,
    DataGatheringAgent,
    AnalysisAgent,
    TheoristSimulationAgent,
    LiteratureReviewerAgent
)

def build_astronomy_graph():
    """Build the LangGraph workflow for the multi-agent system."""
    
    # Initialize agents
    orchestrator = OrchestratorAgent()
    data_agent = DataGatheringAgent()
    analysis_agent = AnalysisAgent()
    simulation_agent = TheoristSimulationAgent()
    literature_agent = LiteratureReviewerAgent()
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("orchestrator", orchestrator.process)
    workflow.add_node("data_gathering", data_agent.process)
    workflow.add_node("analysis", analysis_agent.process)
    workflow.add_node("theorist_simulation", simulation_agent.process)
    workflow.add_node("literature_reviewer", literature_agent.process)
    
    # Define routing logic
    def route_from_orchestrator(state: AgentState) -> str:
        """Route based on orchestrator's decision."""
        next_agent = state.get("next_agent")
        if next_agent:
            return next_agent
        return END
    
    # Add edges
    workflow.set_entry_point("orchestrator")
    workflow.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {
            "data_gathering": "data_gathering",
            "analysis": "analysis",
            "theorist_simulation": "theorist_simulation",
            "literature_reviewer": "literature_reviewer",
            END: END
        }
    )
    
    # Add edges back to orchestrator from each specialist
    for agent in ["data_gathering", "analysis", "theorist_simulation", "literature_reviewer"]:
        workflow.add_edge(agent, "orchestrator")
    
    return workflow.compile()
```

### 5. MCP Server (`src/mcp/server.py`)

```python
import asyncio
from typing import Dict, Any, List
import json
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
from src.workflow.graph_builder import build_astronomy_graph
from src.state.agent_state import AgentState

class AstronomyResearchMCP:
    """MCP server that exposes the multi-agent system as a single tool."""
    
    def __init__(self):
        self.server = Server("astronomy-research-agent")
        self.astronomy_workflow = build_astronomy_graph()
        
        # Register handlers
        self.setup_handlers()
        
    def setup_handlers(self):
        """Set up MCP protocol handlers."""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """Return the astronomy research tool."""
            return [
                Tool(
                    name="astronomy_research",
                    description="Conduct astronomy research using a multi-agent system",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The astronomy research question or task"
                            },
                            "context": {
                                "type": "object",
                                "description": "Additional context or parameters",
                                "properties": {
                                    "data_sources": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Preferred data sources (DESI, LSST, etc.)"
                                    },
                                    "analysis_type": {
                                        "type": "string",
                                        "description": "Type of analysis needed"
                                    },
                                    "include_simulations": {
                                        "type": "boolean",
                                        "description": "Whether to run simulations"
                                    }
                                }
                            }
                        },
                        "required": ["query"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent | ImageContent | EmbeddedResource]:
            """Execute the astronomy research tool."""
            
            if name != "astronomy_research":
                raise ValueError(f"Unknown tool: {name}")
            
            query = arguments.get("query", "")
            context = arguments.get("context", {})
            
            # Initialize state with proper message objects and action tracking
            from langchain.schema import HumanMessage
            from datetime import datetime
            
            initial_state: AgentState = {
                "messages": [HumanMessage(content=query)],
                "action_log": [],  # Complete audit trail
                "current_task": query,
                "task_breakdown": [],
                "data_artifacts": {},  # File references only
                "analysis_results": {},  # Analysis metadata only
                "literature_context": {},  # Paper references
                "simulation_outputs": {},  # Simulation file metadata
                "next_agent": None,
                "final_response": None,
                "metadata": context,
                "start_time": datetime.now().isoformat(),
                "total_tool_calls": 0
            }
            
            # Run the workflow
            final_state = await self.astronomy_workflow.ainvoke(initial_state)
            
            # Format the response
            response_parts = []
            
            # Main response - could be the final_response or constructed from message history
            final_response = final_state.get("final_response")
            if not final_response and final_state.get("messages"):
                # Construct response from conversation history
                ai_messages = [msg.content for msg in final_state["messages"] 
                             if hasattr(msg, 'role') and msg.role == "assistant"]
                final_response = "\n\n".join(ai_messages)
            
            # Add action summary
            action_summary = f"\n\n---\nResearch Summary:\n"
            action_summary += f"Total tool calls: {final_state['total_tool_calls']}\n"
            action_summary += f"Start time: {final_state['start_time']}\n"
            
            # Add file references
            if final_state.get("data_artifacts"):
                action_summary += f"\nData files saved ({len(final_state['data_artifacts'])}):\n"
                for key, file_info in final_state["data_artifacts"].items():
                    action_summary += f"- {file_info['filename']} ({file_info['size_bytes']:,} bytes)\n"
                    action_summary += f"  Preview: {file_info['preview_command']}\n"
            
            response_parts.append(
                TextContent(
                    type="text",
                    text=(final_response or "No response generated") + action_summary
                )
            )
            
            # Add data artifacts as embedded resources (metadata only)
            if final_state.get("data_artifacts"):
                for name, file_info in final_state["data_artifacts"].items():
                    response_parts.append(
                        EmbeddedResource(
                            type="resource",
                            resource={
                                "uri": f"file://{file_info['filename']}",
                                "mimeType": "application/json",
                                "text": json.dumps({
                                    "file_id": file_info['file_id'],
                                    "filename": file_info['filename'],
                                    "size_bytes": file_info['size_bytes'],
                                    "description": file_info.get('description'),
                                    "preview_command": file_info['preview_command']
                                }, indent=2)
                            }
                        )
                    )
            
            # Include action log as a resource for debugging
            if final_state.get("action_log"):
                response_parts.append(
                    EmbeddedResource(
                        type="resource", 
                        resource={
                            "uri": "data://action_log",
                            "mimeType": "application/json",
                            "text": json.dumps(final_state["action_log"], indent=2)
                        }
                    )
                )
            
            return response_parts
    
    async def run(self):
        """Run the MCP server."""
        async with self.server.run_stdio():
            await self.server.wait_for_shutdown()

# Main entry point
async def main():
    mcp_server = AstronomyResearchMCP()
    await mcp_server.run()

if __name__ == "__main__":
    asyncio.run(main())
```

### 6. MCP Client Configuration (`mcp_config.json`)

```json
{
  "mcpServers": {
    "astronomy-research": {
      "command": "python",
      "args": ["-m", "src.mcp.server"],
      "env": {
        "PYTHONPATH": "${workspaceFolder}",
        "OPENAI_API_KEY": "${env:OPENAI_API_KEY}"
      }
    }
  }
}
```

### 7. Requirements (`requirements.txt`)

```
# Core dependencies
langchain>=0.1.0
langgraph>=0.0.20
langchain-openai>=0.0.5
mcp>=0.1.0  # Model Context Protocol SDK
# Note: You may need to install both the MCP server and client SDKs:
# pip install mcp-server mcp-client
# Or from source: pip install git+https://github.com/modelcontextprotocol/python-sdk.git
pydantic>=2.0.0

# Astronomy data access
astroquery>=0.4.6
astropy>=5.3
numpy>=1.24.0
pandas>=2.0.0

# Analysis tools
scipy>=1.11.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Simulation
nbodykit>=0.3.15
camb>=1.5.0

# Literature tools
arxiv>=1.4.8
scholarly>=1.7.0

# Utilities
python-dotenv>=1.0.0
httpx>=0.25.0
aiofiles>=23.0.0
structlog>=23.2.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
```

### 8. Configuration (`config/settings.py`)

```python
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings."""
    
    # API Keys
    openai_api_key: str
    arxiv_api_key: Optional[str] = None
    
    # Agent Configuration
    orchestrator_model: str = "gpt-4"
    specialist_model: str = "gpt-4"
    
    # Data Sources
    desi_base_url: str = "https://data.desi.lbl.gov/public/"
    lsst_base_url: str = "https://lsst.ncsa.illinois.edu/"
    
    # Workflow Configuration
    max_iterations: int = 10
    timeout_seconds: int = 300
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### 9. Running the MCP Server

To run the astronomy research agent as an MCP server:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the MCP server
python -m src.mcp.server
```

For integration with Cursor, add to your `.cursor/mcp_config.json`:

```json
{
  "mcpServers": {
    "astronomy-research": {
      "command": "python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "/path/to/astronomy-research-agent",
      "env": {
        "PYTHONPATH": ".",
        "OPENAI_API_KEY": "${env:OPENAI_API_KEY}"
      }
    }
  }
}
```

## Next Steps for Implementation

1. **Start with Core Infrastructure**:
   - Implement `BaseAgent` and `AgentState`
   - Set up basic LangGraph workflow
   - Create minimal orchestrator

2. **Build One Specialist Agent**:
   - Start with `DataGatheringAgent`
   - Implement 1-2 basic tools (e.g., DESI data access)
   - Test the workflow loop

3. **Expand Incrementally**:
   - Add more tools to existing agents
   - Implement remaining specialist agents
   - Enhance orchestrator's task decomposition

4. **MCP Integration**:
   - Implement the MCP server wrapper
   - Test with MCP client locally
   - Configure Cursor integration via mcp_config.json

5. **Testing Strategy**:
   - Unit tests for individual tools
   - Integration tests for agent interactions
   - End-to-end tests for complete workflows
   - Test MCP protocol compliance

### 10. Example Specialist Agent (`src/agents/data_gathering.py`)

```python
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from src.agents.base import BaseAgent
from src.state.agent_state import AgentState
import json

class DataGatheringAgent(BaseAgent):
    """Agent specialized in gathering astronomy data from various sources."""
    
    def __init__(self, llm: ChatOpenAI = None):
        super().__init__(
            name="data_gathering",
            mcp_tools=["desi-server", "lsst-server", "cmb-server"],  # MCP servers to connect to
            description="Gathers data from astronomy databases and observatories"
        )
        
        self.llm = llm or ChatOpenAI(model="gpt-4", temperature=0)
        
    async def process(self, state: AgentState) -> AgentState:
        """Process data gathering requests."""
        
        # Log start of processing
        self.log_message(state, f"Starting data gathering for task: {state.get('current_task', '')}")
        
        # Extract the current task
        current_task = state.get("current_task", "")
        metadata = state.get("metadata", {})
        
        # Get available tools from MCP servers
        available_tools = {}
        for server_name, client in self.mcp_clients.items():
            tools = await client.list_tools()
            available_tools[server_name] = tools
        
        # Create a prompt for the LLM to decide which tools to use
        tools_description = self._format_tools_description(available_tools)
        
        prompt = f"""You are a data gathering specialist for astronomy research.
        
Current task: {current_task}

Available MCP tools:
{tools_description}

Based on the task, determine:
1. Which MCP servers and tools to use
2. What parameters to pass to each tool

Return your response as JSON with this structure:
{{
    "tool_calls": [
        {{
            "server": "server-name",
            "tool": "tool-name",
            "arguments": {{...}}
        }}
    ],
    "reasoning": "explanation of your choices"
}}"""
        
        # Get LLM decision
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=f"Task: {current_task}")
        ]
        response = await self.llm.ainvoke(messages)
        
        # Parse LLM response
        try:
            plan = json.loads(response.content)
            tool_calls = plan.get("tool_calls", [])
            reasoning = plan.get("reasoning", "")
        except json.JSONDecodeError:
            # Handle parsing error
            tool_calls = []
            reasoning = "Unable to parse tool selection"
        
        # Update conversation with what we're about to do
        state["messages"].append(
            AIMessage(content=f"I'll gather the requested data. {reasoning}")
        )
        
        # Execute the tool calls via MCP
        gathered_files = {}  # Store file metadata, not actual data
        errors = []
        
        for call in tool_calls:
            server_name = call["server"]
            tool_name = call["tool"]
            arguments = call["arguments"]
            
            try:
                # This will automatically log the tool call
                result = await self.call_mcp_tool(server_name, tool_name, arguments, state)
                
                # Extract file metadata if data was saved
                if isinstance(result, dict):
                    save_result = result.get('save_result', {})
                    if save_result.get('status') == 'success':
                        file_key = f"{server_name}_{tool_name}_{save_result['file_id']}"
                        gathered_files[file_key] = {
                            'file_id': save_result['file_id'],
                            'filename': save_result['filename'],
                            'size_bytes': save_result['size_bytes'],
                            'file_type': save_result.get('file_type', 'unknown'),
                            'description': f"Data from {server_name}.{tool_name}",
                            'source': server_name,
                            'created': save_result.get('created'),
                            'preview_command': f"preview_data('{save_result['file_id']}')"
                        }
                        
                        # Also track some metadata about the query
                        if 'total_found' in result:
                            gathered_files[file_key]['total_records'] = result['total_found']
                        if 'query_info' in result:
                            gathered_files[file_key]['query_info'] = result['query_info']
                
            except Exception as e:
                errors.append(f"Error calling {server_name}.{tool_name}: {str(e)}")
                # Error is already logged by call_mcp_tool
        
        # Update state with file references (not actual data)
        state["data_artifacts"] = gathered_files
        
        # Add summary message about what was gathered
        if gathered_files:
            summary = f"Successfully gathered {len(gathered_files)} datasets:\n"
            for key, file_info in gathered_files.items():
                summary += f"- {file_info['filename']} ({file_info['size_bytes']:,} bytes)"
                if 'total_records' in file_info:
                    summary += f" - {file_info['total_records']} records"
                summary += f"\n  Preview: {file_info['preview_command']}\n"
            
            if errors:
                summary += f"\nEncountered {len(errors)} errors:\n"
                for error in errors:
                    summary += f"- {error}\n"
            
            state["messages"].append(AIMessage(content=summary))
            self.log_message(state, f"Data gathering complete: {len(gathered_files)} files saved")
        else:
            error_msg = "Unable to gather data. " + "; ".join(errors)
            state["messages"].append(AIMessage(content=error_msg))
            self.log_message(state, "Data gathering failed")
        
        # Route back to orchestrator
        state["next_agent"] = "orchestrator"
        
        return state
    
    def _format_tools_description(self, available_tools: Dict[str, List]) -> str:
        """Format available tools for LLM prompt."""
        descriptions = []
        for server_name, tools in available_tools.items():
            descriptions.append(f"\n{server_name}:")
            for tool in tools:
                descriptions.append(f"  - {tool.name}: {tool.description}")
                if hasattr(tool, 'inputSchema'):
                    descriptions.append(f"    Parameters: {json.dumps(tool.inputSchema, indent=6)}")
        return "\n".join(descriptions)
```

### 13. Example Analysis Agent (`src/agents/analysis.py`)

```python
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, SystemMessage
from src.agents.base import BaseAgent
from src.state.agent_state import AgentState
import json

class AnalysisAgent(BaseAgent):
    """Agent specialized in analyzing astronomy data."""
    
    def __init__(self, llm: ChatOpenAI = None):
        super().__init__(
            name="analysis",
            mcp_tools=["statistics-server", "correlation-server", "power-spectrum-server"],
            description="Performs statistical analysis on astronomy data"
        )
        self.llm = llm or ChatOpenAI(model="gpt-4", temperature=0)
    
    async def process(self, state: AgentState) -> AgentState:
        """Analyze data using appropriate MCP tools."""
        
        # Log start
        self.log_message(state, "Starting data analysis")
        
        # Get available data from state (file references only)
        data_artifacts = state.get("data_artifacts", {})
        current_task = state.get("current_task", "")
        
        if not data_artifacts:
            state["analysis_results"] = {}
            state["messages"].append(
                AIMessage(content="No data available to analyze. Need to gather data first.")
            )
            state["next_agent"] = "orchestrator"
            return state
        
        # Inform user about available data
        data_summary = "Available data files for analysis:\n"
        for key, file_info in data_artifacts.items():
            data_summary += f"- {file_info['filename']} ({file_info.get('total_records', 'unknown')} records)\n"
        
        state["messages"].append(AIMessage(content=data_summary))
        
        # Query available analysis tools
        available_tools = {}
        for server_name, client in self.mcp_clients.items():
            tools = await client.list_tools()
            available_tools[server_name] = tools
        
        # Let LLM determine analysis approach
        prompt = f"""You are an astronomy data analyst.

Task: {current_task}

Available data files:
{json.dumps({k: {'filename': v['filename'], 'file_id': v['file_id'], 'description': v.get('description')} 
             for k, v in data_artifacts.items()}, indent=2)}

Available analysis tools:
{self._format_tools_description(available_tools)}

Determine which analyses to perform. Return JSON:
{{
    "analyses": [
        {{
            "server": "server-name",
            "tool": "tool-name", 
            "arguments": {{
                "file_id": "id_of_data_file_to_analyze",
                // other tool-specific arguments
            }},
            "purpose": "what this analysis will show"
        }}
    ],
    "summary": "brief explanation of analysis plan"
}}"""
        
        messages = [
            SystemMessage(content=prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        
        # Execute analyses
        analysis_files = {}  # Store analysis output file metadata
        
        try:
            plan = json.loads(response.content)
            
            # Inform user of analysis plan
            state["messages"].append(
                AIMessage(content=f"Analysis plan: {plan.get('summary', 'Running analyses...')}")
            )
            
            for analysis in plan.get("analyses", []):
                try:
                    # Call tool (this logs the action automatically)
                    result = await self.call_mcp_tool(
                        analysis["server"],
                        analysis["tool"],
                        analysis["arguments"],
                        state
                    )
                    
                    # Extract file metadata if analysis was saved
                    if isinstance(result, dict) and 'save_result' in result:
                        save_info = result['save_result']
                        if save_info.get('status') == 'success':
                            analysis_key = f"{analysis['tool']}_{analysis['purpose'].replace(' ', '_')}"
                            analysis_files[analysis_key] = {
                                'file_id': save_info['file_id'],
                                'filename': save_info['filename'],
                                'size_bytes': save_info['size_bytes'],
                                'description': analysis['purpose'],
                                'preview_command': f"preview_data('{save_info['file_id']}')",
                                'source_data': analysis['arguments'].get('file_id')
                            }
                            
                            # Add any summary statistics
                            if 'summary_stats' in result:
                                analysis_files[analysis_key]['summary'] = result['summary_stats']
                
                except Exception as e:
                    self.log_message(state, f"Analysis error: {str(e)}")
                    
        except json.JSONDecodeError as e:
            state["messages"].append(
                AIMessage(content=f"Error planning analysis: {str(e)}")
            )
        
        # Update state with analysis results (file references only)
        state["analysis_results"] = analysis_files
        
        # Summarize what was done
        if analysis_files:
            summary = f"Completed {len(analysis_files)} analyses:\n"
            for key, file_info in analysis_files.items():
                summary += f"- {file_info['description']}\n"
                summary += f"  Output: {file_info['filename']}\n"
                summary += f"  Preview: {file_info['preview_command']}\n"
                if 'summary' in file_info:
                    summary += f"  Key finding: {file_info['summary']}\n"
            
            state["messages"].append(AIMessage(content=summary))
            self.log_message(state, f"Analysis complete: {len(analysis_files)} outputs generated")
        else:
            state["messages"].append(
                AIMessage(content="No analyses were completed successfully.")
            )
        
        state["next_agent"] = "orchestrator"
        return state
    
    def _format_tools_description(self, available_tools: Dict[str, list]) -> str:
        """Format tool descriptions for LLM."""
        descriptions = []
        for server_name, tools in available_tools.items():
            descriptions.append(f"\n{server_name}:")
            for tool in tools:
                descriptions.append(f"  - {tool.name}: {tool.description}")
        return "\n".join(descriptions)
```

### 11. MCP Tool Server Configuration (`config/mcp_configs.py`)

```python
"""Configuration for external MCP tool servers."""

MCP_TOOL_SERVERS = {
    "desi-server": {
        "command": "python",
        "args": ["-m", "desi_mcp.server"],
        "env": {
            "DESI_API_KEY": "${env:DESI_API_KEY}"
        },
        "description": "DESI spectroscopic data access"
    },
    "lsst-server": {
        "command": "python",
        "args": ["-m", "lsst_mcp.server"],
        "env": {
            "LSST_API_KEY": "${env:LSST_API_KEY}"
        },
        "description": "LSST imaging data access"
    },
    "cmb-server": {
        "command": "python",
        "args": ["-m", "cmb_mcp.server"],
        "env": {},
        "description": "CMB data from ACT and other experiments"
    },
    "arxiv-server": {
        "command": "python",
        "args": ["-m", "arxiv_mcp.server"],
        "env": {},
        "description": "ArXiv paper search and retrieval"
    },
    "nbody-server": {
        "command": "python",
        "args": ["-m", "nbody_mcp.server"],
        "env": {},
        "description": "N-body simulations"
    }
}
```

### 12. Updated Graph Builder (`src/workflow/graph_builder.py`)

```python
from langgraph.graph import StateGraph, END
from src.state.agent_state import AgentState
from src.agents import (
    OrchestratorAgent,
    DataGatheringAgent,
    AnalysisAgent,
    TheoristSimulationAgent,
    LiteratureReviewerAgent
)
from config.mcp_configs import MCP_TOOL_SERVERS

async def build_astronomy_graph():
    """Build the LangGraph workflow for the multi-agent system."""
    
    # Initialize agents
    orchestrator = OrchestratorAgent()
    data_agent = DataGatheringAgent()
    analysis_agent = AnalysisAgent()
    simulation_agent = TheoristSimulationAgent()
    literature_agent = LiteratureReviewerAgent()
    
    # Initialize MCP connections for each agent
    agents = [orchestrator, data_agent, analysis_agent, simulation_agent, literature_agent]
    for agent in agents:
        await agent.initialize_mcp_clients(MCP_TOOL_SERVERS)
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes with async wrappers to ensure cleanup
    async def wrapped_process(agent):
        async def process(state):
            try:
                return await agent.process(state)
            except Exception as e:
                # Log error and pass state through
                state["error"] = str(e)
                state["next_agent"] = "orchestrator"
                return state
        return process
    
    workflow.add_node("orchestrator", await wrapped_process(orchestrator))
    workflow.add_node("data_gathering", await wrapped_process(data_agent))
    workflow.add_node("analysis", await wrapped_process(analysis_agent))
    workflow.add_node("theorist_simulation", await wrapped_process(simulation_agent))
    workflow.add_node("literature_reviewer", await wrapped_process(literature_agent))
    
    # Define routing logic
    def route_from_orchestrator(state: AgentState) -> str:
        """Route based on orchestrator's decision."""
        next_agent = state.get("next_agent")
        if next_agent:
            return next_agent
        return END
    
    # Add edges
    workflow.set_entry_point("orchestrator")
    workflow.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {
            "data_gathering": "data_gathering",
            "analysis": "analysis",
            "theorist_simulation": "theorist_simulation",
            "literature_reviewer": "literature_reviewer",
            END: END
        }
    )
    
    # Add edges back to orchestrator from each specialist
    for agent in ["data_gathering", "analysis", "theorist_simulation", "literature_reviewer"]:
        workflow.add_edge(agent, "orchestrator")
    
    compiled_workflow = workflow.compile()
    
    # Store agents for cleanup
    compiled_workflow._agents = agents
    
    return compiled_workflow
```