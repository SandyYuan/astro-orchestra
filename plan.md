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
- **High-level research ideas** (e.g., "I want to model DESI LRG BAO measurements") → Planning Agent → Orchestrator maps plan to agents
- **Specific tasks** (e.g., "Download DESI galaxy data") → Direct to specialist agents
- **Abstract plans** → Orchestrator translates to concrete agent assignments

## Core Component Implementations

The system is built with several key components:

### 1. Base Agent Class (`src/agents/base.py`)
- Abstract base class for all agents
- Handles MCP client connections and tool calls
- Provides logging and state management utilities
- Extracts metadata from tool results without storing large datasets

### 2. Agent State (`src/state/agent_state.py`)
- Shared state structure passed between agents
- Maintains complete audit trail of all actions
- Stores file references and metadata (not actual large datasets)
- Tracks conversation history and workflow progress

### 3. Planning Agent (`src/agents/planning.py`) - PLACEHOLDER
- Transforms rough research ideas into detailed, structured execution plans
- Focuses purely on plan decomposition without knowledge of specific tools/agents
- Breaks down objectives into logical steps with dependencies and success criteria
- Returns structured plans for the orchestrator to map to available resources
- **Note**: Currently a placeholder - implementation exists elsewhere and will be integrated

### 4. Orchestrator Agent (`src/agents/orchestrator.py`)
- Routes between planning and specialist agents based on task type
- Distinguishes high-level research ideas from specific tasks  
- Knows about all available agents and their capabilities
- Maps abstract plans from the planning agent to concrete agent assignments
- Uses Google Gemini to make intelligent routing decisions
- Provides reasoning for routing choices to maintain transparency

### 5. Tool Base Class - REMOVED
(All tools are now external MCP servers, not internal implementations)

### 6. Graph Builder (`src/workflow/graph_builder.py`)
- Builds the LangGraph workflow connecting all agents
- Defines routing logic between orchestrator and specialist agents
- Sets up conditional edges based on orchestrator decisions
- Handles agent initialization and MCP client connections

### 7. MCP Server (`src/mcp/server.py`)
- Exposes the multi-agent system as a single MCP tool
- Handles tool registration and execution requests
- Manages state initialization and workflow execution
- Returns results with action summaries and embedded resources

### 8. MCP Client Configuration (`mcp_config.json`)
- Configures the astronomy research agent as an MCP server for Cursor integration
- Sets environment variables and Python module path
- Enables direct access to the multi-agent system from Cursor

### 9. Dependencies and Requirements
The system requires several key dependency groups:
- **Core**: LangChain, LangGraph, MCP SDK, Pydantic
- **Astronomy**: AstroQuery, AstroPy, NumPy, Pandas
- **Analysis**: SciPy, Scikit-Learn, Matplotlib, Seaborn
- **Simulation**: nbodykit, CAMB
- **Literature**: ArXiv, Scholarly
- **Utilities**: python-dotenv, httpx, aiofiles, structlog
- **Testing**: pytest, pytest-asyncio, pytest-cov

### 10. Configuration (`config/settings.py`)
- Pydantic settings for API keys, model configurations, and data source URLs
- Environment-based configuration with .env file support
- Workflow parameters for timeouts and iteration limits

### 11. Running the MCP Server

To run the astronomy research agent as an MCP server:
1. Install dependencies: `pip install -r requirements.txt`
2. Run the MCP server: `python -m src.mcp.server`
3. Configure Cursor integration by adding the MCP server configuration to `.cursor/mcp_config.json`

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

### 12. Specialist Agent Examples

**Data Gathering Agent (`src/agents/data_gathering.py`)**:
- Connects to DESI, LSST, CMB MCP servers
- Uses LLM to select appropriate tools based on task requirements
- Executes tool calls and tracks file metadata
- Returns data references without storing large datasets in state

**Analysis Agent (`src/agents/analysis.py`)**:
- Connects to statistics, correlation, and power spectrum MCP servers
- Analyzes available data files referenced in state
- Plans and executes multiple analysis operations
- Returns analysis output file references and summaries

### 13. External Tool Server Configuration
The system connects to external MCP tool servers for:
- **DESI Server**: Spectroscopic data access
- **LSST Server**: Imaging data access  
- **CMB Server**: CMB data from ACT and other experiments
- **ArXiv Server**: Paper search and retrieval
- **N-body Server**: Cosmological simulations

Each server is configured with appropriate environment variables and command-line parameters for secure access to astronomy data sources.