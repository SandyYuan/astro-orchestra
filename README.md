# Astro Orchestra - Multi-Agent Astronomy Research System

A sophisticated multi-agent system for conducting astronomy research using the Model Context Protocol (MCP) to orchestrate specialized agents and external astronomical data tools.

## Quick Start

1. **Clone and Setup**:
```bash
cd astro-orchestra
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

2. **Configure Environment**:
```bash
# Create .env file with your Google API key
echo "GOOGLE_API_KEY=your_google_api_key_here" > .env
```

3. **Run MCP Server**:
```bash
python -m src.mcp.server
```

4. **Configure Cursor**:
Add to your `.cursor/mcp_config.json`:
```json
{
  "mcpServers": {
    "astronomy-research": {
      "command": "python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "/path/to/astro-orchestra",
      "env": {
        "PYTHONPATH": ".",
        "GOOGLE_API_KEY": "${env:GOOGLE_API_KEY}"
      }
    }
  }
}
```

## Project Architecture

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 50, 'rankSpacing': 80}}}%%
graph TD
    %% Styling with darker text
    classDef human fill:#FFE4B5,stroke:#FF8C00,stroke-width:3px,color:#333
    classDef orchestrator fill:#E6E6FA,stroke:#9370DB,stroke-width:3px,color:#333
    classDef specialist fill:#B0E0E6,stroke:#4682B4,stroke-width:2px,color:#333
    classDef tool fill:#90EE90,stroke:#228B22,stroke-width:2px,color:#333
    classDef langgraph stroke:#4169E1,stroke-width:3px
    classDef mcp stroke:#32CD32,stroke-width:3px,stroke-dasharray: 5 5
    classDef feedback stroke:#FF6347,stroke-width:3px

    %% Top level - Human Interface
    HC[Human + Cursor]:::human
    
    %% Core Orchestrator
    HC -->|"Research Query"| O[Orchestrator Agent]:::orchestrator
    
    %% Specialist Agents arranged in a row
    O -->|"Routes Tasks"| P[Planning Agent]:::specialist
    O --> DG[Data Gathering Agent]:::specialist
    O --> A[Analysis Agent]:::specialist
    O --> TS[Theorist Simulation Agent]:::specialist
    O --> LR[Literature Reviewer Agent]:::specialist
    
    %% Return paths
    P -->|"Return Results"| O:::langgraph
    DG --> O:::langgraph
    A --> O:::langgraph
    TS --> O:::langgraph
    LR --> O:::langgraph
    
    %% MCP Tool Servers - bottom level
    DG -.->|"MCP Protocol"| DESI[DESI Server]:::tool
    DG -.-> LSST[LSST Server]:::tool
    DG -.-> CMB[CMB Server]:::tool
    A -.-> STATS[Statistics Server]:::tool
    TS -.-> NBODY[N-body Server]:::tool
    LR -.-> ARXIV[ArXiv Server]:::tool
    
    %% Human Feedback Loop
    O ==>|"Pauses for Review<br/>Shows Results"| HC:::feedback
    HC ==>|"Provides Feedback"| O:::feedback
```

Astro Orchestra uses a multi-agent architecture where:

- **Orchestrator Agent**: Breaks down research tasks and coordinates specialists
- **Data Gathering Agent**: Accesses DESI, LSST, CMB databases via MCP tools
- **Analysis Agent**: Performs statistical analysis and computations
- **Theorist Simulation Agent**: Runs cosmological simulations
- **Literature Reviewer Agent**: Searches and synthesizes scientific papers

All agents communicate through a shared state system and use external MCP tool servers for accessing astronomical data and computational resources.

## MCP Integration

The system exposes itself as an MCP server that can be integrated with AI assistants like Cursor. External astronomy tools run as separate MCP servers, allowing for:

- Dynamic tool discovery
- Scalable tool ecosystem
- Separation of concerns
- Easy tool updates and additions

## Development Status

This project is in active development. Core components are being implemented incrementally.

## License

MIT License 