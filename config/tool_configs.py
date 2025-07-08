"""Configuration for external MCP tool servers."""

from typing import Dict, List, Any


MCP_TOOL_SERVERS = {
    "desi-server": {
        "command": "python",
        "args": ["-m", "desi_mcp.server"],
        "env": {
            "DESI_API_KEY": "${env:DESI_API_KEY}"
        },
        "description": "DESI spectroscopic data access",
        "tools": ["search_objects", "get_spectrum", "query_catalog"]
    },
    "lsst-server": {
        "command": "python", 
        "args": ["-m", "lsst_mcp.server"],
        "env": {
            "LSST_API_KEY": "${env:LSST_API_KEY}"
        },
        "description": "LSST imaging data access",
        "tools": ["search_images", "get_photometry", "query_catalog"]
    },
    "cmb-server": {
        "command": "python",
        "args": ["-m", "cmb_mcp.server"],
        "env": {},
        "description": "CMB data from ACT and other experiments",
        "tools": ["get_temperature_maps", "get_polarization_maps", "query_catalogs"]
    },
    "arxiv-server": {
        "command": "python",
        "args": ["-m", "arxiv_mcp.server"],
        "env": {},
        "description": "ArXiv paper search and retrieval",
        "tools": ["search_papers", "get_paper", "get_citations"]
    },
    "nbody-server": {
        "command": "python",
        "args": ["-m", "nbody_mcp.server"],
        "env": {},
        "description": "N-body simulations",
        "tools": ["run_simulation", "analyze_halos", "compute_power_spectrum"]
    },
    "statistics-server": {
        "command": "python",
        "args": ["-m", "stats_mcp.server"],
        "env": {},
        "description": "Statistical analysis tools",
        "tools": ["compute_statistics", "fit_model", "correlation_analysis"]
    }
}


def get_agent_tool_mapping() -> Dict[str, List[str]]:
    """Return mapping of agents to their preferred MCP servers."""
    return {
        "orchestrator": [],  # Orchestrator doesn't use external tools directly
        "data_gathering": ["desi-server", "lsst-server", "cmb-server"],
        "analysis": ["statistics-server", "desi-server", "lsst-server"],
        "theorist_simulation": ["nbody-server", "statistics-server"],
        "literature_reviewer": ["arxiv-server"]
    }


def get_server_config(server_name: str) -> Dict[str, Any]:
    """Get configuration for a specific MCP server."""
    return MCP_TOOL_SERVERS.get(server_name, {})


def get_available_tools(server_name: str) -> List[str]:
    """Get list of available tools for a specific server."""
    config = get_server_config(server_name)
    return config.get("tools", []) 