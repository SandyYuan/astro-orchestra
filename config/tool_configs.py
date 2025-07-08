"""Configuration for external MCP tool servers."""


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

# Agent to MCP server mapping
AGENT_TOOL_MAPPING = {
    "orchestrator": [],  # Orchestrator doesn't use external tools directly
    "data_gathering": ["desi-server", "lsst-server", "cmb-server"],
    "analysis": ["statistics-server", "desi-server", "lsst-server"],
    "theorist_simulation": ["nbody-server", "statistics-server"],
    "literature_reviewer": ["arxiv-server"]
} 