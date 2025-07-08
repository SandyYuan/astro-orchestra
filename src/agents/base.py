"""Base agent class for the multi-agent astronomy research system."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from langchain.schema import BaseMessage
from datetime import datetime
import time
import json

# Mock MCP imports for now - will be replaced with actual MCP SDK
# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client


class MockMCPClient:
    """Mock MCP client for development - replace with actual MCP client."""
    
    def __init__(self, server_name: str):
        self.server_name = server_name
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def list_tools(self):
        """Mock tool listing."""
        return []
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        """Mock tool call."""
        return {
            "status": "success",
            "message": f"Mock call to {self.server_name}.{tool_name}",
            "arguments": arguments
        }


class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""
    
    def __init__(self, name: str, mcp_tools: List[str], description: str):
        """Initialize the base agent.
        
        Args:
            name: Agent identifier
            mcp_tools: List of MCP server names this agent can use
            description: Human-readable description of agent capabilities
        """
        self.name = name
        self.mcp_tools = mcp_tools  # List of MCP server names this agent can use
        self.description = description
        self.mcp_clients = {}  # Will store active MCP client sessions
    
    async def initialize_mcp_clients(self, mcp_configs: Dict[str, Dict[str, Any]]):
        """Initialize MCP client connections for this agent's tools."""
        for tool_name in self.mcp_tools:
            if tool_name in mcp_configs:
                config = mcp_configs[tool_name]
                
                # TODO: Replace with actual MCP client initialization
                # server_params = StdioServerParameters(
                #     command=config["command"],
                #     args=config.get("args", []),
                #     env=config.get("env", {})
                # )
                # client = stdio_client(server_params)
                # self.mcp_clients[tool_name] = await client.__aenter__()
                
                # For now, use mock client
                client = MockMCPClient(tool_name)
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
    
    def format_tools_description(self, available_tools: Dict[str, List]) -> str:
        """Format available tools for LLM prompt."""
        descriptions = []
        for server_name, tools in available_tools.items():
            descriptions.append(f"\n{server_name}:")
            for tool in tools:
                descriptions.append(f"  - {tool.name}: {tool.description}")
                if hasattr(tool, 'inputSchema'):
                    descriptions.append(f"    Parameters: {json.dumps(tool.inputSchema, indent=6)}")
        return "\n".join(descriptions)
    
    @abstractmethod
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the current state and return updated state.
        
        Each agent implements this method to define their specific behavior.
        """
        pass
    
    async def cleanup(self):
        """Clean up MCP client connections."""
        for client in self.mcp_clients.values():
            await client.__aexit__(None, None, None) 