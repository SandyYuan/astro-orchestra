"""Data gathering agent for accessing astronomy databases and observatories."""

from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from src.agents.base import BaseAgent
from src.state.agent_state import AgentState
from config.settings import settings
import json


class DataGatheringAgent(BaseAgent):
    """Agent specialized in gathering astronomy data from various sources."""
    
    def __init__(self, llm: ChatOpenAI = None):
        super().__init__(
            name="data_gathering",
            mcp_tools=["desi-server", "lsst-server", "cmb-server"],  # MCP servers to connect to
            description="Gathers data from astronomy databases and observatories"
        )
        
        self.llm = llm or ChatOpenAI(
            model=settings.specialist_model, 
            temperature=0
        )
        
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
            try:
                tools = await client.list_tools()
                available_tools[server_name] = tools
            except Exception as e:
                self.log_message(state, f"Error connecting to {server_name}: {str(e)}")
                available_tools[server_name] = []
        
        # Create a prompt for the LLM to decide which tools to use
        tools_description = self.format_tools_description(available_tools)
        
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
            "arguments": {{
                // tool-specific parameters
            }},
            "purpose": "why this tool call is needed"
        }}
    ],
    "reasoning": "explanation of your choices",
    "expected_data": "what type of data you expect to gather"
}}

If no suitable tools are available, return an empty tool_calls array and explain why."""
        
        # Get LLM decision
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=f"Task: {current_task}")
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            plan = self._parse_plan(response.content)
            
        except Exception as e:
            self.log_message(state, f"Error getting LLM plan: {str(e)}")
            plan = {"tool_calls": [], "reasoning": "LLM error", "expected_data": "none"}
        
        # Update conversation with what we're about to do
        if plan["tool_calls"]:
            reasoning = plan.get("reasoning", "Gathering requested data")
            expected = plan.get("expected_data", "astronomy data")
            state["messages"].append(
                AIMessage(content=f"I'll gather {expected}. {reasoning}")
            )
        else:
            state["messages"].append(
                AIMessage(content=f"No suitable data gathering tools found. {plan.get('reasoning', '')}")
            )
        
        # Execute the tool calls via MCP
        gathered_files = {}  # Store file metadata, not actual data
        errors = []
        
        for call in plan["tool_calls"]:
            server_name = call["server"]
            tool_name = call["tool"]
            arguments = call["arguments"]
            purpose = call.get("purpose", "data gathering")
            
            try:
                # This will automatically log the tool call
                result = await self.call_mcp_tool(server_name, tool_name, arguments, state)
                
                # Extract file metadata if data was saved
                file_metadata = self._extract_file_metadata(result, server_name, tool_name, purpose)
                if file_metadata:
                    file_key = f"{server_name}_{tool_name}_{len(gathered_files)}"
                    gathered_files[file_key] = file_metadata
                
            except Exception as e:
                error_msg = f"Error calling {server_name}.{tool_name}: {str(e)}"
                errors.append(error_msg)
                # Error is already logged by call_mcp_tool
        
        # Update state with file references (not actual data)
        state["data_artifacts"].update(gathered_files)
        
        # Add summary message about what was gathered
        if gathered_files:
            summary = f"Successfully gathered {len(gathered_files)} datasets:\n"
            for key, file_info in gathered_files.items():
                summary += f"- {file_info['filename']}"
                if 'total_records' in file_info:
                    summary += f" ({file_info['total_records']} records)"
                summary += f": {file_info['description']}\n"
                summary += f"  Preview command: {file_info['preview_command']}\n"
            
            if errors:
                summary += f"\nEncountered {len(errors)} errors:\n"
                for error in errors:
                    summary += f"- {error}\n"
            
            state["messages"].append(AIMessage(content=summary))
            self.log_message(state, f"Data gathering complete: {len(gathered_files)} files saved")
        else:
            error_msg = "Unable to gather data."
            if errors:
                error_msg += " Errors: " + "; ".join(errors)
            state["messages"].append(AIMessage(content=error_msg))
            self.log_message(state, "Data gathering failed")
        
        # Route back to orchestrator
        state["next_agent"] = "orchestrator"
        
        return state
    
    def _parse_plan(self, response_content: str) -> Dict[str, Any]:
        """Parse the LLM's tool selection plan."""
        try:
            plan = json.loads(response_content)
            
            # Validate required fields
            if "tool_calls" not in plan:
                plan["tool_calls"] = []
            if "reasoning" not in plan:
                plan["reasoning"] = "No reasoning provided"
            if "expected_data" not in plan:
                plan["expected_data"] = "astronomy data"
                
            return plan
            
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "tool_calls": [],
                "reasoning": "JSON parsing failed",
                "expected_data": "none"
            }
    
    def _extract_file_metadata(
        self, 
        result: Any, 
        server_name: str, 
        tool_name: str, 
        purpose: str
    ) -> Dict[str, Any] | None:
        """Extract file metadata from tool result."""
        
        if not isinstance(result, dict):
            return None
        
        # Look for save result information
        save_result = result.get('save_result', {})
        if save_result.get('status') == 'success':
            metadata = {
                'file_id': save_result.get('file_id', 'unknown'),
                'filename': save_result.get('filename', f"{server_name}_{tool_name}_output"),
                'size_bytes': save_result.get('size_bytes', 0),
                'file_type': save_result.get('file_type', 'unknown'),
                'description': purpose,
                'source': server_name,
                'tool': tool_name,
                'created': save_result.get('created', 'unknown'),
                'preview_command': f"preview_data('{save_result.get('file_id', 'unknown')}')"
            }
            
            # Add query metadata if available
            if 'total_found' in result:
                metadata['total_records'] = result['total_found']
            if 'query_info' in result:
                metadata['query_info'] = result['query_info']
            if 'data_type' in result:
                metadata['data_type'] = result['data_type']
                
            return metadata
            
        return None 