# Minimal Implementation Plan for Pause/Resume Feature

## Overview
Add pause/resume functionality where the system pauses after each specialist agent completes, reports results to human, and resumes with human feedback.

## 1. State Changes

### File: `src/state/agent_state.py`
Add one field to track human feedback:

```python
class AgentState(TypedDict):
    # ... existing fields ...
    
    # Add this field:
    human_feedback: List[Dict[str, Any]]  # Track all human inputs
```

## 2. Orchestrator Changes

### File: `src/agents/orchestrator.py`
Update the `process` method to pause after specialists and use human feedback:

```python
async def process(self, state: AgentState) -> AgentState:
    """Process state - pause after specialist completion or route to next agent."""
    
    # Check if a specialist just completed
    last_action = state["action_log"][-1] if state["action_log"] else None
    
    if (last_action and 
        last_action["agent"] != "orchestrator" and 
        last_action.get("tool_result")):
        
        # Specialist completed - format results and pause
        agent_name = last_action["agent"]
        
        # Build result summary
        summary = f"## {agent_name.replace('_', ' ').title()} Complete\n\n"
        
        # Add specific results based on agent type
        if agent_name == "data_gathering" and state.get("data_artifacts"):
            for key, file_info in state["data_artifacts"].items():
                summary += f"- Saved: {file_info['filename']} ({file_info['size_bytes']:,} bytes)\n"
                if 'total_records' in file_info:
                    summary += f"  Records: {file_info['total_records']}\n"
        
        elif agent_name == "analysis" and state.get("analysis_results"):
            for key, result in state["analysis_results"].items():
                summary += f"- Analysis: {result.get('description', key)}\n"
                if 'summary' in result:
                    summary += f"  Result: {result['summary']}\n"
        
        # Add similar handling for other agents...
        
        summary += "\nProvide instructions for next steps:"
        
        # Update state and pause
        state["messages"].append(AIMessage(content=summary))
        state["next_agent"] = None  # This triggers pause
        return state
    
    # Not paused - determine next agent
    # Include human feedback in decision
    human_feedback = state.get("human_feedback", [])
    latest_feedback = human_feedback[-1]["content"] if human_feedback else None
    
    context = f"""Current research state:
Task: {state.get('current_task')}
Data files: {len(state.get('data_artifacts', {}))}
Analyses: {len(state.get('analysis_results', {}))}
Recent feedback: {latest_feedback}

Determine next agent (data_gathering, analysis, theorist_simulation, literature_reviewer) or null if complete.
Return JSON: {{"next_agent": "...", "instructions": "...", "reasoning": "..."}}"""
    
    messages = [SystemMessage(content=context)]
    response = await self.llm.ainvoke(messages)
    
    try:
        decision = json.loads(response.content)
        state["next_agent"] = decision.get("next_agent")
        if decision.get("next_agent"):
            state["current_task"] = decision.get("instructions", state["current_task"])
            state["messages"].append(AIMessage(content=decision["reasoning"]))
    except:
        state["next_agent"] = "data_gathering"  # Default
    
    return state
```

## 3. Workflow Changes

### File: `src/workflow/graph_builder.py`
Add checkpointing to enable state persistence:

```python
from langgraph.checkpoint.sqlite import SqliteSaver

async def build_astronomy_graph():
    # ... existing code ...
    
    # Add before compile:
    checkpointer = SqliteSaver.from_conn_string("astronomy_research.db")
    
    # Update compile line:
    compiled_workflow = workflow.compile(checkpointer=checkpointer)
    
    # ... rest of existing code ...
    return compiled_workflow
```

## 4. MCP Server Changes

### File: `src/mcp/server.py`
Update to handle session management and human feedback:

```python
@server.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """Execute astronomy research with pause/resume support."""
    
    if name != "astronomy_research":
        raise ValueError(f"Unknown tool: {name}")
    
    # Get or generate session ID
    session_id = arguments.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
    
    config = {"configurable": {"thread_id": session_id}}
    
    # Check if this is a continuing session
    current_state = await astronomy_workflow.aget_state(config)
    
    if current_state:
        # Resuming - add human feedback
        feedback = arguments.get("query", "continue")
        
        # Update state with human feedback
        updates = {
            "messages": [HumanMessage(content=feedback)],
            "human_feedback": current_state.values.get("human_feedback", []) + [{
                "timestamp": datetime.now().isoformat(),
                "content": feedback
            }]
        }
        
        await astronomy_workflow.aupdate_state(config, updates)
        
        # Resume workflow
        await astronomy_workflow.ainvoke(None, config)
    else:
        # New session
        query = arguments["query"]
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "action_log": [],
            "current_task": query,
            "human_feedback": [],  # Initialize empty
            "data_artifacts": {},
            "analysis_results": {},
            "literature_context": {},
            "simulation_outputs": {},
            "next_agent": None,
            "final_response": None,
            "metadata": arguments.get("context", {}),
            "start_time": datetime.now().isoformat(),
            "total_tool_calls": 0
        }
        
        await astronomy_workflow.ainvoke(initial_state, config)
    
    # Get final state
    final_state = await astronomy_workflow.aget_state(config)
    
    # Extract last AI message
    messages = final_state.values.get("messages", [])
    last_ai_message = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            last_ai_message = msg.content
            break
    
    # Format response with session info
    status = "PAUSED" if final_state.values.get("next_agent") is None else "RUNNING"
    response = f"[Session: {session_id}] [Status: {status}]\n\n"
    response += last_ai_message or "No response generated"
    
    return [types.TextContent(type="text", text=response)]
```

### Update tool schema:
```python
types.Tool(
    name="astronomy_research",
    description="Conduct astronomy research with pause/resume capability",
    inputSchema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Research question or continuation feedback"
            },
            "session_id": {
                "type": "string",
                "description": "Session ID for continuing research (optional)"
            },
            "context": {
                "type": "object",
                "description": "Additional parameters (optional)"
            }
        },
        "required": ["query"]
    }
)
```

## Summary of Changes

1. **State**: Add `human_feedback` list field
2. **Orchestrator**: 
   - Detect specialist completion from action_log
   - Format results and set `next_agent = None` to pause
   - Use human feedback in routing decisions
3. **Workflow**: Add SQLite checkpointing
4. **MCP Server**:
   - Track sessions with thread_id
   - Add human feedback to state when resuming
   - Return session ID in responses

## Usage Pattern

```bash
# First call - starts research
{"query": "Analyze DESI BAO measurements"}
# Returns: "[Session: abc123] [Status: PAUSED]\n\nData Gathering Complete\n- Saved: desi_bao.json (2.3MB)\n  Records: 50,000\n\nProvide instructions for next steps:"

# Continue with feedback
{"query": "Focus on z>2 galaxies and run correlation analysis", "session_id": "abc123"}
# Returns: "[Session: abc123] [Status: PAUSED]\n\nAnalysis Complete\n- Analysis: Correlation function for high-z sample\n  Result: BAO peak detected at 105 Mpc/h\n\nProvide instructions for next steps:"

# Continue again
{"query": "Compare with Planck CDM predictions", "session_id": "abc123"}
# And so on...
```

No additional features - just pause after specialists, show results, gather feedback, and continue.