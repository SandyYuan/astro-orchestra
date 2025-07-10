#!/usr/bin/env python3
"""
Simple chat interface for the Astro Orchestra multi-agent system.
Treats the system as a black box - just sends queries and shows responses.
"""

import asyncio
import os
import uuid
from datetime import datetime
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def chat_with_system():
    """Interactive chat session with the multi-agent system."""
    
    print("=" * 60)
    print("ASTRO ORCHESTRA - AGENT SYSTEM CHAT")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "test_key_for_core_testing":
        print("WARNING: No real GOOGLE_API_KEY found in environment")
        print("System will work in mock mode")
        print()
    
    try:
        # Initialize the workflow system
        from src.workflow.graph_builder import create_workflow_runner, cleanup_workflow
        from src.state.agent_state import create_initial_state
        
        print("✓ System initialized successfully!")
        
        # Session setup
        session_id = str(uuid.uuid4())[:8]
        print(f"Session ID: {session_id}")
        
        print("\nCommands:")
        print("  'quit'/'q' - Exit")
        print("  'state'    - Show current state") 
        print("  'fullstate' - Show complete state with messages")
        print("  'clear'    - Start fresh session")
        print("\nTry queries like:")
        print("  'Download DESI galaxy data'")
        print("  'Analyze the clustering patterns'")
        print("  'Search for papers about dark matter'")
        print("=" * 60)
        
        # Initialize state
        conversation_state = create_initial_state("", {"session_id": session_id})
        workflow = None
        
        while True:
            print(f"\n{'-'*40}")
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
                
            if user_input.lower() == 'clear':
                session_id = str(uuid.uuid4())[:8]
                conversation_state = create_initial_state("", {"session_id": session_id})
                workflow = None  # Reset workflow for new session
                print(f"\n[NEW SESSION: {session_id}]")
                continue
                
            if user_input.lower() == 'state':
                show_state(conversation_state)
                continue
                
            if user_input.lower() == 'fullstate':
                show_state_full(conversation_state)
                continue
                
            if not user_input:
                continue
            
            # Process the request through the system
            conversation_state["current_task"] = user_input
            conversation_state["messages"].append(HumanMessage(content=user_input))
            
            print(f"\n🤖 Processing: {user_input}")
            
            try:
                # Create workflow if needed
                if not workflow:
                    workflow = await create_workflow_runner()
                
                # Run the workflow system
                result_state = None
                step_count = 0
                
                # Execute the workflow
                async for step in workflow.astream(
                    conversation_state,
                    {"thread_id": session_id}
                ):
                    step_count += 1
                    for node_name, node_state in step.items():
                        if node_name == "__end__":
                            result_state = node_state
                            break
                
                # Update conversation state
                if result_state:
                    conversation_state = result_state
                
                # Show the response
                show_response(conversation_state, step_count)
                
            except Exception as e:
                print(f"❌ Error: {e}")
                print("State preserved, continuing...")
        
        # Cleanup
        if workflow:
            print("Cleaning up...")
            await cleanup_workflow(workflow)
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure dependencies are installed: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ System Error: {e}")


def show_response(state, steps):
    """Show the system's response."""
    # Show any AI messages
    messages = state.get("messages", [])
    ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
    
    if ai_messages:
        latest_response = ai_messages[-1].content
        print(f"\n🤖 Response: {latest_response}")
    
    # Show summary of what was accomplished
    artifacts = {
        "data_artifacts": len(state.get("data_artifacts", {})),
        "analysis_results": len(state.get("analysis_results", {})), 
        "simulation_outputs": len(state.get("simulation_outputs", {})),
        "literature_context": len(state.get("literature_context", {}))
    }
    
    total_artifacts = sum(artifacts.values())
    if total_artifacts > 0:
        print(f"\n📊 Generated {total_artifacts} research artifacts:")
        for key, count in artifacts.items():
            if count > 0:
                name = key.replace("_", " ").title()
                print(f"   • {name}: {count}")
    
    print(f"\n✅ Completed in {steps} workflow steps")


def show_state(state):
    """Show current conversation state."""
    print(f"\n{'='*40}")
    print("CONVERSATION STATE")
    print(f"{'='*40}")
    
    session_id = state.get("metadata", {}).get("session_id", "unknown")
    print(f"Session: {session_id}")
    print(f"Current Task: {state.get('current_task', 'None')}")
    print(f"Messages: {len(state.get('messages', []))}")
    print(f"Actions: {len(state.get('action_log', []))}")
    print(f"Tool Calls: {state.get('total_tool_calls', 0)}")
    
    # Show resource counts
    print("\nResearch Artifacts:")
    artifacts = [
        ("Data Files", len(state.get("data_artifacts", {}))),
        ("Analyses", len(state.get("analysis_results", {}))),
        ("Simulations", len(state.get("simulation_outputs", {}))),
        ("Literature", len(state.get("literature_context", {})))
    ]
    
    for name, count in artifacts:
        print(f"  {name}: {count}")


def show_state_full(state):
    """Show complete conversation state including all messages and actions."""
    print(f"\n{'='*60}")
    print("FULL CONVERSATION STATE")
    print(f"{'='*60}")
    
    session_id = state.get("metadata", {}).get("session_id", "unknown")
    print(f"Session: {session_id}")
    print(f"Current Task: {state.get('current_task', 'None')}")
    print(f"Start Time: {state.get('start_time', 'Unknown')}")
    print(f"Next Agent: {state.get('next_agent', 'None')}")
    
    # Show all messages
    messages = state.get("messages", [])
    print(f"\n--- MESSAGES ({len(messages)}) ---")
    for i, msg in enumerate(messages):
        if hasattr(msg, 'content'):
            msg_type = "Human" if msg.__class__.__name__ == "HumanMessage" else "AI"
            content = msg.content
            print(f"\n{i+1}. [{msg_type}]: {content}")
    
    # Show action log
    actions = state.get("action_log", [])
    print(f"\n--- ACTION LOG ({len(actions)}) ---")
    for i, action in enumerate(actions):
        print(f"\n{i+1}. [{action.get('timestamp', 'no-time')}] {action.get('agent', 'unknown')}")
        print(f"   Type: {action.get('action_type', 'unknown')}")
        if action.get('message'):
            print(f"   Message: {action.get('message')}")
        if action.get('tool_call'):
            tool_call = action['tool_call']
            print(f"   Tool: {tool_call.get('tool', 'unknown')} on {tool_call.get('mcp_server', 'unknown')}")
            if tool_call.get('duration_ms'):
                print(f"   Duration: {tool_call['duration_ms']:.2f}ms")
        if action.get('tool_result'):
            result = action['tool_result']
            print(f"   Result: {result.get('status', 'unknown')}")
            if result.get('summary'):
                print(f"   Summary: {result['summary']}")
    
    # Show research artifacts in detail
    print(f"\n--- RESEARCH ARTIFACTS ---")
    
    data_artifacts = state.get("data_artifacts", {})
    if data_artifacts:
        print(f"\nData Files ({len(data_artifacts)}):")
        for key, artifact in data_artifacts.items():
            print(f"  • {key}: {artifact}")
    
    analysis_results = state.get("analysis_results", {})
    if analysis_results:
        print(f"\nAnalysis Results ({len(analysis_results)}):")
        for key, result in analysis_results.items():
            print(f"  • {key}: {result}")
    
    simulation_outputs = state.get("simulation_outputs", {})
    if simulation_outputs:
        print(f"\nSimulation Outputs ({len(simulation_outputs)}):")
        for key, output in simulation_outputs.items():
            print(f"  • {key}: {output}")
    
    literature_context = state.get("literature_context", {})
    if literature_context:
        print(f"\nLiterature Context ({len(literature_context)}):")
        for key, context in literature_context.items():
            print(f"  • {key}: {context}")
    
    # Show human feedback
    human_feedback = state.get("human_feedback", [])
    if human_feedback:
        print(f"\n--- HUMAN FEEDBACK ({len(human_feedback)}) ---")
        for i, feedback in enumerate(human_feedback):
            print(f"{i+1}. {feedback}")
    
    # Show metadata
    metadata = state.get("metadata", {})
    if metadata:
        print(f"\n--- METADATA ---")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    
    print(f"\n{'='*60}")


def show_help():
    """Show help information."""
    print("""
ASTRO ORCHESTRA SYSTEM CHAT
============================

This is a simple chat interface to the multi-agent astronomy research system.
The system handles all routing, execution, and coordination internally.

Commands:
  'quit'/'q' - Exit the chat
  'state'    - Show current conversation state  
  'fullstate' - Show complete state with messages
  'clear'    - Start a fresh session

Example Queries:
  "Download DESI spectroscopic data"
  "Analyze galaxy clustering patterns"
  "Search for papers about dark matter"
  "Run cosmological simulations"
  "What correlations exist in the data?"

The system will automatically:
• Route your request to appropriate agents
• Execute necessary data gathering, analysis, etc.
• Maintain context across the conversation
• Build on previous work in the session

Just ask for what you want - the system handles the rest!
""")


if __name__ == "__main__":
    show_help()
    asyncio.run(chat_with_system()) 