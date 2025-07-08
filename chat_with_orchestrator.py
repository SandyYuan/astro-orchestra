#!/usr/bin/env python3
"""
Interactive chat with the Orchestrator Agent.
See what agents are available, what they do, and how the orchestrator routes queries.
"""

import asyncio
import os
from datetime import datetime
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def chat_with_orchestrator():
    """Interactive chat session with the orchestrator."""
    
    print("=" * 60)
    print("ASTRO ORCHESTRA - ORCHESTRATOR CHAT")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "test_key_for_core_testing":
        print("ERROR: No real GOOGLE_API_KEY found in environment")
        print("Please set your API key in the .env file")
        return
    
    try:
        # Import components
        from src.state.agent_state import AgentState
        from src.agents.orchestrator import OrchestratorAgent
        from src.agents.data_gathering import DataGatheringAgent
        from src.agents.analysis import AnalysisAgent
        from src.agents.theorist_simulation import TheoristSimulationAgent
        from src.agents.literature_reviewer import LiteratureReviewerAgent
        
        print("System initialized successfully!")
        
        # Create orchestrator
        orchestrator = OrchestratorAgent()
        
        # Create all agents to show capabilities
        agents_info = {
            "data_gathering": {
                "agent": DataGatheringAgent(),
                "capabilities": [
                    "Access DESI spectroscopic survey data",
                    "Query LSST imaging databases", 
                    "Retrieve CMB data from ACT/Planck",
                    "Search astronomical object catalogs",
                    "Download observational datasets"
                ]
            },
            "analysis": {
                "agent": AnalysisAgent(),
                "capabilities": [
                    "Statistical analysis of astronomical data",
                    "Correlation studies and clustering analysis",
                    "Power spectrum calculations",
                    "Data fitting and parameter estimation",
                    "Error analysis and uncertainty propagation"
                ]
            },
            "theorist_simulation": {
                "agent": TheoristSimulationAgent(),
                "capabilities": [
                    "N-body cosmological simulations",
                    "Dark matter halo modeling",
                    "Cosmological parameter calculations",
                    "Theoretical model predictions",
                    "Matter power spectrum generation"
                ]
            },
            "literature_reviewer": {
                "agent": LiteratureReviewerAgent(),
                "capabilities": [
                    "ArXiv paper search and retrieval",
                    "Scientific literature synthesis",
                    "Citation analysis and trends",
                    "Research context generation",
                    "Knowledge base creation"
                ]
            }
        }
        
        print("\nAVAILABLE SPECIALIST AGENTS:")
        print("-" * 40)
        for agent_name, info in agents_info.items():
            agent = info["agent"]
            print(f"\n{agent_name.upper().replace('_', ' ')} AGENT:")
            print(f"  Description: {agent.description}")
            print(f"  MCP Tools: {', '.join(agent.mcp_tools) if agent.mcp_tools else 'None'}")
            print("  Capabilities:")
            for capability in info["capabilities"]:
                print(f"    - {capability}")
        
        print("\n" + "=" * 60)
        print("CHAT SESSION STARTED")
        print("Type 'quit' to exit, 'agents' to see available agents again")
        print("=" * 60)
        
        # Initialize conversation state
        conversation_state: AgentState = {
            "messages": [],
            "action_log": [],
            "current_task": "",
            "task_breakdown": [],
            "data_artifacts": {},
            "analysis_results": {},
            "literature_context": {},
            "simulation_outputs": {},
            "next_agent": None,
            "final_response": None,
            "metadata": {"chat_mode": True},
            "start_time": datetime.now().isoformat(),
            "total_tool_calls": 0
        }
        
        while True:
            # Get user input
            print("\n" + "-" * 60)
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
                
            if user_input.lower() == 'agents':
                print("\nAVAILABLE AGENTS:")
                for agent_name, info in agents_info.items():
                    print(f"  - {agent_name}: {info['agent'].description}")
                continue
                
            if not user_input:
                continue
                
            # Add user message to conversation
            conversation_state["messages"].append(HumanMessage(content=user_input))
            conversation_state["current_task"] = user_input
            
            print(f"\nOrchestrator: Thinking about your request...")
            print("             (Making API call to Google Gemini...)")
            
            try:
                # Get orchestrator's decision
                updated_state = await orchestrator.process(conversation_state)
                
                # Show orchestrator's response
                ai_messages = [msg for msg in updated_state["messages"] 
                             if isinstance(msg, AIMessage)]
                
                if ai_messages:
                    latest_response = ai_messages[-1].content
                    print(f"\nOrchestrator: {latest_response}")
                
                # Show routing decision
                next_agent = updated_state.get("next_agent")
                if next_agent and next_agent in agents_info:
                    agent_info = agents_info[next_agent]
                    print(f"\nROUTING DECISION:")
                    print(f"  Next Agent: {next_agent.replace('_', ' ').title()}")
                    print(f"  Reason: {agent_info['agent'].description}")
                    print(f"  Updated Task: {updated_state.get('current_task', 'No specific task')}")
                    
                    # Show what the agent would do
                    print(f"\n  This agent would:")
                    for capability in agent_info['capabilities'][:3]:  # Show first 3
                        print(f"    - {capability}")
                    if len(agent_info['capabilities']) > 3:
                        print(f"    - ... and {len(agent_info['capabilities']) - 3} more capabilities")
                        
                elif updated_state.get("final_response"):
                    print(f"\nRESEARCH COMPLETE:")
                    print(f"  The orchestrator believes the task is finished.")
                    print(f"  Final response provided above.")
                else:
                    print(f"\nUNCERTAIN ROUTING:")
                    print(f"  The orchestrator couldn't decide on a specific agent.")
                    print(f"  Next agent: {next_agent or 'None specified'}")
                
                # Show conversation statistics
                total_messages = len(updated_state["messages"])
                actions = len(updated_state["action_log"])
                print(f"\nCONVERSATION STATS:")
                print(f"  Total messages: {total_messages}")
                print(f"  Actions logged: {actions}")
                print(f"  Data artifacts: {len(updated_state['data_artifacts'])}")
                print(f"  Analysis results: {len(updated_state['analysis_results'])}")
                
                # Update state for next iteration
                conversation_state = updated_state
                
            except Exception as e:
                print(f"\nERROR: {str(e)}")
                print("The orchestrator encountered an error. Please try again.")
        
    except ImportError as e:
        print(f"IMPORT ERROR: {str(e)}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
    except Exception as e:
        print(f"SYSTEM ERROR: {str(e)}")

def show_help():
    """Show help information."""
    print("""
ASTRO ORCHESTRA CHAT HELP
==========================

Commands:
  'agents'     - Show all available specialist agents
  'quit'/'q'   - Exit the chat session
  
Example Queries:
  "I want to study dark matter using DESI data"
  "Can you help me analyze galaxy clustering patterns?"
  "Search for papers about cosmic microwave background"
  "Run a simulation of large scale structure formation"
  "What's the correlation between galaxy mass and environment?"

The orchestrator will:
1. Analyze your request
2. Decide which specialist agent is best suited
3. Show you the routing decision and reasoning
4. Explain what the chosen agent would do

Note: This is a chat interface to test the orchestrator's decision-making.
The specialist agents are not actually executed in this demo.
""")

if __name__ == "__main__":
    show_help()
    asyncio.run(chat_with_orchestrator()) 