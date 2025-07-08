"""Basic example of using Astro Orchestra for a research task."""

import asyncio
from src.workflow.graph_builder import create_workflow_runner
from src.state.agent_state import create_initial_state


async def run_basic_research():
    """Run a basic astronomy research task."""
    
    # Define a research question
    query = "What can we learn about dark matter from galaxy clustering in the DESI survey?"
    
    # Optional context
    context = {
        "data_sources": ["DESI"],
        "analysis_type": "statistical",
        "include_simulations": True
    }
    
    print("Starting astronomy research...")
    print(f"Query: {query}")
    print(f"Context: {context}")
    print("-" * 50)
    
    # Create initial state
    initial_state = create_initial_state(query, context)
    
    # Run the workflow
    async with create_workflow_runner() as workflow:
        final_state = await workflow.ainvoke(initial_state)
    
    # Print results
    print("\nResearch completed!")
    print(f"Final response: {final_state.get('final_response', 'No response')}")
    print(f"Total actions: {len(final_state.get('action_log', []))}")
    print(f"Data files: {len(final_state.get('data_artifacts', {}))}")
    print(f"Analysis results: {len(final_state.get('analysis_results', {}))}")
    
    return final_state


if __name__ == "__main__":
    asyncio.run(run_basic_research()) 