"""Tests for the orchestrator agent."""

import pytest
from src.agents.orchestrator import OrchestratorAgent
from src.state.agent_state import create_initial_state


@pytest.mark.asyncio
async def test_orchestrator_initialization():
    """Test that orchestrator can be initialized."""
    orchestrator = OrchestratorAgent()
    assert orchestrator.name == "orchestrator"
    assert orchestrator.mcp_tools == []
    assert "orchestrator" in orchestrator.description.lower()


@pytest.mark.asyncio
async def test_orchestrator_process():
    """Test that orchestrator can process a basic state."""
    orchestrator = OrchestratorAgent()
    
    # Create initial state
    state = create_initial_state("Test astronomy research question")
    
    # Process should not fail
    result = await orchestrator.process(state)
    
    # Should have made a decision
    assert "next_agent" in result
    assert len(result["action_log"]) > 0
    
    # Should have added a message
    assert len(result["messages"]) >= 1 