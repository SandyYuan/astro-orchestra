"""Planning agent for expanding research ideas into detailed execution plans."""

from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from src.agents.base import BaseAgent
from src.state.agent_state import AgentState
from config.settings import settings
import json


class PlanningAgent(BaseAgent):
    """Agent specialized in expanding research ideas into detailed execution plans."""
    
    def __init__(self, llm: ChatGoogleGenerativeAI = None):
        super().__init__(
            name="planning",
            mcp_tools=["taskmaster-server"],  # Could use task management tools
            description="Expands research ideas into detailed, actionable execution plans"
        )
        self.llm = llm or ChatGoogleGenerativeAI(
            model=settings.specialist_model,
            temperature=0.1,  # Lower temperature for structured planning
            max_tokens=None,
            timeout=None,
            max_retries=2,
            google_api_key=settings.google_api_key
        )
    
    async def process(self, state: AgentState) -> AgentState:
        """Expand a research idea into a detailed execution plan."""
        
        # Log start of planning
        self.log_message(state, "Starting research idea expansion and planning")
        
        current_task = state.get("current_task", "")
        metadata = state.get("metadata", {})
        
        # Create a planning prompt
        planning_prompt = f"""You are a Research Planning Specialist for astronomy. Your job is to take high-level research ideas and expand them into detailed, actionable execution plans.

CURRENT RESEARCH IDEA:
{current_task}

AVAILABLE SPECIALIST AGENTS FOR EXECUTION:
1. DATA_GATHERING: Access DESI, LSST, CMB databases and observational data
2. ANALYSIS: Statistical analysis, correlation studies, power spectra, parameter fitting  
3. THEORIST_SIMULATION: N-body simulations, cosmological modeling, theoretical predictions
4. LITERATURE_REVIEWER: ArXiv search, paper synthesis, research context

YOUR TASK:
1. Analyze the research idea and identify what needs to be done
2. Break it down into specific, actionable steps
3. Determine what data, analysis, simulations, and literature review are needed
4. Create a logical sequence of tasks
5. Identify potential challenges and alternatives

CRITICAL: Return ONLY valid JSON in your response.

{{
    "research_plan": {{
        "title": "Clear title for the research project",
        "objective": "What we're trying to accomplish",
        "approach": "High-level strategy",
        "detailed_steps": [
            {{
                "step_number": 1,
                "description": "What to do in this step",
                "agent_needed": "which specialist agent should handle this",
                "estimated_effort": "low/medium/high",
                "dependencies": ["list of previous steps needed"],
                "outputs_expected": "what this step will produce"
            }}
        ],
        "data_requirements": [
            "specific datasets or observations needed"
        ],
        "analysis_methods": [
            "statistical or computational methods to use"
        ],
        "potential_challenges": [
            "things that could go wrong or be difficult"
        ],
        "success_criteria": [
            "how we'll know if we succeeded"
        ]
    }},
    "next_step": {{
        "agent": "which agent to start with",
        "task": "specific first task to execute",
        "priority": "high/medium/low"
    }},
    "summary": "Brief explanation of the plan for the user"
}}"""

        messages = [
            SystemMessage(content=planning_prompt)
        ]

        # Add the research idea as a human message
        if current_task:
            messages.append(HumanMessage(content=current_task))

        try:
            response = await self.llm.ainvoke(messages)
            
            # Log the planning response
            self.log_message(state, f"Planning response generated ({len(response.content)} chars)")
            
            # Extract JSON from markdown if needed
            content = response.content.strip()
            if content.startswith("```json"):
                start_idx = content.find("```json") + 7
                end_idx = content.rfind("```")
                if end_idx > start_idx:
                    content = content[start_idx:end_idx].strip()
            elif content.startswith("```"):
                start_idx = content.find("```") + 3
                end_idx = content.rfind("```")
                if end_idx > start_idx:
                    content = content[start_idx:end_idx].strip()
            
            plan_data = json.loads(content)
            
            # Extract the plan and next step
            research_plan = plan_data.get("research_plan", {})
            next_step = plan_data.get("next_step", {})
            summary = plan_data.get("summary", "Research plan created")
            
            # Store the plan in state
            state["task_breakdown"] = research_plan.get("detailed_steps", [])
            state["metadata"]["research_plan"] = research_plan
            state["metadata"]["planning_complete"] = True
            
            # Create user-friendly response
            plan_summary = f"{summary}\n\n"
            plan_summary += f"**Research Plan: {research_plan.get('title', 'Untitled')}**\n\n"
            plan_summary += f"Objective: {research_plan.get('objective', 'Not specified')}\n\n"
            plan_summary += f"Approach: {research_plan.get('approach', 'Not specified')}\n\n"
            
            # Add step overview
            steps = research_plan.get("detailed_steps", [])
            if steps:
                plan_summary += f"Execution Plan ({len(steps)} steps):\n"
                for i, step in enumerate(steps[:3], 1):  # Show first 3 steps
                    plan_summary += f"{i}. {step.get('description', 'No description')} → {step.get('agent_needed', 'Unknown agent')}\n"
                if len(steps) > 3:
                    plan_summary += f"... and {len(steps) - 3} more steps\n"
            
            # Add next action
            if next_step:
                plan_summary += f"\nNext: I'll route to the {next_step.get('agent', 'appropriate')} agent to begin with: {next_step.get('task', 'the first task')}"
                
                # Set up routing to the next agent
                state["next_agent"] = next_step.get("agent", "data_gathering")
                state["current_task"] = next_step.get("task", current_task)
            else:
                state["next_agent"] = "orchestrator"  # Back to orchestrator for routing
            
            # Add the plan to conversation history
            state["messages"].append(AIMessage(content=plan_summary))
            self.log_message(state, f"Research plan created with {len(steps)} steps")
            
        except json.JSONDecodeError as e:
            # Handle JSON parsing errors
            error_msg = f"I had trouble creating a structured plan (JSON error: {str(e)}). Let me provide a simpler approach."
            self.log_message(state, f"Planning JSON parsing failed: {str(e)}")
            
            # Fallback: simple task breakdown
            simple_plan = f"{error_msg}\n\nBased on your request '{current_task}', I recommend starting with data gathering to understand what's available, then moving to simulation or analysis as appropriate."
            
            state["messages"].append(AIMessage(content=simple_plan))
            state["next_agent"] = "orchestrator"  # Let orchestrator decide
            
        except Exception as e:
            # Handle other errors
            error_msg = f"I encountered an error while planning: {str(e)}"
            self.log_message(state, f"Planning error: {str(e)}")
            state["messages"].append(AIMessage(content=error_msg))
            state["next_agent"] = "orchestrator"  # Safe fallback
            
        return state 