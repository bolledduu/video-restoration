"""
Multimodal Interview-Analysis Agent Evaluator.
Executes the agent decision pipeline, formats the strict target prompt,
and separates agent reasoning failures from underlying ASD perception errors.
"""

import json
from typing import Dict, Any, Tuple
from config import AGENT_PROMPT_TEMPLATE

class InterviewAnalysisAgent:
    """
    Multimodal Interview-Analysis Agent.
    Receives video frames, target statement, timestamps, transcript, and participant mappings.
    Produces structured attribution decision with uncertainty support.
    """
    def __init__(self, reasoning_noise_prob: float = 0.05):
        # Optional small stochastic reasoning noise (models occasional LLM formatting or prior misinterpretation)
        self.reasoning_noise_prob = reasoning_noise_prob

    def run(
        self,
        statement: str,
        start_time_str: str,
        end_time_str: str,
        asd_result: Dict[str, Any],
        true_speaker: str,
        simulate_reasoning_drift: bool = False
    ) -> Dict[str, Any]:
        """
        Executes interview analysis agent decision.
        Evaluates visual evidence from downstream ASD probabilities.
        """
        asd_decision = asd_result["asd_decision"]
        confidence = asd_result["decision_confidence"]
        
        # Agent reasoning logic:
        # If ASD is uncertain or low confidence, agent reports Uncertain
        if asd_decision == "Uncertain" or confidence < 0.40:
            agent_decision = "Uncertain"
            reasoning = "Visual lip movements are insufficient or contradictory; attribution is uncertain."
        else:
            agent_decision = asd_decision
            reasoning = f"Observed synchronized lip articulatory cycles on {asd_decision} during the statement utterance."

        # Simulate agent reasoning failure (independent of visual evidence) if drift flag is set
        agent_reasoning_error = False
        if simulate_reasoning_drift and agent_decision != "Uncertain":
            # Agent hallucinates opposite speaker due to text semantic bias
            inverted = "Person B" if agent_decision == "Person A" else "Person A"
            agent_decision = inverted
            reasoning = f"[Agent Reasoning Artifact] Linguistic context led to attributing statement to {inverted} despite visual features."
            agent_reasoning_error = True

        return {
            "attributed_participant": agent_decision,
            "start_timestamp": start_time_str,
            "end_timestamp": end_time_str,
            "confidence": float(confidence),
            "reasoning": reasoning,
            "underlying_asd_decision": asd_decision,
            "agent_reasoning_error": agent_reasoning_error
        }

def evaluate_agent_output(
    agent_output: Dict[str, Any],
    true_speaker: str
) -> Dict[str, Any]:
    """
    Evaluates agent output and categorizes error etiology:
    - SUCCESS
    - ABSTENTION
    - PERCEPTUAL_ASD_FAILURE
    - AGENT_REASONING_FAILURE
    """
    pred = agent_output["attributed_participant"]
    asd_pred = agent_output["underlying_asd_decision"]

    if pred == "Uncertain":
        return {
            "status": "ABSTENTION",
            "is_correct": False,
            "is_abstention": True,
            "attribution_error": False,
            "category": "Abstention (Uncertain)"
        }

    is_correct = (pred == true_speaker)
    asd_correct = (asd_pred == true_speaker)

    if is_correct:
        return {
            "status": "SUCCESS",
            "is_correct": True,
            "is_abstention": False,
            "attribution_error": False,
            "category": "Correct Attribution"
        }
    else:
        # Determine whether failure came from ASD or Agent Reasoning
        if not asd_correct:
            category = "Perceptual ASD Failure (Visual/Audio Error)"
        else:
            category = "Agent Reasoning Failure (Decision Layer Error)"

        return {
            "status": "FAILURE",
            "is_correct": False,
            "is_abstention": False,
            "attribution_error": True,
            "category": category
        }
