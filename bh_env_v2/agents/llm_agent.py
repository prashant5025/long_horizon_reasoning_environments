"""
Real LLM Agent Integration — Upgrade 2 (MEDIUM Priority).

Places a real LLM (Claude or OpenAI GPT) in the planning loop.
The HierarchicalPlanner's memory_context feeds the system prompt,
action_space() provides the constrained output format, and
latency budgeting reduces beam_depth for API-bound decisions.

Supports:
  - Anthropic Claude (claude-sonnet-4-20250514, claude-opus-4-20250514)
  - OpenAI GPT (gpt-4o, gpt-4o-mini)
  - Any OpenAI-compatible API endpoint
"""
from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..engine.types import Action, ActionType, EnvID, Event, Observation
from ..engine.environments.base import BaseEnvironment
from ..memory.memory_system import MemorySystem
from ..planning.planner import HierarchicalPlanner
from ..planning.value_fn import ValueFunctionRegistry
from ..skills.skill_library import SkillLibrary, Transition
from .v2_agent import AgentContext, V2Agent


# ═════════════════════════════════════════════════════════════════════════════
#  LLM Interface — Abstract Base
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class LLMResponse:
    """Parsed LLM response."""
    action_type: str
    target_id: Optional[int] = None
    reasoning: str = ""
    raw_text: str = ""
    latency_ms: float = 0.0
    tokens_used: int = 0


class LLMInterface(ABC):
    """Abstract interface for LLM API calls."""

    @abstractmethod
    def query(
        self,
        system_prompt: str,
        user_prompt: str,
        valid_actions: List[str],
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Send a prompt to the LLM and parse the response."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the API key is configured and the service is reachable."""


# ═════════════════════════════════════════════════════════════════════════════
#  Claude LLM (Anthropic)
# ═════════════════════════════════════════════════════════════════════════════

class ClaudeLLM(LLMInterface):
    """Anthropic Claude integration."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        max_tokens: int = 256,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.max_tokens = max_tokens
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package required: pip install anthropic"
                )
        return self._client

    def is_available(self) -> bool:
        return bool(self.api_key)

    def query(
        self,
        system_prompt: str,
        user_prompt: str,
        valid_actions: List[str],
        temperature: float = 0.3,
    ) -> LLMResponse:
        client = self._get_client()

        for attempt in range(self._max_retries):
            try:
                t0 = time.time()
                response = client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                latency = (time.time() - t0) * 1000

                raw_text = response.content[0].text
                tokens = response.usage.input_tokens + response.usage.output_tokens

                return self._parse_response(raw_text, valid_actions, latency, tokens)

            except Exception as e:
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (attempt + 1))
                else:
                    return LLMResponse(
                        action_type=valid_actions[0] if valid_actions else "NOOP",
                        reasoning=f"LLM error after {self._max_retries} retries: {e}",
                        raw_text=str(e),
                    )

    def _parse_response(
        self,
        raw_text: str,
        valid_actions: List[str],
        latency: float,
        tokens: int,
    ) -> LLMResponse:
        """Parse LLM response, extracting action type and reasoning."""
        # Try JSON parsing first
        try:
            data = json.loads(raw_text)
            action = data.get("action", "").upper()
            if action in valid_actions:
                return LLMResponse(
                    action_type=action,
                    target_id=data.get("target_id"),
                    reasoning=data.get("reasoning", ""),
                    raw_text=raw_text,
                    latency_ms=latency,
                    tokens_used=tokens,
                )
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback: search for action keywords in text
        text_upper = raw_text.upper()
        for action in valid_actions:
            if action in text_upper:
                return LLMResponse(
                    action_type=action,
                    reasoning=raw_text[:200],
                    raw_text=raw_text,
                    latency_ms=latency,
                    tokens_used=tokens,
                )

        # Default to first valid action
        return LLMResponse(
            action_type=valid_actions[0] if valid_actions else "NOOP",
            reasoning=f"Could not parse action from: {raw_text[:100]}",
            raw_text=raw_text,
            latency_ms=latency,
            tokens_used=tokens,
        )


# ═════════════════════════════════════════════════════════════════════════════
#  OpenAI LLM
# ═════════════════════════════════════════════════════════════════════════════

class OpenAILLM(LLMInterface):
    """OpenAI GPT integration (also works with compatible APIs)."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 256,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url
        self.max_tokens = max_tokens
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import openai
                kwargs = {"api_key": self.api_key}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self._client = openai.OpenAI(**kwargs)
            except ImportError:
                raise ImportError(
                    "openai package required: pip install openai"
                )
        return self._client

    def is_available(self) -> bool:
        return bool(self.api_key)

    def query(
        self,
        system_prompt: str,
        user_prompt: str,
        valid_actions: List[str],
        temperature: float = 0.3,
    ) -> LLMResponse:
        client = self._get_client()

        for attempt in range(self._max_retries):
            try:
                t0 = time.time()
                response = client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                latency = (time.time() - t0) * 1000

                raw_text = response.choices[0].message.content or ""
                tokens = (response.usage.total_tokens if response.usage else 0)

                return self._parse_response(raw_text, valid_actions, latency, tokens)

            except Exception as e:
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (attempt + 1))
                else:
                    return LLMResponse(
                        action_type=valid_actions[0] if valid_actions else "NOOP",
                        reasoning=f"LLM error after {self._max_retries} retries: {e}",
                        raw_text=str(e),
                    )

    def _parse_response(
        self,
        raw_text: str,
        valid_actions: List[str],
        latency: float,
        tokens: int,
    ) -> LLMResponse:
        """Parse response — same logic as Claude."""
        try:
            data = json.loads(raw_text)
            action = data.get("action", "").upper()
            if action in valid_actions:
                return LLMResponse(
                    action_type=action,
                    target_id=data.get("target_id"),
                    reasoning=data.get("reasoning", ""),
                    raw_text=raw_text,
                    latency_ms=latency,
                    tokens_used=tokens,
                )
        except (json.JSONDecodeError, AttributeError):
            pass

        text_upper = raw_text.upper()
        for action in valid_actions:
            if action in text_upper:
                return LLMResponse(
                    action_type=action,
                    reasoning=raw_text[:200],
                    raw_text=raw_text,
                    latency_ms=latency,
                    tokens_used=tokens,
                )

        return LLMResponse(
            action_type=valid_actions[0] if valid_actions else "NOOP",
            reasoning=f"Could not parse: {raw_text[:100]}",
            raw_text=raw_text,
            latency_ms=latency,
            tokens_used=tokens,
        )


# ═════════════════════════════════════════════════════════════════════════════
#  Prompt Builder
# ═════════════════════════════════════════════════════════════════════════════

class PromptBuilder:
    """Constructs system and user prompts for the LLM planner."""

    _SYSTEM_TEMPLATE = """You are an expert business strategist agent operating in a simulated enterprise environment.

ENVIRONMENT: {env_description}

YOUR ROLE: Make strategic decisions to achieve the goals. You must choose exactly one action per step from the valid action list.

RESPONSE FORMAT: Respond with valid JSON only:
{{"action": "<ACTION_TYPE>", "target_id": <int_or_null>, "reasoning": "<brief_explanation>"}}

RULES:
- Choose ONLY from the valid actions listed
- Consider both short-term and long-term consequences
- Pay attention to shock events and adapt strategy
- Manage stakeholder relationships proactively
- Monitor resource constraints (budget, morale, SLA)"""

    _ENV_DESCRIPTIONS = {
        "sales": "Enterprise Sales Pipeline — Close a $2.4M deal across 11 stakeholders, 6 phases. "
                 "Shocks: champion departure (step ~80), budget freeze (step ~170), competitor threat (step ~260).",
        "pm": "Program Rescue — Deliver a failing $6M program with 4 workstreams, 47 risks. "
              "Shocks: architect departure (step ~100), vendor bankruptcy (step ~220), scope creep (step ~340).",
        "hr_it": "IT Transformation — Migrate 8,000 users with 300 compliance instructions. "
                 "Shock: ransomware attack (step ~240). Maintain SLA while fulfilling all instructions.",
    }

    def system_prompt(self, env_id: str) -> str:
        desc = self._ENV_DESCRIPTIONS.get(env_id, "Unknown environment")
        return self._SYSTEM_TEMPLATE.format(env_description=desc)

    def user_prompt(
        self,
        step: int,
        phase: int,
        state_digest: Dict[str, Any],
        valid_actions: List[str],
        memory_context: str,
        plan_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        parts = [
            f"STEP: {step}  |  PHASE: {phase}",
            "",
            "STATE:",
            self._format_digest(state_digest),
            "",
        ]

        if memory_context:
            parts.extend([
                "MEMORY CONTEXT:",
                memory_context[:1500],  # Truncate to manage token budget
                "",
            ])

        if plan_info:
            parts.extend([
                "PLAN STATUS:",
                f"  Active tasks: {plan_info.get('active_tasks', [])}",
                f"  Goal progress: {plan_info.get('goal_progress', {})}",
                "",
            ])

        parts.extend([
            "VALID ACTIONS:",
            json.dumps(valid_actions),
            "",
            "Choose your action (respond with JSON):",
        ])

        return "\n".join(parts)

    @staticmethod
    def _format_digest(digest: Dict[str, Any]) -> str:
        lines = []
        for k, v in digest.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.2f}")
            elif isinstance(v, (list, dict)):
                lines.append(f"  {k}: {json.dumps(v)[:80]}")
            else:
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
#  LLM Agent — V2Agent with LLM in the Planning Loop
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class LLMAgentConfig:
    """Configuration for the LLM agent."""
    llm_provider: str = "claude"        # "claude" or "openai"
    model: Optional[str] = None         # Override default model
    api_key: Optional[str] = None       # Override env var
    temperature: float = 0.3
    max_tokens: int = 256
    beam_depth: int = 1                 # Reduced for latency (was 3)
    beam_width: int = 2                 # Reduced for latency (was 4)
    llm_every_n_steps: int = 5          # Use LLM every N steps, heuristic otherwise
    fallback_to_heuristic: bool = True  # Fall back if LLM unavailable


class LLMAgent:
    """
    Wraps V2Agent with a real LLM in the planning loop.

    Latency budgeting: beam_depth is reduced from 3 to 1 (each LLM call
    takes ~500ms vs <1ms for heuristic). The LLM is queried every N steps
    with heuristic fallback in between.
    """

    def __init__(
        self,
        ctx: AgentContext,
        config: LLMAgentConfig,
        env_id: str,
    ) -> None:
        self._ctx = ctx
        self._config = config
        self._env_id = env_id
        self._prompt_builder = PromptBuilder()
        self._llm = self._create_llm(config)
        self._base_agent = V2Agent(ctx)
        self._step_count: int = 0
        self._total_tokens: int = 0
        self._total_latency_ms: float = 0.0
        self._llm_decisions: int = 0

    def _create_llm(self, config: LLMAgentConfig) -> LLMInterface:
        if config.llm_provider == "claude":
            return ClaudeLLM(
                model=config.model or "claude-sonnet-4-20250514",
                api_key=config.api_key,
                max_tokens=config.max_tokens,
            )
        elif config.llm_provider == "openai":
            return OpenAILLM(
                model=config.model or "gpt-4o",
                api_key=config.api_key,
                max_tokens=config.max_tokens,
            )
        else:
            raise ValueError(f"Unknown LLM provider: {config.llm_provider}")

    def decide(self, obs: Observation, env: BaseEnvironment) -> Action:
        """Decide using LLM every N steps, heuristic otherwise."""
        self._step_count += 1

        # Use LLM on the scheduled steps, or fall back to V2Agent
        use_llm = (
            self._step_count % self._config.llm_every_n_steps == 0
            and self._llm.is_available()
        )

        if not use_llm:
            return self._base_agent.decide(obs, env)

        # Build LLM prompt
        state_digest = env.state_digest()
        valid_action_types = env.action_space()
        valid_names = [
            a.name if hasattr(a, 'name') else a.action_type.name
            for a in valid_action_types
        ]

        memory_context = self._ctx.memory.context_for_agent(
            obs.step, self._build_query(obs),
        )

        system = self._prompt_builder.system_prompt(self._env_id)
        user = self._prompt_builder.user_prompt(
            step=obs.step,
            phase=obs.phase,
            state_digest=state_digest,
            valid_actions=valid_names,
            memory_context=memory_context,
        )

        # Query LLM
        response = self._llm.query(
            system, user, valid_names,
            temperature=self._config.temperature,
        )

        self._total_tokens += response.tokens_used
        self._total_latency_ms += response.latency_ms
        self._llm_decisions += 1

        # Convert response to Action
        try:
            action_type = ActionType[response.action_type]
        except KeyError:
            # Fallback to heuristic if LLM returned invalid action
            if self._config.fallback_to_heuristic:
                return self._base_agent.decide(obs, env)
            action_type = ActionType.NOOP

        return Action(
            action_type=action_type,
            target_id=response.target_id,
        )

    def post_step(
        self,
        obs: Observation,
        action: Action,
        shaped_reward: float,
        state_digest: Dict[str, Any],
    ) -> None:
        """Delegate post-step to base agent."""
        self._base_agent.post_step(obs, action, shaped_reward, state_digest)

    @property
    def trajectory(self) -> List[Transition]:
        return self._base_agent.trajectory

    def stats(self) -> Dict[str, Any]:
        return {
            "llm_decisions": self._llm_decisions,
            "heuristic_decisions": self._step_count - self._llm_decisions,
            "total_tokens": self._total_tokens,
            "total_latency_ms": self._total_latency_ms,
            "avg_latency_ms": (
                self._total_latency_ms / self._llm_decisions
                if self._llm_decisions > 0 else 0
            ),
            "llm_provider": self._config.llm_provider,
            "model": self._config.model,
        }

    def _build_query(self, obs: Observation) -> str:
        parts = list(obs.tags)
        parts.append(f"phase={obs.phase}")
        parts.append(f"step={obs.step}")
        return " ".join(parts)
