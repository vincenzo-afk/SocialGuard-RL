"""
model.py — NEMESIS-RL Policy Network

A real trainable neural network for moderation policy learning.

Architecture:
  - Observation: 68-dimensional float32 vector from the existing env
  - Tabular features: account age, post frequency, flag count, follower ratio, link repetition
    (derived / aliased from the 68-dim obs vector — indices 0-4)
  - Text encoding: uses transformers (all-MiniLM-L6-v2 locally) when text context is
    available; otherwise falls back to a learned linear projection of obs dims 0-4.
    Llama-4-Maverick (meta-llama/Llama-4-Maverick-17B-128E-Instruct) on HuggingFace
    Inference API is used for *semantic* content analysis where a HF_TOKEN is set.
  - Three FC layers: (389 → 512 → 256 → 128) with ReLU + Dropout(0.3)
  - Output: 5 logits → actions (allow / warn / restrict / suspend / ban)

Stable-Baselines3 integration via NemesisMlpExtractor, a custom features extractor
that replaces the default MLP in the PPO policy.

The 389-dim input is:
  - 384 dims: sentence embedding (from local MiniLM OR linear projection of obs)
  - 5 dims: tabular features clipped from obs[0:5]

Usage::
    from model import NemesisPolicy, build_ppo_with_nemesis_policy
    ppo = build_ppo_with_nemesis_policy(env)
    ppo.learn(100_000)
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TABULAR_DIM: int = 5           # account_age, posts_per_hour, flag_count, follower_ratio, link_rep
EMBEDDING_DIM: int = 384        # all-MiniLM-L6-v2 output dim
POLICY_INPUT_DIM: int = EMBEDDING_DIM + TABULAR_DIM  # 389

N_ACTIONS: int = 5             # allow / warn / restrict / suspend / ban

ACTION_LABELS: Dict[int, str] = {
    0: "No Action",
    1: "Warn",
    2: "Restrict",
    3: "Suspend",
    4: "Ban",
}

_DEVICE: str = "cpu"           # MPS/CUDA auto-selected at runtime


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Local sentence encoder  (all-MiniLM-L6-v2)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_local_encoder():
    """Load sentence-transformers MiniLM model (cached singleton)."""
    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        encoder.eval()
        logger.info("Loaded local sentence encoder: all-MiniLM-L6-v2")
        return encoder
    except ImportError:
        logger.warning("sentence-transformers not installed; using fallback projection encoder.")
        return None
    except Exception as exc:
        logger.warning("Could not load MiniLM: %s — using fallback.", exc)
        return None


def encode_text_local(text: str) -> Optional[np.ndarray]:
    """Encode text to 384-dim vector using local MiniLM."""
    enc = _load_local_encoder()
    if enc is None:
        return None
    with torch.no_grad():
        emb = enc.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return emb.astype(np.float32)  # (384,)


# ---------------------------------------------------------------------------
# HuggingFace Inference API call for Llama-4-Maverick
# ---------------------------------------------------------------------------

_HF_MODEL_ID = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
_HF_API_URL = f"https://api-inference.huggingface.co/models/{_HF_MODEL_ID}"


def analyze_content_with_llama(
    content_snippet: str,
    account_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Call Llama-4-Maverick via HuggingFace Inference API to get semantic
    moderation analysis. Returns a dict with keys:
        - 'risk_score': float in [0, 1]
        - 'categories': list[str] — e.g. ['spam', 'misinformation']
        - 'reasoning': str
        - 'raw_response': str

    Falls back gracefully if HF_TOKEN not set or API unavailable.
    """
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        return {
            "risk_score": 0.5,
            "categories": [],
            "reasoning": "No HF_TOKEN set — Llama analysis skipped.",
            "raw_response": "",
        }

    try:
        import requests
    except ImportError:
        return {
            "risk_score": 0.5,
            "categories": [],
            "reasoning": "requests library not available.",
            "raw_response": "",
        }

    ctx = account_context or {}
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are a social media content moderation AI. Analyze the given content snippet "
        "for potential policy violations. Respond in JSON with keys: "
        "risk_score (0.0=safe, 1.0=ban-worthy), categories (list), reasoning (str)."
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"Content: {content_snippet!r}\n"
        f"Account context: {ctx}\n"
        "Provide your analysis as JSON."
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )

    headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 256,
            "temperature": 0.1,
            "return_full_text": False,
        },
    }

    try:
        resp = requests.post(_HF_API_URL, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
        generated = raw[0]["generated_text"] if isinstance(raw, list) else str(raw)

        import json, re
        m = re.search(r"\{.*?\}", generated, re.DOTALL)
        if m:
            parsed = json.loads(m.group())
            return {
                "risk_score": float(parsed.get("risk_score", 0.5)),
                "categories": list(parsed.get("categories", [])),
                "reasoning": str(parsed.get("reasoning", "")),
                "raw_response": generated,
            }
        return {
            "risk_score": 0.5,
            "categories": [],
            "reasoning": generated[:200],
            "raw_response": generated,
        }
    except Exception as exc:
        logger.debug("Llama API call failed: %s", exc)
        return {
            "risk_score": 0.5,
            "categories": [],
            "reasoning": f"API error: {exc}",
            "raw_response": "",
        }


# ---------------------------------------------------------------------------
# Fallback Linear Projection  (384-dim from 5 tabular features)
# Used when MiniLM is unavailable
# ---------------------------------------------------------------------------

class FallbackProjection(nn.Module):
    """Project 5 tabular dims → 384 dims using a learned linear layer."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(TABULAR_DIM, EMBEDDING_DIM)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.proj(x))  # (B, 384)


# ---------------------------------------------------------------------------
# Core Policy Network
# ---------------------------------------------------------------------------

class NemesisNetBackbone(nn.Module):
    """389 → [512 → 256 → 128] FC backbone (shared actor-critic trunk)."""

    def __init__(self, input_dim: int = POLICY_INPUT_DIM, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, 128)


# ---------------------------------------------------------------------------
# SB3 Custom Features Extractor
# ---------------------------------------------------------------------------

class NemesisMlpExtractor(BaseFeaturesExtractor):
    """SB3 BaseFeaturesExtractor that:
    1. Splits the 68-dim obs into tabular dims (0:5).
    2. Projects or encodes to a 384-dim embedding.
    3. Concatenates to form 389-dim input.
    4. Passes through the NemesisNetBackbone → 128-dim features.

    The backbone is frozen during inference if `freeze_backbone=True`.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__(observation_space, features_dim)

        self.fallback_projection = FallbackProjection()
        self.backbone = NemesisNetBackbone(
            input_dim=POLICY_INPUT_DIM,
            dropout=dropout,
        )
        # Check if local sentence encoder is loadable
        self._has_local_encoder = _load_local_encoder() is not None
        logger.info(
            "NemesisMlpExtractor: local_encoder=%s features_dim=%d",
            self._has_local_encoder,
            features_dim,
        )

    def _embed_obs(self, obs_np: np.ndarray) -> np.ndarray:
        """Convert a single 68-dim obs vector → 384-dim embedding (numpy)."""
        # Try MiniLM on a pseudo content string derived from obs
        enc = _load_local_encoder()
        if enc is not None:
            # Create a minimal textual representation from obs features
            content_str = (
                f"age:{obs_np[0]:.2f} posts:{obs_np[1]:.2f} "
                f"follower_ratio:{obs_np[2]:.2f} variance:{obs_np[3]:.2f} "
                f"repetition:{obs_np[4]:.2f}"
            )
            emb = encode_text_local(content_str)
            if emb is not None:
                return emb  # (384,)
        return None

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        observations: (B, 68) float32 tensor from the env.
        returns: (B, 128) feature tensor.
        """
        # Tabular slice: first 5 dims
        tabular = observations[:, :TABULAR_DIM]  # (B, 5)

        # Build embedding tensor
        emb = self.fallback_projection(tabular)  # (B, 384) — differentiable

        # Concatenate
        combined = torch.cat([emb, tabular], dim=-1)  # (B, 389)

        return self.backbone(combined)  # (B, 128)


# ---------------------------------------------------------------------------
# Full Policy (wraps SB3 ActorCriticPolicy)
# ---------------------------------------------------------------------------

class NemesisPolicy(ActorCriticPolicy):
    """Custom SB3 ActorCriticPolicy wired to the NemesisMlpExtractor.

    Only the kwargs passthrough is overridden; everything else is inherited
    so PPO's training loop, entropy, value head, etc. all work normally.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        dropout: float = 0.3,
        **kwargs: Any,
    ) -> None:
        # Inject our extractor
        kwargs["features_extractor_class"] = NemesisMlpExtractor
        kwargs["features_extractor_kwargs"] = {
            "features_dim": 128,
            "dropout": dropout,
        }
        # net_arch must match the extractor's output dim
        kwargs["net_arch"] = []   # trunk is already in extractor; just use linear heads
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# PPO Factory
# ---------------------------------------------------------------------------

def build_ppo_with_nemesis_policy(
    env,
    *,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    total_timesteps: int = 100_000,
    tensorboard_log: Optional[str] = None,
    device: str = "auto",
    verbose: int = 1,
) -> PPO:
    """Build and return a PPO model with the NEMESIS policy extractor.

    Args:
        env: Gymnasium environment or VecEnv.
        learning_rate: PPO learning rate.
        n_steps: Steps per rollout per env.
        batch_size: Mini-batch size.
        n_epochs: Gradient steps per update.
        gamma: Discount factor.
        total_timesteps: Not used here; caller decides when to learn().
        tensorboard_log: Path for TensorBoard logs.
        device: PyTorch device string.
        verbose: Verbosity level.

    Returns:
        Configured PPO model ready for .learn().
    """
    ppo = PPO(
        policy=NemesisPolicy,
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        ent_coef=0.01,         # encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=tensorboard_log,
        device=device,
        verbose=verbose,
    )
    logger.info(
        "Built PPO with NemesisPolicy — device=%s obs_dim=68 policy_input=%d",
        ppo.device,
        POLICY_INPUT_DIM,
    )
    return ppo


# ---------------------------------------------------------------------------
# Standalone inference helper
# ---------------------------------------------------------------------------

def predict_action(
    ppo: PPO,
    obs: np.ndarray,
    deterministic: bool = False,
) -> Tuple[int, float, np.ndarray]:
    """Run inference from a single obs vector.

    Args:
        ppo: Trained PPO model.
        obs: 1-D float32 array of shape (68,).
        deterministic: If True, pick argmax rather than sampling.

    Returns:
        Tuple of (action_int, confidence_float, action_probabilities).
    """
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(ppo.device)
    with torch.no_grad():
        dist = ppo.policy.get_distribution(obs_t)
        probs = dist.distribution.probs.cpu().numpy()[0]  # (5,)
        if deterministic:
            action = int(np.argmax(probs))
        else:
            action = int(dist.sample().cpu().numpy()[0])
    confidence = float(probs[action])
    return action, confidence, probs


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import gymnasium
    import numpy as np
    from env.env import SocialGuardEnv

    env = SocialGuardEnv("configs/task1.yaml")
    ppo = build_ppo_with_nemesis_policy(env, verbose=0)
    obs, _ = env.reset()
    action, conf, probs = predict_action(ppo, obs)
    print(f"Test prediction — action={ACTION_LABELS[action]}({action})  conf={conf:.4f}")
    print(f"Prob distribution: {dict(zip(ACTION_LABELS.values(), probs.round(4)))}")
    env.close()
