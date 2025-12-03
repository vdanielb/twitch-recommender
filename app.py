import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import streamlit as st

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "als_model.npz"
LOOKUP_PATH = ARTIFACTS_DIR / "lookup.json"


@st.cache_resource(show_spinner=False)
def load_resources() -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[int, str], List[str]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "ALS model artifact not found. Run main.ipynb to regenerate artifacts."
        )
    if not LOOKUP_PATH.exists():
        raise FileNotFoundError(
            "Lookup metadata not found. Run main.ipynb to regenerate artifacts."
        )

    with LOOKUP_PATH.open("r", encoding="utf-8") as fp:
        lookup_payload = json.load(fp)

    artifact_info = lookup_payload.get("artifacts", {})
    popularity_path = ARTIFACTS_DIR / artifact_info.get("popularity_norm", "popularity_norm.npy")
    if not popularity_path.exists():
        raise FileNotFoundError(
            "Popularity weights are missing. Rerun the artifact cell in main.ipynb."
        )

    with np.load(MODEL_PATH, mmap_mode="r") as payload:
        item_factors = np.array(payload["item_factors"], dtype=np.float32)

    popularity_norm = np.load(popularity_path).astype(np.float32)
    if popularity_norm.shape[0] != item_factors.shape[0]:
        raise ValueError("Popularity vector length does not match number of items.")

    item_id_map: Dict[str, int] = {
        name: int(idx) for name, idx in lookup_payload["item_id_map"].items()
    }
    idx_to_item: Dict[int, str] = {int(k): v for k, v in lookup_payload["idx_to_item"].items()}
    streamer_names = sorted(item_id_map.keys())

    return item_factors, popularity_norm, item_id_map, idx_to_item, streamer_names


(
    item_factors,
    popularity_norm,
    item_id_map,
    idx_to_item,
    streamer_names,
) = load_resources()


def _build_virtual_profile(favorite_streamers: Sequence[str]) -> Tuple[np.ndarray | None, List[str]]:
    valid_indices: List[int] = []
    unknown: List[str] = []
    for name in favorite_streamers:
        cleaned = name.strip()
        if not cleaned:
            continue
        idx = item_id_map.get(cleaned)
        if idx is None:
            unknown.append(cleaned)
        else:
            valid_indices.append(idx)

    if not valid_indices:
        return None, unknown

    profile = item_factors[valid_indices].mean(axis=0)
    norm = np.linalg.norm(profile)
    if norm == 0:
        return None, unknown
    return profile / norm, unknown


def hybrid_recommendations_from_streamers(
    favorite_streamers: Sequence[str],
    top_n: int = 10,
    pop_weight: float = 0.01,
) -> Tuple[List[Dict[str, float | str | int]], List[str]]:
    profile, unknown = _build_virtual_profile(favorite_streamers)
    if profile is None:
        return [], unknown

    base_scores = item_factors @ profile
    seed_indices = [item_id_map[name] for name in favorite_streamers if name in item_id_map]
    if seed_indices:
        base_scores[np.array(seed_indices, dtype=np.int64)] = -np.inf

    boosted_scores = base_scores + pop_weight * popularity_norm
    k = min(top_n, boosted_scores.shape[0])
    candidate_indices = np.argpartition(-boosted_scores, range(k))[:k]
    candidate_indices = candidate_indices[np.argsort(-boosted_scores[candidate_indices])]

    recommendations: List[Dict[str, float | str | int]] = []
    for rank, idx in enumerate(candidate_indices, start=1):
        recommendations.append(
            {
                "Rank": rank,
                "Streamer": idx_to_item.get(int(idx), f"Streamer #{idx}"),
                "HybridScore": float(boosted_scores[idx]),
            }
        )
    return recommendations, unknown


st.set_page_config(page_title="Hybrid Twitch Recommender", page_icon="🎮")
st.title("Tell Us Who You Watch")
st.write(
    "Select one or more streamers you already enjoy and the hybrid ALS+pop model will "
    "suggest 10 more channels with overlapping audiences."
)
st.caption("Recommendations are based on 2019 Twitch viewing data.")

if not streamer_names:
    st.error("Streamer list is empty. Regenerate artifacts from main.ipynb.")
else:
    chosen_streamers = st.multiselect(
        "Streamers you already watch",
        streamer_names,
        max_selections=8,
        help="Type to search; select up to 8 channels."
    )
    pop_weight = st.slider(
        "How much do you care about popularity?",
        min_value=0.0,
        max_value=0.10,
        value=0.01,
        step=0.01,
        help="Higher settings favor globally popular streamers."
    )
    deduped_streamers = list(dict.fromkeys(chosen_streamers))

    if not deduped_streamers:
        st.info("Select or type at least one streamer to get recommendations.")
    else:
        results, unknown = hybrid_recommendations_from_streamers(
            deduped_streamers,
            top_n=10,
            pop_weight=pop_weight,
        )
        if unknown:
            st.warning(
                "These streamers were not found in the trained dataset: "
                + ", ".join(unknown)
            )
        if not results:
            st.info("Unable to build a profile from the provided streamers. Try different ones.")
        else:
            st.subheader("Because you already watch:")
            st.caption(
                ", ".join(deduped_streamers[:6]) + ("…" if len(deduped_streamers) > 6 else "")
            )
            st.subheader("Hybrid ALS+pop Recommendations")
            st.dataframe(results, use_container_width=True)
