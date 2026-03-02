"""Ground truth definitions for synthetic Twitter data.

Defines user archetypes, content topics, and engagement rules that
serve as ground truth for verifying model learning.

The engagement rules specify the probability that a user of a given
archetype will take a specific action on content of a given topic.
"""

from dataclasses import dataclass, field
from enum import Enum


class UserArchetype(str, Enum):
    """User behavior archetypes."""
    SPORTS_FAN = "sports_fan"      # Engages heavily with sports content
    POLITICAL_L = "political_L"    # Left-leaning political engagement
    POLITICAL_R = "political_R"    # Right-leaning political engagement
    TECH_BRO = "tech_bro"          # Tech/startup content enthusiast
    LURKER = "lurker"              # Passive consumer, likes but rarely shares
    POWER_USER = "power_user"      # High engagement across all action types


class ContentTopic(str, Enum):
    """Content topics for posts."""
    SPORTS = "sports"
    POLITICS_L = "politics_L"      # Left-leaning political content
    POLITICS_R = "politics_R"      # Right-leaning political content
    TECH = "tech"
    ENTERTAINMENT = "entertainment"
    NEWS = "news"                  # Mixed/neutral news


# User archetype distribution (sum = 100%)
ARCHETYPE_DISTRIBUTION: dict[UserArchetype, float] = {
    UserArchetype.SPORTS_FAN: 0.15,
    UserArchetype.POLITICAL_L: 0.15,
    UserArchetype.POLITICAL_R: 0.15,
    UserArchetype.TECH_BRO: 0.15,
    UserArchetype.LURKER: 0.20,
    UserArchetype.POWER_USER: 0.20,
}

# Content topic distribution (sum = 100%)
TOPIC_DISTRIBUTION: dict[ContentTopic, float] = {
    ContentTopic.SPORTS: 0.25,
    ContentTopic.POLITICS_L: 0.125,
    ContentTopic.POLITICS_R: 0.125,
    ContentTopic.TECH: 0.20,
    ContentTopic.ENTERTAINMENT: 0.20,
    ContentTopic.NEWS: 0.10,
}

# Number of authors per topic
AUTHORS_PER_TOPIC: dict[ContentTopic, int] = {
    ContentTopic.SPORTS: 20,
    ContentTopic.POLITICS_L: 15,
    ContentTopic.POLITICS_R: 15,
    ContentTopic.TECH: 20,
    ContentTopic.ENTERTAINMENT: 20,
    ContentTopic.NEWS: 10,
}


@dataclass
class ActionProbabilities:
    """Probabilities for each action type.

    Only includes actions with non-zero probability.
    Actions not listed are assumed to have probability 0.
    """
    favorite: float = 0.0
    reply: float = 0.0
    repost: float = 0.0
    photo_expand: float = 0.0
    click: float = 0.0
    profile_click: float = 0.0
    vqv: float = 0.0
    share: float = 0.0
    share_via_dm: float = 0.0
    share_via_copy_link: float = 0.0
    dwell: float = 0.0
    quote: float = 0.0
    quoted_click: float = 0.0
    follow_author: float = 0.0
    not_interested: float = 0.0
    block_author: float = 0.0
    mute_author: float = 0.0
    report: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "favorite_score": self.favorite,
            "reply_score": self.reply,
            "repost_score": self.repost,
            "photo_expand_score": self.photo_expand,
            "click_score": self.click,
            "profile_click_score": self.profile_click,
            "vqv_score": self.vqv,
            "share_score": self.share,
            "share_via_dm_score": self.share_via_dm,
            "share_via_copy_link_score": self.share_via_copy_link,
            "dwell_score": self.dwell,
            "quote_score": self.quote,
            "quoted_click_score": self.quoted_click,
            "follow_author_score": self.follow_author,
            "not_interested_score": self.not_interested,
            "block_author_score": self.block_author,
            "mute_author_score": self.mute_author,
            "report_score": self.report,
        }

    def to_array(self) -> list[float]:
        """Convert to array in Phoenix action order."""
        return [
            self.favorite,
            self.reply,
            self.repost,
            self.photo_expand,
            self.click,
            self.profile_click,
            self.vqv,
            self.share,
            self.share_via_dm,
            self.share_via_copy_link,
            self.dwell,
            self.quote,
            self.quoted_click,
            self.follow_author,
            self.not_interested,
            self.block_author,
            self.mute_author,
            self.report,
        ]


# Engagement rules: (archetype, topic) -> action probabilities
# Key: (UserArchetype, ContentTopic or "*" for wildcard)
# Value: ActionProbabilities

ENGAGEMENT_RULES: dict[tuple[UserArchetype, str], ActionProbabilities] = {
    # =========================================================================
    # SPORTS FAN - High engagement with sports, ignores politics
    # =========================================================================
    (UserArchetype.SPORTS_FAN, ContentTopic.SPORTS.value): ActionProbabilities(
        favorite=0.70,
        repost=0.30,
        reply=0.10,
        click=0.50,
        dwell=0.60,
        follow_author=0.15,
        quote=0.05,
    ),
    (UserArchetype.SPORTS_FAN, ContentTopic.POLITICS_L.value): ActionProbabilities(
        favorite=0.05,
        not_interested=0.10,
        dwell=0.10,
    ),
    (UserArchetype.SPORTS_FAN, ContentTopic.POLITICS_R.value): ActionProbabilities(
        favorite=0.05,
        not_interested=0.10,
        dwell=0.10,
    ),
    (UserArchetype.SPORTS_FAN, ContentTopic.TECH.value): ActionProbabilities(
        favorite=0.15,
        click=0.20,
        dwell=0.25,
    ),
    (UserArchetype.SPORTS_FAN, ContentTopic.ENTERTAINMENT.value): ActionProbabilities(
        favorite=0.25,
        repost=0.08,
        click=0.30,
        dwell=0.35,
    ),
    (UserArchetype.SPORTS_FAN, ContentTopic.NEWS.value): ActionProbabilities(
        favorite=0.20,
        click=0.25,
        dwell=0.30,
    ),

    # =========================================================================
    # POLITICAL LEFT - Engages with left content, hostile to right
    # =========================================================================
    (UserArchetype.POLITICAL_L, ContentTopic.POLITICS_L.value): ActionProbabilities(
        favorite=0.65,
        repost=0.45,
        reply=0.25,
        follow_author=0.20,
        quote=0.15,
        click=0.55,
        dwell=0.60,
    ),
    (UserArchetype.POLITICAL_L, ContentTopic.POLITICS_R.value): ActionProbabilities(
        block_author=0.25,
        mute_author=0.15,
        not_interested=0.30,
        reply=0.10,  # Argumentative replies
        report=0.05,
        dwell=0.15,  # Hate-reading
    ),
    (UserArchetype.POLITICAL_L, ContentTopic.SPORTS.value): ActionProbabilities(
        favorite=0.10,
        click=0.15,
        dwell=0.20,
    ),
    (UserArchetype.POLITICAL_L, ContentTopic.TECH.value): ActionProbabilities(
        favorite=0.20,
        repost=0.10,
        click=0.25,
        dwell=0.30,
    ),
    (UserArchetype.POLITICAL_L, ContentTopic.ENTERTAINMENT.value): ActionProbabilities(
        favorite=0.25,
        repost=0.10,
        click=0.30,
        dwell=0.35,
    ),
    (UserArchetype.POLITICAL_L, ContentTopic.NEWS.value): ActionProbabilities(
        favorite=0.35,
        repost=0.20,
        reply=0.10,
        click=0.40,
        dwell=0.45,
    ),

    # =========================================================================
    # POLITICAL RIGHT - Engages with right content, hostile to left
    # =========================================================================
    (UserArchetype.POLITICAL_R, ContentTopic.POLITICS_R.value): ActionProbabilities(
        favorite=0.65,
        repost=0.45,
        reply=0.25,
        follow_author=0.20,
        quote=0.15,
        click=0.55,
        dwell=0.60,
    ),
    (UserArchetype.POLITICAL_R, ContentTopic.POLITICS_L.value): ActionProbabilities(
        block_author=0.25,
        mute_author=0.15,
        not_interested=0.30,
        reply=0.10,  # Argumentative replies
        report=0.05,
        dwell=0.15,  # Hate-reading
    ),
    (UserArchetype.POLITICAL_R, ContentTopic.SPORTS.value): ActionProbabilities(
        favorite=0.15,
        click=0.20,
        dwell=0.25,
    ),
    (UserArchetype.POLITICAL_R, ContentTopic.TECH.value): ActionProbabilities(
        favorite=0.15,
        click=0.20,
        dwell=0.25,
    ),
    (UserArchetype.POLITICAL_R, ContentTopic.ENTERTAINMENT.value): ActionProbabilities(
        favorite=0.20,
        repost=0.05,
        click=0.25,
        dwell=0.30,
    ),
    (UserArchetype.POLITICAL_R, ContentTopic.NEWS.value): ActionProbabilities(
        favorite=0.30,
        repost=0.15,
        reply=0.10,
        click=0.35,
        dwell=0.40,
    ),

    # =========================================================================
    # TECH BRO - High engagement with tech, moderate elsewhere
    # =========================================================================
    (UserArchetype.TECH_BRO, ContentTopic.TECH.value): ActionProbabilities(
        favorite=0.70,
        repost=0.35,
        reply=0.20,
        click=0.60,
        dwell=0.65,
        follow_author=0.18,
        quote=0.12,
        share=0.10,
    ),
    (UserArchetype.TECH_BRO, ContentTopic.SPORTS.value): ActionProbabilities(
        favorite=0.10,
        click=0.15,
        dwell=0.15,
    ),
    (UserArchetype.TECH_BRO, ContentTopic.POLITICS_L.value): ActionProbabilities(
        favorite=0.15,
        click=0.20,
        dwell=0.20,
    ),
    (UserArchetype.TECH_BRO, ContentTopic.POLITICS_R.value): ActionProbabilities(
        favorite=0.15,
        click=0.20,
        dwell=0.20,
    ),
    (UserArchetype.TECH_BRO, ContentTopic.ENTERTAINMENT.value): ActionProbabilities(
        favorite=0.20,
        click=0.25,
        dwell=0.25,
    ),
    (UserArchetype.TECH_BRO, ContentTopic.NEWS.value): ActionProbabilities(
        favorite=0.25,
        repost=0.10,
        click=0.30,
        dwell=0.35,
    ),

    # =========================================================================
    # LURKER - Passive: likes only, never retweets/replies
    # =========================================================================
    (UserArchetype.LURKER, "*"): ActionProbabilities(
        favorite=0.20,
        repost=0.01,  # Almost never
        reply=0.00,   # Never
        click=0.15,
        dwell=0.35,   # Consumes but doesn't engage
        quote=0.00,
        follow_author=0.02,
    ),

    # =========================================================================
    # POWER USER - High engagement across all action types
    # =========================================================================
    (UserArchetype.POWER_USER, "*"): ActionProbabilities(
        favorite=0.45,
        repost=0.35,
        reply=0.25,
        click=0.50,
        dwell=0.55,
        quote=0.15,
        share=0.12,
        share_via_dm=0.08,
        follow_author=0.10,
        profile_click=0.20,
    ),
}


def get_engagement_probs(
    archetype: UserArchetype,
    topic: ContentTopic,
) -> ActionProbabilities:
    """Get engagement probabilities for (archetype, topic) pair.

    First checks for specific (archetype, topic) rule, then falls back
    to wildcard (archetype, "*") rule if available.

    Args:
        archetype: User archetype
        topic: Content topic

    Returns:
        ActionProbabilities for this combination
    """
    # Check for specific rule
    key = (archetype, topic.value)
    if key in ENGAGEMENT_RULES:
        return ENGAGEMENT_RULES[key]

    # Check for wildcard rule
    wildcard_key = (archetype, "*")
    if wildcard_key in ENGAGEMENT_RULES:
        return ENGAGEMENT_RULES[wildcard_key]

    # Default: very low engagement
    return ActionProbabilities(
        favorite=0.05,
        click=0.10,
        dwell=0.15,
    )


@dataclass
class SyntheticUser:
    """A synthetic user with archetype and metadata."""
    user_id: int
    archetype: UserArchetype
    # Embedding is generated based on archetype + noise


@dataclass
class SyntheticPost:
    """A synthetic post with topic and author."""
    post_id: int
    author_id: int
    topic: ContentTopic
    # Embedding is generated based on topic + noise


@dataclass
class SyntheticEngagement:
    """A synthetic engagement event."""
    user_id: int
    post_id: int
    actions: dict[str, float]  # Action name -> 1.0 if taken, 0.0 if not
    timestamp: int


@dataclass
class SyntheticAuthor:
    """An author who creates posts on a specific topic."""
    author_id: int
    primary_topic: ContentTopic
    # Secondary topics with lower probability
    secondary_topics: list[ContentTopic] = field(default_factory=list)


# Expected test outcomes for verification
EXPECTED_TEST_OUTCOMES = {
    # Embedding probe: users of same archetype should cluster
    "user_clustering_silhouette": 0.2,  # Minimum silhouette score

    # Embedding probe: posts of same topic should cluster
    "post_clustering_silhouette": 0.2,

    # Behavioral: archetype-topic preference tests
    "sports_fan_sports_favorite": 0.70,  # Expected ~70% like rate
    "political_L_politics_L_repost": 0.45,  # Expected ~45% retweet
    "political_R_politics_L_block": 0.25,  # Expected ~25% block

    # Action differentiation: lurker vs power user
    "lurker_repost_ratio_max": 0.05,  # Lurker RT ratio < 5%
    "power_user_repost_ratio_min": 0.30,  # Power user RT ratio > 30%

    # Counterfactual: block should reduce ranking
    "block_reduces_ranking_rate": 0.50,  # >50% of cases
}
