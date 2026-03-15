# Architecture

System diagrams for the x-algorithm-enhancements project. Two enhancement features (F1, F2) built on top of xAI's vendored recommendation system.

## System Overview

```mermaid
graph TB
    subgraph vendored ["Vendored Code (xAI, read-only)"]
        HM["Home Mixer<br/><i>Rust orchestration</i>"]
        T["Thunder<br/><i>In-memory post store</i>"]
        P["Phoenix<br/><i>Grok transformer</i>"]
        CP["Candidate Pipeline<br/><i>Framework</i>"]
    end

    subgraph f2 ["F1: KV-Cache Optimization"]
        JIT["JIT Compilation<br/><i>10.3x speedup</i>"]
        KV["KV-Cache<br/><i>9.6x inference</i>"]
        Q["INT8 Quantization<br/><i>58% memory reduction</i>"]
    end

    subgraph f4 ["F2: Multi-Stakeholder Reward Modeling"]
        BT["Bradley-Terry<br/>Training"]
        SU["Stakeholder<br/>Utilities"]
        PF["Pareto Frontier<br/>Analysis"]
        LOSO["LOSO<br/>Partial Observation"]
    end

    P --> JIT --> KV --> Q
    P --> BT --> SU --> PF
    PF --> LOSO
```

## F2: Research Pipeline

Three research directions, each building on the previous:

```mermaid
graph LR
    subgraph d1 ["Direction 1: Identifiability"]
        Labels["Preference<br/>Labels"] --> Train["BT Training<br/><i>4 loss variants</i>"]
        Train --> Weights["18-dim Weight<br/>Vectors"]
        Weights --> Alpha["α-Recovery<br/><i>Spearman=1.0</i>"]
        Alpha --> Stress["Stress Testing<br/><i>1300 runs</i>"]
    end

    subgraph d3 ["Direction 3: Partial Observation"]
        Weights2["Weight<br/>Vectors"] --> LOSO2["LOSO Analysis<br/><i>Hide 1 of K</i>"]
        LOSO2 --> Proxy["Proxy Methods<br/><i>6 tested</i>"]
        Proxy --> VoI["Data Budget<br/><i>25 pairs = -42% regret</i>"]
        VoI --> Bound["Degradation<br/>Bound"]
        Bound --> Scale["K-Scaling<br/><i>F=8, K=3-10</i>"]
    end

    subgraph d2 ["Direction 2: Sensitivity"]
        Weights3["Weight<br/>Vectors"] --> AlphaD["α-Dominance<br/><i>Permutation vs magnitude</i>"]
        AlphaD --> Sweep["Parameter<br/>Sweep"]
        Sweep --> SvD["Spec vs Data<br/><i>Crossover point</i>"]
    end

    d1 --> d3
    d1 --> d2
```

## F2: Data Flow

From synthetic data through BT training to Pareto frontier evaluation:

```mermaid
graph TD
    subgraph data ["Data Generation"]
        SG["Synthetic Generator<br/><i>600 users x 100 content</i><br/><i>6 topics x 5 archetypes</i>"]
        SG --> AP["Action Probabilities<br/><i>[600, 100, 18] tensor</i>"]
    end

    subgraph pref ["Preference Generation"]
        AP --> UF["Utility Functions<br/><i>U = pos - α·neg</i>"]
        UF --> PP["Preference Pairs<br/><i>2000 pairs per stakeholder</i>"]
    end

    subgraph train ["BT Training"]
        PP --> BT2["Bradley-Terry<br/>Loss Optimization"]
        BT2 --> W["Learned Weight<br/>Vectors (18-dim)"]
    end

    subgraph eval ["Evaluation"]
        W --> Scorer["Content Scorer<br/><i>score = w · action_probs</i>"]
        Scorer --> DW["Diversity Weight<br/>Sweep (21 points)"]
        DW --> Frontier["Pareto Frontier<br/><i>user x platform x society</i>"]
    end

    subgraph metrics ["Metrics"]
        Frontier --> Regret["Regret<br/><i>max - achieved</i>"]
        Frontier --> HD["Hausdorff<br/>Distance"]
        Frontier --> Recovery["Recovery<br/>Rate"]
    end
```

## F2: Stakeholder Utility Model

```mermaid
graph LR
    subgraph actions ["18 Actions"]
        Pos["Positive (5)<br/>favorite, repost,<br/>follow, share, reply"]
        Neg["Negative (4)<br/>block, mute,<br/>report, not_interested"]
        Neut["Neutral (9)<br/>click, dwell, vqv,<br/>photo_expand, etc."]
    end

    subgraph stakeholders ["3 Stakeholders"]
        U["User<br/><i>α = 1.0</i><br/><i>U = pos - neg</i>"]
        Pl["Platform<br/><i>α = 0.3</i><br/><i>U = pos - 0.3·neg</i>"]
        S["Society<br/><i>α = 4.0</i><br/><i>U = pos - 4·neg</i>"]
    end

    Pos --> U & Pl & S
    Neg --> U & Pl & S
```

## F1: Optimization Pipeline

```mermaid
graph LR
    Input["Input<br/>Sequence"] --> JIT2["JAX JIT<br/>Compile"]
    JIT2 --> Attn["Multi-Head<br/>Attention"]
    Attn --> Cache["KV-Cache<br/><i>Reuse K,V tensors</i><br/><i>9.6x speedup</i>"]
    Cache --> Quant["INT8<br/>Quantization<br/><i>58% memory</i>"]
    Quant --> Output["Output<br/>Predictions"]
```

*Detailed directory map will be added after the scripts/ restructure.*
