"""Quantization configuration for Phoenix model.

Defines configuration options for different quantization approaches:
- Bit width: FP16, INT8, INT4
- Granularity: per-tensor vs per-channel
- Symmetry: symmetric vs asymmetric
"""

from dataclasses import dataclass
from enum import Enum


class BitWidth(Enum):
    """Quantization bit width options."""
    FP16 = "fp16"   # 16-bit floating point (minimal compression)
    INT8 = "int8"   # 8-bit integer (4x compression)
    INT4 = "int4"   # 4-bit integer (8x compression)


class Granularity(Enum):
    """Quantization granularity options."""
    PER_TENSOR = "per_tensor"    # Single scale for entire tensor
    PER_CHANNEL = "per_channel"  # Scale per output channel (better accuracy)
    PER_GROUP = "per_group"      # Scale per group of weights (e.g., 128 weights)


class Symmetry(Enum):
    """Quantization symmetry options."""
    SYMMETRIC = "symmetric"      # Range [-max, max], zero_point=0
    ASYMMETRIC = "asymmetric"    # Range [min, max], non-zero zero_point


@dataclass
class MixedPrecisionConfig:
    """Layer-specific quantization configuration for mixed precision.

    Allows different bit widths and granularities for different layer types.
    FFN layers typically tolerate more aggressive quantization (INT4) while
    attention layers are more sensitive and benefit from INT8.

    Attributes:
        ffn_bit_width: Bit width for FFN layers (default INT4)
        attention_bit_width: Bit width for attention Q/K/V/O (default INT8)
        embedding_bit_width: Bit width for embeddings (default INT8)
        ffn_granularity: Optional granularity override for FFN (for per-group INT4)
        attention_granularity: Optional granularity override for attention
    """
    ffn_bit_width: BitWidth = BitWidth.INT4
    attention_bit_width: BitWidth = BitWidth.INT8
    embedding_bit_width: BitWidth = BitWidth.INT8

    # Optional layer-specific granularity (allows per-group INT4 for FFN)
    ffn_granularity: Granularity | None = None
    attention_granularity: Granularity | None = None


@dataclass
class QuantizationConfig:
    """Configuration for model quantization.

    Attributes:
        bit_width: Number of bits for quantized values (FP16, INT8, INT4)
        granularity: Per-tensor or per-channel quantization
        symmetry: Symmetric or asymmetric quantization
        quantize_embeddings: Whether to quantize embedding projection matrices
        quantize_attention: Whether to quantize Q/K/V/O projections
        quantize_ffn: Whether to quantize FFN layers
        quantize_unembedding: Whether to quantize final unembedding matrix
        quantize_kv_cache: Whether to quantize KV-cache (stacks with Phase 2b)
        kv_cache_bit_width: Bit width for KV-cache quantization
        name: Human-readable name (auto-generated if None)
    """
    # Core quantization settings
    bit_width: BitWidth = BitWidth.INT8
    granularity: Granularity = Granularity.PER_CHANNEL
    symmetry: Symmetry = Symmetry.SYMMETRIC

    # Layer targeting
    quantize_embeddings: bool = True   # proj_mat_1, proj_mat_2, proj_mat_3
    quantize_attention: bool = True    # Q, K, V, O projections
    quantize_ffn: bool = True          # W_fc1, W_fc1v, W_fc2
    quantize_unembedding: bool = True  # Final output projection

    # KV-cache quantization (extends Phase 2b)
    quantize_kv_cache: bool = False
    kv_cache_bit_width: BitWidth = BitWidth.INT8

    # Per-group quantization settings
    group_size: int = 128  # Number of weights per group for PER_GROUP granularity

    # Mixed precision settings (layer-specific bit widths)
    use_mixed_precision: bool = False
    mixed_precision: MixedPrecisionConfig | None = None

    # Identification
    name: str | None = None

    def __post_init__(self):
        """Auto-generate name if not provided."""
        if self.name is None:
            self.name = self._generate_name()

    def _generate_name(self) -> str:
        """Generate descriptive name from config."""
        if self.use_mixed_precision and self.mixed_precision:
            # Mixed precision naming
            mp = self.mixed_precision
            parts = [f"mixed_{mp.ffn_bit_width.value}ffn_{mp.attention_bit_width.value}attn"]

            if mp.ffn_granularity == Granularity.PER_GROUP:
                parts.append(f"pergroup{self.group_size}")
        else:
            # Standard naming
            parts = [self.bit_width.value]

            if self.bit_width != BitWidth.FP16:
                # Add granularity (shortened)
                if self.granularity == Granularity.PER_TENSOR:
                    gran = "tensor"
                elif self.granularity == Granularity.PER_CHANNEL:
                    gran = "channel"
                else:  # PER_GROUP
                    gran = f"pergroup{self.group_size}"
                parts.append(gran)

                # Add symmetry (shortened)
                sym = "sym" if self.symmetry == Symmetry.SYMMETRIC else "asym"
                parts.append(sym)

        if self.quantize_kv_cache:
            parts.append(f"kv{self.kv_cache_bit_width.value}")

        return "_".join(parts)

    def get_bit_width_int(self) -> int:
        """Get bit width as integer."""
        return {BitWidth.FP16: 16, BitWidth.INT8: 8, BitWidth.INT4: 4}[self.bit_width]

    def get_kv_bit_width_int(self) -> int:
        """Get KV-cache bit width as integer."""
        return {BitWidth.FP16: 16, BitWidth.INT8: 8, BitWidth.INT4: 4}[self.kv_cache_bit_width]


# Pre-defined configurations for comparative study
STUDY_CONFIGS = [
    # FP16 baseline (minimal compression, near-perfect accuracy)
    QuantizationConfig(
        bit_width=BitWidth.FP16,
        name="fp16_baseline",
    ),

    # INT8 variants
    QuantizationConfig(
        bit_width=BitWidth.INT8,
        granularity=Granularity.PER_TENSOR,
        symmetry=Symmetry.SYMMETRIC,
        name="int8_tensor_sym",
    ),
    QuantizationConfig(
        bit_width=BitWidth.INT8,
        granularity=Granularity.PER_CHANNEL,
        symmetry=Symmetry.SYMMETRIC,
        name="int8_channel_sym",
    ),
    QuantizationConfig(
        bit_width=BitWidth.INT8,
        granularity=Granularity.PER_CHANNEL,
        symmetry=Symmetry.ASYMMETRIC,
        name="int8_channel_asym",
    ),

    # INT4 variants (aggressive compression)
    QuantizationConfig(
        bit_width=BitWidth.INT4,
        granularity=Granularity.PER_TENSOR,
        symmetry=Symmetry.SYMMETRIC,
        name="int4_tensor_sym",
    ),
    QuantizationConfig(
        bit_width=BitWidth.INT4,
        granularity=Granularity.PER_CHANNEL,
        symmetry=Symmetry.ASYMMETRIC,
        name="int4_channel_asym",
    ),

    # KV-cache quantization (stacks with Phase 2b)
    QuantizationConfig(
        bit_width=BitWidth.INT8,
        granularity=Granularity.PER_CHANNEL,
        symmetry=Symmetry.SYMMETRIC,
        quantize_kv_cache=True,
        kv_cache_bit_width=BitWidth.INT8,
        name="int8_channel_kv8",
    ),
]


# Extended configurations for Phase 4b comparative study
# Tests all three strategies and their compositions:
# 1. Mixed INT4/INT8 precision
# 2. Per-group INT4 quantization
# 3. KV-cache quantization
EXTENDED_STUDY_CONFIGS = [
    # === Individual Features ===

    # 1. Mixed precision (INT4 FFN, INT8 attention)
    QuantizationConfig(
        use_mixed_precision=True,
        mixed_precision=MixedPrecisionConfig(
            ffn_bit_width=BitWidth.INT4,
            attention_bit_width=BitWidth.INT8,
            embedding_bit_width=BitWidth.INT8,
        ),
        name="mixed_int4_ffn_int8_attn",
    ),

    # 2. Per-group INT4
    QuantizationConfig(
        bit_width=BitWidth.INT4,
        granularity=Granularity.PER_GROUP,
        group_size=128,
        symmetry=Symmetry.SYMMETRIC,
        name="int4_pergroup128_sym",
    ),

    # 3. INT8 with KV-cache quantization (already in STUDY_CONFIGS)
    # Included here for complete comparison
    QuantizationConfig(
        bit_width=BitWidth.INT8,
        granularity=Granularity.PER_CHANNEL,
        symmetry=Symmetry.SYMMETRIC,
        quantize_kv_cache=True,
        kv_cache_bit_width=BitWidth.INT8,
        name="int8_channel_sym_kv8",
    ),

    # === Two-way Compositions ===

    # 4. Mixed precision + KV-cache
    QuantizationConfig(
        use_mixed_precision=True,
        mixed_precision=MixedPrecisionConfig(
            ffn_bit_width=BitWidth.INT4,
            attention_bit_width=BitWidth.INT8,
            embedding_bit_width=BitWidth.INT8,
        ),
        quantize_kv_cache=True,
        kv_cache_bit_width=BitWidth.INT8,
        name="mixed_int4_int8_kv8",
    ),

    # 5. Per-group INT4 + KV-cache
    QuantizationConfig(
        bit_width=BitWidth.INT4,
        granularity=Granularity.PER_GROUP,
        group_size=128,
        symmetry=Symmetry.SYMMETRIC,
        quantize_kv_cache=True,
        kv_cache_bit_width=BitWidth.INT8,
        name="int4_pergroup_kv8",
    ),

    # 6. Mixed precision with per-group for INT4 FFN layers
    QuantizationConfig(
        use_mixed_precision=True,
        mixed_precision=MixedPrecisionConfig(
            ffn_bit_width=BitWidth.INT4,
            attention_bit_width=BitWidth.INT8,
            embedding_bit_width=BitWidth.INT8,
            ffn_granularity=Granularity.PER_GROUP,
        ),
        group_size=128,
        name="mixed_int4_pergroup_int8_attn",
    ),

    # === Three-way Composition (Maximum Compression) ===

    # 7. All three: Mixed precision + per-group INT4 FFN + KV-cache
    QuantizationConfig(
        use_mixed_precision=True,
        mixed_precision=MixedPrecisionConfig(
            ffn_bit_width=BitWidth.INT4,
            attention_bit_width=BitWidth.INT8,
            embedding_bit_width=BitWidth.INT8,
            ffn_granularity=Granularity.PER_GROUP,
        ),
        group_size=128,
        quantize_kv_cache=True,
        kv_cache_bit_width=BitWidth.INT8,
        name="mixed_int4_pergroup_int8_kv8",
    ),
]
