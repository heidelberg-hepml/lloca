import pytest
import torch

from lloca.backbone.attention import LLoCaAttention
from lloca.backbone.particletransformer import Block, ParticleTransformer, SequenceTrimmer
from lloca.framesnet.equi_frames import LearnedPDFrames, LearnedRestFrames, LearnedSO13Frames
from lloca.framesnet.frames import InverseFrames
from lloca.framesnet.nonequi_frames import IdentityFrames
from lloca.reps.tensorreps import TensorReps
from lloca.reps.tensorreps_transform import TensorRepsTransform
from lloca.utils.rand_transforms import rand_lorentz
from tests.constants import (
    FRAMES_PREDICTOR,
    LOGM2_MEAN_STD,
    MILD_TOLERANCES,
    REPS,
    TOLERANCES,
)
from tests.helpers import equivectors_builder, sample_particle, sweep
from tests.hep import get_tagging_features

BLOCK_SWEEP = sweep(
    dict(
        FramesPredictor=FRAMES_PREDICTOR[0],
        attn_reps=REPS[0],
        logm2_mean=0,
        logm2_std=1,
    ),
    ("FramesPredictor", FRAMES_PREDICTOR),
    ("attn_reps", REPS),
    ("logm2_mean,logm2_std", LOGM2_MEAN_STD),
)


@pytest.mark.parametrize("batch_dims", [[10]])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize(*BLOCK_SWEEP)
def test_block_invariance_equivariance(
    FramesPredictor,
    batch_dims,
    logm2_std,
    logm2_mean,
    attn_reps,
    num_heads,
):
    dtype = torch.float64

    assert len(batch_dims) == 1
    equivectors = equivectors_builder()
    predictor = FramesPredictor(equivectors=equivectors).to(dtype=dtype)

    fm_test = sample_particle(batch_dims, logm2_std, logm2_mean, dtype=dtype)
    predictor.equivectors.init_standardization(fm_test)

    # define block
    in_reps = TensorReps("1x1n")
    attn_reps = TensorReps(attn_reps)
    trafo = TensorRepsTransform(TensorReps(in_reps))
    linear_in = torch.nn.Linear(in_reps.dim, attn_reps.dim * num_heads).to(dtype=dtype)
    linear_out = torch.nn.Linear(attn_reps.dim * num_heads, in_reps.dim).to(dtype=dtype)
    attention = LLoCaAttention(attn_reps, num_heads)
    ParT_block = Block(attention=attention, embed_dim=attn_reps.dim * num_heads).to(dtype)
    ParT_block.eval()  # turn off dropout

    def block_wrapper(x, frames, fourmomenta):
        x = x.unsqueeze(0)
        mask = torch.ones_like(x[..., 0])
        frames = frames.reshape(1, *frames.shape)
        # reference (jet) momentum in the global frame
        p_global = torch.einsum("...ij,...j->...i", frames.inv, fourmomenta.unsqueeze(0))
        attention.prepare_frames(frames, p_ref=p_global.sum(dim=-2))
        x = ParT_block(x=x, padding_mask=mask)
        x = x.squeeze(0)
        return x

    # get global transformation
    random = rand_lorentz([1], dtype=dtype)
    random = random.repeat(*batch_dims, 1, 1)

    # sample Lorentz vectors
    fm = sample_particle(batch_dims, logm2_std, logm2_mean, dtype=dtype)
    frames = predictor(fm)
    fm_local = trafo(fm, frames)

    # block - global
    x_local = linear_in(fm_local)
    x_prime_local = block_wrapper(x_local, frames, fm_local)
    fm_prime_local = linear_out(x_prime_local)
    # back to global
    fm_prime_global = trafo(fm_prime_local, InverseFrames(frames))
    fm_prime_tr_global = torch.einsum("...ij,...j->...i", random, fm_prime_global)

    # global - block
    fm_transformed = torch.einsum("...ij,...j->...i", random, fm)
    frames_transformed = predictor(fm_transformed)
    fm_tr_local = trafo(fm_transformed, frames_transformed)
    x_tr_local = linear_in(fm_tr_local)
    x_tr_prime_local = block_wrapper(x_tr_local, frames_transformed, fm_tr_local)
    fm_tr_prime_local = linear_out(x_tr_prime_local)
    # back to global frame
    fm_tr_prime_global = trafo(fm_tr_prime_local, InverseFrames(frames_transformed))

    # test feature invariance before the operation
    torch.testing.assert_close(x_local, x_tr_local, **TOLERANCES)

    # test feature invariance after the operation
    torch.testing.assert_close(x_tr_prime_local, x_prime_local, **TOLERANCES)

    # test equivariance of outputs
    torch.testing.assert_close(fm_tr_prime_global, fm_prime_tr_global, **TOLERANCES)


@pytest.mark.parametrize(
    "FramesPredictor",
    [
        LearnedSO13Frames,
        LearnedPDFrames,
        LearnedRestFrames,
    ],
)
@pytest.mark.parametrize("batch_dims", [[10]])
@pytest.mark.parametrize("logm2_mean,logm2_std", LOGM2_MEAN_STD)
def test_ParT_invariance(
    FramesPredictor,
    batch_dims,
    logm2_std,
    logm2_mean,
):
    dtype = torch.float64

    assert len(batch_dims) == 1
    equivectors = equivectors_builder()
    predictor = FramesPredictor(equivectors=equivectors).to(dtype=dtype)

    fm_test = sample_particle(batch_dims, logm2_std, logm2_mean, dtype=dtype)
    predictor.equivectors.init_standardization(fm_test)

    # define ParT
    in_reps = TensorReps("1x1n")
    trafo = TensorRepsTransform(TensorReps(in_reps))
    model = ParticleTransformer(
        input_dim=7,
        num_classes=1,
        attn_reps="8x0n+2x1n",
        num_layers=2,
    ).to(dtype=dtype)
    model.eval()  # turn off dropout

    def ParT_wrapper(p_local, frames):
        # jet-relative features need the jet momentum per particle, not a batch index
        jet = p_local.sum(dim=-2, keepdim=True).expand_as(p_local)
        fts_local = get_tagging_features(p_local, jet)
        fts_local = fts_local.transpose(-1, -2).unsqueeze(0)
        # reference (jet) momentum in the global frame (energy-first)
        p_ref = torch.einsum("nij,nj->ni", frames.inv, p_local).sum(dim=0, keepdim=True)
        p_local = p_local[..., [1, 2, 3, 0]]
        p_local = p_local.transpose(-1, -2).unsqueeze(0)
        mask = torch.ones_like(p_local[..., [0], :])
        frames = frames.reshape(1, *frames.shape)
        x = model(x=fts_local, v=p_local, frames=frames, mask=mask, p_ref=p_ref)
        x = x.transpose(-1, -2).squeeze(0)
        return x

    # get global transformation
    random = rand_lorentz([1], dtype=dtype)
    random = random.repeat(*batch_dims, 1, 1)

    # sample Lorentz vectors
    fm = sample_particle(batch_dims, logm2_std, logm2_mean, dtype=dtype)
    frames = predictor(fm)
    fm_local = trafo(fm, frames)

    # ParT
    score_prime_local = ParT_wrapper(fm_local, frames)

    # global - ParT
    fm_transformed = torch.einsum("...ij,...j->...i", random, fm)
    frames_transformed = predictor(fm_transformed)
    fm_tr_local = trafo(fm_transformed, frames_transformed)
    score_tr_prime_local = ParT_wrapper(fm_tr_local, frames_transformed)

    # test feature invariance before the operation
    torch.testing.assert_close(fm_local, fm_tr_local, **TOLERANCES)

    # test equivariance of scores
    torch.testing.assert_close(score_tr_prime_local, score_prime_local, **MILD_TOLERANCES)


# Configuration options of the ported ParT, one per entry. They are independent knobs, so each
# is exercised once on top of the small base model rather than in a product; ``train_mode`` is
# not a ParT argument but selects the module mode, which some options only take effect in.
PART_OPTIONS = [
    dict(checkpoint_blocks=True),
    dict(compile=True, compile_kwargs=dict(mode="default")),
    # pairwise interaction features, one entry per feature set and coordinate system
    dict(pair_input_type="ee"),
    dict(pair_input_type="xyzt"),
    dict(pair_input_type="xyzt:spherical"),
    dict(pair_input_type="xyzt:cylindrical"),
    dict(pair_input_dim=8),  # the full pp set: adds the boost, opening-angle and rapidity features
    dict(pair_extra_dim=2),  # extra pair features handed in as a sparse (uu, uu_idx) pair
    dict(pair_extra_dim=2, for_inference=True),  # the same through the dense (ONNX) pair embedding
    dict(remove_self_pair=True),
    dict(pair_input_dim=8, remove_self_pair=True, for_inference=True),
    dict(normalize_input=False),
    dict(use_conv_embed=True),
    # ParT versions: v2 switches the FFN to SwiGLU, v3 to RMSNorm without bias, v3.5 adds
    # query/key normalization and an elementwise attention-output gate
    dict(version=2),
    dict(version=3),
    dict(version=3.5),
    # block options that are inactive in eval mode or without an explicit request
    dict(
        train_mode=True,
        block_params=dict(
            drop_path_rate=0.1,
            layer_scale_init_values=1e-5,
            scale_attn_mask=True,
            headwise_attn_output_gate=True,
        ),
    ),
    dict(block_ids_with_attn_mask=[0]),
    # heads and output modes
    dict(fc_params=((32, 0.1),)),
    dict(fc_params=((32, 0.1, "swiglu"),)),
    dict(num_cls_layers=0),  # average pooling instead of class attention
    dict(cls_block_params=dict(dropout=0.0)),
    dict(fc_params=None),  # no classifier head: ParT as a feature extractor
    dict(for_inference=True),  # dense pair embedding and a softmax output
    dict(for_segmentation=True),  # per-particle instead of per-jet scores
    dict(weight_init="timm"),
]

SHAPE_SWEEP = sweep(
    dict(FramesPredictor=IdentityFrames, part_kwargs={}),
    ("FramesPredictor", [LearnedPDFrames]),
    ("part_kwargs", PART_OPTIONS),
)


@pytest.mark.parametrize("batch_dims", [[10]])
@pytest.mark.parametrize("logm2_mean,logm2_std", [LOGM2_MEAN_STD[0]])
@pytest.mark.parametrize(*SHAPE_SWEEP)
def test_ParT_shape(
    FramesPredictor,
    batch_dims,
    logm2_std,
    logm2_mean,
    part_kwargs,
):
    assert len(batch_dims) == 1
    part_kwargs = dict(part_kwargs)
    train_mode = part_kwargs.pop("train_mode", False)
    pair_extra_dim = part_kwargs.get("pair_extra_dim", 0)

    kwargs = {}
    if FramesPredictor == LearnedPDFrames:
        kwargs["equivectors"] = equivectors_builder()
    predictor = FramesPredictor(**kwargs)

    fm_test = sample_particle(batch_dims, logm2_std, logm2_mean)
    if FramesPredictor == LearnedPDFrames:
        predictor.equivectors.init_standardization(fm_test)

    # define ParT
    in_reps = TensorReps("1x1n")
    trafo = TensorRepsTransform(TensorReps(in_reps))
    model = ParticleTransformer(
        input_dim=7,
        num_classes=1,
        attn_reps="8x0n+2x1n",
        embed_dims=(32, 64, 32),
        pair_embed_dims=(16, 16, 16),
        num_heads=2,
        num_layers=2,
        **part_kwargs,
    )
    model.train(train_mode)  # eval turns off dropout, train exercises the stochastic paths

    def extra_pair_features(num_particles):
        """Dense (uu, uu_idx) pair features, in the sparse layout ParT expects."""
        if not pair_extra_dim:
            return None, None
        i, j = torch.meshgrid(*(torch.arange(num_particles),) * 2, indexing="ij")
        uu_idx = torch.stack([i.flatten(), j.flatten()]).unsqueeze(0)
        return torch.randn(1, pair_extra_dim, uu_idx.shape[-1]), uu_idx

    def ParT_wrapper(p_local, frames):
        # jet-relative features need the jet momentum per particle, not a batch index
        jet = p_local.sum(dim=-2, keepdim=True).expand_as(p_local)
        fts_local = get_tagging_features(p_local, jet)
        fts_local = fts_local.transpose(-1, -2).unsqueeze(0)
        # reference (jet) momentum in the global frame (energy-first)
        p_ref = torch.einsum("nij,nj->ni", frames.inv, p_local).sum(dim=0, keepdim=True)
        p_local = p_local[..., [1, 2, 3, 0]]
        p_local = p_local.transpose(-1, -2).unsqueeze(0)
        mask = torch.ones_like(p_local[..., [0], :])
        frames = frames.reshape(1, *frames.shape)
        uu, uu_idx = extra_pair_features(p_local.shape[-1])
        x = model(
            x=fts_local, v=p_local, frames=frames, mask=mask, uu=uu, uu_idx=uu_idx, p_ref=p_ref
        )
        x = x.transpose(-1, -2).squeeze(0)
        return x

    # sample Lorentz vectors
    fm = sample_particle(batch_dims, logm2_std, logm2_mean)
    frames = predictor(fm)
    fm_local = trafo(fm, frames)

    # ParT
    out = ParT_wrapper(fm_local, frames)
    if part_kwargs.get("for_segmentation"):
        assert out.shape == (batch_dims[0], 1)  # one score per particle
    elif part_kwargs.get("fc_params", ()) is None:
        assert out.shape == (model.embed_dim, 1)  # pooled embedding, no classifier head
    else:
        assert out.shape == (1,)  # one score per jet


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(version=3.6), "include_global_token"),
        (dict(trim=True), "trim=True is not supported"),
        (dict(pair_input_type="nope", pair_input_dim=4), "Invalid value for"),
    ],
)
def test_ParT_rejects_unsupported_options(kwargs, match):
    """Options that ParT accepts upstream but that LLoCa cannot support must fail loudly."""
    with pytest.raises((NotImplementedError, AssertionError, RuntimeError), match=match):
        ParticleTransformer(input_dim=7, num_classes=1, attn_reps="4x0n", num_layers=1, **kwargs)


def test_ParT_loads_legacy_attention_checkpoint():
    """Attention._load_from_state_dict renames the packed in_proj_* tensors of older ParT
    checkpoints, which stored them as buffers of nn.MultiheadAttention rather than a Linear."""
    model = ParticleTransformer(input_dim=7, num_classes=1, attn_reps="4x0n", num_layers=1)
    state_dict = model.state_dict()
    for key in list(state_dict):
        if "in_proj." in key:
            state_dict[key.replace("in_proj.", "in_proj_")] = state_dict.pop(key)

    model.load_state_dict(state_dict)  # raises on any missing or unexpected key


@pytest.mark.parametrize("training", [True, False])
def test_sequence_trimmer_drops_padding_only(training):
    """The trimmer is unused by LLoCa (the local frames are not permuted along), but it is part
    of the ported ParT, so pin that it only ever removes padding."""
    num_particles, num_real = 40, 12
    trimmer = SequenceTrimmer(enabled=True, warmup_steps=1, round_to_32=True, num_extra_tokens=1)
    trimmer.train(training)

    x = torch.randn(2, 3, num_particles)
    v = torch.randn(2, 4, num_particles)
    uu = torch.randn(2, 1, num_particles, num_particles)
    mask = torch.zeros(2, 1, num_particles)
    mask[..., :num_real] = 1

    # the warmup pass is a no-op, and builds an all-ones mask when none is given
    assert trimmer(x, v, None, uu)[2].all()

    x_out, v_out, mask_out, uu_out = trimmer(x, v, mask, uu)
    seq_len = x_out.shape[-1]
    assert seq_len < num_particles  # padding was trimmed
    assert v_out.shape[-1] == mask_out.shape[-1] == seq_len
    assert uu_out.shape[-2:] == (seq_len, seq_len)
    assert mask_out.sum() == mask.sum()  # no real particle was dropped
