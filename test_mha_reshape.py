"""Unit tests for the term-major -> time-major reshape and the MHA encoder.

Verifies the reshape is a correct block transpose (a wrong reshape would let
the model train silently on misaligned frame fragments), plus smoke-tests the
encoder, the MHA actor/critic trunks, the full ActorCriticMHA forward path,
and that the runner globals expose ActorCriticMHA so eval(class_name) resolves.

Run with the IsaacLab venv python (no sim launch needed):
  /home/ordis/projects/IsaacLab/.venv/bin/python test_mha_reshape.py
"""
import torch

from rsl_rl.networks.mha import AvgL1Norm, LinearMHAEncoder, to_time_major
from rsl_rl.modules.actor_critic_mha import ActorCriticMHA, MHAActor, MHACritic

ACTOR_DIMS = [3, 3, 3, 29, 29, 29]       # 96 single-step -> 480 flat
CRITIC_DIMS = [3, 3, 3, 3, 29, 29, 29]   # 99 single-step -> 495 flat
H = 5
B = 4


def build_known_term_major(dims, n_history, batch):
    """Term-major flat tensor where frame t of every term is constant t.

    So the correct time-major[:, t, :] is a vector of all t's. This encodes a
    known temporal order into the term-major layout so we can check the reshape
    recovers it frame-by-frame.
    """
    single = sum(dims)
    flat = torch.zeros(batch, n_history * single)
    off = 0
    for d in dims:
        for t in range(n_history):
            flat[:, off + t * d : off + (t + 1) * d] = float(t)
        off += n_history * d
    return flat


def test_to_time_major():
    x = build_known_term_major(ACTOR_DIMS, H, B)
    assert x.shape == (B, 480)
    tm = to_time_major(x, ACTOR_DIMS, H)
    assert tm.shape == (B, H, sum(ACTOR_DIMS)), tm.shape
    for t in range(H):
        frame = tm[:, t, :]
        assert torch.all(frame == float(t)), f"frame {t} not uniform: {frame.unique()}"
    # current frame is the last (tail) slot, per the env's term-major layout
    assert torch.all(tm[:, -1, :] == float(H - 1))
    print("[ok] to_time_major: shape + per-frame values + current-frame-at-tail")


def test_encoder_smoke():
    x = build_known_term_major(ACTOR_DIMS, H, B)
    past = to_time_major(x, ACTOR_DIMS, H)[:, :-1, :]
    enc = LinearMHAEncoder(
        sum(ACTOR_DIMS), H - 1, hidden_dim=32, nhead=4, is_learnable_pos_embedding=True, dropout=0.1
    )
    z = enc(past)
    assert z.shape == (B, 32), z.shape
    print("[ok] LinearMHAEncoder forward ->", tuple(z.shape))


def test_actor_critic_smoke():
    x_a = build_known_term_major(ACTOR_DIMS, H, B)
    x_c = build_known_term_major(CRITIC_DIMS, H, B)
    actor = MHAActor(term_dims=ACTOR_DIMS, num_actions=29, n_history=H,
                     enc_hidden=256, nheads=8, pos_emb=True, dropout=0.0,
                     hidden_dims=[512, 256, 128], activation="elu")
    critic = MHACritic(term_dims=CRITIC_DIMS, n_history=H,
                       enc_hidden=256, nheads=8, pos_emb=True, dropout=0.0,
                       hidden_dims=[512, 256, 128], activation="elu")
    mean = actor(x_a)
    val = critic(x_c)
    assert mean.shape == (B, 29), mean.shape
    assert val.shape == (B, 1), val.shape
    print("[ok] MHAActor ->", tuple(mean.shape), " MHACritic ->", tuple(val.shape))


def test_actor_critic_mha_full():
    # Mock the obs / obs_groups the runner hands to the policy constructor.
    obs = {"policy": torch.zeros(B, 480), "critic": torch.zeros(B, 495)}
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    ac = ActorCriticMHA(
        obs, obs_groups, num_actions=29,
        actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
        activation="elu", n_history=5, nheads=8, use_critic_mha=True,
        actor_term_dims=ACTOR_DIMS, critic_term_dims=CRITIC_DIMS,
    )
    act = ac.act(obs)            # inherited: get_actor_obs -> normalizer -> MHAActor -> sample
    val = ac.evaluate(obs)       # inherited: get_critic_obs -> normalizer -> MHACritic
    assert act.shape == (B, 29), act.shape
    assert val.shape == (B, 1), val.shape
    print("[ok] ActorCriticMHA.act ->", tuple(act.shape), " .evaluate ->", tuple(val.shape))


def test_mismatch_raises():
    x = torch.zeros(B, 479)  # wrong last dim
    raised = False
    try:
        to_time_major(x, ACTOR_DIMS, H)
    except ValueError:
        raised = True
    assert raised, "expected ValueError on dim mismatch"
    print("[ok] to_time_major raises on dim mismatch")


def test_runner_resolves_class():
    import rsl_rl.modules as mods
    from rsl_rl.runners import on_policy_runner

    assert hasattr(mods, "ActorCriticMHA"), "rsl_rl.modules missing ActorCriticMHA"
    assert hasattr(on_policy_runner, "ActorCriticMHA"), (
        "on_policy_runner globals missing ActorCriticMHA -- eval(class_name) would fail"
    )
    print("[ok] rsl_rl.modules + on_policy_runner globals expose ActorCriticMHA")


if __name__ == "__main__":
    test_to_time_major()
    test_encoder_smoke()
    test_actor_critic_smoke()
    test_actor_critic_mha_full()
    test_mismatch_raises()
    test_runner_resolves_class()
    print("ALL TESTS PASSED")
