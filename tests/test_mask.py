"""The load-bearing tests. If test_state_is_independent_of_suffix fails, the
entire 'encode once, answer N questions' property is a lie."""
import torch, pytest
from transformers import Qwen3Config, Qwen3Model

from systemone.model.mask import block_allow, batch_block_mask


def test_quadrants():
    a = block_allow(n_state=5, n_total=8)
    assert a[:5, :5].all(),        "state block must be bidirectional"
    assert not a[:5, 5:].any(),    "state must NOT attend forward to suffix"
    assert a[5:, :5].all(),        "suffix must see the state"
    assert a[5:, 5:].all(),        "options must see each other"


def test_padding_masked_as_keys():
    m = batch_block_mask([3], [6], 8, torch.float32)
    assert (m[0, 0, :, 6:] < -1e30).all(), "padding must be unattendable"


@pytest.fixture(scope="module")
def tiny():
    cfg = Qwen3Config(vocab_size=256, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16)
    torch.manual_seed(0)
    return Qwen3Model(cfg).eval()


def _run(model, ids, n_state):
    mask = batch_block_mask([n_state], [ids.size(1)], ids.size(1), torch.float32)
    with torch.no_grad():
        return model(input_ids=ids, attention_mask=mask,
                     use_cache=False).last_hidden_state


def test_state_is_independent_of_suffix(tiny):
    """THE test. Change the question/options; the state's hidden states must
    not move by a single float. That independence is the cache."""
    torch.manual_seed(1)
    n_state, T = 12, 24
    ids = torch.randint(0, 256, (1, T))

    h1 = _run(tiny, ids, n_state)

    ids2 = ids.clone()
    ids2[:, n_state:] = torch.randint(0, 256, (1, T - n_state))  # new suffix
    h2 = _run(tiny, ids2, n_state)

    torch.testing.assert_close(h1[:, :n_state], h2[:, :n_state],
                               rtol=0, atol=0)
    assert not torch.allclose(h1[:, n_state:], h2[:, n_state:]), \
        "suffix should have changed"


def test_options_see_each_other(tiny):
    """Contrastive scoring requires a later option to affect an earlier one.
    Under a causal mask this test fails - which is the whole reason for the
    bidirectional suffix block."""
    torch.manual_seed(2)
    n_state, T = 8, 20
    ids = torch.randint(0, 256, (1, T))
    h1 = _run(tiny, ids, n_state)

    ids2 = ids.clone()
    ids2[:, -1] = (ids2[:, -1] + 7) % 256      # perturb the LAST option only
    h2 = _run(tiny, ids2, n_state)

    early = slice(n_state, n_state + 3)
    assert not torch.allclose(h1[:, early], h2[:, early]), \
        "early suffix positions must react to a later option"
