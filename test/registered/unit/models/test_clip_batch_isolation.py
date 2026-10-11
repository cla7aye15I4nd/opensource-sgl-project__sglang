import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.models import clip
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class QKV(nn.Module):
    def forward(self, x):
        return torch.cat((x, x, x), dim=-1), None


class Projection(nn.Module):
    def forward(self, x):
        return x, None


class TestCLIPBatchIsolation(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        config = SimpleNamespace(
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=2,
            num_hidden_layers=2,
            layer_norm_eps=1e-5,
            vocab_size=32,
            max_position_embeddings=16,
            attention_dropout=0.0,
        )
        # Keep the production embeddings, attention, encoder, and pooling;
        # substitute only the distributed projections and MLP.
        with (
            get_parallel().override(
                tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0
            ),
            patch.object(clip, "QKVParallelLinear", return_value=QKV()),
            patch.object(clip, "RowParallelLinear", return_value=Projection()),
            patch.object(
                clip, "CLIPMLP", side_effect=lambda *args, **kw: nn.Identity()
            ),
        ):
            self.model = clip.CLIPModel.__new__(clip.CLIPModel)
            nn.Module.__init__(self.model)
            self.model.text_model = clip.CLIPTextTransformer(config)
            self.model.text_projection = nn.Identity()
            self.model.pooler = clip.Pooler(clip.PoolingType.LAST, normalize=True)
            self.model.eval()
        self.model.text_model.register_forward_hook(self.capture_hidden_states)

    def capture_hidden_states(self, module, args, output):
        self.hidden_states = output.detach().clone()

    def embed(self, sequences):
        lengths = torch.tensor([len(seq) for seq in sequences])
        batch = SimpleNamespace(
            mm_inputs=None,
            batch_size=len(sequences),
            extend_seq_lens=lengths,
            dimensions=None,
        )
        with torch.no_grad():
            return self.model(
                torch.cat([torch.tensor(seq) for seq in sequences]),
                torch.cat([torch.arange(len(seq)) for seq in sequences]),
                batch,
            ).embeddings

    def test_embeddings_do_not_depend_on_batch_neighbors(self):
        a, b, c = [1, 4, 2], [1, 7, 8, 9, 2], [1, 12, 13, 2]
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            self.model.to(dtype=dtype)
            for sequences in ([a, b], [b, a], [c, b], [a, b, c]):
                with self.subTest(dtype=dtype, sequences=sequences):
                    embeddings, hidden_states = [], []
                    for seq in sequences:
                        embeddings.append(self.embed([seq]))
                        hidden_states.append(self.hidden_states)
                    torch.testing.assert_close(
                        self.embed(sequences), torch.cat(embeddings)
                    )
                    # Check every token, not only the pooled last token, so losing
                    # within-request causality also fails this comparison.
                    torch.testing.assert_close(
                        self.hidden_states, torch.cat(hidden_states, dim=1)
                    )


if __name__ == "__main__":
    unittest.main()
