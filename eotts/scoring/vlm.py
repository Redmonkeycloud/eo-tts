"""V3: a vision-language model as verifier, scored by the probability of "Yes" (VQAScore).

The model sees the image and the question

    Does this figure show "{caption}"? Please answer yes or no.

and the score is the probability it assigns to the token "Yes" as the first token of its answer,
softmax over the full vocabulary. Question and answer are the defaults of VQAScore (Lin et al.,
"Evaluating Text-to-Visual Generation with Image-to-Text Generation", ECCV 2024; the
``t2v_metrics`` package). Unlike the CLIP verifiers (V1, V2), which truncate captions at 77
tokens, the VLM reads the whole Git-10M caption.

Numerics: bf16 weights, the language-model head in float32. Scores of the same pairs differ
slightly with batch composition (bf16 attention; up to ~0.1 in logit when prompts of different
lengths are padded together). Sample scoring therefore puts exactly the K samples of one tile in
a batch (one caption, one image size, no padding), and the validity check scores pairs one at a
time, so every score is reproducible.

Candidates (all fit a 12 GB GPU in bf16; revisions pinned). Which one becomes V3 is decided by
``scripts/12_vlm_validity.py`` on real tiles only, before V3 scores any generated sample
(protocol v2, amendment 2).
"""
from __future__ import annotations

import torch

QUESTION = 'Does this figure show "{}"? Please answer yes or no.'
ANSWER = "Yes"
NEGATIVE = "No"          # only for the diagnostic P("Yes") + P("No"), not part of the score

VLMS = {
    "v3_qwen25vl_3b": dict(repo="Qwen/Qwen2.5-VL-3B-Instruct",
                           revision="66285546d2b821cf421d4f5eb2576359d3770cd3"),
    "v3_qwen3vl_4b": dict(repo="Qwen/Qwen3-VL-4B-Instruct",
                          revision="ebb281ec70b05090aa6165b016eac8ec08e71b17"),
    "v3_internvl3_2b": dict(repo="OpenGVLab/InternVL3-2B-hf",
                            revision="cb57a075cb75a2e6d1b668b128d48bb00ae321d2"),
}


class VlmVerifier:
    def __init__(self, name: str, device: str, dtype=None):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        spec = VLMS[name]
        self.name, self.device = name, device
        dtype = dtype or torch.bfloat16
        self.processor = AutoProcessor.from_pretrained(spec["repo"], revision=spec["revision"])
        self.processor.tokenizer.padding_side = "left"        # last position = first answer token
        self.model = AutoModelForImageTextToText.from_pretrained(
            spec["repo"], revision=spec["revision"], dtype=dtype).to(device).eval()
        self.answer_id, self.negative_id = (self._single_token(w) for w in (ANSWER, NEGATIVE))

    def _single_token(self, word):
        ids = self.processor.tokenizer.encode(word, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"{self.name}: '{word}' is {len(ids)} tokens, expected 1")
        return ids[0]

    def prompts(self, captions):
        msgs = [[{"role": "user", "content": [{"type": "image"},
                                              {"type": "text", "text": QUESTION.format(c)}]}]
                for c in captions]
        return [self.processor.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                for m in msgs]

    @torch.no_grad()
    def score(self, images, captions, return_no=False):
        """P("Yes") for each (image, caption) pair, in [0, 1]; with return_no also P("No")."""
        inputs = self.processor(text=self.prompts(captions), images=[im.convert("RGB") for im in images],
                                padding=True, return_tensors="pt").to(self.device)
        # The language-model head runs in float32 on the last hidden state: in bf16 the logits
        # are rounded to steps of 0.125 at their typical size, which alone moved P("Yes") by up
        # to 0.03 between batched and single scoring.
        out = self.model(**inputs, output_hidden_states=True, logits_to_keep=1)
        h = out.hidden_states[-1][:, -1].float()
        head = self.model.get_output_embeddings()
        logits = h @ head.weight.float().T
        if getattr(head, "bias", None) is not None:
            logits = logits + head.bias.float()
        prob = torch.softmax(logits, dim=-1)
        yes = prob[:, self.answer_id].cpu()
        return (yes, prob[:, self.negative_id].cpu()) if return_no else yes
