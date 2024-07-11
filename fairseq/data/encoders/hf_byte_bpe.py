# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.data.encoders import register_bpe
from fairseq import file_utils
import os

@register_bpe('hf_byte_bpe')
class HuggingFaceByteLevelBPE(object):

    @staticmethod
    def add_args(parser):
        # fmt: off
        # bpe_merges: str = field(default=os.environ.get("BPE_MERGES"), metadata={"help": "path to merges.txt"})
        # bpe_vocab: str = field(default=os.environ.get("BPE_VOCAB"), metadata={"help": "path to vocab.json"})
        parser.add_argument('--bpe-merges', help='path to merges.txt', default=os.environ.get("BPE_MERGES"))
        parser.add_argument('--bpe-vocab', help='path to vocab.json', default=os.environ.get("BPE_VOCAB"))
        parser.add_argument('--bpe-add-prefix-space', action='store_true',
                            help='add prefix space before encoding')
        # fmt: on

    def __init__(self, args):
        try:
            from tokenizers import ByteLevelBPETokenizer
        except ImportError:
            raise ImportError(
                'Please install huggingface/tokenizers with: '
                'pip install tokenizers'
            )

        bpe_vocab = file_utils.cached_path(args.bpe_vocab)
        bpe_merges = file_utils.cached_path(args.bpe_merges)

        self.bpe = ByteLevelBPETokenizer(
            bpe_vocab,
            bpe_merges,
            add_prefix_space=getattr(args, 'bpe_add_prefix_space', False),
        )
        self.bpe.add_special_tokens(["<s>", "<pad>", "</s>", "<unk>", "<mask>"])

    def encode(self, x: str) -> str:
        return ' '.join(map(str, self.bpe.encode(x).ids))

    def decode(self, x: str) -> str:
        if "<mask>" in x.split() or "<unk>" in x.split():
            decoded = x
        else:
            decoded = self.bpe.decode([ int(tok) for tok in x.split() ])
        
        return decoded

    def is_beginning_of_word(self, x: str) -> bool:
        if x in ['<unk>', '<s>', '</s>', '<pad>']:
            # special elements are always considered beginnings
            # HACK: this logic is already present in fairseq/tasks/masked_lm.py
            # but these special tokens are also contained in the hf_byte_bpe
            # vocabulary which causes duplicate special tokens. This hack makes
            # sure that they are all taken into account.
            return True
        return self.decode(x).startswith(' ')
