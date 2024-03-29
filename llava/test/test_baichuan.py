import unittest

import torch
from transformers import AutoTokenizer

from llava import conversation as conversation_lib
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN
from llava.train.train import DataArguments, LazySupervisedDataset


class DummyProcessor:
    def __init__(self):
        self.crop_size = 224

    def preprocess(self, _, return_tensors):
        assert return_tensors
        return {
            'pixel_values': torch.randn(
                1, 3, 224, 224
            )
        }


class TestBaichuan(unittest.TestCase):

    # https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/blob/main/generation_utils.py#L35
    #   seems to suggest the Baichuan model expects:
    #       1) no system message
    #       2) user / assistant ids with no colons after

    # https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/blob/main/tokenizer_config.json
    #   does not add <bos> or <eos> tokens but the model generates <eos> tokens so we add them here

    def test_chat_dataset_processing(self):
        conversation_lib.default_conversation = conversation_lib.conv_templates["baichuan_2_chat"]

        tokenizer = AutoTokenizer.from_pretrained(
            'baichuan-inc/Baichuan2-7B-Chat', trust_remote_code=True
        )
        tokenizer.add_tokens(
            [DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True
        )

        args = DataArguments(
            is_multimodal=True,
            data_path='llava/test/fixtures/en_pretrain_sample.json',
            image_folder='llava/test/fixtures/sampled_images'
        )
        args.mm_use_im_start_end = False
        args.image_processor = DummyProcessor()

        dataset = LazySupervisedDataset(
            args.data_path, tokenizer, args
        )

        self.assertEqual(len(dataset), 5)

        # two turns

        processed = dataset[0]
        input_ids = processed['input_ids'].squeeze().tolist()
        input_ids[
            input_ids.index(-200)
        ] = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_PATCH_TOKEN)

        labels = [
            label for label in processed['labels'].squeeze().tolist() if label != -100
        ]

        self.assertEqual(
            [
                '<reserved_106>', '<im_patch>', '\n', 'Create', '▁a', '▁compact', '▁narrative',
                '▁representing', '▁the', '▁image', '▁presented', '.',
                '<reserved_107>', 'l', 'arch', '▁trees', '▁in', '▁autumn', '▁colours', '▁along', '▁the', '▁trail', '</s>'
            ],
            tokenizer.convert_ids_to_tokens(input_ids)
        )
        self.assertEqual(
            [
                'l', 'arch', '▁trees', '▁in', '▁autumn', '▁colours', '▁along', '▁the', '▁trail', '</s>'
            ],
            tokenizer.convert_ids_to_tokens(labels)
        )

        # four turns

        processed = dataset[1]
        input_ids = processed['input_ids'].squeeze().tolist()
        input_ids[
            input_ids.index(-200)
        ] = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_PATCH_TOKEN)

        labels = [
            label for label in processed['labels'].squeeze().tolist() if label != -100
        ]

        self.assertEqual(
            [
                '<reserved_106>', '<im_patch>', '\n', 'Create', '▁a', '▁compact', '▁narrative',
                '▁representing', '▁the',  '▁image', '▁presented', '.',
                 '<reserved_107>', 'l', 'arch', '▁trees', '▁in', '▁autumn', '▁colours', '▁along', '▁the', '▁trail', '</s>',
                 '<reserved_106>', 'How', '▁tall', '▁are', '▁the', '▁trees', '?',
                 '<reserved_107>', 'Some', '▁are', '▁tall', ',', '▁some', '▁are', '▁short', '.', '</s>'
             ],
            tokenizer.convert_ids_to_tokens(input_ids)
        )
        self.assertEqual(
            [
                'l', 'arch', '▁trees', '▁in', '▁autumn', '▁colours', '▁along', '▁the', '▁trail', '</s>',
                'Some', '▁are', '▁tall', ',', '▁some', '▁are', '▁short', '.', '</s>'
            ],
            tokenizer.convert_ids_to_tokens(labels)
        )
