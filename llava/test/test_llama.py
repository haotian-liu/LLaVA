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


class TestLlama(unittest.TestCase):


    def test_chat_dataset_processing(self):
        conversation_lib.default_conversation = conversation_lib.conv_templates["llama_2"]

        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf", trust_remote_code=True
        )
        tokenizer.add_tokens(
            [DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True
        )
        tokenizer.pad_token = tokenizer.unk_token

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
                '<s>', '▁[', 'INST', ']', '▁<<', 'SY', 'S', '>>', '<0x0A>', 'You', '▁are', '▁a', '▁helpful', ',',
                '▁respect', 'ful', '▁and', '▁honest', '▁assistant', '.', '▁Always', '▁answer', '▁as', '▁help',
                'fully', '▁as', '▁possible', ',', '▁while', '▁being', '▁safe', '.', '▁', '▁Your', '▁answers',
                '▁should', '▁not', '▁include', '▁any', '▁harm', 'ful', ',', '▁un', 'eth', 'ical', ',', '▁rac',
                'ist', ',', '▁sex', 'ist', ',', '▁to', 'xic', ',', '▁dangerous', ',', '▁or', '▁illegal',
                '▁content', '.', '▁Please', '▁ensure', '▁that', '▁your', '▁responses', '▁are', '▁soci',
                'ally', '▁un', 'bi', 'ased', '▁and', '▁positive', '▁in', '▁nature', '.', '<0x0A>', '<0x0A>',
                'If', '▁a', '▁question', '▁does', '▁not', '▁make', '▁any', '▁sense', ',', '▁or', '▁is', '▁not',
                '▁fact', 'ually', '▁coh', 'er', 'ent', ',', '▁explain', '▁why', '▁instead', '▁of', '▁answering',
                '▁something', '▁not', '▁correct', '.', '▁If', '▁you', '▁don', "'", 't', '▁know', '▁the', '▁answer',
                '▁to', '▁a', '▁question', ',', '▁please', '▁don', "'", 't', '▁share', '▁false', '▁information',
                '.', '<0x0A>', '<', '</', 'SY', 'S', '>>', '<0x0A>', '<0x0A>', '<im_patch>', '▁', '<0x0A>',
                'Create', '▁a', '▁compact', '▁narr', 'ative', '▁representing', '▁the', '▁image', '▁presented',
                '.', '▁[', '/', 'INST', ']',
                '▁l', 'arch', '▁trees', '▁in', '▁aut', 'umn', '▁colours', '▁along', '▁the', '▁trail', '▁', '</s>'
            ],
            tokenizer.convert_ids_to_tokens(input_ids)
        )
        self.assertEqual(
            [
                '▁l', 'arch', '▁trees', '▁in', '▁aut', 'umn', '▁colours', '▁along', '▁the', '▁trail', '▁', '</s>'
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
                '<s>', '▁[', 'INST', ']', '▁<<', 'SY', 'S', '>>', '<0x0A>', 'You', '▁are', '▁a', '▁helpful', ',',
                '▁respect', 'ful', '▁and', '▁honest', '▁assistant', '.', '▁Always', '▁answer', '▁as', '▁help',
                'fully', '▁as', '▁possible', ',', '▁while', '▁being', '▁safe', '.', '▁', '▁Your', '▁answers',
                '▁should', '▁not', '▁include', '▁any', '▁harm', 'ful', ',', '▁un', 'eth', 'ical', ',', '▁rac',
                'ist', ',', '▁sex', 'ist', ',', '▁to', 'xic', ',', '▁dangerous', ',', '▁or', '▁illegal', '▁content',
                '.', '▁Please', '▁ensure', '▁that', '▁your', '▁responses', '▁are', '▁soci', 'ally', '▁un', 'bi',
                'ased', '▁and', '▁positive', '▁in', '▁nature', '.', '<0x0A>', '<0x0A>', 'If', '▁a', '▁question',
                '▁does', '▁not', '▁make', '▁any', '▁sense', ',', '▁or', '▁is', '▁not', '▁fact', 'ually', '▁coh',
                'er', 'ent', ',', '▁explain', '▁why', '▁instead', '▁of', '▁answering', '▁something', '▁not',
                '▁correct', '.', '▁If', '▁you', '▁don', "'", 't', '▁know', '▁the', '▁answer', '▁to', '▁a',
                '▁question', ',', '▁please', '▁don', "'", 't', '▁share', '▁false', '▁information', '.', '<0x0A>',
                '<', '</', 'SY', 'S', '>>', '<0x0A>', '<0x0A>', '<im_patch>', '▁', '<0x0A>', 'Create', '▁a',
                '▁compact', '▁narr', 'ative', '▁representing', '▁the', '▁image', '▁presented', '.',
                '▁[', '/', 'INST', ']', '▁l', 'arch', '▁trees', '▁in', '▁aut', 'umn', '▁colours', '▁along',
                '▁the', '▁trail', '▁', '</s>', '<s>', '▁[', 'INST', ']', '▁How', '▁tall', '▁are', '▁the',
                '▁trees', '?', '▁[', '/', 'INST', ']', '▁Some', '▁are', '▁tall', ',', '▁some', '▁are',
                '▁short', '.', '▁', '</s>'
            ],
            tokenizer.convert_ids_to_tokens(input_ids)
        )
        self.assertEqual(
            [
                [
                    '▁l', 'arch', '▁trees', '▁in', '▁aut', 'umn', '▁colours', '▁along', '▁the', '▁trail', '▁', '</s>',
                    '▁Some', '▁are', '▁tall', ',', '▁some', '▁are', '▁short', '.', '▁', '</s>'
                ]
            ],
            tokenizer.convert_ids_to_tokens(labels)
        )
