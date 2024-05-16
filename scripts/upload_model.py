import argparse
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path


def upload(args):
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, None, model_name, device_map='cpu')

    if args.export_hub_model_id is not None:
        model.push_to_hub(
            args.export_hub_model_id,
            token=args.hf_hub_token,
            max_shard_size="{}GB".format(args.export_size),
            safe_serialization=(not args.export_legacy_format),
        )

    try:
        tokenizer.padding_side = "left"  # restore padding side
        tokenizer.init_kwargs["padding_side"] = "left"
        if args.export_hub_model_id is not None:
            tokenizer.push_to_hub(args.export_hub_model_id, token=args.hf_hub_token)
    except Exception:
        print("Cannot save tokenizer, please copy the files manually.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--export_hub_model_id", type=str, required=True)
    parser.add_argument("--hf_hub_token", type=str, required=True)
    parser.add_argument("--export_size", type=int, default=5)
    parser.add_argument("--export_legacy_format", type=bool, default=False)
    args = parser.parse_args()

    upload(args)
