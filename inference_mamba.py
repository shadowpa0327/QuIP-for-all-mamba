from transformers import AutoTokenizer
from quantizer import load_quantized_model
from eval_utils.lm_eval_wrapper import eval_mamba_zero_shot
import argparse
from transformers import AutoTokenizer
from quantizer import load_quantized_model
from eval_utils.lm_eval_wrapper import eval_mamba_zero_shot
import json
import os
def main(args):
    quant_model = load_quantized_model(args.quant_dir).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.quant_dir)
    results = eval_mamba_zero_shot(
        quant_model,
        tokenizer,
        "mamba",
        batch_size=1,
        task_list=args.task_list
    )

    if args.log_dir:
        with open(os.path.join(args.log_dir, args.quant_dir.split('/')[-1]), 'w') as f:
            json.dump(results, f, indent=4)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant_dir", type=str, help="Directory of the quantized model")
    parser.add_argument("--task", type=str, help="Task to evaluate on")
    parser.add_argument(
        '--task_list', type=lambda s: [item for item in s.split(',')], default=["lambada_openai"],
        help='Task to be evaled, e.g., --task_list lambada_openai,hellaswag,arc_easy,arc_challenge,piqa,winogrande'
    )
    parser.add_argument(
        '--log_dir', type=str,
    )
    args = parser.parse_args()
    main(args)