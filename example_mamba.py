#from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import MambaConfig, MambaForCausalLM
from transformers import AutoTokenizer
import torch
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)  
from quantizer import QuipQuantizer
import argparse


def main(args):
    assert "mamba" in args.model_name.lower(), "Only support Mamba models"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    #NOTE(This is the model from huggingface instead of official mamba.)
    model = MambaForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16).cuda()
    model.config.use_cache = False

    quant = QuipQuantizer(codebook=args.codebook, dataset=args.dataset, ft_epochs=0, nsamples=args.nsamples, batch_size=args.batch_size, 
                      block_name_to_quantize="backbone.layers", modules_to_not_convert=["dt_proj"])
    quant.quantize_model(model, tokenizer, args.quant_dir)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mamba Script')
    parser.add_argument('--model_name', type=str, default='state-spaces/mamba-2.8b-hf', help='Model name')
    parser.add_argument('--quant_dir', type=str, default='mamba_2.8b_2bit_quip', help='Quantization directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--nsamples', type=int, default=32, help='Number of samples')
    parser.add_argument('--dataset', type=str, default='redpajama', help='Dataset')
    parser.add_argument('--codebook', type=str, default='E8P12', help='Codebook')
    args = parser.parse_args()

    main(args)