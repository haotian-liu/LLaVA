import bitsandbytes as bnb
from bitsandbytes.nn.modules import Params4bit, Int8Params
import torch 

def Params4bitCuda(self, device):
    self.data = self.data.cuda(device)
    self.quant_state[0] = self.quant_state[0].cuda(device)
    self.quant_state[4][0] = self.quant_state[4][0].cuda(device)
    self.quant_state[4][1][0] = self.quant_state[4][1][0].cuda(device)
    self.quant_state[4][1][1] = self.quant_state[4][1][1].cuda(device)

    self.quant_state[6] = self.quant_state[6].cuda(device)
    return self

class Linear4bitOnline(torch.nn.Module):
    def __init__(self, weight, bias, quant_type):
        super().__init__()
        self.weight = Params4bit(
            weight.data, requires_grad=False, compress_statistics=True, quant_type=quant_type
        )
        self.compute_dtype = None
        #self.weight.cuda(weight.device)
        self.bias = bias

    def forward(self, x: torch.Tensor):
        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        if getattr(self.weight, "quant_state", None) is None:
            print(
                "FP4 quantization state not initialized. Please call .cuda() or .to(device) on the LinearFP4 layer first."
            )
        inp_dtype = x.dtype
        if self.compute_dtype is not None:
            x = x.to(self.compute_dtype)

        bias = None if self.bias is None else self.bias.to(self.compute_dtype)
        out = bnb.matmul_4bit(
            x, self.weight.t(), bias=bias, quant_state=self.weight.quant_state
        )

        out = out.to(inp_dtype)

        return out
    
class Linear8bitLtOnline(torch.nn.Module):
    def __init__(
        self,
        weight,
        bias,
        has_fp16_weights=True,
        memory_efficient_backward=False,
        threshold=0.0,
        index=None,
    ):
        super().__init__()
        assert (
            not memory_efficient_backward
        ), "memory_efficient_backward is no longer required and the argument is deprecated in 0.37.0 and will be removed in 0.39.0"
        self.state = bnb.MatmulLtState()
        self.index = index

        # Necessary for stacked layers
        self.state.threshold = threshold
        self.state.has_fp16_weights = has_fp16_weights
        self.state.memory_efficient_backward = memory_efficient_backward
        if threshold > 0.0 and not has_fp16_weights:
            self.state.use_pool = True

        self.weight = Int8Params(
            weight.data,
            has_fp16_weights=has_fp16_weights,
            requires_grad=has_fp16_weights,
        )
        self.bias = bias

    def init_8bit_state(self):
        self.state.CB = self.weight.CB
        self.state.SCB = self.weight.SCB
        self.weight.CB = None
        self.weight.SCB = None

    def forward(self, x: torch.Tensor):
        self.state.is_training = self.training
        if self.weight.CB is not None:
            self.init_8bit_state()

        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)
        
        out = bnb.matmul(x, self.weight, bias=self.bias, state=self.state)

        if not self.state.has_fp16_weights:
            if self.state.CB is not None and self.state.CxB is not None:
                # we converted 8-bit row major to turing/ampere format in the first inference pass
                # we no longer need the row-major weight
                del self.state.CB
                self.weight.data = self.state.CxB
        return out
    
def quantize_offline(model, bits: int):
    assert (bits == 4), f'bits: {bits} is not supported'
    
    for i, layer in enumerate(model.model.layers):
        layer.self_attn.W_pack = bnb.nn.Linear4bit(
                            layer.self_attn.W_pack.weight.shape[1],
                            layer.self_attn.W_pack.weight.shape[0],
                            False,
                            torch.float16,
                            compress_statistics=True,
                            quant_type="nf4",
                        )
        layer.self_attn.o_proj = bnb.nn.Linear4bit(
                            layer.self_attn.o_proj.weight.shape[1],
                            layer.self_attn.o_proj.weight.shape[0],
                            False,
                            torch.float16,
                            compress_statistics=True,
                            quant_type="nf4",
                        )

        layer.mlp.gate_proj = bnb.nn.Linear4bit(
                            layer.mlp.gate_proj.weight.shape[1],
                            layer.mlp.gate_proj.weight.shape[0],
                            False,
                            torch.float16,
                            compress_statistics=True,
                            quant_type="nf4",
                        )
        layer.mlp.down_proj = bnb.nn.Linear4bit(
                            layer.mlp.down_proj.weight.shape[1],
                            layer.mlp.down_proj.weight.shape[0],
                            False,
                            torch.float16,
                            compress_statistics=True,
                            quant_type="nf4",
                        )
        layer.mlp.up_proj = bnb.nn.Linear4bit(
                            layer.mlp.up_proj.weight.shape[1],
                            layer.mlp.up_proj.weight.shape[0],
                            False,
                            torch.float16,
                            compress_statistics=True,
                            quant_type="nf4",
                        )
    return model

def quantize_online(model, bits: int):
    def quant(weight, bias=None):
        if bits == 8:
            linear = Linear8bitLtOnline(
                weight,
                bias,
                has_fp16_weights=False,
                threshold=6.0,
            )
            if bias is not None:
                linear.bias = torch.nn.Parameter(bias)
        elif bits == 4:
            linear = Linear4bitOnline(
                weight,
                bias,
                quant_type="nf4", #fp4/nf4
            )
        else:
            raise ValueError("quantize only support 4/8 bit")
        return linear

    for i, layer in enumerate(model.model.layers):
        layer.self_attn.W_pack = quant(layer.self_attn.W_pack.weight)
        layer.self_attn.o_proj = quant(layer.self_attn.o_proj.weight)
        layer.mlp.gate_proj = quant(layer.mlp.gate_proj.weight)
        layer.mlp.down_proj = quant(layer.mlp.down_proj.weight)
        layer.mlp.up_proj = quant(layer.mlp.up_proj.weight)
    return model

def init_model_weight_int4(config, model, state_dict):
    #replace Params4bit.cuda with Params4bitCuda
    Params4bit.cuda = Params4bitCuda

    for i in range(config.num_hidden_layers):
        weight_data = state_dict[f'model.layers.{i}.self_attn.W_pack.weight.data']
        weight_quant_state = state_dict[f'model.layers.{i}.self_attn.W_pack.weight.quant_state']
        model.model.layers[i].self_attn.W_pack.weight = Params4bit(weight_data, requires_grad=False, quant_state=weight_quant_state)
        
        weight_data = state_dict[f'model.layers.{i}.self_attn.o_proj.weight.data']
        weight_quant_state = state_dict[f'model.layers.{i}.self_attn.o_proj.weight.quant_state']
        model.model.layers[i].self_attn.o_proj.weight = Params4bit(weight_data, requires_grad=False, quant_state=weight_quant_state)
        
        weight_data = state_dict[f'model.layers.{i}.mlp.gate_proj.weight.data']
        weight_quant_state = state_dict[f'model.layers.{i}.mlp.gate_proj.weight.quant_state']
        model.model.layers[i].mlp.gate_proj.weight = Params4bit(weight_data, requires_grad=False, quant_state=weight_quant_state)
        
        weight_data = state_dict[f'model.layers.{i}.mlp.up_proj.weight.data']
        weight_quant_state = state_dict[f'model.layers.{i}.mlp.up_proj.weight.quant_state']
        model.model.layers[i].mlp.up_proj.weight = Params4bit(weight_data, requires_grad=False, quant_state=weight_quant_state)
        
        weight_data = state_dict[f'model.layers.{i}.mlp.down_proj.weight.data']
        weight_quant_state = state_dict[f'model.layers.{i}.mlp.down_proj.weight.quant_state']
        model.model.layers[i].mlp.down_proj.weight = Params4bit(weight_data, requires_grad=False, quant_state=weight_quant_state)
        
        model.model.layers[i].input_layernorm.weight = state_dict[f'model.layers.{i}.input_layernorm.weight']
        model.model.layers[i].post_attention_layernorm.weight = state_dict[f'model.layers.{i}.post_attention_layernorm.weight']
    
    model.model.embed_tokens.weight = state_dict['model.embed_tokens.weight']
    model.model.norm.weight = state_dict['model.norm.weight']
    model.lm_head.weight = state_dict['lm_head.weight'] 
    return model