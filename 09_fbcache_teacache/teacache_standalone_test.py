#!/usr/bin/env python3
"""
TeaCache Standalone Test for FLUX.1-schnell
Based on the standalone TeaCache implementation from ali-vilab/TeaCache
"""

import time
import torch
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union
from diffusers import FluxPipeline
from diffusers.models import FluxTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers

logger = logging.get_logger(__name__)

def teacache_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """TeaCache-enabled forward method for FluxTransformer2DModel"""
        
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        # TeaCache logic
        if self.enable_teacache:
            inp = hidden_states.clone()
            temb_ = temb.clone()
            modulated_inp, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.transformer_blocks[0].norm1(inp, emb=temb_)
            
            if self.cnt == 0 or self.cnt == self.num_steps-1:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
            else: 
                coefficients = [4.98651651e+02, -2.83781631e+02,  5.58554382e+01, -3.82021401e+00, 2.64230861e-01]
                rescale_func = np.poly1d(coefficients)
                self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance = 0
            self.previous_modulated_input = modulated_inp 
            self.cnt += 1 
            if self.cnt == self.num_steps:
                self.cnt = 0           
        
        if self.enable_teacache:
            if not should_calc:
                hidden_states += self.previous_residual
            else:
                ori_hidden_states = hidden_states.clone()
                # Process transformer blocks
                for index_block, block in enumerate(self.transformer_blocks):
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

                hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

                # Process single transformer blocks
                for index_block, block in enumerate(self.single_transformer_blocks):
                    hidden_states = block(
                        hidden_states=hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

                hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                self.previous_residual = hidden_states - ori_hidden_states
        else:
            # Standard forward pass without TeaCache
            for index_block, block in enumerate(self.transformer_blocks):
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

            for index_block, block in enumerate(self.single_transformer_blocks):
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

def main():
    print("TeaCache Standalone Test for FLUX.1-schnell")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Test parameters
    prompt = "A futuristic city skyline at sunset with flying cars"
    num_inference_steps = 4  # Using 4 steps for FLUX.1-schnell
    seed = 42
    
    # TeaCache thresholds to test
    teacache_configs = [
        {"threshold": 0.25, "expected_speedup": "1.5x"},
        {"threshold": 0.4, "expected_speedup": "1.8x"},
        {"threshold": 0.6, "expected_speedup": "2.0x"},
    ]
    
    print("\nLoading FLUX.1-schnell pipeline...")
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    
    # Monkey patch the forward method
    FluxTransformer2DModel.forward = teacache_forward
    
    results = []
    
    # 1. Baseline test (no TeaCache)
    print("\n1. Baseline test (no TeaCache)...")
    pipeline.transformer.__class__.enable_teacache = False
    
    start_time = time.time()
    image_baseline = pipeline(
        prompt,
        num_inference_steps=num_inference_steps,
        height=1024,
        width=1024,
        generator=torch.Generator("cuda").manual_seed(seed)
    ).images[0]
    baseline_time = time.time() - start_time
    
    image_baseline.save("output/teacache_baseline.png")
    print(f"Baseline time: {baseline_time:.3f}s")
    results.append(("Baseline", baseline_time, 1.0, "output/teacache_baseline.png"))
    
    # 2. TeaCache tests
    for i, config in enumerate(teacache_configs):
        threshold = config["threshold"]
        expected_speedup = config["expected_speedup"]
        
        print(f"\n{i+2}. TeaCache test (threshold={threshold}, expected {expected_speedup})...")
        
        # Enable TeaCache
        pipeline.transformer.__class__.enable_teacache = True
        pipeline.transformer.__class__.cnt = 0
        pipeline.transformer.__class__.num_steps = num_inference_steps
        pipeline.transformer.__class__.rel_l1_thresh = threshold
        pipeline.transformer.__class__.accumulated_rel_l1_distance = 0
        pipeline.transformer.__class__.previous_modulated_input = None
        pipeline.transformer.__class__.previous_residual = None
        
        start_time = time.time()
        image_teacache = pipeline(
            prompt,
            num_inference_steps=num_inference_steps,
            height=1024,
            width=1024,
            generator=torch.Generator("cuda").manual_seed(seed)
        ).images[0]
        teacache_time = time.time() - start_time
        
        speedup = baseline_time / teacache_time
        filename = f"output/teacache_thresh_{threshold}.png"
        image_teacache.save(filename)
        
        print(f"TeaCache time: {teacache_time:.3f}s")
        print(f"Speedup: {speedup:.2f}x")
        results.append((f"TeaCache {threshold}", teacache_time, speedup, filename))
    
    # 3. Performance summary
    print("\n" + "="*60)
    print("TEACACHE PERFORMANCE SUMMARY")
    print("="*60)
    print(f"{'Configuration':<20} {'Time (s)':<10} {'Speedup':<10} {'Image':<25}")
    print("-"*60)
    
    for config, exec_time, speedup, filename in results:
        print(f"{config:<20} {exec_time:<10.3f} {speedup:<10.2f}x {filename:<25}")
    
    print(f"\nPrompt: {prompt}")
    print(f"Steps: {num_inference_steps}")
    print(f"Resolution: 1024x1024")
    print(f"Model: FLUX.1-schnell")
    print(f"Device: {torch.cuda.get_device_name()}")
    
    # Calculate best speedup
    best_speedup = max(results[1:], key=lambda x: x[2])
    print(f"\nBest TeaCache speedup: {best_speedup[2]:.2f}x with threshold {best_speedup[0].split()[-1]}")

if __name__ == "__main__":
    main()
