#!/usr/bin/env python3
"""
Test script to validate that hidden states returned by vLLM match those from HuggingFace.
This script generates text using both vLLM and HuggingFace Transformers and compares
the hidden states from the last layer.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
import argparse


def test_hidden_states(model_name="facebook/opt-125m", prompt="Hello, my name is", max_tokens=10):
    """
    Test that vLLM returns the same hidden states as HuggingFace.
    
    Args:
        model_name: The model to test
        prompt: The input prompt
        max_tokens: Number of tokens to generate
    """
    print(f"Testing model: {model_name}")
    print(f"Prompt: {prompt}")
    print(f"Max tokens: {max_tokens}")
    print("-" * 50)
    
    # Initialize HuggingFace model and tokenizer
    print("Loading HuggingFace model...")
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Generate with HuggingFace and get hidden states
    print("Generating with HuggingFace...")
    inputs = hf_tokenizer(prompt, return_tensors="pt").to(hf_model.device)
    
    with torch.no_grad():
        # Generate tokens one by one to match vLLM's behavior
        hf_outputs = hf_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  # Use greedy decoding for deterministic results
            output_hidden_states=True,
            return_dict_in_generate=True
        )
    
    # Extract hidden states from HuggingFace
    # The hidden_states is a tuple of length (num_generated_tokens + 1)
    # Each element is a tuple of tensors for each layer
    hf_hidden_states = hf_outputs.hidden_states
    
    # Get the last layer hidden states for each generated token
    hf_last_hidden = []
    for token_hidden_states in hf_hidden_states:
        if token_hidden_states is not None:
            # Get the last layer's hidden states (last element in the tuple)
            # Shape: [batch_size, seq_len, hidden_dim]
            last_layer = token_hidden_states[-1]
            # Get the last token's hidden state
            last_token_hidden = last_layer[:, -1, :].cpu().numpy()
            hf_last_hidden.append(last_token_hidden)
    
    if hf_last_hidden:
        hf_last_hidden = np.concatenate(hf_last_hidden, axis=0)
    else:
        hf_last_hidden = np.array([])
    
    hf_generated_text = hf_tokenizer.decode(hf_outputs.sequences[0], skip_special_tokens=True)
    print(f"HuggingFace output: {hf_generated_text}")
    print(f"HuggingFace hidden states shape: {hf_last_hidden.shape if len(hf_last_hidden) > 0 else 'None'}")
    
    # Initialize vLLM with hidden states enabled
    print("\nLoading vLLM model...")
    vllm_model = LLM(
        model=model_name,
        return_hidden_states=True,
        dtype="float16",
        gpu_memory_utilization=0.5  # Use less memory to allow both models
    )
    
    # Generate with vLLM
    print("Generating with vLLM...")
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,  # Greedy decoding
        top_p=1.0,
    )
    
    vllm_outputs = vllm_model.generate([prompt], sampling_params)
    vllm_output = vllm_outputs[0]
    
    print(f"vLLM output: {vllm_output.prompt + vllm_output.outputs[0].text}")
    
    # Extract hidden states from vLLM
    vllm_hidden_states = vllm_output.hidden_states
    
    if vllm_hidden_states is not None:
        vllm_hidden_states_np = vllm_hidden_states.cpu().numpy()
        print(f"vLLM hidden states shape: {vllm_hidden_states_np.shape}")
        
        # Compare the hidden states
        print("\n" + "=" * 50)
        print("Comparing hidden states...")
        
        # We need to be careful about the comparison:
        # 1. vLLM might include prompt tokens in hidden states
        # 2. There might be slight numerical differences due to different implementations
        
        # Get the number of generated tokens
        num_generated = len(vllm_output.outputs[0].token_ids)
        
        # Extract only the hidden states for generated tokens
        # Assuming vLLM returns all hidden states (prompt + generated)
        if vllm_hidden_states_np.shape[0] > num_generated:
            # Take the last num_generated hidden states
            vllm_generated_hidden = vllm_hidden_states_np[-num_generated:]
        else:
            vllm_generated_hidden = vllm_hidden_states_np
        
        print(f"Comparing last {num_generated} generated tokens")
        print(f"vLLM generated hidden shape: {vllm_generated_hidden.shape}")
        print(f"HF generated hidden shape: {hf_last_hidden.shape}")
        
        # Compare shapes
        if vllm_generated_hidden.shape == hf_last_hidden.shape:
            print("✓ Shapes match!")
            
            # Compare values (with tolerance for numerical differences)
            if hf_last_hidden.size > 0:
                # Calculate statistics
                abs_diff = np.abs(vllm_generated_hidden - hf_last_hidden)
                max_diff = np.max(abs_diff)
                mean_diff = np.mean(abs_diff)
                rel_diff = abs_diff / (np.abs(hf_last_hidden) + 1e-8)
                max_rel_diff = np.max(rel_diff)
                
                print(f"Maximum absolute difference: {max_diff:.6f}")
                print(f"Mean absolute difference: {mean_diff:.6f}")
                print(f"Maximum relative difference: {max_rel_diff:.6f}")
                
                # Check if differences are within acceptable tolerance
                tolerance = 1e-2  # Adjust based on model precision
                if max_diff < tolerance:
                    print(f"✓ Hidden states match within tolerance ({tolerance})!")
                else:
                    print(f"✗ Hidden states differ more than tolerance ({tolerance})")
                    print("Note: Some difference is expected due to implementation details")
                    
                # Show a sample of values for debugging
                print("\nSample values (first 5 dimensions of first token):")
                print(f"vLLM:  {vllm_generated_hidden[0, :5]}")
                print(f"HF:    {hf_last_hidden[0, :5]}")
        else:
            print("✗ Shape mismatch!")
            print("This might be due to different tokenization or output format")
    else:
        print("✗ vLLM did not return hidden states!")
        print("Make sure return_hidden_states is properly implemented")
    
    print("\n" + "=" * 50)
    print("Test completed!")
    
    # Clean up
    del hf_model
    del vllm_model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Test vLLM hidden states implementation")
    parser.add_argument("--model", type=str, default="facebook/opt-125m",
                       help="Model to test (default: facebook/opt-125m)")
    parser.add_argument("--prompt", type=str, default="Hello, my name is",
                       help="Input prompt (default: 'Hello, my name is')")
    parser.add_argument("--max-tokens", type=int, default=10,
                       help="Number of tokens to generate (default: 10)")
    
    args = parser.parse_args()
    
    try:
        test_hidden_states(
            model_name=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens
        )
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()