# Arbitrary Graph-Attention Transformers

Transformer models with arbitrary graph attention patterns in JAX.

TODO list:
 * [ ] training code
   * [x] instantiate the model
   * [x] jit + evaluate the model
   * [x] training loop
   * [x] training working with long inputs
   * [ ] update training loop
 * [x] add attention patterns
   * [x] fully connected attn pattern
   * [x] window attn pattern
   * [x] longformer-style attn pattern
     * [ ] add dilation support
     * [ ] add block window support
 * [ ] optimizations
   * [ ] add a more efficient global / local attention computation
   * [x] accept attention pattern defined per layer or share between layers (tieing)
   * [x] tie the positional embeddings between layers
   * [x] support LoRA fine-tuning (lorax)
 * [x] unit tests
