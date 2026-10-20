# quantization Research

Questions I want to answer:

If we're using different inference and training kernels, particularly with low-bit quantization, how do we ensure that the rollout policy matches the training policy? Could numerical differences introduce nonzero KL divergence even before a policy update?

What if we quantize different weights, matrices, or layers independently and measure how much each contributes to the degradation in model performance?

Making reinforcement learning cheaper without destabilizing training?

Commands:
source .venv/bin/activate
