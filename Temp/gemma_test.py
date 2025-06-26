# Common imports
import os
import jax
import jax.numpy as jnp

# Gemma imports
from gemma import gm

model = gm.nn.Gemma3_1B()
params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_1B_IT)

tokenizer = gm.text.Gemma3Tokenizer()

prompt = tokenizer.encode('One word to describe Paris: \n\n', add_bos=True)
prompt = jnp.asarray(prompt)

# Run the model
out = model.apply(
    {'params': params},
    tokens=prompt,
    return_last_only=True,  # Only predict the last token
)


# Sample a token from the predicted logits
next_token = jax.random.categorical(
    jax.random.key(1),
    out.logits
)
output = tokenizer.decode(next_token)

print(f'Next token: {output}')