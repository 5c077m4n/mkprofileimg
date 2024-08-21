import os

from diffusers import FluxPipeline
from torch import Generator, bfloat16


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", torch_dtype=bfloat16
    )
    pipe.enable_model_cpu_offload(device="cpu")

    prompt = "Can you please generate an ultra-realistic portrait of a young, happy caucasian person, with broad shoulders, wearing an plain oversized t shirt, glasses, bald, with a short beard?"
    result = pipe(
        prompt=prompt,
        guidance_scale=0.0,
        height=768,
        width=1360,
        num_inference_steps=4,
        max_sequence_length=256,
        generator=Generator("cpu").manual_seed(0),
    )

    for index, img in enumerate(result.images):
        img.save(f"flux-schnell-{index}.png")


if __name__ == "__main__":
    main()
