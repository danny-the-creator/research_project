import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

LOCAL_PATH_LORA = "../saved_models/lora"
LOCAL_PATH_SEFT = "../saved_models/seft"

USE_SEFT = False


def load_latest(use_seft=USE_SEFT):
    model_path = LOCAL_PATH_SEFT if use_seft else LOCAL_PATH_LORA
    latest_model = os.path.join(model_path, os.listdir(LOCAL_PATH_LORA)[-1])
    # print(model_path)

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.float16,
    #     bnb_4bit_use_double_quant=False,
    # )

    model = AutoModelForCausalLM.from_pretrained(
        latest_model,
        # quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(latest_model)

    return model, tokenizer

if __name__ == '__main__':
    model, tokenizer = load_latest()

    # print(model)
    # print(tokenizer)


    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # 2. Set up the simplified pipeline

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        # temperature=0.7,
        # top_p=0.9,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
    )

    # 3. Showcase prompt

    while True:
        print("Type your question here:")
        command = input(">>> ")
        if command == "exit":
            break

        messages = [
            {"role": "user", "content": f"{command}"},
        ]

        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        print("\n--- Generating  Text ---")
        output = generator(formatted_prompt)
        # Extract and print output text cleanly
        generated_text = output[0]["generated_text"][len(formatted_prompt):]
        print(generated_text.strip())
        print("---------------------------------")