from utils import llm_complete, llm_output_text


# Example models
# These models were available at the point of experimentation, check official updates to see if a model has been removed for your own use
# e.g., refer to https://platform.claude.com/docs/en/api/models/list
CLAUDE_MODELS = ['claude-3-haiku-20240307', 'claude-3-sonnet-20240229', 'claude-3-opus-20240229', "claude-3-5-sonnet-20240620",\
                 "claude-sonnet-4-6", "claude-opus-4-6", "claude-opus-4-5-20251101", "claude-haiku-4-5-20251001",\
                 "claude-sonnet-4-5-20250929", "claude-opus-4-1-20250805", "claude-opus-4-20250514"]
GPT_MODELS = ["gpt-3.5-turbo-1106", "gpt-4-1106-preview", "gpt-4-0125-preview", "gpt-4-0613", "gpt-4o-2024-05-13",\
              "o3-mini-2025-01-31", "gpt-5-2025-08-07"]
LLAMA_MODELS = ["meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"]

llm_options={'max_tokens': 20, 'temperature': 0.0, "logprobs": True, "n":1}


if __name__ == "__main__":
    # Test claude
    res1 = llm_complete([{"role": "system", "content": "You are a creative, but rigorous author. You start every sentence with 'Soooo'."},
                         {"role": "user", "content": "What is your name?"}],
                       model=CLAUDE_MODELS[0],
                       options=llm_options)
    print("CLAUDE RESPONDED:")
    print(llm_output_text(res1))

    # Test GPT
    res2 = llm_complete([{"role": "system", "content": "You are a creative, but rigorous author. You start every sentence with 'Soooo'."},
            {"role": "user", "content": "What is your name?"}],
                       model=GPT_MODELS[0],
                       options=llm_options)
    print("GPT RESPONDED:")
    print(llm_output_text(res2))

    # Test GPT (with reasoning summary for gpt-5 and o-series models)
    res2 = llm_complete([{"role": "system", "content": "You are a creative, but rigorous author. You start every sentence with 'Soooo'."},
            {"role": "user", "content": "What is your name? How many o's are there in your name?"}],
                       model=GPT_MODELS[-1],
                       options=llm_options)
    print("GPT (reasoning model) RESPONDED:")
    print(llm_output_text(res2))

    # Test Llama
    res3 = llm_complete([{"role": "system",
                          "content": "You are a creative, but rigorous author. You start every sentence with 'Soooo'."},
                         {"role": "user", "content": "What is your name?"}],
                        model=LLAMA_MODELS[0],
                        options=llm_options)
    print("LLAMA RESPONDED:")
    print(llm_output_text(res3))




