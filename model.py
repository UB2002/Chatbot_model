# from transformers import AutoModelForCausalLM, AutoTokenizer
#from dotenv import load_dotenv
#import os
#load_dotenv()

# # model_id = "Qwen/Qwen3-4B"
# model_id = "Qwen/Qwen3-1.7B"
#access_token = os.getenv("ACCESS_TOKEN")


# tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
# model = AutoModelForCausalLM.from_pretrained(model_id, token=access_token)


# print("saving the model")


# tokenizer.save_pretrained(f"Qwen1.7B_tokenizer/{model_id}")
# model.save_pretrained(f"Qwen1.7B_model{model_id}")



from transformers import AutoModelForCausalLM, AutoTokenizer

class QwenChatbot:
    def __init__(self, model_id="Qwen/Qwen3-0.6B"):
        self.tokenizer = AutoTokenizer.from_pretrained(f"tokenizer/{model_id}", local_files_only = True)
        self.model = AutoModelForCausalLM.from_pretrained(f"model/{model_id}", local_files_only = True)
        self.history = []
        

    def generate_response(self, user_input):
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(text, return_tensors="pt")
        response_ids = self.model.generate(**inputs, max_new_tokens=32768)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response