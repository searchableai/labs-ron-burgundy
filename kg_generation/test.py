from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast.from_pretrained("t5-large")

sent = "I am Jing Gu"

breakpoint()