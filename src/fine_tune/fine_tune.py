import os
import torch
import torch.nn as nn

from dotenv import load_dotenv
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HUGGING_FACE_API_KEY")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device:\n", device, "\n")

modelName = "google/gemma-3-1b-it"

tokenizer = AutoTokenizer.from_pretrained(modelName)

inputIds = tokenizer("Hello world!").get("input_ids")

print(inputIds, "\n")

# Query for which training is required
inputConversation = [
    {
        "role": "user",
        "content": "Where to learn GenAI ?"
    },
    {
        "role": "assistant",
        "content": "Best place to learn is "
    }
]

# Use chat template to get prompt string
# conversation=inputConversation: This is the input conversation, which is a list of dictionaries representing a dialogue between a user and an assistant.
# tokenize=False: This indicates that the method should not tokenize the conversation into input IDs (tokens) but instead return the formatted text as a string.
# continue_final_message=True: This likely specifies that the final message in the conversation should be left open-ended, allowing the model to continue generating text.

inputTagText = tokenizer.apply_chat_template(conversation=inputConversation, tokenize=False,
                                             continue_final_message=True, )
inputTokenText = tokenizer.apply_chat_template(conversation=inputConversation, tokenize=True, )

# How model takes input in tags format
print("Input Tags Text:\n", inputTagText, "\n")

# How model takes input in Tokens
print("Input Token Text:\n", inputTokenText, "\n")

# inputDeTokens=tokenizer.apply_chat_template(
#     conversation=inputConversation,
#     tokenize=False,
#     continue_final_message=True,
# )
# print("Input DeTokens:\n", inputDeTokens,"\n")

outputLabel = "Flutter master course by Shivam Srivastava"

fullConversation = inputTagText + outputLabel + tokenizer.eos_token

print("Full Conversation:\n", fullConversation, "\n")

fullConversationToken = tokenizer(fullConversation, return_tensors="pt", add_special_tokens=False).to(device).get(
    "input_ids")
print("Full Conversation Token:\n", fullConversationToken, "\n")

answerInputsIds = fullConversationToken[:, :-1].to(device)
answerTargetIds = fullConversationToken[:, 1:].to(device)

print("answerInputsIds:\n", answerInputsIds, "\n")
print("answerTargetIds:\n", answerTargetIds, "\n")


def calculateLoss(logits, labels):
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    crossEntropy = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
    return crossEntropy


autoModel = AutoModelForCausalLM.from_pretrained(modelName, torch_dtype=torch.bfloat16,
                                                 attn_implementation='eager').to(device)

# start training model
autoModel.train()

# use optimizer
optimizer = AdamW(autoModel.parameters(), lr=3e-5, weight_decay=0.01)

for _ in range(10):
    out = autoModel(input_ids=answerInputsIds)
    loss = calculateLoss(out.logits, answerTargetIds).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print("Loss ", loss.item())

print("answerTargetIds:\n", answerTargetIds, "\n")

inputPrompt2 = "Where to learn GenAI ?"
# inputTokens2 = tokenizer(inputPrompt2, return_tensors="pt").get("input_ids").to(device)
# print("Input Tokens2:\n", inputTokens2, "\n")
#
# outputTokens = autoModel.generate(inputTokens2)
# print("Output Tokens:\n", outputTokens, "\n")
#
# outputText = tokenizer.batch_decode(outputTokens)
# print("Output Text:\n", outputText, "\n")

finalQuestionInputTags = tokenizer.apply_chat_template(conversation=inputConversation, tokenize=False )
print("finalQuestionInputTags:\n", finalQuestionInputTags, "\n")

finalInputTokens=tokenizer(finalQuestionInputTags,return_tensors="pt").to(device)
print("finalInputTokens:\n", finalInputTokens, "\n")

finalOutputToken = autoModel.generate(finalInputTokens["input_ids"])

print("finalOutputToken:\n", finalOutputToken, "\n")

finalOutputDecoded = tokenizer.batch_decode(finalOutputToken, skip_special_token=True)

print("finalOutputDecoded:\n", finalOutputDecoded, "\n")
print("Final Answer:\n", finalOutputDecoded, "\n")
