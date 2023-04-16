import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import colorful as cf
from translate import Translator
import config
cf.use_true_colors()
cf.use_style('monokai')


class CosmoAgent:
    def __init__(self):
        print(cf.bold | cf.purple("Loading VALEra..."))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/cosmo-xl")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("allenai/cosmo-xl").to(self.device)
        self.conversation_history = []

    def chat(self, situation_narrative, role_instruction):
        print(cf.green("Chat with Cosmo! Input [RESET] to reset the conversation history and [END] to end the conversation."))
        while True:
            translator = Translator(from_lang=config.LANGUAGE, to_lang='en')
            translator1 = Translator(from_lang='en', to_lang=config.LANGUAGE)
            input1 = input("You: ")
            user_input = translator.translate(input1)
            if user_input == "[RESET]":
                self.reset_history()
                print(cf.green("[Conversation history cleared. Chat with Cosmo!]"))
                continue
            if user_input == "[END]":
                self.reset_history()
                break
            text1 = self.generate(situation_narrative, role_instruction, user_input)
            response = translator1.translate(text1)
            print(cf.blue("Valera: " + response))


def main():
    print(cf.bold | cf.blue("Welcome to Chat with VALEra!"))
    cosmo = CosmoAgent()
    cosmo.run()
    

if __name__ == '__main__':
   main()
