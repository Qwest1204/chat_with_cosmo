import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import colorful as cf
from translate import Translator
import config
import asyncio
import logging
from aiogram import Bot, Dispatcher, types
import aiogram
from aiogram.filters.command import Command
cf.use_true_colors()
cf.use_style('monokai')

logging.basicConfig(level=logging.INFO)
bot = Bot(token=config.TOKEN)
dp = Dispatcher()

@dp.message(Command("start"))
class CosmoAgent:
    async def cmd_start(message: types.Message): 
        await message.answer('Loading VALERA')
        def __init__(self):
            
            print(cf.bold | cf.purple("Loading VALERA ..."))
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained("allenai/cosmo-xl")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("allenai/cosmo-xl").to(self.device)
            self.conversation_history = []
    async def main():
        await dp.start_polling(bot)


    def observe(self, observation):
        self.conversation_history.append(observation)

    def set_input(self, situation_narrative="", role_instruction=""):
        input_text = " <turn> ".join(self.conversation_history)

        if role_instruction != "":
            input_text = "{} <sep> {}".format(role_instruction, input_text)

        if situation_narrative != "":
            input_text = "{} <sep> {}".format(situation_narrative, input_text)

        return input_text

    def generate(self, situation_narrative, role_instruction, user_response):

        self.observe(user_response)

        input_text = self.set_input(situation_narrative, role_instruction)

        inputs = self.tokenizer([input_text], return_tensors="pt").to(self.device)
        outputs = self.model.generate(inputs["input_ids"], max_new_tokens=128, temperature=1.0, top_p=.95, do_sample=True)
        cosmo_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        self.observe(cosmo_response)

        return cosmo_response


    def run(self):
        def get_valid_input(prompt, default):
            while True:
                user_input = input(prompt)
                if user_input in ["Y", "N", "y", "n"]:
                    return user_input
                if user_input == "":
                    return default
        
        while True:
            continue_chat = ""
            situation_narrative = input(cf.yellow("Input the situation description/narrative of the conversation (You can leave it empty):"))
            if situation_narrative == "":
                situation_narrative = "Cosmo is having a friendly conversation with a friend."
            role_instruction = input(cf.orange("Which role should Cosmo take? Who is Cosmo talking to? (You can leave it empty):"))
            if role_instruction == "":
                role_instruction = "You are Cosmo and you are talking to a friend."

            self.chat(situation_narrative, role_instruction)
            continue_chat = get_valid_input(cf.purple("Start a new conversation with new setup? [Y/N]:"), "Y")
            if continue_chat in ["N", "n"]:
                break

        print(cf.blue("Cosmo: See you!"))

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
            print(cf.blue("Cosmo: " + response))

def main():
    print(cf.bold | cf.blue("Welcome to SODAverse!"))
    cosmo = CosmoAgent()
    cosmo.run()

if __name__ == '__main__':
   asyncio.run(main())
