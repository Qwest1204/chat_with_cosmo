import logging
import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import colorful as cf
from translate import Translator
import config
cf.use_true_colors()
cf.use_style('monokai')

# Устанавливаем уровень логирования
logging.basicConfig(level=logging.INFO)

# Создаем объект бота
bot = Bot(token='6159322821:AAHSRHucxja62pfDzN1zZolyig7GZArCtVg')

# Создаем объект диспетчера
dp = Dispatcher(bot)
messages = []
# Обработчик команды /start
class CosmoAgent:
    @dp.message_handler(commands=['start'])
    async def send_welcome(message: types.Message):
       await message.reply("valera loading....")
       def __init__(self):
        print(cf.bold | cf.purple("Loading VALEra..."))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/cosmo-xl")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("allenai/cosmo-xl").to(self.device)
        self.conversation_history = []


    # Обработчик текстовых сообщений
    # Обработчик любых сообщений

    @dp.message_handler()
    async def save_message(message: types.Message):
        """
        Сохраняем сообщение в переменную
        """
        messages.append(message.text)
        await message.answer("Сообщение сохранено!")

    # Обработчик команды /show_messages
    @dp.message_handler(commands=['show_messages'])
    async def show_messages(message: types.Message):
        def chat(self, situation_narrative, role_instruction):
            print(cf.green("Chat with Cosmo! Input [RESET] to reset the conversation history and [END] to end the conversation."))
            while True:
                translator = Translator(from_lang=config.LANGUAGE, to_lang='en')
                translator1 = Translator(from_lang='en', to_lang=config.LANGUAGE)
                input1 = messages
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
        await message.reply("Valera: " + response)


    def main():
        print(cf.bold | cf.blue("Welcome to Chat with VALEra!"))
        cosmo = CosmoAgent()
        cosmo.run()
        
        

# Запускаем бота
if __name__ == '__main__':
    asyncio.run(dp.start_polling())