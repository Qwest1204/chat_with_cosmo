import openai
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor

token = 'Сюда вставляем токен полученный от Ботфазер'
openai.api_key = 'сюда вставляем api от openAI'

bot = Bot(token)
dp = Dispatcher(bot)
@dp.message_handler(commands=['start'])
async def start_command(message : types.Message):
    await message.answer("Привет! Я телеграм бот ChatGPT Хак Анонима\n\nПодпишись на канал https://youtube.com/@HackAnonim\n\nhttps://t.me/hack_anonim")


@dp.message_handler()
async def send(message : types.Message):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=message.text,
        temperature=0.9,
        max_tokens=1000,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.6,
        stop=["You:"]
                                        )
    await message.answer(response['choices'][0]['text'])

executor.start_polling(dp, skip_updates=True)