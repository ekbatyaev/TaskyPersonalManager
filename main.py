import logging
import asyncio
from aiogram import Bot, Dispatcher
from aiogram.filters.command import Command
from aiogram.fsm.context import FSMContext
from aiogram.filters.state import State, StatesGroup
from aiogram import Router, F
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, Message, CallbackQuery
from aiogram.utils.chat_action import ChatActionSender
from aiogram.fsm.storage.memory import MemoryStorage
from tokens_file import telegram_bot_token

# Обработчик следования пользователя
user_router = Router()

# Логирование состояний
logging.basicConfig(level=logging.INFO)

# Объект бота
bot = Bot(token=telegram_bot_token)
# Диспетчер
dp = Dispatcher(storage=MemoryStorage())

# Регистрация маршрутизатора
dp.include_router(user_router)

#Глобальные функции для получения внешних данных

# Классы States

class MainStates(StatesGroup):
    start_state = State()
    problem_types = State()

class TaskCreation(StatesGroup):
    get_title = State()
    title_retrival = State()
    get_description = State()
    description_retrival = State()
    get_deadline = State()
    deadline_retrival = State()

# Клавиатуры

#Клавиатуры стандартного меню

def get_started():
    keyboard_list = [
        [InlineKeyboardButton(text="Начать работу", callback_data='Начать работу')]
    ]
    keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_list)
    return keyboard

# Клавиатура главных опций

def get_user_option():
    keyboard_list = [
        [InlineKeyboardButton(text='Создать новую задачу', callback_data='create_task')],
        [InlineKeyboardButton(text='Найти существующую задачу', callback_data='find_task')],
        [InlineKeyboardButton(text='Вывести список всех текущих задач', callback_data='get_task_list')]
    ]
    keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_list)
    return keyboard

# Клавиатуры блока создания новой задачи

def back_to_main_option():
    keyboard_list = [
        [InlineKeyboardButton(text='Вернуться назад', callback_data='back_to_main')]
    ]
    keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_list)
    return keyboard

def task_options():
    keyboard_list = [
        [InlineKeyboardButton(text='Изменить', callback_data='alter')],
        [InlineKeyboardButton(text='Продолжить', callback_data='continue')]
    ]
    keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_list)
    return keyboard

# Команды

#Команда /start

@user_router.message(Command("start"))
async def start_process(message: Message, state: FSMContext):
    await state.clear()
    async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
        await asyncio.sleep(0.5)
        first_message = await message.answer(f"*Привет, вышкинец!*" + "\n\n" +
                                             f"Меня зовут *Tasky* 🤖 и я умею: \n📌составлять задачи\n📌ставить дедлайны\n📌делать напоминания",
                                             reply_markup=get_started(), parse_mode="Markdown")
        await state.update_data(last_message_id=first_message.message_id)
        await state.update_data(message_edit = first_message)
    await state.set_state(MainStates.start_state)

# States


@user_router.callback_query(F.data == 'Начать работу', MainStates.start_state)
async def task_choice_process(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    message_edit = data.get("message_edit")
    await asyncio.sleep(0.5)
    task_question = await message_edit.edit_text(f"Выбери свою задачу: ",
                                                    reply_markup = get_user_option(), parse_mode = "Markdown")
    await state.update_data(last_message_id=task_question.message_id)
    await state.set_state(MainStates.problem_types)

# Блоки создания новой задачи

@user_router.callback_query(F.data == 'create_task', MainStates.problem_types)
async def creating_new_task(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    last_message_id = data.get("last_message_id")
    if last_message_id:
        await bot.delete_message(chat_id = call.from_user.id, message_id=last_message_id)  # Удаление последнего сообщения
    await asyncio.sleep(0.5)
    create_task_message = await call.message.answer(f"Ура, у нас новая задача! Пожалуйста, напиши название задачи", reply_markup=back_to_main_option(), parse_mode="Markdown")
    await state.update_data(last_message_id=create_task_message.message_id)
    await state.set_state(TaskCreation.get_title)

@user_router.callback_query(F.data == 'back_to_main', TaskCreation.get_title)
async def back_to_main_options(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    last_message_id = data.get("last_message_id")
    if last_message_id:
        await bot.delete_message(chat_id=call.from_user.id, message_id=last_message_id)  # Удаление последнего сообщения
    await asyncio.sleep(0.5)
    task_question = await call.message.answer(f"Выбери свою задачу: ",
                                                 reply_markup=get_user_option(), parse_mode="Markdown")
    await state.update_data(last_message_id=task_question.message_id)
    await state.set_state(MainStates.problem_types)


@user_router.message(F.text, TaskCreation.get_title)
async def get_title(message: Message, state: FSMContext):
    async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
        task_name_confirmation = await message.answer("Cпасибо!\n" + f"*Название твоей задачи: *" + "\n\n" + message.text, reply_markup = task_options(),
                                       parse_mode="Markdown")
        await state.update_data(task_name = message.text)
        await state.update_data(last_message_id=task_name_confirmation.message_id)
        await state.set_state(TaskCreation.title_retrival)

@user_router.callback_query(F.data == 'alter', TaskCreation.title_retrival)
async def altering_task_name(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    last_message_id = data.get("last_message_id")
    if last_message_id:
        await bot.delete_message(chat_id=call.from_user.id, message_id=last_message_id)  # Удаление последнего сообщения
    await asyncio.sleep(0.5)
    task_name_question = await call.message.answer("Пожалуйста, напиши название задачи", parse_mode="Markdown")
    await state.update_data(last_message_id=task_name_question.message_id)
    await state.set_state(TaskCreation.get_title)

@user_router.callback_query(F.data == 'continue', TaskCreation.title_retrival)
async def get_task_description(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    last_message_id = data.get("last_message_id")
    if last_message_id:
        await bot.delete_message(chat_id=call.from_user.id, message_id=last_message_id)  # Удаление последнего сообщения
    await asyncio.sleep(0.5)
    description_quesiton = await call.message.answer(f"Пришло время самого интересного 🔥 - *описания*.\nНапиши описание своей задачи", parse_mode="Markdown")
    await state.update_data(last_message_id=description_quesiton.message_id)
    await state.set_state(TaskCreation.get_description)

@user_router.message(F.text, TaskCreation.get_description)
async def get_description(message: Message, state: FSMContext):
    async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
        task_description_confirmation = await message.answer("Отлично 👍\n" + f"*Описание твоей задачи: *" + "\n\n" + message.text, reply_markup = task_options(),
                                       parse_mode="Markdown")
        await state.update_data(task_description = message.text)
        await state.update_data(last_message_id = task_description_confirmation.message_id)
        await state.set_state(TaskCreation.description_retrival)

@user_router.callback_query(F.data == 'alter', TaskCreation.description_retrival)
async def altering_task_name(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    last_message_id = data.get("last_message_id")
    if last_message_id:
        await bot.delete_message(chat_id=call.from_user.id, message_id=last_message_id)  # Удаление последнего сообщения
    await asyncio.sleep(0.5)
    task_description_question = await call.message.answer("Пожалуйста, напиши описание своей задачи", parse_mode="Markdown")
    await state.update_data(last_message_id=task_description_question.message_id)
    await state.set_state(TaskCreation.get_description)

@user_router.callback_query(F.data == 'continue', TaskCreation.description_retrival)
async def get_task_description(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    last_message_id = data.get("last_message_id")
    if last_message_id:
        await bot.delete_message(chat_id=call.from_user.id, message_id=last_message_id)  # Удаление последнего сообщения
    await asyncio.sleep(0.5)
    description_quesiton = await call.message.answer(f"Осталось самое главное - *дедлайн* ⏰\nНапиши дату в формате: *12:45-17.05*", parse_mode="Markdown")
    await state.update_data(last_message_id=description_quesiton.message_id)
    await state.set_state(TaskCreation.get_description)

@user_router.message(F.text, TaskCreation.get_deadline)
async def get_deadline(message: Message, state: FSMContext):
    async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
        data = await state.get_data()
        last_message_id = data.get("last_message_id")
        if last_message_id:
            await bot.delete_message(chat_id=message.from_user.id, message_id=last_message_id)  # Удаление последнего сообщения
        common_task_message = await message.answer(
            f"Твой задача выглядит так:\n\n" + f"*Название: *" + data.get("task_name") + "\n" + f"*Описание: *" + data.get("task_description") +
            "\n" + f"*Дедлайн: *" + message.text, parse_mode="Markdown")
        await state.update_data( task_deadline = message.text)
        await state.update_data(last_message_id = common_task_message.message_id)
        await state.set_state(TaskCreation.deadline_retrival)

# Запуск процесса поллинга новых апдейтов
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())