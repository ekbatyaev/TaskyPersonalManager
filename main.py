import re
import json
import logging
import asyncio
import datetime
import torch
from aiogram import Bot, Dispatcher
from aiogram.filters.command import Command
from aiogram.fsm.context import FSMContext
from aiogram.filters.state import State, StatesGroup
from aiogram import Router, F
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, Message, CallbackQuery
from aiogram.utils.chat_action import ChatActionSender
from transformers import AutoTokenizer, AutoModel
from aiogram.fsm.storage.memory import MemoryStorage
from torch.nn.functional import cosine_similarity
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

# Путь к данным задач пользователей

user_path = "user_tasks/tasks.json"

# Модель для поиска релевантной информации

MODEL_NAME= "intfloat/multilingual-e5-large"

# Глобальные функции для получения и обработки данных

async def get_embeddings(texts):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # Берем CLS-токен

async def search(user_question, user_id):
    data = await load_data(user_path)
    print(data, user_id)
    print(data.get(user_id))

    if data.get(user_id) is None:
        return "Вопрос не найден"
    tasks = list(data[user_id].keys())
    tasks_embeddings = await get_embeddings(tasks)
    user_question_emb = await get_embeddings(user_question)
    database_tasks_embed = torch.stack([torch.as_tensor(cell) for cell in tasks_embeddings])
    similarities = cosine_similarity(
        user_question_emb.unsqueeze(1), database_tasks_embed.unsqueeze(0), dim=-1
    )
    if similarities.numel() == 0:
        raise ValueError("Ошибка: similarities пустой!")
    best_match_idx = similarities.argmax().item()  # Индекс наибольшего сходства
    if best_match_idx >= len(tasks_embeddings):
        raise IndexError(f"Ошибка: best_match_idx ({best_match_idx}) выходит за границы {len(tasks_embeddings)}")

    best_match = tasks[best_match_idx]
    best_score = similarities.flatten()[best_match_idx].item()  # Исправлено для корректного доступа
    print("answer: " + str(user_question) + " " + str(best_score) + " " + str(best_match))
    print(best_score, best_match)
    if best_score < 0.85:
        return "Вопрос не найден"
    return best_match


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
    extend_deadline = State()
    deadline_retrival = State()
    overall_task_retrival = State()

class TaskSearch(StatesGroup):
    get_query = State()
    query_retrival = State()
    task_changing = State()
    field_changing = State()
    alter_field_retrival = State()
    extension_retrival = State()
# Клавиатуры

#Клавиатуры стандартного меню

def get_started():
    keyboard_list = [
        [InlineKeyboardButton(text="Начать работу 🚀 ", callback_data='start_work')]
    ]
    keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_list)
    return keyboard

# Клавиатура главных опций

def get_user_option():
    keyboard_list = [
        [InlineKeyboardButton(text='Создать новую задачу 🆕', callback_data='create_task')],
        [InlineKeyboardButton(text='Найти существующую задачу 🔍', callback_data='find_task')],
        [InlineKeyboardButton(text='Вывести список всех задач 📋', callback_data='get_task_list')]
    ]
    keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_list)
    return keyboard

# Клавиатуры блока создания новой задачи

def back_to_main_option():
    keyboard_list = [
        [InlineKeyboardButton(text='Вернуться в главное меню 🏠', callback_data='back_to_main')]
    ]
    keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_list)
    return keyboard

def extend_options():
    keyboard_list = [
        [InlineKeyboardButton(text='Да ✅', callback_data='answer_yes')],
        [InlineKeyboardButton(text='Нет ❌', callback_data='answer_no')]
    ]
    keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_list)
    return keyboard

def task_options():
    keyboard_list = [
        [InlineKeyboardButton(text='Продолжить ➡️', callback_data='continue')],
        [InlineKeyboardButton(text='Изменить ✏️', callback_data='alter')]
    ]
    keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_list)
    return keyboard

def task_creation_end_options():
    keyboard_list = [
        [InlineKeyboardButton(text='Создать задачу ✅', callback_data='make_task')],
        [InlineKeyboardButton(text='Перезаписать 🔄', callback_data='rewrite')]
    ]
    keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_list)
    return keyboard

# Клавиатуры поиска существующей задачи

def task_search_options():
    keyboard_list = [
        [InlineKeyboardButton(text='Просмотреть задачу 👀', callback_data='watch_task')],
        [InlineKeyboardButton(text='Найти другую задачу 🔎', callback_data='find_another_task')],
        [InlineKeyboardButton(text='Вывести все задачи 📋', callback_data='write_all')],
    ]
    keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_list)
    return keyboard

def founded_task_options():
    keyboard_list = [
        [InlineKeyboardButton(text='Задача выполнена ✅', callback_data='task_completed')],
        [InlineKeyboardButton(text='Изменить задачу ✏️', callback_data='change_task')],
        [InlineKeyboardButton(text='Удалить задачу 🗑', callback_data='delete_task')]
    ]
    keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_list)
    return keyboard

def change_task_options():
    keyboard_list = [
        [InlineKeyboardButton(text='Изменить название ✏️', callback_data='change_name')],
        [InlineKeyboardButton(text='Изменить описание 📝', callback_data='change_description')],
        [InlineKeyboardButton(text='Изменить дедлайн ⏳', callback_data='change_deadline')],
        [InlineKeyboardButton(text='Изменить продление ✍️', callback_data='change_extension')],
        [InlineKeyboardButton(text='Вернуться назад 🔙', callback_data='back')],
    ]
    keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_list)
    return keyboard

def back_to_change_options():
    keyboard_list = [
        [InlineKeyboardButton(text='Изменить еще одно поле ✏️', callback_data='change_another_field')],
        [InlineKeyboardButton(text='Вернуться в главное меню 🏠', callback_data='back_to_main_menu')]
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
                                             f"Меня зовут *Tasky* 🤖 и я *умею*: \n📌составлять задачи\n📌ставить дедлайны\n📌делать напоминания",
                                             reply_markup=get_started(), parse_mode="Markdown")
        await state.update_data(last_message_id=first_message.message_id)
    await state.set_state(MainStates.start_state)

# States


@user_router.callback_query(F.data == 'start_work', MainStates.start_state)
async def task_choice_process(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    last_message_id = data.get("last_message_id")
    if last_message_id:
        await bot.delete_message(chat_id=call.from_user.id, message_id=last_message_id)  # Удаление последнего сообщения
    await asyncio.sleep(0.5)
    task_question = await call.message.answer(f"Выбери свою задачу: ",
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
    create_task_message = await call.message.answer(f"Пожалуйста, напиши *название задачи*", reply_markup=back_to_main_option(), parse_mode="Markdown")
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
        task_name_confirmation = await message.answer(f"*Название твоей задачи: *" + message.text, reply_markup = task_options(),
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
    description_question = await call.message.answer(f"Пришло время самого интересного 🔥 - *описания*.\nНапиши описание своей задачи", parse_mode="Markdown")
    await state.update_data(last_message_id=description_question.message_id)
    await state.set_state(TaskCreation.get_description)

@user_router.message(F.text, TaskCreation.get_description)
async def get_description(message: Message, state: FSMContext):
    async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
        task_description_confirmation = await message.answer(f"*Описание твоей задачи: *" + "\n\n" + message.text, reply_markup = task_options(),
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
    await state.set_state(TaskCreation.get_deadline)


@user_router.message(F.text, TaskCreation.get_deadline)
async def get_deadline(message: Message, state: FSMContext):
    async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
        data = await state.get_data()
        last_message_id = data.get("last_message_id")
        if last_message_id:
            await bot.delete_message(chat_id=message.from_user.id, message_id=last_message_id)  # Удаление последнего сообщения

        if await check_deadline_format(message.text):
            task_deadline_message = await message.answer("Дедлайн задачи 🔥: " + message.text,
                                                         reply_markup=task_options()
                                                         , parse_mode="Markdown")
            await state.update_data(task_deadline=message.text)
            await state.update_data(last_message_id=task_deadline_message.message_id)
            await state.set_state(TaskCreation.deadline_retrival)
        else:
            task_deadline_question = await message.answer("Пожалуйста, напиши дату в формате: *12:45-17.05*",
                                                               parse_mode="Markdown")
            await state.update_data(last_message_id = task_deadline_question.message_id)
            await state.set_state(TaskCreation.get_deadline)

@user_router.callback_query(F.data == 'alter', TaskCreation.deadline_retrival)
async def altering_task_deadline(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    last_message_id = data.get("last_message_id")
    if last_message_id:
        await bot.delete_message(chat_id=call.from_user.id, message_id=last_message_id)  # Удаление последнего сообщения
    await asyncio.sleep(0.5)
    task_deadline_question = await call.message.answer("Пожалуйста, напиши дату в формате: *12:45-17.05*", parse_mode="Markdown")
    await state.update_data(last_message_id=task_deadline_question.message_id)
    await state.set_state(TaskCreation.get_deadline)

@user_router.callback_query(F.data == 'continue', TaskCreation.deadline_retrival)
async def extending_deadline(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    last_message_id = data.get("last_message_id")
    if last_message_id:
        await bot.delete_message(chat_id=call.from_user.id, message_id=last_message_id)  # Удаление последнего сообщения
    await asyncio.sleep(0.5)
    extend_deadline_message = await call.message.answer(
                    "Продлить ли дедлайн на день после его прохождения?", reply_markup=  extend_options(), parse_mode="Markdown")
    await state.update_data(last_message_id = extend_deadline_message.message_id)
    await state.set_state(TaskCreation.extend_deadline)

@user_router.callback_query(F.data.count("answer"), TaskCreation.extend_deadline)
async def get_extend_option(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    last_message_id = data.get("last_message_id")
    task_name = data.get("task_name")
    task_description = data.get("task_description")
    task_deadline = data.get("task_deadline")
    if last_message_id:
        await bot.delete_message(chat_id=call.from_user.id, message_id=last_message_id)  # Удаление последнего сообщения
    await asyncio.sleep(0.5)
    if call.data == "answer_yes":
        extend_deadline_option = "Да"
    else:
        extend_deadline_option = "Нет"
    common_task_message = await call.message.answer(
        f"Твоя задача выглядит так:\n\n" + f"*Название: *" + task_name + "\n" + f"*Описание: *" + task_description +
        "\n" + f"*Дедлайн: *" + task_deadline + "\n" + f"*Продление дедлайна: *" + extend_deadline_option, reply_markup=task_creation_end_options(),
        parse_mode="Markdown")
    await state.update_data(extend_option = extend_deadline_option)
    await state.update_data(last_message_id=common_task_message.message_id)
    await state.set_state(TaskCreation.overall_task_retrival)

@user_router.callback_query(F.data == 'rewrite', TaskCreation.overall_task_retrival)
async def task_rewriting(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    last_message_id = data.get("last_message_id")
    if last_message_id:
        await bot.delete_message(chat_id=call.from_user.id, message_id=last_message_id)  # Удаление последнего сообщения
    await asyncio.sleep(0.5)
    create_task_message = await call.message.answer(f"Начнем все сначала)\nПожалуйста, напиши название задачи", parse_mode="Markdown")
    await state.update_data(last_message_id=create_task_message.message_id)
    await state.set_state(TaskCreation.get_title)

@user_router.callback_query(F.data == 'make_task', TaskCreation.overall_task_retrival)
async def task_creation_confirm(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    last_message_id = data.get("last_message_id")
    task_name = data.get("task_name")
    task_description = data.get("task_description")
    task_deadline = data.get("task_deadline")
    task_extend_option = data.get("extend_option")
    if last_message_id:
        await bot.delete_message(chat_id=call.from_user.id, message_id=last_message_id)  # Удаление последнего сообщения
    user_tasks = await load_data(user_path)
    if user_tasks.get(str(call.from_user.id)) is None:
        user_tasks[str(call.from_user.id)] = {
            task_name: [task_description, task_deadline, task_extend_option, "Создана"]
        }
    else:
        user_tasks[str(call.from_user.id)][task_name] = [task_description, task_deadline, task_extend_option, "Создана"]
    await save_data(user_path, user_tasks)
    await asyncio.sleep(0.5)
    await call.message.answer(f"*Задача создана*!🎉", parse_mode="Markdown")
    await asyncio.sleep(0.5)
    task_question = await call.message.answer(f"Выбери свою новую задачу: ",
                                              reply_markup=get_user_option(), parse_mode="Markdown")
    await state.update_data(last_message_id=task_question.message_id)
    await state.set_state(MainStates.problem_types)

# Блок поиска задачи

@user_router.callback_query(F.data == 'find_task', MainStates.problem_types)
async def task_search(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    last_message_id = data.get("last_message_id")
    if last_message_id:
        await bot.delete_message(chat_id = call.from_user.id, message_id=last_message_id)  # Удаление последнего сообщения
    await asyncio.sleep(0.5)
    query_message = await call.message.answer(f"Напиши название нужной тебе задачи", reply_markup = back_to_main_option(), parse_mode="Markdown")
    await state.update_data(last_message_id = query_message.message_id)
    await state.set_state(TaskSearch.get_query)

@user_router.callback_query(F.data == 'back_to_main', TaskSearch.get_query)
async def back_to_main(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    last_message_id = data.get("last_message_id")
    await asyncio.sleep(0.5)
    if last_message_id:
        await bot.delete_message(chat_id=call.from_user.id, message_id=last_message_id)  # Удаление последнего сообщения
    await asyncio.sleep(0.5)
    task_question = await call.message.answer(f"Выбери свою новую задачу: ",
                                              reply_markup=get_user_option(), parse_mode="Markdown")
    await state.update_data(last_message_id=task_question.message_id)
    await state.set_state(MainStates.problem_types)

@user_router.message(F.text, TaskSearch.get_query)
async def probable_task(message: Message, state: FSMContext):
    async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
        await asyncio.sleep(0.5)
        wait_message  = await message.answer(f"_Ведется поиск..._", parse_mode="Markdown")
        search_result = await search(message.text, str(message.from_user.id))
        await asyncio.sleep(0.5)
        if search_result != "Вопрос не найден":
            user_tasks = await load_data(user_path)
            task_description_confirmation = await wait_message.edit_text(
                text=f"По твоему запросу я нашел эту задачу: " + "\n\n"
                     + f"*Название: *" + search_result + "\n" + f"*Дедлайн: *" + user_tasks[str(message.from_user.id)][search_result][1], reply_markup=task_search_options(), parse_mode="Markdown")
            await state.update_data(founded_task=search_result)
            await state.update_data(last_message_id=task_description_confirmation.message_id)
            await state.update_data(message_edit = task_description_confirmation)
            await state.update_data(user_tasks=user_tasks)
            await state.set_state(TaskSearch.query_retrival)
        else:
            await wait_message.edit_text(
                text=f"Я не смог найти твою задачу 🙁" + "\n\n", parse_mode="Markdown")
            task_question = await message.answer(f"Выбери свою новую задачу: ",
                                                 reply_markup=get_user_option(), parse_mode="Markdown")
            await state.update_data(last_message_id=task_question.message_id)
            await state.set_state(MainStates.problem_types)

@user_router.callback_query(F.data == 'watch_task', TaskSearch.query_retrival)
async def looking_at_task(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    last_message_id = data.get("last_message_id")
    founded_task = data.get("founded_task")
    user_tasks = await load_data(user_path)
    if last_message_id:
        await bot.delete_message(chat_id = call.from_user.id, message_id=last_message_id)  # Удаление последнего сообщения
    await asyncio.sleep(0.5)
    task_storage = user_tasks[str(call.from_user.id)][founded_task]
    common_task_message = await call.message.answer(
        f"Твоя задача выглядит так:\n\n" + f"*Название: *" + founded_task + "\n" + f"*Описание: *" + task_storage[0] +
        "\n" + f"*Дедлайн: *" + task_storage[1] + "\n" + f"*Продление дедлайна: *" + task_storage[2] +  "\n" + f"*Статус: *" + task_storage[3],
        reply_markup=founded_task_options(),
        parse_mode="Markdown")
    await state.update_data(message_edit = common_task_message)
    await state.set_state(TaskSearch.query_retrival)

@user_router.callback_query(F.data == 'find_another_task', TaskSearch.query_retrival)
async def looking_at_task(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    user_tasks = data.get("user_tasks")
    founded_task = data.get("founded_task")
    message_to_edit = data.get("message_edit")
    await asyncio.sleep(0.5)
    await message_to_edit.edit_text(
        text=f"По твоему запросу я нашел эту задачу: " + "\n\n"
             +f"*Название: *" +  founded_task + "\n" + f"*Дедлайн: *" + user_tasks[str(call.from_user.id)][founded_task][1] , reply_markup=None, parse_mode="Markdown")
    await asyncio.sleep(0.5)
    query_message = await call.message.answer(f"Напиши название нужной тебе задачи", reply_markup = back_to_main_option(), parse_mode="Markdown")
    await state.update_data(last_message_id=query_message.message_id)
    await state.set_state(TaskSearch.get_query)

@user_router.callback_query(F.data == 'write_all', TaskSearch.query_retrival)
async def get_list_of_tasks(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    user_tasks = data.get("user_tasks")
    founded_task = data.get("founded_task")
    message_to_edit = data.get("message_edit")
    await asyncio.sleep(0.5)
    await message_to_edit.edit_text(
        text=f"По твоему запросу я нашел эту задачу: " + "\n\n"
             +f"*Название: *" +  founded_task + "\n" + f"*Дедлайн: *" + user_tasks[str(call.from_user.id)][founded_task][1] , reply_markup=None, parse_mode="Markdown")
    await asyncio.sleep(0.5)
    tasks = list(user_tasks[str(call.from_user.id)].keys())
    list_of_task = ""
    for i, task in enumerate(tasks):
        list_of_task += f"*{i + 1})* " + task + "\n" + "*Дедлайн: *" + user_tasks[str(call.from_user.id)][task][1] + "\n"
    await call.message.answer(f"Твой список задач 📋: \n\n" + list_of_task, parse_mode="Markdown")
    await asyncio.sleep(0.5)
    query_message = await call.message.answer(f"Напиши название нужной тебе задачи", reply_markup = back_to_main_option(), parse_mode="Markdown")
    await state.update_data(last_message_id=query_message.message_id)
    await state.set_state(TaskSearch.get_query)

@user_router.callback_query(F.data == 'task_completed', TaskSearch.query_retrival)
async def completing_task(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    founded_task = data.get("founded_task")
    user_tasks = data.get("user_tasks")
    message_to_edit = data.get("message_edit")
    user_tasks[str(call.from_user.id)][founded_task][3] = "Выполнена"
    await message_to_edit.edit_text(text = f"Твоя задача выглядит так:\n\n" + f"*Название: *" + founded_task + "\n"
                                                    + f"*Описание: *" + user_tasks[str(call.from_user.id)][founded_task][0] + "\n" + f"*Дедлайн: *" +
                                                    user_tasks[str(call.from_user.id)][founded_task][1] + "\n" + f"*Продление дедлайна: *"
                                                    + user_tasks[str(call.from_user.id)][founded_task][2] +  "\n"
                                                    + f"*Статус: *" + user_tasks[str(call.from_user.id)][founded_task][3], reply_markup=None, parse_mode="Markdown")
    await save_data(user_path, user_tasks)
    await asyncio.sleep(0.5)
    await call.message.answer(f"Статус задачи *изменен*", parse_mode="Markdown")
    await asyncio.sleep(0.5)
    task_question = await call.message.answer(f"Выбери свою новую задачу: ",
                                              reply_markup=get_user_option(), parse_mode="Markdown")
    await state.update_data(last_message_id=task_question.message_id)
    await state.set_state(MainStates.problem_types)

@user_router.callback_query(F.data == 'change_task', TaskSearch.query_retrival)
async def task_altering_process(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    founded_task = data.get("founded_task")
    user_tasks = data.get("user_tasks")
    message_to_edit = data.get("message_edit")
    await asyncio.sleep(0.5)
    await message_to_edit.edit_text(
        text=f"Твоя задача выглядит так:\n\n" + f"*Название: *" + founded_task + "\n"
             + f"*Описание: *" + user_tasks[str(call.from_user.id)][founded_task][0] + "\n" + f"*Дедлайн: *" +
             user_tasks[str(call.from_user.id)][founded_task][1] + "\n" + f"*Продление дедлайна: *"
             + user_tasks[str(call.from_user.id)][founded_task][2] + "\n"
             + f"*Статус: *" + user_tasks[str(call.from_user.id)][founded_task][3], reply_markup=change_task_options(),
        parse_mode="Markdown")
    await state.set_state(TaskSearch.task_changing)


@user_router.callback_query(F.data == 'delete_task', TaskSearch.query_retrival)
async def task_altering_process(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    founded_task = data.get("founded_task")
    user_tasks = data.get("user_tasks")
    message_to_edit = data.get("message_edit")
    await asyncio.sleep(0.5)
    await message_to_edit.edit_text(
        text=f"Твоя задача выглядит так:\n\n" + f"*Название: *" + founded_task + "\n"
             + f"*Описание: *" + user_tasks[str(call.from_user.id)][founded_task][0] + "\n" + f"*Дедлайн: *" +
             user_tasks[str(call.from_user.id)][founded_task][1] + "\n" + f"*Продление дедлайна: *"
             + user_tasks[str(call.from_user.id)][founded_task][2] + "\n"
             + f"*Статус: *" + user_tasks[str(call.from_user.id)][founded_task][3], reply_markup=None,
        parse_mode="Markdown")
    del user_tasks[str(call.from_user.id)][founded_task]
    await save_data(user_path, user_tasks)
    await asyncio.sleep(0.5)
    await call.message.answer(f"Задача *удалена*", parse_mode="Markdown")
    await asyncio.sleep(0.5)
    task_question = await call.message.answer(f"Выбери свою новую задачу: ",
                                              reply_markup=get_user_option(), parse_mode="Markdown")
    await state.update_data(last_message_id=task_question.message_id)
    await state.set_state(MainStates.problem_types)


@user_router.callback_query(F.data.count("name") + F.data.count("description") + + F.data.count("deadline") != 0 , TaskSearch.task_changing)
async def changing_task_field(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    founded_task = data.get("founded_task")
    user_tasks = data.get("user_tasks")
    message_to_edit = data.get("message_edit")
    await message_to_edit.edit_text(
        text=f"Твоя задача выглядит так:\n\n" + f"*Название: *" + founded_task + "\n"
             + f"*Описание: *" + user_tasks[str(call.from_user.id)][founded_task][0] + "\n" + f"*Дедлайн: *" +
             user_tasks[str(call.from_user.id)][founded_task][1] + "\n" + f"*Продление дедлайна: *"
             + user_tasks[str(call.from_user.id)][founded_task][2] + "\n"
             + f"*Статус: *" + user_tasks[str(call.from_user.id)][founded_task][3], reply_markup=None,
        parse_mode="Markdown")
    await asyncio.sleep(0.5)
    new_field_question = await call.message.answer("Напиши новый вариант", parse_mode="Markdown")
    await state.update_data(last_message_id = new_field_question.message_id)
    await state.update_data(alt_field = call.data)
    await state.set_state(TaskSearch.field_changing)

@user_router.callback_query(F.data.count("extension"), TaskSearch.task_changing)
async def changing_task_field(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    founded_task = data.get("founded_task")
    user_tasks = data.get("user_tasks")
    message_to_edit = data.get("message_edit")
    await message_to_edit.edit_text(
        text=f"Твоя задача выглядит так:\n\n" + f"*Название: *" + founded_task + "\n"
             + f"*Описание: *" + user_tasks[str(call.from_user.id)][founded_task][0] + "\n" + f"*Дедлайн: *" +
             user_tasks[str(call.from_user.id)][founded_task][1] + "\n" + f"*Продление дедлайна: *"
             + user_tasks[str(call.from_user.id)][founded_task][2] + "\n"
             + f"*Статус: *" + user_tasks[str(call.from_user.id)][founded_task][3], reply_markup=None,
        parse_mode="Markdown")
    await asyncio.sleep(0.5)
    extend_deadline_message = await call.message.answer(
        "Продлить ли дедлайн на день после его прохождения?", reply_markup=extend_options(), parse_mode="Markdown")
    await state.update_data(last_message_id=extend_deadline_message.message_id)
    await state.update_data(alt_field = call.data)
    await state.set_state(TaskSearch.extension_retrival)


@user_router.message(F.text, TaskSearch.field_changing)
async def new_field(message: Message, state: FSMContext):
    async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
        data = await state.get_data()
        founded_task = data.get("founded_task")
        user_tasks = data.get("user_tasks")
        alt_field = data.get("alt_field")
        incorrect_type = False
        if alt_field.count("name"):
            user_tasks[str(message.from_user.id)][message.text] = user_tasks[str(message.from_user.id)][founded_task]
            del user_tasks[str(message.from_user.id)][founded_task]
            await state.update_data(founded_task = message.text)
        elif alt_field.count("description"):
            user_tasks[str(message.from_user.id)][founded_task][0] = message.text
        elif alt_field.count("deadline"):
            if await check_deadline_format(message.text):
                user_tasks[str(message.from_user.id)][founded_task][1] = message.text
            else:
                correction_message = await message.answer("Пожалуйста, напиши дату в формате: *12:45-17.05*",
                                                              parse_mode="Markdown")
                incorrect_type = True

        if not incorrect_type:
            data = await state.get_data()
            await save_data(user_path, user_tasks)
            # Обновленное поле
            founded_task = data.get("founded_task")
            overall_message = await message.answer(f"Твоя задача выглядит так:\n\n" + f"*Название: *" + founded_task + "\n"
                                                   + f"*Описание: *" + user_tasks[str(message.from_user.id)][founded_task][
                                                       0] + "\n" + f"*Дедлайн: *" +
                                                   user_tasks[str(message.from_user.id)][founded_task][
                                                       1] + "\n" + f"*Продление дедлайна: *"
                                                   + user_tasks[str(message.from_user.id)][founded_task][2] + "\n"
                                                   + f"*Статус: *" + user_tasks[str(message.from_user.id)][founded_task][3],
                                                   reply_markup=back_to_change_options(),
                                                   parse_mode="Markdown")
            await state.update_data(message_edit=overall_message)
            await state.update_data(user_tasks=user_tasks)
            await state.set_state(TaskSearch.alter_field_retrival)

@user_router.callback_query(F.data == "change_another_field", TaskSearch.alter_field_retrival)
async def changing_another_one_field(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    user_tasks = data.get("user_tasks")
    founded_task = data.get("founded_task")
    message_to_edit = data.get("message_edit")
    await asyncio.sleep(0.5)
    await message_to_edit.edit_text(text = f"Твоя задача выглядит так:\n\n" + f"*Название: *" + founded_task + "\n"
                 + f"*Описание: *" + user_tasks[str(call.from_user.id)][founded_task][0] + "\n" + f"*Дедлайн: *" +
                 user_tasks[str(call.from_user.id)][founded_task][1] + "\n" + f"*Продление дедлайна: *"
                 + user_tasks[str(call.from_user.id)][founded_task][2] + "\n"
                 + f"*Статус: *" + user_tasks[str(call.from_user.id)][founded_task][3],
                                             reply_markup=change_task_options(),
        parse_mode="Markdown")
    await state.set_state(TaskSearch.task_changing)

@user_router.callback_query(F.data == "back_to_main_menu", TaskSearch.alter_field_retrival)
async def changing_another_one_field(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    message_to_edit = data.get("message_edit")
    user_tasks = data.get("user_tasks")
    founded_task = data.get("founded_task")
    await asyncio.sleep(0.5)
    await message_to_edit.edit_text(text = f"Твоя задача выглядит так:\n\n" + f"*Название: *" + founded_task + "\n"
                 + f"*Описание: *" + user_tasks[str(call.from_user.id)][founded_task][0] + "\n" + f"*Дедлайн: *" +
                 user_tasks[str(call.from_user.id)][founded_task][1] + "\n" + f"*Продление дедлайна: *"
                 + user_tasks[str(call.from_user.id)][founded_task][2] + "\n"
                 + f"*Статус: *" + user_tasks[str(call.from_user.id)][founded_task][3], reply_markup=None,
        parse_mode="Markdown")
    await asyncio.sleep(0.5)
    task_question = await call.message.answer(f"Выбери свою новую задачу: ",
                                              reply_markup=get_user_option(), parse_mode="Markdown")
    await state.update_data(last_message_id=task_question.message_id)
    await state.set_state(MainStates.problem_types)

@user_router.callback_query(F.data.count("answer"), TaskSearch.extension_retrival)
async def extension_field_changing(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    last_message_id = data.get("last_message_id")
    user_tasks = data.get("user_tasks")
    founded_task = data.get("founded_task")
    message_to_edit = data.get("message_edit")
    if last_message_id:
        await bot.delete_message(chat_id=call.from_user.id, message_id=last_message_id)  # Удаление последнего сообщения
    await asyncio.sleep(0.5)
    if call.data == "answer_yes":
        extend_deadline_option = "Да"
    else:
        extend_deadline_option = "Нет"
    user_tasks[str(call.from_user.id)][founded_task][2] = extend_deadline_option
    await save_data(user_path, user_tasks)
    await message_to_edit.edit_text(text = f"Твоя задача выглядит так:\n\n" + f"*Название: *" + founded_task + "\n"
                                           + f"*Описание: *" + user_tasks[str(call.from_user.id)][founded_task][
                                               0] + "\n" + f"*Дедлайн: *" +
                                           user_tasks[str(call.from_user.id)][founded_task][
                                               1] + "\n" + f"*Продление дедлайна: *"
                                           + user_tasks[str(call.from_user.id)][founded_task][2] + "\n"
                                           + f"*Статус: *" + user_tasks[str(call.from_user.id)][founded_task][3],
                                           reply_markup=back_to_change_options(),
                                           parse_mode="Markdown")
    await state.update_data(user_tasks=user_tasks)
    await state.set_state(TaskSearch.alter_field_retrival)


# Блок вывода списка всех задач

@user_router.callback_query(F.data == 'get_task_list', MainStates.problem_types)
async def get_all_list(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    last_message_id = data.get("last_message_id")
    if last_message_id:
        await bot.delete_message(chat_id = call.from_user.id, message_id=last_message_id)  # Удаление последнего сообщения
    await asyncio.sleep(0.5)
    user_storage = await load_data(user_path)
    user_tasks = list(user_storage[str(call.from_user.id)].keys())
    list_of_task = ""
    for i, task in enumerate(user_tasks):
        list_of_task += f"*{i+1})* " + task + "\n" + "*Дедлайн: *" + user_storage[str(call.from_user.id)][task][1] + "\n"
    await call.message.answer(f"Твой список задач 📋: \n\n" + list_of_task, parse_mode="Markdown")
    await asyncio.sleep(0.5)
    task_question = await call.message.answer(f"Выбери свою новую задачу: ",
                                              reply_markup=get_user_option(), parse_mode="Markdown")
    await state.update_data(last_message_id=task_question.message_id)
    await state.set_state(MainStates.problem_types)
# Функции

# Проверка на валидность формата введенного дедлайна

async def check_deadline_format(input_str):
    pattern = r"^(\d{2}):(\d{2})-(\d{2})\.(\d{2})$"
    match = re.match(pattern, input_str)

    if not match:
        return False

    hours, minutes, day, month = map(int, match.groups())
    if not (0 <= hours <= 23 and 0 <= minutes <= 59):
        return False
    if not (1 <= day <= 31 and 1 <= month <= 12):
        return False
    now = datetime.datetime.now()
    current_year = now.year
    try:
        deadline = datetime.datetime(current_year, month, day, hours, minutes)
    except ValueError:
        return False

    return deadline > now

#Загрузка данных
async def load_data(data_file):
    try:
        with open(data_file, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}  # Если файл не существует, возвращаем пустой список

#Сохранение данных
async def save_data(data_file, data):
    with open(data_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# Запуск процесса поллинга новых апдейтов
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())