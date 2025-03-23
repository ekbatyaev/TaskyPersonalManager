import re
import json
import logging
import asyncio
import torch
import json
from aiogram import Bot, Dispatcher
from aiogram.filters.command import Command
from aiogram.fsm.context import FSMContext
from aiogram.filters.state import State, StatesGroup
from datetime import datetime
from aiogram import Router, F
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, Message, CallbackQuery
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from aiogram.utils.chat_action import ChatActionSender
from transformers import AutoTokenizer, AutoModel
from aiogram.fsm.storage.memory import MemoryStorage
from torch.nn.functional import cosine_similarity
from tokens_file import telegram_bot_token

# Глобальные переменные кафки

KAFKA_BROKER = "localhost:9092"
TOPIC_CREATE = "task-create"
TOPIC_DELETE = "task-delete"
TOPIC_UPDATE_STATUS = "task-update-status"
TOPIC_UPDATE = "task-update"
TOPIC_GET_ACTIVE = "task-get-active"
TOPIC_GET_HISTORY = "task-get-history"
TOPIC_GET_ACTIVE_RESPONSE = "task-get-active-response"
TOPIC_GET_HISTORY_RESPONSE = "task-get-history-response"
producer = None
task_queue = None
task_results = {}
task_events = {}

# Функции Кафки


async def start_kafka():
    """Запуск Kafka consumer и producer"""

    consumer = AIOKafkaConsumer(
        TOPIC_GET_ACTIVE_RESPONSE,
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        group_id="task-get-active-response",  # Должно совпадать с Spring
        auto_offset_reset="earliest"
    )

    await consumer.start()
    asyncio.create_task(kafka_consumer(consumer))

    global producer
    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )
    producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BROKER)
    await producer.start()



async def stop_kafka():
    """Остановка Kafka producer"""
    if producer:
        await producer.stop()

async def send_to_kafka(data, topic):
    """Отправка сообщения в Kafka"""
    if producer:
        await producer.send_and_wait(topic, json.dumps(data).encode("utf-8"))

async def accept_message(user_id, message):
    print("Hi")
    task_results[user_id] = message.value.get("tasks", [])
    print(task_results)

async def kafka_consumer(consumer):
    """Фоновая задача для обработки сообщений Kafka"""
    async for message in consumer:
        message_data = json.loads(message.value.decode("utf-8"))
        user_id = message_data.get("userId")
        print(user_id, message_data, message)
        if user_id:
            # await accept_message(user_id, message)
            task_results[user_id] = message_data.get("tasks", [])
            if user_id in task_events:
                task_events[user_id].set()  # Разрешаем основному потоку продолжить


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

async def search(user_question, user_id, data):
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
    deadline_retrival = State()
    overall_task_retrival = State()

class TaskSearch(StatesGroup):
    get_query = State()
    query_retrival = State()
    task_changing = State()
    field_changing = State()
    alter_field_retrival = State()
    extension_retrival = State()
    get_type_task_list = State()
    history_list_retrival = State()
    find_history_task = State()

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
        [InlineKeyboardButton(text='Найти актуальную задачу 🔍', callback_data='find_task')],
        [InlineKeyboardButton(text='Список актуальных задач 📋', callback_data='get_actual_task_list')],
        [InlineKeyboardButton(text='Вывести историю задач 📚 ', callback_data='get_history_task_list')],
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
        [InlineKeyboardButton(text='Список актуальных задач 📋', callback_data='write_all_actual')],
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
        [InlineKeyboardButton(text='Вернуться назад 🏠', callback_data='back')]
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

# Выбор типа задач из истории

def get_task_history_option():
    keyboard_list = [

        [InlineKeyboardButton(text='Выполненные ✅', callback_data='completed_tasks')],
        [InlineKeyboardButton(text='Просроченные ⏰', callback_data='overdue_tasks')],
        [InlineKeyboardButton(text='Отложенные️ 🕒', callback_data='deferred_tasks')]
    ]
    keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_list)
    return keyboard

def back_history_options():
    keyboard_list = [
        [InlineKeyboardButton(text='Найти задачу 🔍', callback_data='find_task')],
        [InlineKeyboardButton(text='Выбрать другую категорию️ 🔄', callback_data='another_category')],
        [InlineKeyboardButton(text='Вернуться в главное меню 🏠', callback_data='back_to_main_menu')]
    ]
    keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_list)
    return keyboard

def find_history_options():
    keyboard_list = [
        [InlineKeyboardButton(text='Найти другую задачу 🔍', callback_data='find_another_task')],
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
        await asyncio.sleep(0.5)
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
        await asyncio.sleep(0.5)
        task_description_confirmation = await message.answer(f"*Описание твоей задачи 📝: *" + "\n\n" + message.text, reply_markup = task_options(),
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
        await asyncio.sleep(0.5)
        if await check_deadline_format(message.text):
            task_deadline_message = await message.answer("Дедлайн задачи ⏳: " + message.text,
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
async def deadline_next_section(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    last_message_id = data.get("last_message_id")
    task_name = data.get("task_name")
    task_description = data.get("task_description")
    task_deadline = data.get("task_deadline")
    if last_message_id:
        await bot.delete_message(chat_id=call.from_user.id, message_id=last_message_id)  # Удаление последнего сообщения
    await asyncio.sleep(0.5)
    common_task_message = await call.message.answer(
        f"Твоя задача выглядит так:\n\n" + f"*Название ✨: *" + task_name + "\n" + f"*Описание 📝: *" + task_description +
        "\n" + f"*Дедлайн ⏳: *" + task_deadline,
        reply_markup=task_creation_end_options(),
        parse_mode="Markdown")
    await state.update_data(last_message_id=common_task_message.message_id)
    await state.set_state(TaskCreation.overall_task_retrival)

@user_router.callback_query(F.data == 'rewrite', TaskCreation.overall_task_retrival)
async def task_rewriting(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    last_message_id = data.get("last_message_id")
    if last_message_id:
        await bot.delete_message(chat_id=call.from_user.id, message_id=last_message_id)  # Удаление последнего сообщения
    await asyncio.sleep(0.5)
    create_task_message = await call.message.answer(f"Начнем все сначала)\nПожалуйста, напиши *название задачи*", parse_mode="Markdown")
    await state.update_data(last_message_id=create_task_message.message_id)
    await state.set_state(TaskCreation.get_title)

@user_router.callback_query(F.data == 'make_task', TaskCreation.overall_task_retrival)
async def task_creation_confirm(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    last_message_id = data.get("last_message_id")
    task_name = data.get("task_name")
    task_description = data.get("task_description")
    task_deadline = data.get("task_deadline")
    if last_message_id:
        await bot.delete_message(chat_id=call.from_user.id, message_id=last_message_id)  # Удаление последнего сообщения
    await asyncio.sleep(0.5)
    # Отправка данных в Kafka
    kafka_data = {
        "userId": str(call.from_user.id),
        "task": {
            "title": task_name,
            "description": task_description,
            "deadline": await convert_to_iso_datetime(task_deadline)
        }
    }

    asyncio.create_task(send_to_kafka(kafka_data, TOPIC_CREATE))  # Асинхронно отправляем в Kafka
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
        # Получение данных из кафки

        task_events[str(message.from_user.id)] = asyncio.Event()  # Создаём событие для ожидания ответа
        await send_to_kafka({"userId": str(message.from_user.id)}, TOPIC_GET_ACTIVE)
        try:
            await asyncio.wait_for(task_events[str(message.from_user.id)].wait(), timeout=10)  # Ждём ответ до 10 секунд
        except asyncio.TimeoutError:
            await wait_message.edit_text(text="⏳ Сервер долго не отвечает. Попробуй позже.")
            await asyncio.sleep(0.5)
            task_question = await message.message.answer(f"Выбери свою новую задачу: ",
                                                      reply_markup=get_user_option(), parse_mode="Markdown")
            await state.update_data(last_message_id=task_question.message_id)
            await state.set_state(MainStates.problem_types)
            return

        tasks = task_results.pop(str(message.from_user.id), [])  # Забираем результат и удаляем его
        print(tasks)
        user_tasks = {}
        #Преобразование тасок в старый формат
        for task in tasks:
            user_tasks[task.get("title")] = [task.get("description"), task.get("deadline"), task.get("id")]

        search_result = await search(message.text, str(message.from_user.id), user_tasks)
        await asyncio.sleep(0.5)
        if search_result != "Вопрос не найден":
            task_description_confirmation = await wait_message.edit_text(
                text=f"По твоему запросу я нашел эту задачу: " + "\n\n"
                     + f"*Название ✨: *" + search_result + "\n" + f"*Дедлайн ⏳: *" + user_tasks[search_result][1], reply_markup=task_search_options(), parse_mode="Markdown")
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
    user_tasks = data.get("user_tasks")
    last_message_id = data.get("last_message_id")
    founded_task = data.get("founded_task")
    if last_message_id:
        await bot.delete_message(chat_id = call.from_user.id, message_id=last_message_id)  # Удаление последнего сообщения
    await asyncio.sleep(0.5)
    task_storage = user_tasks[str(call.from_user.id)][founded_task]
    common_task_message = await call.message.answer(
        f"Твоя задача выглядит так:\n\n" + f"*Название ✨: *" + founded_task + "\n" + f"*Описание 📝: *" + task_storage[0] +
        "\n" + f"*Дедлайн ⏳: *" + task_storage[1],
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
             +f"*Название ✨: *" +  founded_task + "\n" + f"*Дедлайн ⏳: *" + user_tasks[founded_task][1] , reply_markup=None, parse_mode="Markdown")
    await asyncio.sleep(0.5)
    query_message = await call.message.answer(f"Напиши название нужной тебе задачи", reply_markup = back_to_main_option(), parse_mode="Markdown")
    await state.update_data(last_message_id=query_message.message_id)
    await state.set_state(TaskSearch.get_query)

@user_router.callback_query(F.data == 'write_all_actual', TaskSearch.query_retrival)
async def get_list_of_tasks(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    user_tasks = data.get("user_tasks")
    founded_task = data.get("founded_task")
    message_to_edit = data.get("message_edit")
    await asyncio.sleep(0.5)
    await message_to_edit.edit_text(
        text=f"По твоему запросу я нашел эту задачу: " + "\n\n"
             +f"*Название ✨: *" +  founded_task + "\n" + f"*Дедлайн ⏳: *" + user_tasks[founded_task][1] , reply_markup=None, parse_mode="Markdown")
    await asyncio.sleep(0.5)
    task_events[str(call.from_user.id)] = asyncio.Event()  # Создаём событие для ожидания ответа
    await send_to_kafka({"userId": str(call.from_user.id)}, TOPIC_GET_ACTIVE)

    database_think_message = await call.message.answer(f"_Получение данных из базы..._", parse_mode="Markdown")
    try:
        await asyncio.wait_for(task_events[str(call.from_user.id)].wait(), timeout=10)  # Ждём ответ до 10 секунд
    except asyncio.TimeoutError:
        await database_think_message.edit_text(text="⏳ Сервер долго не отвечает. Попробуй позже.")
        await asyncio.sleep(0.5)
        task_question = await call.message.answer(f"Выбери свою новую задачу: ",
                                                  reply_markup=get_user_option(), parse_mode="Markdown")
        await state.update_data(last_message_id=task_question.message_id)
        await state.set_state(MainStates.problem_types)
        return

    user_tasks = task_results.pop(str(call.from_user.id), [])  # Забираем результат и удаляем его
    print(user_tasks)
    if not user_tasks:
        await database_think_message.edit_text(text="У тебя пока нет задач 📋")
    else:
        list_of_task = ""
        for i, task in enumerate(user_tasks):
            list_of_task += f"*{i + 1})* " + task.get("title") + "\n" + "*Дедлайн ⏳: *" + task.get("deadline") + "\n"
        await database_think_message.edit_text(text=f"Твой список задач 📋: \n\n" + list_of_task, parse_mode="Markdown")
        await asyncio.sleep(0.5)

    task_question = await call.message.answer(f"Выбери свою новую задачу: ",
                                              reply_markup=get_user_option(), parse_mode="Markdown")
    await state.update_data(last_message_id=task_question.message_id)
    await state.set_state(MainStates.problem_types)

# Меняем статус задачи на выполнена

@user_router.callback_query(F.data == 'task_completed', TaskSearch.query_retrival)
async def completing_task(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    founded_task = data.get("founded_task")
    user_tasks = data.get("user_tasks")
    message_to_edit = data.get("message_edit")
    await message_to_edit.edit_text(text = f"Твоя задача выглядит так:\n\n" + f"*Название ✨: *" + founded_task + "\n"
                                                    + f"*Описание 📝: *" + user_tasks[founded_task][0] + "\n" + f"*Дедлайн ⏳: *" +
                                                    user_tasks[founded_task][1], reply_markup=None, parse_mode="Markdown")
    await asyncio.sleep(0.5)

    # Отправка данных в Kafka
    kafka_data = {
        "userId": str(call.from_user.id),
        "taskld": user_tasks[founded_task][2],
        "status": "completed"
    }


    asyncio.create_task(send_to_kafka(kafka_data, TOPIC_UPDATE_STATUS))  # Асинхронно отправляем в Kafka

    await call.message.answer(f"Статус задачи *изменен*", parse_mode="Markdown")
    await asyncio.sleep(0.5)

    task_question = await call.message.answer(f"Выбери свою новую задачу: ",
                                              reply_markup=get_user_option(), parse_mode="Markdown")
    await state.update_data(last_message_id=task_question.message_id)
    await state.update_data(status_update_flag = True)
    await state.set_state(MainStates.problem_types)

@user_router.callback_query(F.data == 'change_task', TaskSearch.query_retrival)
async def task_altering_process(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    founded_task = data.get("founded_task")
    user_tasks = data.get("user_tasks")
    message_to_edit = data.get("message_edit")
    await asyncio.sleep(0.5)
    await message_to_edit.edit_text(
        text=f"Твоя задача выглядит так:\n\n" + f"*Название ✨: *" + founded_task + "\n"
             + f"*Описание 📝: *" + user_tasks[founded_task][0] + "\n" + f"*Дедлайн ⏳: *" +
             user_tasks[founded_task][1], reply_markup=change_task_options(),
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
        text=f"Твоя задача выглядит так:\n\n" + f"*Название ✨: *" + founded_task + "\n"
             + f"*Описание 📝: *" + user_tasks[founded_task][0] + "\n" + f"*Дедлайн ⏳: *" +
             user_tasks[founded_task][1], reply_markup=None,
        parse_mode="Markdown")
    del user_tasks[str(call.from_user.id)][founded_task]
    await asyncio.sleep(0.5)

    # Отправка данных в Kafka
    kafka_data = {
        "userId": str(call.from_user.id),
        "taskld": user_tasks[founded_task][2],
    }

    asyncio.create_task(send_to_kafka(kafka_data, TOPIC_DELETE))  # Асинхронно отправляем в Kafka

    await call.message.answer(f"Задача *удалена*", parse_mode="Markdown")
    await asyncio.sleep(0.5)
    task_question = await call.message.answer(f"Выбери свою новую задачу: ",
                                              reply_markup=get_user_option(), parse_mode="Markdown")
    await state.update_data(last_message_id=task_question.message_id)
    await state.set_state(MainStates.problem_types)


@user_router.callback_query(F.data.count("change") , TaskSearch.task_changing)
async def changing_task_field(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    founded_task = data.get("founded_task")
    user_tasks = data.get("user_tasks")
    message_to_edit = data.get("message_edit")
    await message_to_edit.edit_text(
        text=f"Твоя задача выглядит так:\n\n" + f"*Название ✨: *" + founded_task + "\n"
             + f"*Описание 📝: *" + user_tasks[founded_task][0] + "\n" + f"*Дедлайн ⏳: *" +
             user_tasks[founded_task][1], reply_markup=None,
        parse_mode="Markdown")
    await asyncio.sleep(0.5)
    new_field_question = await call.message.answer("Напиши новый вариант", parse_mode="Markdown")
    await state.update_data(last_message_id = new_field_question.message_id)
    await state.update_data(alt_field = call.data)
    await state.set_state(TaskSearch.field_changing)


@user_router.message(F.text, TaskSearch.field_changing)
async def new_field(message: Message, state: FSMContext):
    async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
        data = await state.get_data()
        founded_task = data.get("founded_task")
        user_tasks = data.get("user_tasks")
        alt_field = data.get("alt_field")
        incorrect_type = False
        if alt_field.count("name"):
            user_tasks[message.text] = user_tasks[founded_task]
            del user_tasks[founded_task]
            await state.update_data(founded_task = message.text)
        elif alt_field.count("description"):
            user_tasks[founded_task][0] = message.text
        elif alt_field.count("deadline"):
            if await check_deadline_format(message.text):
                user_tasks[founded_task][1] = message.text
            else:
                await message.answer("Пожалуйста, напиши дату в формате: *12:45-17.05*",
                                                              parse_mode="Markdown")
                incorrect_type = True

        if not incorrect_type:
            data = await state.get_data()
            # Обновленное поле
            founded_task = data.get("founded_task")
            if data.get("status_update_flag"):
                status = "completed"
            else:
                status = "active"


            await asyncio.sleep(0.5)
            kafka_data = {
                "userId": str(message.from_user.id),
                "status": status,
                "task": {
                    "id": user_tasks[founded_task][2],
                    "title": founded_task,
                    "description": user_tasks[founded_task][0],
                    "deadline": convert_to_iso_datetime(user_tasks[founded_task][1])
                }
            }
            asyncio.create_task(send_to_kafka(kafka_data, TOPIC_UPDATE))  # Асинхронно отправляем в Kafka
            overall_message = await message.answer(f"Твоя задача выглядит так:\n\n" + f"*Название ✨: *" + founded_task + "\n"
                                                   + f"*Описание 📝: *" + user_tasks[founded_task][0] + "\n" + f"*Дедлайн ⏳: *" +
                                                   user_tasks[founded_task][1],
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
    await message_to_edit.edit_text(text = f"Твоя задача выглядит так:\n\n" + f"*Название ✨: *" + founded_task + "\n"
                 + f"*Описание 📝: *" + user_tasks[founded_task][0] + "\n" + f"*Дедлайн ⏳: *" +
                 user_tasks[founded_task][1], reply_markup=change_task_options(),
        parse_mode="Markdown")
    await state.set_state(TaskSearch.task_changing)

@user_router.callback_query(F.data == "back_to_main_menu", TaskSearch.alter_field_retrival)
async def changing_another_one_field(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    message_to_edit = data.get("message_edit")
    user_tasks = data.get("user_tasks")
    founded_task = data.get("founded_task")
    await asyncio.sleep(0.5)
    await message_to_edit.edit_text(text = f"Твоя задача выглядит так:\n\n" + f"*Название ✨: *" + founded_task + "\n"
                 + f"*Описание 📝: *" + user_tasks[founded_task][0] + "\n" + f"*Дедлайн ⏳: *" +
                 user_tasks[founded_task][1], reply_markup=None,
        parse_mode="Markdown")
    await asyncio.sleep(0.5)
    task_question = await call.message.answer(f"Выбери свою новую задачу: ",
                                              reply_markup=get_user_option(), parse_mode="Markdown")
    await state.update_data(last_message_id=task_question.message_id)
    await state.set_state(MainStates.problem_types)


# Блок вывода списка всех актуальных задач

@user_router.callback_query(F.data == 'get_actual_task_list', MainStates.problem_types)
async def get_all_list(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    user_id = call.from_user.id
    last_message_id = data.get("last_message_id")
    if last_message_id:
        await bot.delete_message(chat_id = call.from_user.id, message_id=last_message_id)  # Удаление последнего сообщения
    await asyncio.sleep(0.5)
    task_events[user_id] = asyncio.Event()  # Создаём событие для ожидания ответа
    await send_to_kafka({"userId": str(call.from_user.id)}, TOPIC_GET_ACTIVE)

    database_think_message = await call.message.answer(f"_Получение данных из базы..._", parse_mode="Markdown")
    try:
        await asyncio.wait_for(task_events[user_id].wait(), timeout=10)  # Ждём ответ до 10 секунд
    except asyncio.TimeoutError:
        await database_think_message.edit_text(text = "⏳ Сервер долго не отвечает. Попробуй позже.")
        await asyncio.sleep(0.5)
        task_question = await call.message.answer(f"Выбери свою новую задачу: ",
                                                  reply_markup=get_user_option(), parse_mode="Markdown")
        await state.update_data(last_message_id=task_question.message_id)
        await state.set_state(MainStates.problem_types)
        return

    user_tasks = task_results.pop(user_id, [])  # Забираем результат и удаляем его
    print(user_tasks)
    if not user_tasks:
        await database_think_message.edit_text(text = "У тебя пока нет задач 📋")
    else:
        list_of_task = ""
        for i, task in enumerate(user_tasks):
            list_of_task += f"*{i+1})* " + task.get("title") + "\n" + "*Дедлайн ⏳: *" + task.get("deadline") + "\n"
        await database_think_message.edit_text(text = f"Твой список задач 📋: \n\n" + list_of_task, parse_mode="Markdown")
        await asyncio.sleep(0.5)

    task_question = await call.message.answer(f"Выбери свою новую задачу: ",
                                              reply_markup=get_user_option(), parse_mode="Markdown")
    await state.update_data(last_message_id=task_question.message_id)
    await state.set_state(MainStates.problem_types)

# Блок историю задач

@user_router.callback_query(F.data == 'get_history_task_list', MainStates.problem_types)
async def get_history_task_option(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    last_message_id = data.get("last_message_id")
    if last_message_id:
        await bot.delete_message(chat_id = call.from_user.id, message_id=last_message_id)  # Удаление последнего сообщения
    await asyncio.sleep(0.5)
    task_question = await call.message.answer(f"Выбери какие задачи тебе нужны: ",
                                              reply_markup=get_user_option(), parse_mode="Markdown")
    await state.update_data(last_message_id=task_question.message_id)
    await state.set_state(MainStates.problem_types)

@user_router.callback_query(F.data == 'get_history_task_list', MainStates.problem_types)
async def get_history_task_option(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    last_message_id = data.get("last_message_id")
    if last_message_id:
        await bot.delete_message(chat_id = call.from_user.id, message_id=last_message_id)  # Удаление последнего сообщения
    await asyncio.sleep(0.5)
    task_question = await call.message.answer(f"Выбери какие задачи тебе нужны: ",
                                              reply_markup=get_user_option(), parse_mode="Markdown")
    await state.update_data(last_message_id=task_question.message_id)
    await state.set_state(TaskSearch.get_type_task_list)

@user_router.callback_query(F.data.count("tasks"), TaskSearch.get_type_task_list)
async def get_history_task_option(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    user_id = call.from_user.id
    last_message_id = data.get("last_message_id")
    if last_message_id:
        await bot.delete_message(chat_id=call.from_user.id, message_id=last_message_id)  # Удаление последнего сообщения
    await asyncio.sleep(0.5)
    status = ""

    if call.data == "completed_tasks":
        status = "completed"
    elif call.data == "overdue_tasks":
        status = "overdue"
    elif call.data == "deferred_tasks":
        status = "backlog"

    task_events[user_id] = asyncio.Event()  # Создаём событие для ожидания ответа
    await send_to_kafka({"userId": str(call.from_user.id), "status": status}, TOPIC_GET_HISTORY)

    database_think_message = await call.message.answer(f"_Получение данных из базы..._", parse_mode="Markdown")
    try:
        await asyncio.wait_for(task_events[user_id].wait(), timeout=10)  # Ждём ответ до 10 секунд
    except asyncio.TimeoutError:
        await database_think_message.edit_text(text="⏳ Сервер долго не отвечает. Попробуй позже.")
        await asyncio.sleep(0.5)
        task_question = await call.message.answer(f"Выбери свою новую задачу: ",
                                                  reply_markup=get_user_option(), parse_mode="Markdown")
        await state.update_data(last_message_id=task_question.message_id)
        await state.set_state(MainStates.problem_types)
        return

    tasks = task_results.pop(user_id, [])  # Забираем результат и удаляем его
    print(tasks)
    user_tasks = {}
    for task in tasks:
        user_tasks[task.get("title")] = [task.get("description"), task.get("deadline"), task.get("id")]
    list_of_task = ""
    if not tasks:
        answer_message = await database_think_message.edit_text(text="У тебя пока нет задач 📋")
    else:
        for i, task in enumerate(tasks):
            list_of_task += f"*{i + 1})* " + task.get("title") + "\n" + "*Дедлайн ⏳: *" + task.get("deadline") + "\n"
        answer_message = await database_think_message.edit_text(text=f"Твой список задач 📋: \n\n" + list_of_task, reply_markup = back_history_options(), parse_mode="Markdown")
        await asyncio.sleep(0.5)
    await state.update_data(task_list = list_of_task)
    await state.update_data(user_tasks = user_tasks)
    await state.update_data(message_to_edit = answer_message)
    await state.set_state(TaskSearch.history_list_retrival)

@user_router.callback_query(F.data == 'another_category', TaskSearch.history_list_retrival)
async def choice_another_category(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    message_to_edit = data.get("message_to_edit")
    list_of_task = data.get("list")
    if list_of_task == "У тебя пока нет задач 📋":
        await message_to_edit.edit_text(text=list_of_task, reply_markup=None, parse_mode = "Markdown")
    else:
        await message_to_edit.edit_text(text = f"Твой список задач 📋: \n\n" + list_of_task, reply_markup = None, parse_mode = "Markdown" )
    await asyncio.sleep(0.5)
    task_question = await call.message.answer(f"Выбери какие задачи тебе нужны: ",
                                              reply_markup=get_user_option(), parse_mode="Markdown")
    await state.update_data(last_message_id=task_question.message_id)
    await state.set_state(TaskSearch.get_type_task_list)

@user_router.callback_query(F.data == 'back_to_main_menu', TaskSearch.history_list_retrival)
async def back_to_main_options(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    message_to_edit = data.get("message_to_edit")
    list_of_task = data.get("list")
    if list_of_task == "У тебя пока нет задач 📋":
        await message_to_edit.edit_text(text=list_of_task, reply_markup=None, parse_mode = "Markdown")
    else:
        await message_to_edit.edit_text(text = f"Твой список задач 📋: \n\n" + list_of_task, reply_markup = None, parse_mode = "Markdown" )
    await asyncio.sleep(0.5)
    task_question = await call.message.answer(f"Выбери свою задачу: ",
                                                 reply_markup=get_user_option(), parse_mode="Markdown")
    await state.update_data(last_message_id=task_question.message_id)
    await state.set_state(MainStates.problem_types)

@user_router.callback_query(F.data == 'find_task', TaskSearch.history_list_retrival)
async def task_search(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    last_message_id = data.get("last_message_id")
    if last_message_id:
        await bot.delete_message(chat_id = call.from_user.id, message_id=last_message_id)  # Удаление последнего сообщения
    await asyncio.sleep(0.5)
    query_message = await call.message.answer(f"Напиши название нужной тебе задачи", reply_markup = back_to_main_option(), parse_mode="Markdown")
    await state.update_data(last_message_id = query_message.message_id)
    await state.set_state(TaskSearch.find_history_task)

@user_router.callback_query(F.data == 'back_to_main', TaskSearch.find_history_task)
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

@user_router.message(F.text, TaskSearch.find_history_task)
async def probable_task(message: Message, state: FSMContext):
    async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
        data = await state.get_data()
        await asyncio.sleep(0.5)
        user_tasks = data.get("user_tasks")
        wait_message  = await message.answer(f"_Ведется поиск..._", parse_mode="Markdown")
        search_result = await search(message.text, str(message.from_user.id), user_tasks)
        await asyncio.sleep(0.5)
        if search_result != "Вопрос не найден":
            task_description_confirmation = await wait_message.edit_text(
                text=f"По твоему запросу я нашел эту задачу: " + "\n\n"
                     + f"*Название ✨: *" + search_result  + "\n"
                     + f"*Описание 📝: *" + user_tasks[search_result][0] + f"*Дедлайн ⏳: *" + user_tasks[search_result][1], reply_markup=find_history_options(), parse_mode="Markdown")
            await state.update_data(founded_task=search_result)
            await state.update_data(message_edit = task_description_confirmation)
            await state.update_data(user_tasks=user_tasks)
            await state.set_state(TaskSearch.history_list_retrival)
        else:
            lose_message = await wait_message.edit_text(
                text=f"Я не смог найти твою задачу 🙁" + "\n\n",reply_markup =  find_history_options(), parse_mode="Markdown")
            await state.update_data(message_edit=lose_message)
            await state.set_state(TaskSearch.find_history_task)


@user_router.callback_query(F.data == 'find_another_task', TaskSearch.find_history_task)
async def looking_at_task(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    user_tasks = data.get("user_tasks")
    founded_task = data.get("founded_task")
    message_to_edit = data.get("message_edit")
    await asyncio.sleep(0.5)
    await message_to_edit.edit_text(
        text=f"По твоему запросу я нашел эту задачу: " + "\n\n"
             +f"*Название ✨: *" +  founded_task + "\n" + f"*Описание 📝: *" + user_tasks[founded_task][0] + f"*Дедлайн ⏳: *" + user_tasks[founded_task][1] , reply_markup=None, parse_mode="Markdown")
    await asyncio.sleep(0.5)
    query_message = await call.message.answer(f"Напиши название нужной тебе задачи", reply_markup = back_to_main_option(), parse_mode="Markdown")
    await state.update_data(last_message_id=query_message.message_id)
    await state.set_state(TaskSearch.find_history_task)

@user_router.callback_query(F.data == 'back_to_main_menu', TaskSearch.find_history_task)
async def back_to_main(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    user_tasks = data.get("user_tasks")
    founded_task = data.get("founded_task")
    message_to_edit = data.get("message_edit")
    await asyncio.sleep(0.5)
    await message_to_edit.edit_text(
        text=f"По твоему запросу я нашел эту задачу: " + "\n\n"
             + f"*Название ✨: *" + founded_task + "\n" + f"*Описание 📝: *" + user_tasks[founded_task][
                 0] + f"*Дедлайн ⏳: *" + user_tasks[founded_task][1], reply_markup=None, parse_mode="Markdown")
    await asyncio.sleep(0.5)
    task_question = await call.message.answer(f"Выбери свою новую задачу: ",
                                              reply_markup=get_user_option(), parse_mode="Markdown")
    await state.update_data(last_message_id=task_question.message_id)
    await state.set_state(MainStates.problem_types)


# Функции

# Функции кафки

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
    now = datetime.now()
    current_year = now.year
    try:
        deadline = datetime(current_year, month, day, hours, minutes)
    except ValueError:
        return False

    return deadline > now


# Конвертация даты в новый формат

async def convert_to_iso_datetime(deadline_str):
    """Конвертирует строку '12:34-16.04' в ISO 8601 формат"""
    try:
        time_part, date_part = deadline_str.split("-")
        parsed_datetime = datetime.strptime(date_part + "." + str(datetime.now().year) + " " + time_part, "%d.%m.%Y %H:%M")
        return parsed_datetime.isoformat()
    except ValueError:
        return None

# Запуск бота и кафки

async def main():
    await start_kafka()
    try:
        print("Kafka producer запущен 🚀")
        await dp.start_polling(bot)  # Запуск бота
    finally:
        await stop_kafka()
        print("Kafka producer остановлен ❌")

if __name__ == "__main__":
    asyncio.run(main())