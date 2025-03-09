import re
import datetime


async def check_deadline_format(input_str):
    pattern = r"^(\d{2}):(\d{2})-(\d{2})\.(\d{2})$"
    match = re.match(pattern, input_str)

    if not match:
        return False

    hours, minutes, day, month = map(int, match.groups())

    # Проверяем корректность времени и даты
    if not (0 <= hours <= 23 and 0 <= minutes <= 59):
        return False
    if not (1 <= day <= 31 and 1 <= month <= 12):
        return False

    # Получаем текущую дату и время
    now = datetime.datetime.now()

    # Создаем объект даты дедлайна в текущем году
    current_year = now.year
    try:
        deadline = datetime.datetime(current_year, month, day, hours, minutes)
    except ValueError:
        return False  # Некорректная дата (например, 30 февраля)

    # Проверяем, что дедлайн позже текущего времени
    return deadline > now


# Тесты
import asyncio

test_cases = [
    "12:45-17.01",  # Будущий срок
    "23:59-31.12",  # Граничное значение
    "00:00-01.01",  # Начало года
    "24:00-15.07",  # Неверный час
    "12:60-10.08",  # Неверные минуты
    "12:30-32.01",  # Неверный день
    "09:15-15.13",  # Неверный месяц
    "12:45-01.01",  # Прошедший срок
]


async def run_tests():
    for test in test_cases:
        result = await check_deadline_format(test)
        print(f"{test}: {'Correct' if result else 'Incorrect'}")


asyncio.run(run_tests())
