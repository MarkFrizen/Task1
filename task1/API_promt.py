import json
import time
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests

# Проверка наличия необходимых библиотек при импорте
# Проверяет что openai установлен иначе выводит ошибку и завершает работу
try:
    from openai import OpenAI, APIConnectionError
except ImportError as e:
    print("ОШИБКА: Не найдена необходимая библиотека. Установите: pip install openai requests")
    print("Детали: " + str(e))
    exit(1)

# Настройки подключения к серверу и пути к входному-выходному файлам
# Определяет адрес LM Studio ключ модели таймаут и имена файлов
BASE_URL = "http://localhost:1234/v1"
API_KEY = "lm-studio"
MODEL_NAME = "qwen/qwen3.5-9b"
TIMEOUT_MINUTES = 10
TIMEOUT = TIMEOUT_MINUTES * 60
INPUT_FILE = "input.csv"
OUTPUT_FILE = "output.json"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 0.9

# Клиент OpenAI для работы с локальным сервером LM Studio
# Инициализирует клиент для отправки запросов к языковой модели
client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=TIMEOUT)

# Оценка числа токенов в тексте примерно по формуле 3 символа на токен
# Возвращает приблизительное количество токенов в переданном тексте
def count_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 3)

# Проверка доступности сервера и наличия нужной модели
# Запражает список моделей на сервере и проверяет что нужная модель доступна
def check_server_ready() -> bool:
    try:
        resp = requests.get(f"{BASE_URL}/models", timeout=5)
        resp.raise_for_status()
        models_data = resp.json()
        models = models_data.get("data", []) if isinstance(models_data, dict) else models_data
        model_found = any(m.get("id") == MODEL_NAME for m in models)
        if not model_found:
            print("ERROR: Модель " + MODEL_NAME + " не найдена.")
            print("Доступные модели: " + str([m.get("id") for m in models]))
            return False
        print("OK: Сервер LM Studio доступен и модель найдена.")
        return True
    except Exception as e:
        print("ERROR: Сервер не отвечает. Проверьте, запущен ли LM Studio. Детали: " + str(e))
        return False

# Отправка текста в LLM с техниками zero few shot и CoT возвращает сырой ответ
# Формирует промпт в зависимости от выбранной техники и вызывает модель
def call_model(
        text: str,
        technique: str = "zero",
        temperature: float = 0.0,
        top_p: float = 0.9
) -> str:
    system_prompt = (
        "Ты строгий ассистент. Твой ответ ДОЛЖЕН быть валидным JSON объектом с полями category и entities. "
        "Не пиши никакого текста вне JSON. Не используй markdown блоки. "
        "Формат: {\"category\": \"КЛАСС\", \"entities\": [\"сущность1\", \"сущность2\"]}"
    )
    user_prompt = ""

    # Few-shot: добавляем примеры в промпт
    if technique == "few":
        examples = [
            {"text": "Компания Apple выпустила новый iPhone 15 Pro Max в черном цвете.", "category": "ТЕХНОЛОГИИ", "entities": ["Apple", "iPhone 15 Pro Max"]},
            {"text": "Курс доллара вырос до 92 рублей на фоне новостей из ЦБ.", "category": "ФИНАНСЫ", "entities": ["доллар", "ЦБ"]}
        ]
        examples_str = "\n".join([f"Текст: {e['text']}\nОтвет: {json.dumps(e, ensure_ascii=False)}" for e in examples])
        user_prompt = (
                "Классифицируй текст и извлеки сущности. Верни ТОЛЬКО JSON. "
                "Примеры:\n" + examples_str + "\n\n"
                                              "Текст для обработки:\n" + text
        )

    # Chain of thought: просим модель описать рассуждения перед ответом
    elif technique == "cot":
        user_prompt = (
                "Классифицируй текст и извлеки сущности. Сначала напиши краткий план рассуждений step_by_step, затем дай финальный ответ в формате JSON. "
                "Текст для обработки:\n" + text
        )

    # Zero-shot: без примеров
    else:
        user_prompt = (
                "Классифицируй текст и извлеки сущности. Верни ТОЛЬКО JSON. "
                "Формат: {\"category\": \"КЛАСС\", \"entities\": [\"сущность1\", \"сущность2\"]} "
                "Текст для обработки:\n" + text
        )

    # Запрос к LLM с параметрами temperature и top_p
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=1024
    )
    choice = response.choices[0]
    result = choice.message.content if choice.message else None
    return result if result else ""

# Очистка ответа от markdown парсинг JSON безопасная обработка ошибок
# Удаляет обёртку markdown и пытается распарсить текст как JSON
def parse_json_response(raw_text: str) -> Optional[Dict[str, Any]]:
    if not raw_text or not raw_text.strip():
        return {"error": "empty_response", "raw_output": raw_text}
    clean_text = raw_text.strip()
    # Удаляем markdown-обёртку ```json ... ``` при наличии
    if clean_text.startswith("```"):
        lines = clean_text.splitlines()
        start_idx = next((i for i, line in enumerate(lines) if "{" in line), None)
        rev_end_idx = next((i for i, line in enumerate(reversed(lines)) if "}" in line), None)
        if rev_end_idx is not None:
            end_idx = len(lines) - 1 - rev_end_idx
        else:
            end_idx = None
        if start_idx is not None and end_idx is not None and start_idx <= end_idx:
            clean_text = "\n".join(lines[start_idx:end_idx+1])
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        return {"error": "invalid_json", "raw_output": clean_text[:200]}

# Чтение CSV-файла и возврат списка очищенных строк как словарей
# Использует DictReader для получения словарей с очищенными значениями
def load_csv_data(filepath: str) -> List[Dict[str, str]]:
    data = []
    try:
        with open(filepath, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                clean_row = {k: (v.strip() if v else "") for k, v in row.items()}
                data.append(clean_row)
    except FileNotFoundError:
        print("ERROR: Файл " + filepath + " не найден. Создайте файл с данными.")
    return data

# Сохранение данных в CSV-файл
# Записывает список словарей в CSV с использованием DictWriter
def save_csv_data(filepath: str, data: List[Dict[str, str]]) -> None:
    if not data:
        with open(filepath, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = list(data[0].keys())
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

# Добавление новой записи
# Запрашивает текст у пользователя и добавляет запись в список
def add_record(data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    print("\n--- Добавление новой записи ---")
    text = input("Введите текст: ").strip()
    if not text:
        print("ОШИБКА: Текст не может быть пустым.")
        return data
    record = {"text": text}
    data.append(record)
    print("Запись добавлена.")
    return data

# Редактирование записи по индексу
# Показывает список записей запрашивает номер и заменяет текст
def edit_record(data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    print("\n--- Редактирование записи ---")
    if not data:
        print("НЕТ записей для редактирования.")
        return data
    for i, row in enumerate(data, 1):
        text_val = row.get("text", "")
        preview = text_val[:60] + ("..." if len(text_val) > 60 else "")
        print(f"  {i}. {preview}")
    try:
        idx = int(input("Номер записи для редактирования: ")) - 1
        if 0 <= idx < len(data):
            old_text = data[idx].get("text", "")
            print(f"Текущий текст: {old_text}")
            new_text = input("Новый текст: ").strip()
            if not new_text:
                print("ОШИБКА: Текст не может быть пустым.")
                return data
            data[idx]["text"] = new_text
            print("Запись изменена.")
        else:
            print("ОШИБКА: Неверный номер записи.")
    except ValueError:
        print("ОШИБКА: Введите корректный номер.")
    return data

# Удаление записи по индексу
# Показывает список записей запрашивает номер и удаляет запись
def delete_record(data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    print("\n--- Удаление записи ---")
    if not data:
        print("НЕТ записей для удаления.")
        return data
    for i, row in enumerate(data, 1):
        text_val = row.get("text", "")
        preview = text_val[:60] + ("..." if len(text_val) > 60 else "")
        print(f"  {i}. {preview}")
    try:
        idx = int(input("Номер записи для удаления: ")) - 1
        if 0 <= idx < len(data):
            removed = data.pop(idx)
            print(f"Удалено: {removed.get('text', '')[:60]}")
        else:
            print("ОШИБКА: Неверный номер записи.")
    except ValueError:
        print("ОШИБКА: Введите корректный номер.")
    return data

# Управление записями в input.csv
# Предоставляет меню для просмотра добавления редактирования и удаления записей
def manage_csv(filepath: str) -> List[Dict[str, str]]:
    data = load_csv_data(filepath)
    if not data:
        print("В файле пока нет записей.")
    else:
        print(f"Загружено записей: {len(data)}")
    while True:
        print("\n=== Управление записями ===")
        print("1. Показать все записи")
        print("2. Добавить запись")
        print("3. Редактировать запись")
        print("4. Удалить запись")
        print("5. Сохранить и выйти")
        choice = input("Выберите действие (1-5): ").strip()
        if choice == "1":
            for i, row in enumerate(data, 1):
                text_val = row.get("text", "")
                preview = text_val[:80] + ("..." if len(text_val) > 80 else "")
                print(f"  {i}. {preview}")
            if not data:
                print("  (нет записей)")
        elif choice == "2":
            data = add_record(data)
            save_csv_data(filepath, data)
        elif choice == "3":
            data = edit_record(data)
            save_csv_data(filepath, data)
        elif choice == "4":
            data = delete_record(data)
            save_csv_data(filepath, data)
        elif choice == "5":
            save_csv_data(filepath, data)
            print("Изменения сохранены.")
            break
        else:
            print("ОШИБКА: Неверный выбор.")
    return load_csv_data(filepath)

# Главный цикл проверка сервера загрузка данных обработка сохранение результатов
# Координирует весь процесс от проверки подключения до сохранения итогов
def main():
    print("=== ЗАПУСК СКРИПТА ===")
    if not check_server_ready():
        print("Скрипт остановлен из-за проблем с подключением к серверу.")
        return

    # Подготовка пустого файла результатов
    output_path = Path(OUTPUT_FILE)
    output_path.write_text("[]", encoding="utf-8")
    print("Файл " + OUTPUT_FILE + " подготовлен.")
    data = load_csv_data(INPUT_FILE)
    if not data:
        print("Нет данных для обработки. Проверьте файл input.csv.")
        return
    print("Загружено записей: " + str(len(data)))

    # Интерактивное управление записями
    print("\nПерейдите к управлению записями в " + INPUT_FILE + "?")
    manage = input("Добавить/изменить/удалить записи? (y/n): ").strip().lower()
    if manage == "y":
        data = manage_csv(INPUT_FILE)
        if not data:
            print("Нет данных для обработки. Проверьте файл " + INPUT_FILE + ".")
            return
        print("Загружено записей для обработки: " + str(len(data)))
    
    # Параметры эксперимента
    technique = "few"
    temperature = 0.0
    top_p = 0.9
    print("Запуск эксперимента: technique=" + technique + ", temp=" + str(temperature) + ", top_p=" + str(top_p))
    results = []

    # Цикл обработки каждой строки CSV
    for idx, row in enumerate(data, start=1):
        text_content = row.get("text") or row.get("content") or row.get(list(row.keys())[0])
        if not text_content:
            continue
        print("--- Обработка текста " + str(idx) + " из " + str(len(data)) + " ---")
        tokens_count = count_tokens(text_content)
        print("Оценка токенов: " + str(tokens_count))
        try:
            start_time = time.time()
            raw_response = call_model(text_content, technique, temperature, top_p)
            elapsed = time.time() - start_time
            parsed_data = parse_json_response(raw_response)
            result_entry = {
                "index": idx,
                "input_text": text_content,
                "raw_output": raw_response,
                "structured_output": parsed_data,
                "tokens_input": tokens_count,
                "processing_time_sec": round(elapsed, 2),
                "technique": technique,
                "temperature": temperature,
                "top_p": top_p
            }
            results.append(result_entry)
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            status = "OK" if (isinstance(parsed_data, dict) and "error" not in parsed_data) else "PARSE_ERROR"
            if status == "OK":
                print(status + ": Обработано за " + str(round(elapsed, 2)) + " сек.")
            else:
                print("Обработано за " + str(round(elapsed, 2)) + " сек.")
        except APIConnectionError as e:
            print("ERROR: Connection на строке " + str(idx) + ": " + str(e))
            results.append({"index": idx, "input_text": text_content, "error": "connection_error", "details": str(e)})
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print("ERROR: Критическая ошибка на строке " + str(idx) + ": " + str(e))
            results.append({"index": idx, "input_text": text_content, "error": str(e)})
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
    print("=== ЗАВЕРШЕНО ===")
    print("Всего успешно обработано: " + str(len(results)))
    print("Результаты сохранены в " + OUTPUT_FILE)
if __name__ == "__main__":
    main()