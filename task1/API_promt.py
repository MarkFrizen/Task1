import json
import time
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests

# Зависимости: openai, tenacity, requests
try:
    from openai import OpenAI
    from tenacity import retry, stop_after_attempt, wait_exponential
except ImportError as e:
    print("ОШИБКА: Не найдена необходимая библиотека. Установите зависимости: pip install openai tenacity requests")
    print("Детали ошибки: " + str(e))
    exit(1)

# Настройки подключения к LLM-серверу и пути к файлам
BASE_URL = "http://192.168.0.140:1234/v1"
API_KEY = "lm-studio"
MODEL_NAME = "qwen/qwen3.5-9b"
TIMEOUT = 90
INPUT_FILE = "input.csv"
OUTPUT_FILE = "output.json"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 0.9

# Клиент OpenAI для взаимодействия с локальным сервером (LM Studio)
client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=TIMEOUT)

# Оценивает число токенов в тексте (приблизительно) и проверяет готовность LLM-сервера
def count_tokens(text: str) -> int:
    """
    Быстрая оценка количества токенов.
    Формула: примерно 3 символа = 1 токен.
    """
    if not text:
        return 0
    return max(1, len(text) // 3)

def check_server_ready() -> bool:
    """
    Проверяет, что LM Studio запущен и искомая модель загружена.
    """
    try:
        resp = requests.get(f"{BASE_URL}/models", timeout=5)
        resp.raise_for_status()
        models_data = resp.json()

        # Получаем список моделей из ответа
        models = models_data.get("data", []) if isinstance(models_data, dict) else models_data
        model_found = any(m.get("id") == MODEL_NAME for m in models)
        if not model_found:
            print("ERROR: Модель " + MODEL_NAME + " не найдена.")
            print("Доступные модели: " + str([m.get("id") for m in models]))
            return False
        print("OK: Сервер LM Studio доступен и модель найдена.")
        return True
    except Exception as e:
        print("ERROR: Сервер не отвечает или недоступен. Проверьте, запущен ли LM Studio. Детали: " + str(e))
        return False

# -------------------- ЯДРО ИНЖЕНЕРИИ ПРОМПТОВ --------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def call_model(
        text: str,
        technique: str = "zero",
        temperature: float = 0.0,
        top_p: float = 0.9
) -> str:
    """
    Отправляет текст в LLM с техникой zero-/few-shot или chain-of-thought, возвращает сырой ответ.
    """

    # Системный промпт: требует строгого формата JSON
    system_prompt = (
        "Ты строгий ассистент. Твой ответ ДОЛЖЕН быть валидным JSON объектом. "
        "Не пиши никакого текста вне JSON. Не используй markdown блоки. "
        "Если не можешь ответить, верни JSON с полем error: не удалось извлечь данные."
    )
    user_prompt = ""

    if technique == "few":
        examples = [
            {"text": "Компания Apple выпустила новый iPhone 15 Pro Max в черном цвете.", "category": "ТЕХНОЛОГИИ", "entities": ["Apple", "iPhone 15 Pro Max"]},
            {"text": "Курс доллара вырос до 92 рублей на фоне новостей из ЦБ.", "category": "ФИНАНСЫ", "entities": ["доллар", "ЦБ"]}
        ]
        examples_str = "\n".join([f"Текст: {e['text']}\nОтвет: {e['category']}, {e['entities']}" for e in examples])
        user_prompt = (
                "Классифицируй текст и извлеки сущности. Используй формат JSON. "
                "Примеры:\n" + examples_str + "\n\n"
                                              "Текст для обработки:\n" + text
        )
    elif technique == "cot":
        user_prompt = (
                "Классифицируй текст и извлеки сущности. Сначала напиши краткий план рассуждений step_by_step, затем дай финальный ответ в формате JSON. "
                "Текст для обработки:\n" + text
        )
    else: # zero-shot
        user_prompt = (
                "Классифицируй текст и извлеки сущности. Верни ТОЛЬКО JSON. "
                "Текст для обработки:\n" + text
        )

    # Отправка запроса к LLM
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=512
    )
    return response.choices[0].message.content

def parse_json_response(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    Очищает ответ от markdown-разметки и парсит JSON, возвращает dict или ошибку.
    """
    clean_text = raw_text.strip()

    # Очистка от markdown блоков, если модель их добавила
    if clean_text.startswith("```"):
        lines = clean_text.splitlines()

        # Ищем первую строку, содержащую открывающую фигурную скобку
        start_idx = next((i for i, line in enumerate(lines) if "{" in line), None)

        # Ищем последнюю строку, содержащую закрывающую фигурную скобку.
        # ВАЖНО: Здесь исправлена ошибка. Мы используем reversed, но потом конвертируем индекс.
        # Переменная line корректно определена в этом генераторе.
        rev_end_idx = next((i for i, line in enumerate(reversed(lines)) if "}" in line), None)
        if rev_end_idx is not None:
            end_idx = len(lines) - 1 - rev_end_idx
        else:
            end_idx = None
        if start_idx is not None and end_idx is not None and start_idx <= end_idx:
            # ВАЖНО: Используем \n (один слэш). Двойной слэш сломает JSON.
            clean_text = "\n".join(lines[start_idx:end_idx+1])
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        # Безопасный возврат ошибки вместо исключения
        return {"error": "invalid_json", "raw_output": clean_text[:200]}

# Загружает строки из CSV-файла в список словарей
def load_csv_data(filepath: str) -> List[Dict[str, str]]:
    """
    Загрузка данных из CSV файла.
    """
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

# -------------------- ОСНОВНОЙ ЦИКЛ ВЫПОЛНЕНИЯ --------------------

def main():
    """
    Запускает проверку сервера, загружает CSV, обрабатывает каждый текст через LLM и сохраняет результаты в JSON.
    """
    print("=== ЗАПУСК СКРИПТА ===")
    if not check_server_ready():
        print("Скрипт остановлен из-за проблем с подключением к серверу.")
        return

    # Инициализация пустого JSON файла результатов
    output_path = Path(OUTPUT_FILE)
    output_path.write_text("[]", encoding="utf-8")
    print("Файл " + OUTPUT_FILE + " подготовлен.")
    data = load_csv_data(INPUT_FILE)
    if not data:
        print("Нет данных для обработки. Проверьте файл input.csv.")
        return
    print("Загружено записей: " + str(len(data)))

    # Параметры эксперимента
    technique = "few"
    temperature = 0.0
    top_p = 0.9
    print("Запуск эксперимента: technique=" + technique + ", temp=" + str(temperature) + ", top_p=" + str(top_p))
    results = []
    for idx, row in enumerate(data, start=1):
        # Определение поля с текстом
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

            # Построчная запись в файл (защита от потери данных)
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            status = "OK" if (isinstance(parsed_data, dict) and "error" not in parsed_data) else "PARSE_ERROR"
            print(status + ": Обработано за " + str(elapsed) + " сек.")
        except Exception as e:
            print("ERROR: Критическая ошибка на строке " + str(idx) + ": " + str(e))
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
    print("=== ЗАВЕРШЕНО ===")
    print("Всего успешно обработано: " + str(len(results)))
    print("Результаты сохранены в " + OUTPUT_FILE)
if __name__ == "__main__":
    main()