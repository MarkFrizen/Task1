import json
import csv
import time
import re
import tiktoken
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# -------------------- НАСТРОЙКИ --------------------
BASE_URL = "http://192.168.0.140:1234/v1"   # Ваш локальный сервер
API_KEY = "not-needed"
MODEL_NAME = "qwen/qwen3.5-9b"
TIMEOUT = 600

# Создаём клиент один раз
client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=TIMEOUT)

# -------------------- ПРОМПТЫ --------------------
"""Возвращает системный промпт в зависимости от техники и задачи.
technique: "zero", "few", "cot"
task: "classify" (тональность) или "extract" (сущности)"""
def get_prompt(technique: str, task: str) -> str:
    base_instruction = {
        "classify": (
            "Ты классификатор тональности. Верни ТОЛЬКО JSON в формате:\n"
            '{"sentiment": "positive|negative|neutral", "confidence": число от 0 до 1}'
        ),
        "extract": (
            "Ты извлекатель именованных сущностей. Верни ТОЛЬКО JSON в формате:\n"
            '{"persons": [список имён], "locations": [список мест], "organizations": [список организаций]}'
        )
    }[task]
    if technique == "zero":
        return base_instruction
    elif technique == "few":
        # Примеры для Few-shot
        examples = {
            "classify": (
                "Примеры:\n"
                'Текст: "Акции компании выросли на 5% после отчёта." → '
                '{"sentiment": "positive", "confidence": 0.95}\n'
                'Текст: "Ураган разрушил десятки домов, жертв нет." → '
                '{"sentiment": "negative", "confidence": 0.9}\n'
                'Текст: "Сегодня облачно, температура +15°C." → '
                '{"sentiment": "neutral", "confidence": 0.8}\n\n'
                "Теперь классифицируй следующий текст, верни ТОЛЬКО JSON."
            ),
            "extract": (
                "Примеры:\n"
                'Текст: "Илон Маск посетил Берлин и встретился с канцлером." → '
                '{"persons": ["Илон Маск"], "locations": ["Берлин"], "organizations": []}\n'
                'Текст: "Microsoft и Google объявили о партнёрстве в Лондоне." → '
                '{"persons": [], "locations": ["Лондон"], "organizations": ["Microsoft", "Google"]}\n\n'
                "Теперь извлеки сущности из следующего текста, верни ТОЛЬКО JSON."
            )
        }[task]
        return examples + "\n\n" + base_instruction
    elif technique == "cot":
        # Chain-of-Thought: просим сначала объяснить рассуждения, затем дать JSON
        cot_instruction = (
            "Реши задачу пошагово. Сначала кратко опиши свои рассуждения, "
            "затем в конце дай финальный ответ в виде JSON в формате:\n"
            "{\"sentiment\": \"positive|negative|neutral\", \"confidence\": число от 0 до 1}"
        )
        return cot_instruction

    else:
        raise ValueError(f"Неизвестная техника: {technique}")

# -------------------- ПОДСЧЁТ ТОКЕНОВ --------------------
def count_tokens(text: str, model: str = MODEL_NAME) -> int:
    """Подсчитывает количество токенов в тексте с помощью tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")  # fallback
    return len(encoding.encode(text))

# -------------------- ВЫЗОВ МОДЕЛИ С ПОВТОРАМИ --------------------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),  # повторяем при любой ошибке
    reraise=True
)
def call_model(text: str, system_prompt: str, temperature: float = 0.0, top_p: float = 1.0) -> dict:
    """
    Отправляет запрос к модели и возвращает словарь с ответом и метаданными.
    При ошибках повторяет до 3 раз с задержкой.
    """
    # Проверяем длину - предупреждаем, если слишком много
    total_tokens = count_tokens(system_prompt + text)
    if total_tokens > 3500:
        print(f"[WARNING] Внимание: {total_tokens} токенов, может превысить лимит контекста.")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]
    print(f"\nОтправка запроса: {text[:60]}... (токенов: {total_tokens})")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=2048,
        timeout=TIMEOUT
    )

    # Извлекаем сообщение
    message = response.choices[0].message
    raw_content = message.content or ""
    # Проверяем reasoning_content только если content пустой
    if not raw_content and hasattr(message, 'reasoning_content') and message.reasoning_content:
        raw_content = message.reasoning_content
    print(f"Получен ответ (длина {len(raw_content)} символов).")

    # Пытаемся получить использованные токены из ответа
    usage = response.usage if hasattr(response, 'usage') else None
    if usage:
        total_tokens_used = usage.total_tokens
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
    else:
        # Если нет, считаем приблизительно
        total_tokens_used = count_tokens(system_prompt + text) + count_tokens(raw_content)
        prompt_tokens = count_tokens(system_prompt + text)
        completion_tokens = count_tokens(raw_content)
    return {
        "content": raw_content,
        "total_tokens": total_tokens_used,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": response.choices[0].finish_reason
    }

# -------------------- ПАРСИНГ JSON --------------------
def parse_json_response(text: str) -> dict | None:
    """Извлекает JSON из текста ответа модели."""
    if not text:
        return None
    
    # Ищем все возможные JSON-объекты в тексте
    import json
    
    # Находим все позиции открывающих скобок
    start_positions = []
    depth = 0
    for i, char in enumerate(text):
        if char == '{':
            if depth == 0:
                start_positions.append(i)
            depth += 1
        elif char == '}':
            depth -= 1
    
    # Проверяем каждый возможный JSON
    for start in start_positions:
        # Находим соответствующую закрывающую скобку
        depth = 0
        end = -1
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end != -1:
            json_str = text[start:end+1]
            try:
                parsed = json.loads(json_str)
                # Проверяем, содержит ли JSON нужные ключи
                if 'sentiment' in parsed and 'confidence' in parsed:
                    return parsed
                # Если это cot формат, проверяем результат внутри
                if 'reasoning' in parsed and 'result' in parsed:
                    result = parsed['result']
                    if isinstance(result, dict) and 'sentiment' in result and 'confidence' in result:
                        return result
            except json.JSONDecodeError:
                continue
    return None

# -------------------- ПАРСИНГ MARKDOWN ТАБЛИЦЫ --------------------
"""Парсит Markdown-таблицу из ответа и возвращает список словарей.
    Ожидается, что таблица имеет заголовок и разделитель."""
def parse_markdown_table(text: str) -> list[dict] | None:
    lines = text.strip().split('\n')
    # Ищем строки, содержащие '|'
    table_lines = [line for line in lines if '|' in line]
    if len(table_lines) < 3:
        return None
    # Проверяем, что вторая строка - разделитель
    if not re.search(r'\|[\s\-:]+\|', table_lines[1]):
        return None
    headers = [h.strip() for h in table_lines[0].split('|') if h.strip()]
    rows = []
    for row_line in table_lines[2:]:
        cells = [c.strip() for c in row_line.split('|') if c.strip()]
        if len(cells) == len(headers):
            rows.append(dict(zip(headers, cells)))
    return rows if rows else None

# -------------------- ОБРАБОТКА ФАЙЛА --------------------
"""
    Обрабатывает CSV файл и сохраняет результаты.
    Если output_format == "markdown", модель попросит вернуть Markdown-таблицу,
    и мы её распарсим.
    """
def process_file(
        input_file: str,
        output_file: str,
        task: str,
        technique: str = "zero",
        temperature: float = 0.0,
        top_p: float = 1.0,
        output_format: str = "json"   # "json" или "markdown"
):
    system_prompt = get_prompt(technique, task)

    # Если нужен Markdown, модифицируем промпт
    if output_format == "markdown":
        system_prompt += (
            "\nВерни ответ в виде Markdown-таблицы с колонками: "
            "текст, тональность, уверенность (или сущности)."
        )
    results = []
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)  # пропускаем заголовок
        for idx, row in enumerate(reader, 1):
            if not row or not row[0]:
                continue
            text = row[0]
            print(f"\n--- Обработка текста {idx} ---")
            try:
                response_data = call_model(
                    text,
                    system_prompt,
                    temperature=temperature,
                    top_p=top_p
                )
                raw = response_data["content"]

                # Парсим в зависимости от формата
                if output_format == "json":
                    parsed = parse_json_response(raw)
                else:  # markdown
                    parsed = parse_markdown_table(raw)

                prediction = parsed if parsed else None

                # Логируем
                print(f"Предсказание: {prediction}")
                print(f"Токенов использовано: {response_data['total_tokens']}")
                results.append({
                    "id": idx,
                    "text": text,
                    "prediction": prediction,
                    "raw_response": raw,
                    "tokens_used": response_data["total_tokens"],
                    "finish_reason": response_data["finish_reason"]
                })
            except Exception as e:
                print(f"[ERROR] Ошибка при обработке текста {idx}: {e}")
                results.append({
                    "id": idx,
                    "text": text,
                    "error": str(e),
                    "prediction": None
                })

            # Небольшая пауза, чтобы не перегружать сервер
            time.sleep(0.5)

    # Сохраняем результат
    if output_file:
        output = {
            "task": task,
            "technique": technique,
            "temperature": temperature,
            "top_p": top_p,
            "output_format": output_format,
            "total": len(results),
            "results": results
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] Результаты сохранены в {output_file}")
    
    # Возвращаем результаты для сбора в один файл
    return results

# -------------------- ТОЧКА ВХОДА --------------------
if __name__ == "__main__":
    print("=== ЗАПУСК СКРИПТА ===")

    # Параметры для эксперимента
    TASK = "classify"          # или "extract"
    TECHNIQUES = ["zero", "few", "cot"]
    TEMPERATURES = [0.0, 0.5]  # разные температуры для сравнения
    TOP_P = 0.9
    OUTPUT_FILE = "output.json"  # единый файл для всех результатов
    
    # Очищаем output.json перед началом экспериментов
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump({}, f, ensure_ascii=False)
    print(f"[INFO] Файл {OUTPUT_FILE} очищен")

    # Собираем все результаты в один список
    all_results = {
        "task": TASK,
        "total_experiments": len(TECHNIQUES) * len(TEMPERATURES),
        "experiments": []
    }

    # Прогоняем несколько экспериментов
    for technique in TECHNIQUES:
        for temp in TEMPERATURES:
            print(f"\n[EXPERIMENT] Эксперимент: technique={technique}, temperature={temp}")
            experiment_results = process_file(
                input_file="input.csv",
                output_file="",  # не сохраняем промежуточные файлы
                task=TASK,
                technique=technique,
                temperature=temp,
                top_p=TOP_P,
                output_format="json"
            )
            all_results["experiments"].append({
                "technique": technique,
                "temperature": temp,
                "results": experiment_results
            })

    # Сохраняем все результаты в один файл
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Все результаты сохранены в {OUTPUT_FILE}")
    print("\n=== ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ ===")