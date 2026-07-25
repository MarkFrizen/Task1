import json
import csv
from openai import OpenAI

# --- НАСТРОЙКИ ---
# Время ожидания ответа от сервера (60 минут)
TIMEOUT_MINUTES = 60

# Подключаемся к локальному серверу LM Studio
# base_url - адрес локального сервера, где запущена модель
# api_key - фиктивный ключ (LM Studio не требует реальной аутентификации)
# timeout - время ожидания ответа от модели
client = OpenAI(
    base_url="http://192.168.8.11:1234/v1",
    api_key="lm-studio",
    timeout=TIMEOUT_MINUTES * 60
)

# Имя модели, которая загружена в LM Studio
MODEL_NAME = "qwen/qwen3.6-27b"

def get_prompt(task):
    """
    Генерирует системный промпт (инструкцию) для модели.
    
    Zero-shot подход: модель выполняет задачу только на основе инструкции,
    без примеров.
    
    Args:
        task (str): Тип задачи - "classify" или "extract"
    
    Returns:
        str: Инструкция для модели
    """
    if task == "classify":
        # Задача: классификация тональности текста
        # Модель должна вернуть JSON с тональностью
        # и уровнем уверенности (число от 0 до 1)
        return (
            "Ты классификатор тональности. Верни ТОЛЬКО JSON.\n"
            '{"sentiment": "positive|negative|neutral", "confidence": число от 0 до 1}'
        )
    elif task == "extract":
        # Задача: извлечение именованных сущностей
        # Модель должна вернуть JSON с людьми, местами и организациями
        return (
            "Ты извлечатель сущностей. Верни ТОЛЬКО JSON.\n"
            '{"persons": [], "locations": [], "organizations": []}'
        )

def call_model(text, prompt):
    """
    Отправляет текст в модель через API и возвращает ответ.
    
    Args:
        text (str): Текст для обработки
        prompt (str): Инструкция для модели
    
    Returns:
        str | None: Ответ модели (в виде строки) или None в случае ошибки
    
    Process:
        1. Отправляет запрос с системной инструкцией и пользовательским текстом
        2. Извлекает ответ из первого варианта ответа (choices[0])
        3. Возвращает содержимое ответа или None при ошибке
    """
    try:
        # Отправляем запрос к модели
        # messages - список сообщений: system (инструкция) и user (текст)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
            temperature=0,      # Детерминированный вывод (без креативности)
            max_tokens=512      # Максимум 512 токенов в ответе
        )
        
        # Извлекаем ответ: response -> choices[0] -> message -> content
        return response.choices[0].message.content

    except Exception as e:
        # В случае любой ошибки (сетевой, таймаут и т.д.) возвращаем None
        return None

def parse_response(text):
    """
    Извлекает JSON из текста ответа модели.
    
    Модель может добавлять пояснения перед/после JSON, например:
    "Вот результат:\n{"sentiment": "positive"}\nНадеюсь, это поможет"
    
    Args:
        text (str): Ответ от модели в виде строки
    
    Returns:
        dict | None: Распарсенный JSON или None, если не удалось
    """
    # Если ответ пустой, сразу возвращаем None
    if not text:
        return None
    
    # Ищем начало JSON (символ '{')
    start = text.find('{')
    # Ищем конец JSON (последний символ '}')
    end = text.rfind('}')
    
    # Если нашли и начало, и конец
    if start != -1 and end != -1:
        try:
            # Извлекаем подстроку от start до end+1 (включая {})
            json_str = text[start:end+1]
            # Парсим JSON строку в Python словарь
            return json.loads(json_str)
        except:
            # Если JSON некорректный, возвращаем None
            return None
    
    # Если не нашли JSON, возвращаем None
    return None

def process_file(input_file, output_file, task):
    """
    Обрабатывает CSV файл с текстами и сохраняет результаты в JSON.
    
    Args:
        input_file (str): Путь к входному CSV файлу
        output_file (str): Путь к выходному JSON файлу
        task (str): Тип задачи ("classify" или "extract")
    
    Process:
        1. Получаем промпт для задачи
        2. Читаем CSV файл построчно
        3. Для каждой строки вызываем модель
        4. Парсим ответ и сохраняем в results
        5. Сохраняем финальный JSON с метаданными
    """
    # Получаем промпт для текущей задачи
    prompt = get_prompt(task)
    results = []  # Список результатов для всех строк
    
    # Открываем входной CSV файл
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        
        # Пропускаем первую строку (заголовок CSV)
        next(reader)
        
        # Обрабатываем каждую строку
        for i, row in enumerate(reader):
            # Пропускаем пустые строки
            if not row or not row[0]:
                continue
            text = row[0]  # Берём первый столбец как текст
            
            # Вызываем модель и получаем сырый ответ
            raw = call_model(text, prompt)
            
            # Парсим JSON из ответа
            prediction = parse_response(raw)
            
            # Добавляем запись в результаты
            results.append({
                "id": i + 1,           # Порядковый номер (начинается с 1)
                "text": text,          # Исходный текст
                "prediction": prediction  # Результат от модели
            })
    
    # Формируем финальную структуру данных
    output = {
        "task": task,           # Тип задачи
        "total": len(results),  # Общее количество обработанных текстов
        "results": results      # Массив результатов
    }
    
    # Сохраняем в JSON файл
    # ensure_ascii=False - сохраняем кириллицу как есть
    # indent=2 - делаем красивое форматирование (отступы)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

# Точка входа: скрипт запускается напрямую
if __name__ == "__main__":
    # Обрабатываем input.csv и сохраняем в output.json
    # Задача: классификация тональности
    process_file("input.csv", "output.json", "classify")