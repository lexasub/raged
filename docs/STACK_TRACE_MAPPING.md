# Smart Stack Trace Mapping

Сервис для автоматического анализа стектрейсов с маппингом на AST-ноды, извлечением кода и рекомендациями по исправлению.

## Возможности

- **Парсинг стектрейсов** для 4 языков: Python, C++, Java, Rust
- **Маппинг на AST** — каждый фрейм сопоставляется с узлом в графе кода
- **Извлечение кода** — получение snippet'ов кода для каждого фрейма
- **Анализ root cause** — определение категории ошибки,severity, причин
- **Рекомендации по фиксу** — конкретные шаги для исправления
- **Поиск похожих проблем** — семантический поиск аналогичных ошибок в кодебазе

## Поддерживаемые форматы стектрейсов

### Python
```
Traceback (most recent call last):
  File "main.py", line 42, in <module>
    result = process_data(data)
  File "processor.py", line 15, in process_data
    return transform(item)
ValueError: Invalid input
```

### C++ (GDB/LLDB style)
```
terminate called after throwing an instance of 'std::out_of_range'
Stack trace:
#0  0x00007fff5fbff6c0 in std::vector<int>::at(unsigned long) at vector.h:1134
#1  0x00007fff5fbff700 in processData(std::vector<int>&) at processor.cpp:25
#2  0x00007fff5fbff740 in main at main.cpp:15
```

### Java
```
java.lang.NullPointerException: Cannot invoke method on null object
    at com.example.UserService.getUser(UserService.java:42)
    at com.example.Main.main(Main.java:15)
Caused by: java.lang.IllegalArgumentException: Invalid argument
    at com.example.UserService.validateId(UserService.java:55)
```

### Rust
```
thread 'main' panicked at 'index out of bounds: len is 3 but index is 5', src/main.rs:42:5
stack backtrace:
   0: rust_begin_unwind
   1: my_crate::process_array
              at src/main.rs:42:5
   2: my_crate::main
              at src/main.rs:10:1
```

## Использование

### CLI

```bash
# Анализ из файла
ast-rag analyze-stacktrace error.log

# Анализ из stdin
echo "$STACKTRACE" | ast-rag analyze-stacktrace

# Вывод в JSON
ast-rag analyze-stacktrace error.log -o json

# Вывод в текстовом формате
ast-rag analyze-stacktrace error.log -o text

# Пропустить AST маппинг для скорости
ast-rag analyze-stacktrace error.log --no-ast-mapping

# Подробный вывод
ast-rag analyze-stacktrace error.log -v
```

### Python API

```python
from ast_rag.stack_trace import StackTraceService
from ast_rag.graph_schema import create_driver
from ast_rag.embeddings import EmbeddingManager
from ast_rag.models import ProjectConfig

# Инициализация
config = ProjectConfig.model_validate_json(open("ast_rag_config.json").read())
driver = create_driver(config.neo4j)
embed = EmbeddingManager(config.qdrant, config.embedding, neo4j_driver=driver)

service = StackTraceService(driver, embed)

# Анализ стектрейса
trace = """
Traceback (most recent call last):
  File "main.py", line 42, in <module>
    result = process_data(data)
ValueError: Invalid input
"""

report = service.analyze(trace)

# Вывод результатов
print(report.to_markdown())  # Markdown для человека
print(report.to_json())      # JSON для машины

# Доступ к деталям
print(f"Error: {report.error_type}")
print(f"Root cause: {report.root_cause.likely_cause}")
print(f"Suggested fix: {report.root_cause.suggested_fix}")
print(f"Mapped frames: {report.mapped_frames}/{report.total_frames}")

# Анализ из файла
report = service.analyze_from_file("error.log")
```

## Архитектура

```
┌─────────────────────────────────────────────────────────┐
│                    Stack Trace Input                     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              StackTraceParserFactory                     │
│  ┌──────────┬──────────┬──────────┬──────────┐          │
│  │  Python  │   C++    │   Java   │   Rust   │          │
│  │  Parser  │  Parser  │  Parser  │  Parser  │          │
│  └──────────┴──────────┴──────────┴──────────┘          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   StackFrames[]                          │
│  - frame_index, function_name, class_name               │
│  - file_path, line_number, language                     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              AST Mapping (StackTraceService)             │
│  1. Find by file_path + line_number                     │
│  2. Find by function/class name                         │
│  3. Semantic search                                     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Code Snippet Retrieval                      │
│  - get_code_snippet(file, start_line, end_line)         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Root Cause Analysis                         │
│  - Categorize error (null_pointer, out_of_bounds, ...)  │
│  - Determine severity (critical, high, medium, low)     │
│  - Generate likely cause explanation                    │
│  - Suggest fixes                                        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Similar Issues Search                       │
│  - Semantic search by error type + message              │
│  - Find related code patterns                           │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  StackTraceReport                        │
│  - error_type, message, language                        │
│  - root_cause: {category, severity, fix, confidence}    │
│  - call_chain: [StackFrame with code snippets]          │
│  - similar_issues: [SimilarIssue]                       │
│  - summary                                              │
└─────────────────────────────────────────────────────────┘
```

## Модель данных

### StackFrame
```json
{
  "frame_index": 0,
  "function_name": "process_data",
  "class_name": "DataProcessor",
  "file_path": "/path/to/processor.py",
  "line_number": 42,
  "language": "python",
  "frame_type": "method_call",
  "code_snippet": "...",
  "ast_node_id": "abc123",
  "ast_node_qualified_name": "DataProcessor.process_data"
}
```

### RootCause
```json
{
  "error_type": "NullPointerException",
  "error_message": "Cannot invoke method on null",
  "likely_cause": "A null reference was accessed...",
  "severity": "high",
  "category": "null_pointer",
  "suggested_fix": "1. Add null checks...\n2. Use Optional...",
  "confidence": 0.85,
  "related_frames": [0, 1]
}
```

### StackTraceReport
```json
{
  "error_type": "NullPointerException",
  "message": "Cannot invoke method on null",
  "language": "java",
  "root_cause": {...},
  "call_chain": [...],
  "similar_issues": [...],
  "summary": "...",
  "total_frames": 5,
  "mapped_frames": 3
}
```

## Категории ошибок

| Категория | Примеры | Severity |
|-----------|---------|----------|
| `null_pointer` | NullPointerException, NoneType | high |
| `out_of_bounds` | IndexError, out_of_range | high |
| `type_error` | TypeError, ClassCastException | medium |
| `value_error` | ValueError, IllegalArgumentException | medium |
| `key_error` | KeyError, NoSuchElement | low |
| `attribute_error` | AttributeError, MissingProperty | low |
| `file_error` | FileNotFoundError, IOException | medium |
| `memory_error` | MemoryError, OutOfMemory | critical |
| `concurrency` | ConcurrentModification, Deadlock | critical |
| `panic` | panic, assertion failed | critical |

## Интеграция с analyze_text

Сервис использует существующий API `analyze_text` для дополнительного контекста:

```python
# В StackTraceService.analyze()
text_results = self._analyze_with_text_api(stacktrace)
if text_results and not report.similar_issues:
    report.similar_issues = self._convert_text_results_to_issues(text_results)
```

Это позволяет находить релевантный код даже когда точный маппинг на AST не удался.

## Тесты

```bash
# Запустить тесты
pytest tests/test_stack_trace.py -v

# Тесты парсеров
pytest tests/test_stack_trace.py::TestPythonParser -v
pytest tests/test_stack_trace.py::TestJavaParser -v
pytest tests/test_stack_trace.py::TestCppParser -v
pytest tests/test_stack_trace.py::TestRustParser -v

# Тесты моделей
pytest tests/test_stack_trace.py::TestStackFrame -v
pytest tests/test_stack_trace.py::TestStackTraceReport -v
```

## Примеры

См. `ast_rag/stack_trace/examples.py` для примеров стектрейсов и использования.

## Расширение

### Добавление нового парсера

```python
from .models import StackFrame, Language
from .parsers import StackTraceParser

class GoParser(StackTraceParser):
    def detect_language(self, stacktrace: str) -> Language:
        # Логика детектирования
        return Language.UNKNOWN  # или новый enum
    
    def extract_error_info(self, stacktrace: str) -> tuple[str, str]:
        # Извлечение типа и сообщения ошибки
        return "Error", ""
    
    def parse(self, stacktrace: str) -> list[StackFrame]:
        # Парсинг фреймов
        return []

# Регистрация в фабрике
StackTraceParserFactory._parsers[Language.GO] = GoParser
```

## Ограничения

- Требуется запущенный Neo4j для AST маппинга
- Требуется запущенный Qdrant для семантического поиска
- Точность маппинга зависит от полноты индексации кодебазы
- Некоторые форматы стектрейсов могут требовать доработки парсеров

## Будущие улучшения

- [ ] Поддержка Go, TypeScript, C#
- [ ] Парсинг JSON/XML логов ошибок
- [ ] Интеграция с GitHub Issues для поиска похожих проблем
- [ ] ML-модель для классификации ошибок
- [ ] Автоматическое создание PR с фиксом
- [ ] Статистика частых ошибок по проекту
