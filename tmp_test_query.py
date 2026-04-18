from pathlib import Path
from rag_pipeline import parse_records, structured_query_engine

text = Path('data/students.txt').read_text(encoding='utf-8')
records = parse_records(text)

queries = [
    'marks of tanya mehta',
    'marks of priya mehta',
    'student id 250',
    'tanya mehta',
]
for q in queries:
    print('Q:', q)
    print(structured_query_engine(records, q))
    print('---')
