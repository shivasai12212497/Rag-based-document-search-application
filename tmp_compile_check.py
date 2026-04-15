import py_compile
try:
    py_compile.compile(r'd:/project/rag_pipeline.py', doraise=True)
    print('ok')
except Exception as e:
    import traceback
    traceback.print_exc()
