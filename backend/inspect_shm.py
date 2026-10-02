import pefile
from pathlib import Path
import sys

dll_path = Path(r"C:\Users\D_26\Documents\GitHub\Glukometr\backend\.venv\Lib\site-packages\torch\lib\shm.dll")

pe = pefile.PE(str(dll_path))
print("=== Импорты shm.dll ===\n")

for entry in pe.DIRECTORY_ENTRY_IMPORT:
    dll_name = entry.dll.decode()
    print(f"{dll_name}:")
    for imp in entry.imports[:30]:
        fn = imp.name.decode() if imp.name else f"ordinal {imp.ordinal}"
        print(f"    {fn}")
    print()