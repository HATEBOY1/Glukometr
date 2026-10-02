import pefile
from pathlib import Path

for dll_name in ("VCRUNTIME140.dll", "VCRUNTIME140_1.dll"):
    dll = Path(r"C:\Windows\System32") / dll_name
    print(f"\n=== {dll} ===")
    if not dll.exists():
        print("  ФАЙЛ НЕ НАЙДЕН")
        continue

    pe = pefile.PE(str(dll))
    found = False
    if hasattr(pe, "DIRECTORY_ENTRY_EXPORT"):
        for exp in pe.DIRECTORY_ENTRY_EXPORT.symbols:
            if exp.name and b"__CxxFrameHandler4" in exp.name:
                found = True
                print(f"  НАЙДЕНО: {exp.name.decode()}")
    print(f"  __CxxFrameHandler4: {'ЕСТЬ' if found else 'ОТСУТСТВУЕТ'}")