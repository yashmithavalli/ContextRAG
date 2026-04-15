import traceback
import sys

try:
    import app.main
    print("Success")
except Exception as e:
    with open("my_error.txt", "w", encoding="utf-8") as f:
        f.write(traceback.format_exc())
    print("Logged to my_error.txt")
