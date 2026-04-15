import traceback

try:
    import app.main
    print("Successfully imported app.main")
except Exception as e:
    with open("error_log.txt", "w") as f:
        f.write(traceback.format_exc())
    print("Error written to error_log.txt")
