import os
if __name__ == "__main__":
    BASE_IMAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    print(BASE_IMAGE_DIR)