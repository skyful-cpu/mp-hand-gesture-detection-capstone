import src.Image as Image

def main_function():
    cmd = input('''press 's' to start: ''')

    if cmd == 's':
        filename = input('type filename: ')
        image = Image.Image()
        image.get_landmarks()

    if cmd == 'q':
        exit()

if __name__ == '__main__':
    while True:
        main_function()