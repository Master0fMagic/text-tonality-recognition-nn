import net


def main():
    model = net.load()

    while True:
        text = input('Input text to predict tonality or ctrl+C to exit\n')
        predicted = net.parse_tonality(net.predict(model, text))
        print(f'Predicted tonality is {predicted}')


if __name__ == '__main__':
    main()
