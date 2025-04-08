import argparse

from src.tokenizer.encoder_decoder import Encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-encoder', type=str, default=None, nargs='?')
    parser.add_argument('-decoder', type=str, default=None, nargs='?')

    args = parser.parse_args()

    encoder = Encoder()


    # switch statement
    if args.encoder:
        encoder.encode(args.encoder)
    elif args.decoder:
        encoder.decode(args.decoder)
    else:
        print('Enter valid command')


if __name__ == "__main__":
    main()
