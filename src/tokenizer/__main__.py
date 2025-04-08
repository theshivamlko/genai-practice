

import  argparse

from src.tokenizer.encoder_decoder import Encoder

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument( 'encoder',type=str,default=None,nargs='?')

    args=parser.parse_args()

    encoder=Encoder()



    # switch statement
    if args.encoder=='encoder':
        encoder.encode("Hello World")
    elif args.encoder=='decoder':
        encoder.decode("Hello World")
    else:
        print('Enter valid command')




if __name__ == "__main__":
    main()
