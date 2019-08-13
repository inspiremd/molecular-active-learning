import argparse




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_in', type=str, required=True)
    parser.add_argument('--prev_model', type=str, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--scratch_dir', type=str, required=True)
    return parser.parse_args()

def main(args):
    print("learning stuff.")

if __name__ == '__main__':
    args = get_args()
    main(args)