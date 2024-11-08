import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="3D DOSY reconstruction")

    parser.add_argument('--num_samples', type=int, default=2000)

    parser.add_argument('--batch_size', type=int, default=8)
    
    parser.add_argument('--Num_dim1', type=int, default=128, help='first dimension of the data')
    parser.add_argument('--Num_dim2', type=int, default=128, help='second dimension of the data')
    parser.add_argument('--base_D', type=float, default=0)
    parser.add_argument('--max_D', type=float, default=14)
    parser.add_argument('--num_D', type=int, default=3)
    parser.add_argument('--min_sep', type=float, default=1)
    parser.add_argument('--label_size', type=int, default=70)
    parser.add_argument('--signal_dim', type=int, default=6)
    parser.add_argument('--max_b', type=float, default=0.3)
    parser.add_argument('--max_J', type=int, default=70)  # todo

    parser.add_argument('--loze', type=bool, default=True)  # todo

    parser.add_argument('--sig', type=float, default=0.1)
    parser.add_argument('--snr', type=int, default=30)
    # # 数据集路径
    parser.add_argument('--data_path', type=str,default='dataset')
    parser.add_argument('--result_path', type=str,default='result')
    parser.add_argument('--num_datasets', type=int, default=15)
    parser.add_argument('--num_epochs', type=int, default=40)

    # parser.add_argument('--shuffle', action='store_true', help='是否打乱数据，默认为False')
    # parser.add_argument('--pin_memory', action='store_true', help='是否使用内存锁定，默认为False')

    args,_ = parser.parse_known_args(args=[])
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
