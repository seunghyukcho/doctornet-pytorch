def add_model_args(parser):
    group = parser.add_argument_group('model')
    group.add_argument('--n_classes', type=int,
            help="Number of classes for classification.")
    group.add_argument('--n_annotators', type=int,
            help="Number of annotators that labeled the data.")
    group.add_argument('--feature_dim', type=int, default=2048,
            help="Dimension of feature from feature extractor (Default: 2048).")
    group.add_argument('--weight_type', type=str, default='W', choices=['W', 'I'],
            help="Method of averaging weights. W: independent with feature, I: dependent with feature (Default: W).")
    group.add_argument('--bottleneck_dim', type=int, default=None,
            help="Dimension of bottleneck layer.")


def add_dn_train_args(parser):
    group = parser.add_argument_group('doctornet')
    group.add_argument('--dn_epochs', type=int, default=10,
            help="Number of epochs for training DoctorNet (Default: 10).")
    group.add_argument('--dn_lr', type=float, default=1e-3,
            help="Learning rate for training DoctorNet (Default: 1e-3).")
    group.add_argument('--dn_l1', type=float, default=0.0,
            help="L1 regularizer rate (Default: 0.0).")
    group.add_argument('--dn_l2', type=float, default=0.0,
            help="L2 regularizer(weight decay in Adam) rate (Default: 0.0).")
    group.add_argument('--dn_entropy_weight', type=float, default=0.0,
            help="Confidence penalty weight (Default: 0.0).")

def add_aw_train_args(parser):
    group = parser.add_argument_group('averaging weights')
    group.add_argument('--aw_epochs', type=int, default=10,
            help="Number of epochs for training averaging weights logits (Default: 10).")
    group.add_argument('--aw_lr', type=float, default=3e-2,
            help="Learning rate for training averaging weights logits (Default: 3e-2).")
    group.add_argument('--aw_l1', type=float, default=0.0,
            help="L1 regularizer rate (Default: 0.0).")
    group.add_argument('--aw_l2', type=float, default=0.0,
            help="L2 regularizer(weight decay in Adam) rate (Default: 0.0).")
    group.add_argument('--aw_entropy_weight', type=float, default=0.0,
            help="Confidence penalty weight (Default: 0.0).")


def add_train_args(parser):
    group = parser.add_argument_group('train')
    group.add_argument('--seed', type=int, default=7777,
            help="Random seed (Default: 7777).")
    group.add_argument('--batch_size', type=int, default=32,
            help="Number of instances in a batch (Default: 32).")
    group.add_argument('--log_interval', type=int, default=1,
            help="Log interval (Default: 1).")
    group.add_argument('--train_data', type=str,
            help="Root directory of train data.")
    group.add_argument('--valid_data', type=str,
            help="Root directory of validation data.")
    group.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda',
            help="Device going to use for training (Default: cuda).")
    group.add_argument('--save_dir', type=str, default='checkpoints/',
            help="Folder going to save model checkpoints (Default: checkpoints/).")
    group.add_argument('--log_dir', type=str, default='logs/',
            help="Folder going to save logs (Default: logs/).")

    add_dn_train_args(parser)
    add_aw_train_args(parser)


def add_test_args(parser):
    group = parser.add_argument_group('test')
    group.add_argument('--batch_size', type=int, default=32,
                       help="Number of instances in a batch (Default: 32).")
    group.add_argument('--test_data', type=str,
                       help="Root directory of test data.")
    group.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu',
                       help="Device going to use for training (Default: cpu).")
    group.add_argument('--ckpt_dir', type=str,
                       help="Directory which contains the checkpoint and args.json.")

