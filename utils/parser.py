import argparse
parser = argparse.ArgumentParser(description='PyTorch View Adaptive LSTM Model')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=400,
                    help='upper epoch limit')     
parser.add_argument('--batch_size', type=int, default=256,
                    help='batch size')     
parser.add_argument('--hidden_unit', type=int, default=100,
                    help='hidden units')         
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--gradnorm', type=float, default=1.0)
parser.add_argument('--trans', action='store_true',
                    help='use trans branch')
parser.add_argument('--rota', action='store_true',
                    help='use rota branch')
#parser.add_argument('--gradclip', type=bool, default = True,
#                    help='use gradient clipping')
parser.add_argument('--gradclip', dest='gradclip', action='store_true')
parser.add_argument('--no-gradclip', dest='gradclip', action='store_false')
parser.set_defaults(gradclip=True)

parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--log_interval', type=int, default=200, metavar='N',
                    help='report interval')                     
parser.add_argument('--checkpoint_folder', type=str, default='/home2/wuxiao/va_lstm/va_lstm/experiments/')
parser.add_argument('--loadfrompath', action='store_true',
                    help='use model from savepath')
parser.add_argument('--version', type=str, default='')
parser.add_argument('--dataset_dir', default='/home2/wuxiao/va_lstm/va_lstm/dataset/', help="root directory for all the datasets")
parser.add_argument('--dataset_name', default='NTU-RGB-D-CV', help="dataset name ") # NTU-RGB-D-CS,NTU-RGB-D-CV
parser.add_argument('--num_workers', type = int, default=4)
parser.add_argument('--gpu_id', type = int, default = -1)
parser.add_argument('--debug', action='store_true',
                    help='check properties of input data')
parser.add_argument('--stages', nargs='*', default=['2'], help='stages to go through')
parser.add_argument('--epochs_per_stage', nargs='*', default = ['150'], help='epochs per stage')
#parser.add_argument('--lr_decay', type = bool, default = True, help='using scheduler to decay lr')
parser.add_argument('--lr_decay', dest='lr_decay', action='store_true')
parser.add_argument('--no-lr_decay', dest='lr_decay', action='store_false')
parser.set_defaults(lr_decay=True)

parser.add_argument('--optim_mode', type=str, default='max')
parser.add_argument('--visualize', action='store_true')
#parser.add_argument('--pengfei', type = bool, default = True)
parser.add_argument('--pengfei', dest='pengfei', action='store_true')
parser.add_argument('--no-pengfei', dest='pengfei', action='store_false')
parser.set_defaults(pengfei=True)

#parser.add_argument('--mean_after_fc', type = bool, default = True)
parser.add_argument('--mean_after_fc', dest='mean_after_fc', action='store_true')
parser.add_argument('--no-mean_after_fc', dest='mean_after_fc', action='store_false')
parser.set_defaults(mean_after_fc=True)

parser.add_argument('--separated_lstm', action = 'store_true')
parser.add_argument('--pre_proc', type = str, default = 'strans', help='strans, ftrans, raw')
parser.add_argument('--backbone', type = str, default = 'lstm', help = 'lstm, cnn')
#parser.add_argument('--downsampling', default = False, action = 'store_true')
#parser.add_argument('--mask_person', default = False, action = 'store_true')
#parser.add_argument('--mask_frame', default = False, action = 'store_true')
parser.add_argument('--downsampling', dest='downsampling', action='store_true')
parser.add_argument('--no-downsampling', dest='downsampling', action='store_false')
parser.set_defaults(downsampling=False)

parser.add_argument('--mask_person', dest='mask_person', action='store_true')
parser.add_argument('--no-mask_person', dest='mask_person', action='store_false')
parser.set_defaults(mask_person=True)

parser.add_argument('--mask_frame', dest='mask_frame', action='store_true')
parser.add_argument('--no-mask_frame', dest='mask_frame', action='store_false')
parser.set_defaults(mask_frame=True)

parser.add_argument('--more_hidden', dest='more_hidden', action='store_true')
parser.add_argument('--no-more_hidden', dest='more_hidden', action='store_false')
parser.set_defaults(more_hidden=False)

