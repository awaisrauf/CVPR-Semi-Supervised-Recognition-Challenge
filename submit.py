from dataloader.data import get_test_data
import torch
from utils.args import args_for_train_tl
import os
from functions import make_final_submission

# get all the input variables
args = args_for_train_tl()
# ToDo: load args from experiemnt_ags file from args.resume
# https://stackoverflow.com/questions/28348117/using-argparse-and-json-together
# if args.load_json:
#     with open(args.load_json, 'rt') as f:
#         t_args = argparse.Namespace()
#         t_args.__dict__.update(json.load(f))
#         args = parser.parse_args(namespace=t_args)
state = {k: v for k, v in args._get_kwargs()}


############################################
# Create and Load Model
############################################
from models.models import get_model
model, image_size = get_model(args, num_classes=200, train=False, use_pretrained=False)
model = model.cuda()
model = torch.nn.DataParallel(model)


####
test_loader = get_test_data(batch_size=args.train_batch, image_size=image_size)


# Resume
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
    args.checkpoint = os.path.dirname(args.resume)
    checkpoint = torch.load(args.resume)
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])


print('==> Creating Submission..')
preds = []
Idxs = []
for i, (Idx, images) in enumerate(test_loader):
    print(".", end='')
    images = images.cuda()
    with torch.no_grad():
        y_preds = model(images)

        # print(y_preds.shape)
    bs = y_preds.shape[0]
    temp_pred = y_preds.to('cpu').numpy()
    for example in range(bs):
        preds.append(temp_pred[example, :])
        Idxs.append(Idx.to('cpu').numpy()[example])

submission_file_path = args.submission_file
make_final_submission(Idxs, preds, submission_file_path)


