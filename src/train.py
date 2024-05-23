import csv
import argparse
import warnings
import torch.optim as optim

from utils import *
from models import *
from dataloader import *
from train_and_eval import *

warnings.filterwarnings("ignore", category=Warning)


def main():
    g, labels, idx_train, idx_val, idx_test = load_data(param['dataset'], seed=param['seed'])
    feats = g.ndata["feat"].to(device)
    labels = labels.to(device)
    param['feat_dim'] = g.ndata["feat"].shape[1]
    param['label_dim'] = labels.int().max().item() + 1

    # feature masking
    if param["feat_missing_rate"] > 0:
        feats[idx_test] = mask_features(feats[idx_test], param["feat_missing_rate"])

    # training data split
    if param["label_rate"] > 0 and param["exp_setting"] == "tran":
        random_state = np.random.RandomState(param['seed'])
        remaining_size = labels.shape[0] - idx_val.shape[0] - idx_test.shape[0]
        max_label_rate = float(remaining_size / labels.shape[0])
        assert max_label_rate >= param["label_rate"]
        train_examples_per_class = int(param["label_rate"] * labels.shape[0] / param['label_dim'])
        idx_train, idx_val, idx_test = get_train_val_test_split(
            random_state=random_state,
            labels=F.one_hot(labels),
            train_examples_per_class=train_examples_per_class,
            val_size=idx_val.shape[0],
            test_size=idx_test.shape[0])

    folder_name = "none"
    if param['challenge'] == "feat":
        folder_name = f"{param['feat_missing_rate'] * 100}%"
    elif param['challenge'] == "label":
        folder_name = f"{param['label_rate'] * 100}%"

    if param['exp_setting'] == "tran":
        output_dir = Path.cwd().joinpath("../outputs", "transductive", param['dataset'], param["challenge"],
                                         folder_name,
                                         f"{param['teacher']}_{param['student']}", f"seed_{param['seed']}")
        indices = (idx_train, idx_val, idx_test)
    elif param['exp_setting'] == "ind":
        output_dir = Path.cwd().joinpath("../outputs", "inductive", param['dataset'], param["challenge"],
                                         folder_name,
                                         f"{param['teacher']}_{param['student']}", f"seed_{param['seed']}")
        indices = graph_split(idx_train, idx_val, idx_test, labels, param)
    else:
        raise ValueError(f"Unknown experiment setting! {param['exp_setting']}")

    check_writable(output_dir, overwrite=False)

    criterion_l = torch.nn.NLLLoss()
    criterion_t = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    evaluator = get_evaluator(param["dataset"])

    if param['ablation_mode'] == 0:
        #  Teacher training -> Student Training
        model_t = Model(param, model_type='teacher').to(device)
        optimizer_t = optim.Adam(model_t.parameters(), lr=float(1e-2), weight_decay=float(param["weight_decay"]))
        out_t, _, test_teacher, _, state_teacher = train_teacher(param, model_t, g, feats, labels, indices, criterion_l,
                                                                 evaluator,
                                                                 optimizer_t)

        model_s = Model(param, model_type='student').to(device)
        optimizer_s = optim.Adam(model_s.parameters(), lr=float(param["learning_rate"]),
                                 weight_decay=float(param["weight_decay"]))
        _, test_acc, test_val, test_best, _ = train_student(param, model_s, g, feats, labels, out_t, indices,
                                                            criterion_l, criterion_t, evaluator, optimizer_s)

        return test_teacher, test_acc, test_val, test_best


    elif param['ablation_mode'] == 1:
        #  Teacher training only
        model_t = Model(param, model_type='teacher').to(device)
        optimizer_t = optim.Adam(model_t.parameters(), lr=float(1e-2), weight_decay=float(param["weight_decay"]))
        out_t, test_acc, test_val, test_best, state_t = train_teacher(param, model_t, g, feats, labels, indices,
                                                                      criterion_l, evaluator, optimizer_t)

        np.savez(output_dir.joinpath("out_teacher"), out_t.detach().cpu().numpy())
        torch.save(state_t, output_dir.joinpath("model_teacher"))

        return test_val, test_acc, test_val, test_best


    elif param['ablation_mode'] == 2:
        #  Teacher Loading -> Student Training
        if not os.path.exists(output_dir.joinpath("out_teacher.npz")):
            model_t = Model(param, model_type='teacher').to(device)
            optimizer_t = optim.Adam(model_t.parameters(), lr=float(1e-2), weight_decay=float(param["weight_decay"]))
            out_t, test_acc, test_val, test_best, state_t = train_teacher(param, model_t, g, feats, labels, indices,
                                                                          criterion_l, evaluator, optimizer_t)

            np.savez(output_dir.joinpath("out_teacher"), out_t.detach().cpu().numpy())
            torch.save(state_t, output_dir.joinpath("model_teacher"))

        out_t = load_out_t(output_dir).to(device)
        model_t = Model(param, model_type='teacher').to(device)
        state_t = torch.load(output_dir.joinpath("model_teacher"))
        model_t.load_state_dict(state_t)

        if param['exp_setting'] == 'tran':
            test_teacher = evaluator(out_t[indices[2]].log_softmax(dim=1), labels[indices[2]])
        else:
            test_teacher = evaluator(out_t[indices[4]].log_softmax(dim=1), labels[indices[4]])

        model_s = Model(param, model_type='student').to(device)
        optimizer_s = optim.Adam(model_s.parameters(), lr=float(param["learning_rate"]),
                                 weight_decay=float(param["weight_decay"]))
        out_s, test_acc, test_val, test_best, state_s = train_student(param, model_s, g, feats, labels, out_t, indices,
                                                                      criterion_l, criterion_t, evaluator, optimizer_s
                                                                      )

        np.savez(output_dir.joinpath("out_student"), out_s.detach().cpu().numpy())
        torch.save(state_s, output_dir.joinpath("model_student"))

        return test_teacher, test_acc, test_val, test_best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch DGL implementation")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--teacher", type=str, default="GCN")
    parser.add_argument("--student", type=str, default="AdaGMLP")
    parser.add_argument("--split_rate", type=float, default=0.1)
    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--hidden_dim_s", type=int, default=64)
    parser.add_argument("--dropout_t", type=float, default=0.8)
    parser.add_argument("--dropout_s", type=float, default=0.6)

    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--max_epoch", type=int, default=2000)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_mode", type=int, default=0)
    parser.add_argument("--data_mode", type=int, default=0)
    parser.add_argument("--ablation_mode", type=int, default=2)

    # AdaGMLP specific
    parser.add_argument("--lamb", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--selective", type=int, default=1)
    parser.add_argument("--lamb_a", type=float, default=0.5)
    parser.add_argument("--aug_feat_missing_rate", type=float, default=0.)

    # different setting
    parser.add_argument("--exp_setting", type=int, default=0)
    parser.add_argument("--challenge", type=int, default=-1)
    parser.add_argument("--label_rate", type=float, default=0.)
    parser.add_argument("--feat_missing_rate", type=float, default=0.)


    args = parser.parse_args()
    param = args.__dict__

    if param['dataset'] == 'ogbn-arxiv':
        param['norm_type'] = 'batch'
    else:
        param['norm_type'] = 'none'

    if param['exp_setting'] == 0:
        param['exp_setting'] = 'tran'
    else:
        param['exp_setting'] = 'ind'

    if args.challenge == 0:
        param['challenge'] = 'label'
    elif args.challenge == 1:
        param['challenge'] = 'feat'
    else:
        param['challenge'] = 'public'

    device = get_device(param)

    if param['save_mode'] == 0:
        set_seed(param['seed'])
        test_teacher, test_acc, test_val, test_best = main()

    else:
        test_acc_list = []
        test_val_list = []
        test_best_list = []
        test_teacher_list = []

        for seed in range(5):
            param['seed'] += seed*10
            set_seed(param['seed'])
            test_teacher, test_acc, test_val, test_best = main()

            test_acc_list.append(test_acc)
            test_val_list.append(test_val)
            test_best_list.append(test_best)
            test_teacher_list.append(test_teacher)

    column = None
    if not os.path.exists(f'../PerformMetrics_{param["dataset"]}.csv'):
        column = ["Time"]
        for v, _ in param.items():
            column.append(v)
        column.append("test_acc_list")
        column.append("test_val_list")
        column.append("test_best_list")
        column.append("test_teacher_list")
        column.append("mean_test_acc")
        column.append("mean_tese_val")
        column.append("mean_test_best")
        column.append("mean_test_teacher")
        column.append("std_test_acc")
        column.append("std_test_val")
        column.append("std_test_best")
        column.append("std_test_teacher")

    outFile = open(f'../PerformMetrics_{param["dataset"]}.csv', 'a+', newline='')
    writer = csv.writer(outFile, dialect='excel')
    if column is not None:
        writer.writerow(column)

    results = [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())]
    for v, k in param.items():
        results.append(k)

    if param['save_mode'] == 0:
        results.append(str(test_acc))
        results.append(str(test_val))
        results.append(str(test_best))
        results.append(str(test_teacher))

    else:
        results.append(str(test_acc_list))
        results.append(str(test_val_list))
        results.append(str(test_best_list))
        results.append(str(test_teacher_list))
        results.append(str(np.mean(test_acc_list)))
        results.append(str(np.mean(test_val_list)))
        results.append(str(np.mean(test_best_list)))
        results.append(str(np.mean(test_teacher_list)))
        results.append(str(np.std(test_acc_list)))
        results.append(str(np.std(test_val_list)))
        results.append(str(np.std(test_best_list)))
        results.append(str(np.std(test_teacher_list)))
    writer.writerow(results)
