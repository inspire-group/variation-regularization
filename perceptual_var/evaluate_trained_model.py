
from typing import Dict, List
import torch
import csv
import argparse

from perceptual_advex.utilities import add_dataset_model_arguments, \
    get_dataset_model
from perceptual_advex.attacks import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Adversarial training evaluation')

    add_dataset_model_arguments(parser, include_checkpoint=True)
    parser.add_argument('--attacks', metavar='attack', type=str, nargs='+',
                        help='attack names')
    parser.add_argument('--union_source', action='store_true', default=False)
    parser.add_argument('--source', type=str, nargs='+')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='number of examples/minibatch')
    parser.add_argument('--parallel', type=int, default=1,
                        help='number of GPUs to train on')
    parser.add_argument('--num_batches', type=int, required=False,
                        help='number of batches (default entire dataset)')
    parser.add_argument('--per_example', action='store_true', default=False,
                        help='output per-example accuracy')
    parser.add_argument('--output', type=str, help='output CSV')

    args = parser.parse_args()

    dataset, model = get_dataset_model(args)
    _, val_loader = dataset.make_loaders(1, args.batch_size, only_val=True)

    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    attack_names: List[str] = args.attacks
    attacks = [eval(attack_name) for attack_name in attack_names]
    if args.source:
        sources = [eval(source_name) for source_name in args.source]
    else:
        sources = []

    # Parallelize
    if torch.cuda.is_available():
        device_ids = list(range(args.parallel))
        model = nn.DataParallel(model, device_ids)
        attacks = [nn.DataParallel(attack, device_ids) for attack in attacks]

    source_correct: Dict[str, List[torch.Tensor]] = \
        {source_name: [] for source_name in args.source}
    batches_correct: Dict[str, List[torch.Tensor]] = \
        {attack_name: [] for attack_name in attack_names}

    union_correct_all = []
    for batch_index, (inputs, labels) in enumerate(val_loader):
        print(f'BATCH {batch_index:05d}')

        if (
            args.num_batches is not None and
            batch_index >= args.num_batches
        ):
            break

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        union_correct = 1
        union_source = 1
        for source_name, source in zip(args.source, sources):
            adv_inputs = source(inputs, labels)
            with torch.no_grad():
                adv_logits = model(adv_inputs)
            batch_correct = (adv_logits.argmax(1) == labels).detach()
            union_correct *= batch_correct
            union_source *= batch_correct
            batch_accuracy = batch_correct.float().mean().item()
            print(f'ATTACK {source_name}',
                  f'accuracy = {batch_accuracy * 100:.2f}',
                  sep='\t')
            source_correct[source_name].append(batch_correct)
        if not args.union_source:
            union_correct = 1

        for attack_name, attack in zip(attack_names, attacks):
            adv_inputs = attack(inputs, labels)
            with torch.no_grad():
                adv_logits = model(adv_inputs)
            batch_correct = (adv_logits.argmax(1) == labels).detach()
            if args.union_source:
                batch_correct *= (union_source > 0)
            union_correct *= batch_correct
            batch_accuracy = batch_correct.float().mean().item()
            print(f'ATTACK {attack_name}',
                  f'accuracy = {batch_accuracy * 100:.2f}',
                  sep='\t')
            batches_correct[attack_name].append(batch_correct)
        union_correct_all.append(union_correct)
    print('OVERALL')
    accuracies = []
    attacks_correct: Dict[str, torch.Tensor] = {}
    print('SOURCE')
    for source_name in args.source:
        accuracy = torch.cat(source_correct[source_name]).float().mean().item()
        print(f'ATTACK {source_name}',
              f'accuracy = {accuracy * 100:.2f}',
              sep='\t')

    print('TARGET')
    for attack_name in attack_names:
        attacks_correct[attack_name] = torch.cat(batches_correct[attack_name])
        accuracy = attacks_correct[attack_name].float().mean().item()
        print(f'ATTACK {attack_name}',
              f'accuracy = {accuracy * 100:.2f}',
              sep='\t')
        accuracies.append(accuracy)
    union_correct_all = torch.cat(union_correct_all)
    union_acc = union_correct_all.float().mean().item()
    print('UNION ',
          f'accuracy = {union_acc * 100:.2f}',
          sep= '\t')
    if args.output:
        with open(args.output, 'w') as out_file:
            out_csv = csv.writer(out_file)
            out_csv.writerow(attack_names)
            if args.per_example:
                for example_correct in zip(*[
                    attacks_correct[attack_name] for attack_name in attack_names
                ]):
                    out_csv.writerow(
                        [int(attack_correct.item()) for attack_correct
                         in example_correct])
            out_csv.writerow(accuracies)
