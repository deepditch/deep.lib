from download_roaddamage_images import main as download
from train import main as train
from ToCoreml import main as convert
from UploadModel import main as upload
import os
import re
import argparse

def get_previous_best_fmeasure(logfile):
    if not os.path.exists(logfile):
        return .57

    with open(logfile, 'r') as f:
        lines = f.read().splitlines()
        last_line = lines[-1]
        return float(last_line)

def get_best_model_in_dir(dir):
    best = ("", 0.0)

    for model in os.listdir(dir):
        if not model.endswith('.ckpt.tar'): continue
        score = float(re.findall("\d+\.\d+", model)[0])
        if score > best[1]: best = model, score

    return best

def log_new_best(logfile, score):
    with open(logfile, 'w') as f:
        f.write('%.2f \n' % score)

def main(args):
    print("Downloading")
    download(args)

    print("Training")
    try:
        pass
        train(args)
    except(e):
        print(e)
        print("Continuing")
    

    model, score = get_best_model_in_dir(args.model_dir)
    prev_best = get_previous_best_fmeasure(args.model_dir + "/best_model.txt")

    print(f"Best Model: {model}")

    if score > prev_best:
        print(f"Converting {model} to CoreML")
        args.model_file = args.model_dir + '/road_damage_model.mlmodel'
        convert(model, args.model_file)

        print(f"Uploading {model}")
        upload(args)

        log_new_best(args.model_dir + "/best_model.txt", score)
    else:
        print(f"Discarding {model}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--email')
    parser.add_argument('--password')
    parser.add_argument('--data_dir', metavar='Data Directory')
    parser.add_argument('--model_dir', metavar='Data Directory')
    parser.add_argument('--batch_size', type=int, metavar='Batch Size')

    args = parser.parse_args()

    main(args)