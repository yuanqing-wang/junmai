import json
import glob

def run(args):
    path = args.path
    files = glob.glob(path + "/*/*/result.json")
    results = []
    for file in files:
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                result = json.loads(line)
                results.append(result)
    results.sort(key=lambda x: x["val_loss_energy"])
    print(results[0])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="analysis")
    parser.add_argument("--path", type=str)
    args = parser.parse_args()
    run(args)

    
