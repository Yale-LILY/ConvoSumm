import json

modes = ["train", "test", "val"]
for mode in modes:
    with open(f"corpus/{mode}.json") as inputf, open(f"corpus/{mode}.source", "w") as outputf, open(f"corpus/{mode}.target", "w") as outputft:
        data = json.load(inputf)
        for count, example in enumerate(data):
            print(count)
            utterances = example['dialogue'].split("\r\n")
            utterances = [x.strip().replace("\n", " ") for x in utterances]
            utterances_str = " </s> ".join(utterances)
            outputf.write(utterances_str + "\n")
            summary = example['summary']
            outputft.write(summary.strip().replace("\n", " ") + "\n")
