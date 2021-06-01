
import json

for mode in ["train", "val", "test"]:
    with open(f"{mode}.jsonl") as inputf, open(f"{mode}.source", "w") as outputfs, open(f"{mode}.target", "w") as outputft:
        for line in inputf:
            data = json.loads(line)
            texts_ = data['texts']
            roles_ = data['roles']
            predictions = data['predictions']

            roles = [x for count, x in enumerate(roles_) if predictions[count] != 2]
            texts = [x for count, x in enumerate(texts_) if predictions[count] != 2]
    
            prev_role = ""
            output = ""
            for role, text in zip(roles, texts):
                if role != prev_role:
                    output += role + " said "  + text + " "
                else:
                    output +=  text + " "
                prev_role = role
    
            output = " ".join(output.split())
            outputfs.write(output.replace("\n", " ").strip() + "\n")
    
            summary = data['summary']
            outputft.write(summary.replace("\n", " ").strip() + "\n")
