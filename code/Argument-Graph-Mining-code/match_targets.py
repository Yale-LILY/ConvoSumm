
import os

modes = ["train", "val", "test"]
short2long = {}
for mode in modes:
    with open(f"/project/fas/radev/af726/convosumm/Argument-Graph-Mining/longformer/data/{mode}.target") as inputf:
        for line in inputf:
            line = line.strip()
            line_tmp = ''.join(e for e in line if e.isalnum()).lower()
            short2long[line_tmp[:300]] = line

print(len(short2long))

good = 0
#out_dir = "/project/fas/radev/af726/convosumm/Argument-Graph-Mining/ami-sep-graph-newsumm/"
out_dir = "/project/fas/radev/af726/convosumm/Argument-Graph-Mining/ami-sep-graph-newsumm-nosep/"
for mode in modes:  
    with open(f"/project/fas/radev/af726/convosumm/Argument-Graph-Mining/ami-sep-graph/{mode}.target") as inputft, open(f"/project/fas/radev/af726/convosumm/Argument-Graph-Mining/ami-sep-graph/{mode}.source") as inputfs, open(os.path.join(out_dir, f"{mode}.source"), "w") as outputfs, open(os.path.join(out_dir, f"{mode}.target"), "w") as outputft:
        for (lines, linet) in zip(inputfs, inputft):
            linet = linet.strip()
            try:
                #long_summ = short2long[line[:300].lower()]
                line_tmp = ''.join(e for e in linet if e.isalnum()).lower()
                long_summ = short2long[line_tmp[:300]]
                good += 1
            except:
                long_summ = short2long["theprojectmanagerintroducedtheupcomingprojecttotheteamandthentheteammembersparticipatedinanexerciseinwhichtheydrewtheirfavoriteanimalsanddiscussedwhytheylikedthoseparticularanimalstheprojectmanagerdiscussedtheprojectfinancesandtheteamengagedinabrainstormingsessionaboutvariousfeaturestoconsiderindesi"]
                good += 1
            #import pdb;pdb.set_trace()
            lines = lines.replace("PM -> ", "PM said ")
            lines = lines.replace("UI -> ", "UI said ")
            lines = lines.replace("ID -> ", "ID said ")
            lines = lines.replace("ME -> ", "ME said ")
            lines = lines.replace(" -> ", " ")
            lines = lines.replace("->", "")
            outputfs.write(lines.replace("\n", " ") + "\n")
            outputft.write(long_summ.replace("\n", " ") + "\n")
print(good)
