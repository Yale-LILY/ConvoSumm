
from pathlib import Path
import recap_argument_graph as ag
import networkx as nx
import re
import sys

folder = sys.argv[1]

with open(f"{folder}/train.graph", "w") as outputftrain, open(f"{folder}/val.graph", "w") as outputfval, open(f"{folder}/test.graph", "w") as outputftest:
    for split in ["train", "dev", "test"]:
        print(split)
        for i in range(300):
            try:
                json_file = f"{folder}/CONVERSATION-{split}.source.graph_remove{i}.json"
                graph = ag.Graph.from_json(open(json_file))

                gnx = graph.to_nx()
                gdict = graph.to_dict()
                id2text = {node['id']: node['text'] for node in gdict['nodes']}
                ids = [node['id'] for node in gdict['nodes']]
                texts = [node['text'] for node in gdict['nodes']]

                try:
                    startid = ids[texts.index('CONVERSATION')]
                except:
                    startid = ids[texts.index('Conversation')]

                gedges = list(nx.dfs_edges(gnx, startid))
                done = []
                output_text = ""
                for edge in gedges:
                    n1, n2 = edge
                    n1_text = id2text[n1]
                    if n1 in done:
                        n1_text = ""
                    else:
                        done.append(n1)
                    n2_text = id2text[n2]
                    if n2 in done:
                        n2_text = ""
                    else:
                        done.append(n2)
                    if n1_text == "Conversation" and n2_text == "Issue":
                        output_text += "     -> Issue " 
                        continue
                    elif n1_text == "" and n2_text == "Issue":
                        output_text += "     -> Issue " 
                        continue
                    elif n1_text == "Conversation":
                        output_text += "     -> " + n2_text + " "
                    elif n1_text == "Issue": 
                        output_text += "Issue -> " + n2_text + " "
                    elif n1_text == "Default Inference":
                        output_text += " " + n2_text + " "
                    elif n2_text == "Default Inference":
                        output_text += " " + n1_text + " "
                    else:
                        output_text += " " + n1_text + "     -> " + n2_text + " "
                output_text = output_text
                line = output_text

                line = line.replace("CONVERSATION -> ", "")
                line = line.replace("Conversation ->", "")
                line = line.replace("Conversation ", "")
                #line = line.replace(" -> ", "")
                #line = line.replace(" ->", "")
                line = line.replace(" <n> ", " ")
                line = line.replace(" <e> ", " ")
                line = line.replace("<s> ", "")
                line =  line.replace("\n", " ")
                #line = re.sub(' +', ' ', line).strip()
                line = line.replace("Conversation PM", "PM").replace("Conversation ME", "ME").replace("Conversation UI", "UI").replace("Conversation ID", "ID")
                #line = re.sub(' +', ' ', line).strip()

                #line = cur_info + " " + line
                line = line.replace("-> ->", "->")
                

                line = line.replace(" <n> ", " ")
                line = line.replace(" <e> ", " ")
                line = line.replace("<s> ", "")
                line =  line.replace("\n", " ")
                line = re.sub(' +', ' ', line).strip()

                if split == "train":
                    outputf = outputftrain
                elif split == "dev":
                    outputf = outputfval
                else:
                    outputf = outputftest

                outputf.write(line + "\n")
            except Exception as e:
                print(e)
                continue
